/**
 * A bidirectional JSON-RPC peer over a byte channel.
 *
 * Both ends open requests, so this is a peer rather than a client or a server.
 * The Rust core asks this runtime to start a run; mid-run the runtime asks the
 * core whether a tool call may proceed and then to perform it. Those two
 * conversations interleave over the same pipe, which is why replies are
 * correlated by id rather than by order.
 *
 * Transport is injected rather than reached for, so the whole thing is testable
 * against a pair of in-memory queues with no child process involved.
 */

import {
  type ErrorFrame,
  type Frame,
  type RequestFrame,
  type ResultFrame,
  ErrorCode,
  FrameDecoder,
  encodeFrame,
  isError,
  isNotification,
  isRequest,
  isResult,
} from "./protocol.js";

/** The byte channel a peer talks over. */
export interface PeerTransport {
  /** Called for each outbound line. Must deliver whole lines, in order. */
  write(line: string): void;
  /** Registers the inbound sink. Called once, during construction. */
  onData(sink: (chunk: string) => void): void;
  /** Registers the end-of-stream sink. */
  onClose(sink: () => void): void;
}

export type RequestHandler = (params: unknown) => Promise<unknown> | unknown;
export type NotificationHandler = (params: unknown) => void;

/** Raised when the far side replies with an error frame. */
export class RpcError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "RpcError";
    this.code = code;
  }
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

export class RpcPeer {
  readonly #transport: PeerTransport;
  readonly #decoder = new FrameDecoder();
  readonly #pending = new Map<string, Pending>();
  readonly #requestHandlers = new Map<string, RequestHandler>();
  readonly #notificationHandlers = new Map<string, NotificationHandler>();
  /**
   * Ids are unique per originating peer only, so both ends may use "1". They are
   * never compared across directions -- a reply is matched against the map of
   * requests *this* peer opened.
   */
  #nextId = 1;
  #closed = false;
  #onFatal: (error: unknown) => void = () => {};

  constructor(transport: PeerTransport) {
    this.#transport = transport;
    transport.onData((chunk) => this.#ingest(chunk));
    transport.onClose(() => this.close(new RpcError(ErrorCode.PeerClosed, "Peer closed the channel")));
  }

  /**
   * Called when the channel desynchronises.
   *
   * Separate from per-request rejection because a malformed frame is not one
   * request failing, it is the connection no longer being trustworthy. The
   * caller decides whether that means exit.
   */
  onFatal(sink: (error: unknown) => void): void {
    this.#onFatal = sink;
  }

  handle(method: string, handler: RequestHandler): void {
    this.#requestHandlers.set(method, handler);
  }

  onNotification(method: string, handler: NotificationHandler): void {
    this.#notificationHandlers.set(method, handler);
  }

  /** Opens a request and resolves with the far side's result. */
  request(method: string, params?: unknown): Promise<unknown> {
    if (this.#closed) {
      return Promise.reject(new RpcError(ErrorCode.PeerClosed, `Cannot call ${method}: channel closed`));
    }
    const id = String(this.#nextId++);
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      try {
        this.#transport.write(encodeFrame({ id, method, params }));
      } catch (cause) {
        this.#pending.delete(id);
        reject(cause);
      }
    });
  }

  /** Sends a one-way message. Never throws for a closed channel: events are best-effort. */
  notify(method: string, params?: unknown): void {
    if (this.#closed) return;
    try {
      this.#transport.write(encodeFrame({ method, params }));
    } catch {
      // A failed notification must not take down a run. The far side losing an
      // event is a display problem; losing the run is a work problem.
    }
  }

  /** Rejects everything outstanding and stops accepting new work. */
  close(reason?: unknown): void {
    if (this.#closed) return;
    this.#closed = true;
    const error = reason ?? new RpcError(ErrorCode.PeerClosed, "Channel closed");
    for (const [, pending] of this.#pending) {
      pending.reject(error);
    }
    this.#pending.clear();
  }

  get closed(): boolean {
    return this.#closed;
  }

  #ingest(chunk: string): void {
    let frames: Frame[];
    try {
      frames = this.#decoder.push(chunk);
    } catch (error) {
      this.#onFatal(error);
      this.close(error);
      return;
    }
    for (const frame of frames) {
      // `.catch`, not `void`.
      //
      // `#serve` catches what a handler throws, but it calls `#respond` and
      // `#reply` *outside* that try — so a write that fails (EPIPE on a closed
      // stdout) or a result that will not serialise (a circular reference, a
      // BigInt) escaped as an unhandled rejection. A notification handler that
      // throws did the same. Node's default for an unhandled rejection is to
      // terminate the process, so the whole runtime died and the core saw the
      // pipe close with no explanation.
      //
      // Reported through the fatal path instead, which is the one that already
      // knows how to close down and tell the core why.
      this.#dispatch(frame).catch((error: unknown) => {
        this.#onFatal(error);
        this.close(error);
      });
    }
  }

  /**
   * Requests are tested first on purpose.
   *
   * The four frame shapes overlap structurally -- a request and a result both
   * carry `id` -- so eliminating the others first narrows the union to nothing
   * and the compiler stops believing a request can still be here. Asking the
   * most specific question first keeps the narrowing honest.
   */
  async #dispatch(frame: Frame): Promise<void> {
    if (isRequest(frame)) {
      await this.#serve(frame);
      return;
    }
    if (isResult(frame) || isError(frame)) {
      this.#settle(frame);
      return;
    }
    if (isNotification(frame)) {
      this.#notificationHandlers.get(frame.method)?.(frame.params);
    }
  }

  #settle(frame: ResultFrame | ErrorFrame): void {
    const pending = this.#pending.get(frame.id);
    // An unmatched reply means the far side is answering something we did not
    // ask. Dropped rather than fatal: the likeliest cause is a reply arriving
    // after a local timeout, and killing the run over it helps nobody.
    if (!pending) return;
    this.#pending.delete(frame.id);
    if (isResult(frame)) {
      pending.resolve(frame.result);
    } else {
      pending.reject(new RpcError(frame.error.code, frame.error.message));
    }
  }

  async #serve(frame: RequestFrame): Promise<void> {
    const handler = this.#requestHandlers.get(frame.method);
    if (!handler) {
      this.#reply(frame.id, {
        code: ErrorCode.UnknownMethod,
        message: `No handler for ${frame.method}`,
      });
      return;
    }
    try {
      const result = await handler(frame.params);
      this.#respond(frame.id, result);
    } catch (error) {
      this.#reply(frame.id, {
        code: error instanceof RpcError ? error.code : ErrorCode.Internal,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  #respond(id: string, result: unknown): void {
    if (this.#closed) return;
    this.#transport.write(encodeFrame({ id, result: result ?? null }));
  }

  #reply(id: string, error: { code: string; message: string }): void {
    if (this.#closed) return;
    this.#transport.write(encodeFrame({ id, error }));
  }
}
