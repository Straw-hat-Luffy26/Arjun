# Third-Party Notices

ARJUN incorporates third-party software. This file records those components and
their licences, as their licences require.

## OpenClaw

A pruned copy of [openclaw/openclaw](https://github.com/openclaw/openclaw) is
vendored at `agent-runtime/vendor/openclaw/`, pinned to commit
`ed56f3c001a6d18427bc399493edabc7166233bd`.

Portions of that copy have been modified: cloud provider adapters, their SDK
dependencies, cloud transport layers, chat channels and the plugin host shim are
removed, and two registries are narrowed to a single OpenAI-compatible transport.
`agent-runtime/vendor/README.md` records every change. Modified files carry an
"ARJUN sovereign build" comment at the point of change.

The upstream licence is reproduced verbatim at
`agent-runtime/vendor/openclaw/LICENSE`:

> MIT License
>
> Copyright (c) 2026 OpenClaw Foundation
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

OpenClaw records its own incorporated third-party code in its
`THIRD_PARTY_NOTICES.md`; that file is not reproduced here because the adapted
code it covers (Pi/pi-mono) is not part of the vendored subset.

## Agent runtime npm dependencies

The Node agent runtime resolves 50 third-party packages at runtime. The
authoritative, versioned list is `agent-runtime/package-lock.json`; regenerate a
readable inventory with:

```bash
npm --prefix agent-runtime ls --all --omit=dev
```

Direct runtime dependencies:

| Package | Version | Licence | Used for |
|---|---|---|---|
| `openai` | 7.5.0 | Apache-2.0 | OpenAI-compatible client, pointed at local llama-server / vLLM / SGLang endpoints |
| `typebox` | 1.3.15 | MIT | Tool and message schema validation |
| `zod` | 4.4.3 | MIT | Model catalogue schema validation |
| `ipaddr.js` | 2.5.0 | MIT | Network policy IP classification |
| `partial-json` | 0.1.7 | MIT | Incremental parsing of streamed tool-call arguments |
| `libphonenumber-js` | 1.13.11 | MIT | Pulled by `normalization-core`; not used by ARJUN |
| `mdast-util-from-markdown` | 2.0.3 | MIT | Reasoning-tag partitioning |
| `mdast-util-gfm-table` | 2.0.0 | MIT | Reasoning-tag partitioning |
| `micromark-extension-gfm-table` | 2.1.1 | MIT | Reasoning-tag partitioning |

The remaining packages are transitive dependencies of the three markdown parsers,
predominantly the `micromark` and `unist` families, all MIT.

No cloud model provider SDK is present. `agent-runtime/scripts/audit-vendor.mjs`
enforces that.

## Desktop surface npm dependencies

The React surface in `src/` is bundled into the application at build time by
Vite. Nothing here is fetched at runtime; the egress gate
(`scripts/check-egress.mjs`) still holds, because none of these packages
constructs an outbound client and the one module permitted to do so is
unchanged.

The authoritative, versioned list is `package-lock.json`, and every resolved
component — 457 npm entries — is inventoried in `evidence/sbom.cdx.json`.

Direct dependencies:

| Package | Version | Licence | Used for |
|---|---|---|---|
| `react` | 19.2.8 | MIT | The surface |
| `react-dom` | 19.2.8 | MIT | The surface |
| `react-router-dom` | 7.18.2 | MIT | Screen routing |
| `@tauri-apps/api` | 2.11.1 | Apache-2.0 OR MIT | IPC to the Rust process |
| `@tauri-apps/plugin-log` | 2.9.0 | MIT OR Apache-2.0 | Log bridge |
| `@tauri-apps/plugin-opener` | 2.5.4 | MIT OR Apache-2.0 | Opening a produced file in the file manager |
| `@tauri-apps/plugin-sql` | 2.4.0 | MIT OR Apache-2.0 | Local conversation store |
| `lucide-react` | 1.28.0 | ISC | Icons |
| `three` | 0.185.1 | MIT | The orb on the chat surface |
| `@react-three/fiber` | 9.7.0 | MIT | React binding for the above |
| `mermaid` | 11.17.2 | MIT | Laying out and drawing `mermaid` fences in an assistant reply |
| `pdfjs-dist` | 6.3.289 | Apache-2.0 | Rendering a produced PDF in the chat |
| `@tauri-apps/plugin-dialog` | 2.6.0 | MIT OR Apache-2.0 | Opening the platform save dialog when a file is exported |

### pdf.js, and the save dialog

Two dependencies were added together, for one capability: reading a produced
file and getting a copy of it out of the application.

- **`pdfjs-dist` 6.3.289 — Apache-2.0.** Mozilla's PDF renderer, the one Firefox
  ships. `artifact_preview.rs` answers a PDF with no body on purpose — there is
  no PDF reader in the Rust process — which left the one lane that produces a
  *report* as the one lane whose output could not be read without leaving the
  application. Loaded by dynamic `import()` in
  `src/components/chat/PdfView.tsx`, so it is code-split and a session that
  never opens a PDF never loads it.

  It parses and rasterises on a web worker. That worker is emitted by the
  bundler as an asset of this application and referenced by a same-origin URL,
  which is what keeps it inside `script-src 'self'`; the CDN URL the library's
  own documentation suggests would be refused by the content security policy,
  and would be an egress this product does not permit in any case.

- **`@tauri-apps/plugin-dialog` 2.6.0 (`MIT OR Apache-2.0`), with
  `tauri-plugin-dialog` 2.6.0 on the Rust side.** Pinned exactly, and to the
  same version on both sides. A Tauri plugin is two halves of one thing talking
  over IPC, and the CLI refuses to start on a major/minor mismatch — `2.7.3`
  against `2.6.0` was caught that way rather than at the moment somebody
  clicked Save as…. Opens the operating system's
  own save dialog so a person can choose where their file goes. The capability
  in `src-tauri/capabilities/default.json` grants **`dialog:allow-save` only** —
  not `dialog:allow-open`, which would let the surface ask for a file to *read*,
  a reach this application has no use for and should not hold.

  The Rust plugin brings the larger share of the new `cargo` entries in
  `evidence/sbom.cdx.json`; most are the Windows API bindings the platform
  picker needs.

### Mermaid, and what it brings with it

`mermaid` is the largest of these and the only one with a substantial
dependency tree of its own: it resolves 113 packages. It is loaded by dynamic
`import()` in `src/components/chat/MermaidGraph.tsx`, so it is code-split into
its own chunks and a session that never shows a diagram never loads it.

Two of those packages are worth naming rather than leaving in the count:

- **`dompurify` 3.4.15 — `MPL-2.0 OR Apache-2.0`.** A dual licence; ARJUN takes
  it under Apache-2.0, which carries no file-level copyleft obligation. It is
  what Mermaid sanitises its own output with, before
  `sanitizeDiagramSvg` checks it again against an allowlist.
- **`khroma` 2.1.0 — MIT, but not machine-readable.** The package omits the
  `license` field from its `package.json`, so `evidence/sbom.cdx.json` records
  it as unstated. The licence is not actually missing: `node_modules/khroma/license`
  contains the MIT text, "Copyright (c) 2019-present Fabio Spampinato, Andrew
  Maney". Recorded here because an unstated licence in an SBOM shipped as
  evidence should be answered rather than left as a blank for a reader to
  wonder about.

Everything else in that tree is MIT or ISC — the `d3`, `unist`, `micromark` and
`langium` families, `cytoscape`, `katex`, `dagre-d3-es`, `roughjs` and `marked`.

The `MPL-2.0` and `CC-BY-4.0` components in the SBOM (`lightningcss` and its
platform binaries, `caniuse-lite`) are build-time tooling pulled in by Tailwind
and Vite. They are not part of the shipped bundle and predate this dependency.

## Python document sidecar

`sidecars/document_sidecar/` uses Docling and pypdf. Their licences apply as
distributed; record versions in the deployment bundle's SBOM.
