/**
 * The Document & Vision Analyst's administration (plan P06).
 *
 * What reads pages on this machine — the OCR model's exact files and the page
 * analyser — and, for every model with an image projector, whether it is
 * vision-ready. A model is vision-ready only after an image probe on this
 * machine read a random token back; a projector on disk is not enough, and
 * nothing here can make it so without that probe.
 */

import { getBackendService } from './api';

/** Exactly what reads a scanned page. */
export interface OcrIdentity {
  modelId: string;
  weightsFile: string;
  weightsSha256: string | null;
  weightsBytes: number;
  projectorFile: string | null;
  projectorBytes: number | null;
  detent: string;
  requestFingerprint: string;
  prompt: string;
}

/** One image probe, pass or fail. */
export interface ReadinessRecord {
  modelId: string;
  weightsFile: string;
  weightsBytes: number;
  weightsSha256: string | null;
  projectorFile: string;
  projectorBytes: number;
  projectorSha256: string;
  probeToken: string;
  prompt: string;
  answerExcerpt: string;
  passed: boolean;
  reason: string;
  elapsedMs: number;
  at: string;
  transport: string;
  actor: string;
}

export interface VisionCandidate {
  modelId: string;
  name: string;
  projector: string | null;
  ready: boolean;
  reason: string | null;
  lastProbe: ReadinessRecord | null;
}

export interface ProjectorBinding {
  modelId: string;
  projector: string;
  projectorBytes: number;
  projectorType: string | null;
  projectionDim: number;
  modelArchitecture: string;
  modelEmbedding: number;
  verifiedAt: string;
  verifiedBy: string;
}

export interface DocumentAnalystStatus {
  ocr: OcrIdentity | null;
  ocrUnavailable: string | null;
  ocrTransport: string;
  pageAnalyser: string | null;
  pageAnalyserUnavailable: string | null;
  vision: VisionCandidate[];
  interpreter: string | null;
  projectorBindings: ProjectorBinding[];
}

export interface ProjectorBindOutcome {
  binding: ProjectorBinding;
  restartRequired: boolean;
  visionReady: boolean;
}

export const documentAnalystService = {
  /** What reads pages here, and why anything that cannot, cannot. */
  status(): Promise<DocumentAnalystStatus> {
    return getBackendService().invoke<DocumentAnalystStatus>('document_analyst_status');
  },

  /**
   * Binds an image projector to a model after checking both files' headers.
   * Any filename; in force at the next start; never by itself vision-ready.
   */
  bindProjector(modelId: string, projectorPath: string): Promise<ProjectorBindOutcome> {
    return getBackendService().invoke<ProjectorBindOutcome>('registry_bind_projector', {
      modelId,
      projectorPath,
    });
  },

  /** Shows a model a random token in an image and records whether it read it. */
  probe(modelId: string): Promise<ReadinessRecord> {
    return getBackendService().invoke<ReadinessRecord>('vision_probe_model', { modelId });
  },
};
