import { getBackendService } from './api';
import type { Classification } from './registry.service';

/** A folder or share this installation reads. Mirrors `knowledge::connector::Collection`. */
export interface Collection {
  id: string;
  name: string;
  kind: 'localFolder' | 'networkShare';
  root: string;
  owner: string;
  classification: Classification;
  /** Narrows who may search it; never widens the classification's clearance. */
  restrictedToRoles: string[];
  retentionDays?: number | null;
  enabled: boolean;
}

export interface CollectionView extends Collection {
  lastSyncedAt: string | null;
  currentSources: number;
}

/** What one sync did. Mirrors `knowledge::ingest::SyncReport`. */
export interface SyncReport {
  plan: string;
  outcome: {
    documentsRead: number;
    chunksIndexed: number;
    pagesNeedingReview: number;
    flaggedForInjection: string[];
    failures: { file: string; reason: string }[];
    retired: string[];
    supersededSha256s: string[];
    withdrawnSha256s: string[];
    skipped: number;
  };
  memoryInvalidated: number;
  memoryStale: number;
  unreadable: string[];
}

export type ProviderState = 'qualified' | 'unqualified' | 'unavailable';

export interface QualificationCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface QualificationRecord {
  identity: { modelId: string; profile: string; weightsSha256: string; dimensions: number; pooling: string };
  spaceKey: string;
  passed: boolean;
  checks: QualificationCheck[];
  orderedCorrectly: number;
  probes: number;
  measuredAt: string;
}

/** Which half of retrieval runs here. Mirrors `commands::retrieval::RetrievalStatus`. */
export interface RetrievalStatus {
  provider: {
    state: ProviderState;
    identity?: QualificationRecord['identity'];
    spaceKey?: string;
    detail: string;
    qualification?: QualificationRecord;
  };
  /** Over the passages this reader may see. */
  coverage?: { embedded: number; failed: number; total: number };
  reindexing: boolean;
  lastReindex?: { spaceKey: string; embedded: number; failed: number; remaining: number; stoppedBecause?: string };
  indexRevision: number;
}

export const retrievalService = {
  collections(): Promise<CollectionView[]> {
    return getBackendService().invoke<CollectionView[]>('knowledge_collections');
  },

  save(collection: Collection): Promise<void> {
    return getBackendService().invoke<void>('knowledge_collection_save', { collection });
  },

  remove(collectionId: string): Promise<number> {
    return getBackendService().invoke<number>('knowledge_collection_remove', { collectionId });
  },

  sync(collectionId: string): Promise<SyncReport> {
    return getBackendService().invoke<SyncReport>('knowledge_collection_sync', { collectionId });
  },

  status(): Promise<RetrievalStatus> {
    return getBackendService().invoke<RetrievalStatus>('knowledge_retrieval_status');
  },

  qualify(): Promise<QualificationRecord> {
    return getBackendService().invoke<QualificationRecord>('knowledge_embedding_qualify');
  },

  reindex(): Promise<{ spaceKey: string; embedded: number; failed: number; remaining: number }> {
    return getBackendService().invoke('knowledge_reindex');
  },
};
