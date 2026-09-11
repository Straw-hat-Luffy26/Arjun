/**
 * Models already on this machine.
 *
 * The download half of this file is gone. It carried `start_model_download`,
 * `pause_model_download`, `resume_model_download`, `cancel_model_download`,
 * `get_active_downloads` and the `download:progress` listener — every one of
 * them a HuggingFace fetch — and this build reaches no model catalogue at all.
 * Weights arrive by a reviewed offline transfer into the model directory and
 * are picked up by "Detect models" on the Models screen.
 *
 * What is left reads and removes what is already installed, and touches
 * nothing off this machine.
 */

import { invoke } from '@tauri-apps/api/core';
import type { InstalledModel, StorageSummary } from '../types/download';

export async function getInstalledModels(): Promise<InstalledModel[]> {
  return invoke('get_installed_models');
}

export async function deleteInstalledModel(
  providerId: string,
  modelId: string,
  quantization: string
): Promise<void> {
  return invoke('delete_installed_model', { providerId, modelId, quantization });
}

export async function getStorageSummary(): Promise<StorageSummary> {
  return invoke('get_storage_summary');
}
