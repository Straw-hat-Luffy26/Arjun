/**
 * ISarathiClient — Master SDK Abstraction Interface
 * 
 * Sarathi is completely UI-agnostic.
 * Any frontend (React, Vue, Svelte, Solid, Electron, Webview, or CLI)
 * interacts ONLY with this interface.
 * 
 * Zero business logic is stored in the UI layer.
 */

import { AppConfig, AppPaths } from '../types/config';
import { AppState } from '../types/app-state';
import { Theme } from '../types/theme';
import { Setting, ActivityLogEntry } from '../types/database';
import { HardwareProfile, SystemValidationResult } from '../types/system';

export interface ISarathiConfigService {
  getConfig(): Promise<AppConfig>;
  setConfig(config: AppConfig): Promise<void>;
  getConfigValue<T = unknown>(key: string): Promise<T>;
  setConfigValue(key: string, value: unknown): Promise<void>;
  getAppPaths(): Promise<AppPaths>;
  resetConfig(): Promise<void>;
}

export interface ISarathiSystemService {
  getAppInfo(): Promise<{ name: string; version: string }>;
  getAppState(): Promise<AppState>;
  logActivity(action: string, category: string, details?: string): Promise<void>;
}

export interface ISarathiDatabaseService {
  getSetting(key: string): Promise<Setting | null>;
  setSetting(key: string, value: string, type?: string): Promise<void>;
  getAllSettings(): Promise<Setting[]>;
  getRecentActivity(limit?: number): Promise<ActivityLogEntry[]>;
}

export interface ISarathiThemeService {
  getTheme(): Promise<Theme>;
  setTheme(theme: Theme): Promise<void>;
  getSystemTheme(): 'dark' | 'light';
  applyTheme(theme: 'dark' | 'light'): void;
}

// ─── PHASE 2 SERVICE INTERFACES ───

export interface ISarathiSystemAnalyzerService {
  getHardwareProfile(): Promise<HardwareProfile | null>;
  analyzeSystem(): Promise<HardwareProfile>;
  overrideHardwareValue(fieldPath: string, value: unknown): Promise<HardwareProfile>;
  revertHardwareOverride(fieldPath: string): Promise<HardwareProfile>;
  validateSystem(): Promise<SystemValidationResult>;
}

export interface ISarathiModelManagerService {
  listModels(): Promise<unknown[]>;
  getRecommendations(): Promise<unknown[]>;
}

export interface ISarathiModelProviderService {
  getProviders(): Promise<unknown[]>;
  searchProviderModels(providerId: string, query: string): Promise<unknown[]>;
}

export interface ISarathiAIEngineService {
  loadModel(providerId: string, modelId?: string, quantization?: string): Promise<unknown>;
  unloadModel(): Promise<void>;
  chat(messages?: unknown[]): Promise<unknown>;
}

/**
 * Master Sarathi Client Interface
 */
export interface ISarathiClient {
  readonly config: ISarathiConfigService;
  readonly system: ISarathiSystemService;
  readonly database: ISarathiDatabaseService;
  readonly theme: ISarathiThemeService;
  
  // Modules
  readonly systemAnalyzer: ISarathiSystemAnalyzerService;
  readonly modelManager: ISarathiModelManagerService;
  readonly modelProviders: ISarathiModelProviderService;
  readonly aiEngine: ISarathiAIEngineService;
}
