import { ISarathiClient, ISarathiSystemAnalyzerService } from './ISarathiClient';
import * as configService from '../services/config.service';
import * as appService from '../services/app.service';
import * as dbService from '../services/database.service';
import * as themeService from '../services/theme.service';
import * as systemService from '../services/system.service';
import { registryService } from '../services/registry.service';
import { getBackendService } from '../services/api';
import * as providerService from '../services/provider.service';
import * as aiService from '../services/ai.service';
import { AppConfig } from '../types/config';
import { Theme } from '../types/theme';

export class SarathiTauriClient implements ISarathiClient {
  readonly config = {
    getConfig: () => configService.getConfig(),
    setConfig: (config: AppConfig) => configService.setConfig(config),
    getConfigValue: <T = unknown>(key: string) => configService.getConfigValue<T>(key),
    setConfigValue: (key: string, value: unknown) => configService.setConfigValue(key, value),
    getAppPaths: () => configService.getAppPaths(),
    resetConfig: () => configService.resetConfig(),
  };

  readonly system = {
    getAppInfo: () => appService.getAppInfo(),
    getAppState: () => appService.getAppState(),
    logActivity: (action: string, category: string, details?: string) => appService.logActivity(action, category, details),
  };

  readonly database = {
    getSetting: (key: string) => dbService.getSetting(key),
    setSetting: (key: string, value: string, type?: string) => dbService.setSetting(key, value, type || 'string'),
    getAllSettings: () => dbService.getAllSettings(),
    getRecentActivity: (limit?: number) => dbService.getRecentActivity(limit || 10),
  };

  readonly theme = {
    getTheme: () => themeService.getTheme(),
    setTheme: (theme: Theme) => themeService.setTheme(theme),
    getSystemTheme: () => themeService.getSystemTheme(),
    applyTheme: (theme: 'dark' | 'light') => themeService.applyTheme(theme),
  };

  // Phase 2: System Analyzer
  readonly systemAnalyzer: ISarathiSystemAnalyzerService = {
    getHardwareProfile: () => systemService.getHardwareProfile(),
    analyzeSystem: () => systemService.analyzeSystem(),
    overrideHardwareValue: (fieldPath: string, value: unknown) => systemService.overrideHardwareValue(fieldPath, value),
    revertHardwareOverride: (fieldPath: string) => systemService.revertHardwareOverride(fieldPath),
    validateSystem: () => systemService.validateSystem(),
  };

  // Phase 3: Model Manager
  //
  // These used to be wired to `services/model.service.ts`, whose four functions
  // were stubs: `listModels` and `getRecommendations` returned `[]`, and
  // `getModelCompatibility` returned `{ modelId, score: 0 }` — a fabricated
  // number on a public interface. They are now the real commands.
  //
  // `getModelCompatibility` is gone rather than reimplemented: no backend
  // computes a per-model compatibility score, so there was nothing to point it
  // at. `get_compatible_packages` answers a different question (which packages
  // will run at all) and is not a score.
  readonly modelManager = {
    listModels: async () => registryService.listModels(),
    getRecommendations: async () =>
      getBackendService().invoke<unknown[]>('get_model_recommendations', {}),
  };

  // Phase 4: Model Providers
  readonly modelProviders = {
    getProviders: async () => providerService.getProviders(),
    searchProviderModels: async (pId: string, q: string) => providerService.searchProviderModels(pId, q),
  };

  // Phase 5: AI Engine
  readonly aiEngine = {
    loadModel: async (providerId: string, modelId?: string, quantization?: string) =>
      aiService.loadModel(providerId, modelId, quantization),
    unloadModel: async () => aiService.unloadModel(),
    chat: async (msgs?: unknown[]) => aiService.chat(msgs as any),
  };

}
