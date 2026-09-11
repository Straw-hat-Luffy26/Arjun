import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Cpu,
  Zap,
  Database,
  HardDrive,
  Terminal,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Sliders,
  Check,
  X,
  Info,
  Server,
  Layers,
  Activity,
  ShieldCheck,
  Box,
  ArrowRight
} from 'lucide-react';
import { Card, Button, Badge, Dialog, Input, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import * as systemService from '../services/system.service';
import { HardwareProfile, CpuInfo, GpuInfo, MemoryInfo, StorageInfo, OsInfo, SoftwareEnvironment, AIRuntimeInfo, OverrideValue } from '../types/system';
import { CapabilityReadout } from '../components/system/CapabilityReadout';
import {
  formatBytes,
  formatFrequency,
  formatPercentage,
  formatNumber,
  classNames
} from '../utils/helpers';
import styles from './SystemInfo.module.css';

function unwrapOverride<T>(field: OverrideValue<T> | T | undefined, fallback: T): T {
  if (field === undefined || field === null) return fallback;
  if (typeof field === 'object' && field !== null && 'detected' in field) {
    const ov = field as OverrideValue<T>;
    if (ov.isOverridden || (ov as any).is_overridden) {
      if (ov.overridden !== null && ov.overridden !== undefined) {
        return ov.overridden;
      }
    }
    return ov.detected ?? fallback;
  }
  return (field as T) ?? fallback;
}

const emptyCpu: CpuInfo = {
  manufacturer: 'Unknown',
  model: 'Unknown',
  architecture: 'Unknown',
  physicalCores: 0,
  logicalProcessors: 0,
  baseFrequencyMhz: 0,
  boostFrequencyMhz: 0,
  virtualizationSupported: false,
  simdCapabilities: []
};

const emptyMemory: MemoryInfo = {
  totalBytes: 0,
  availableBytes: 0,
  usedBytes: 0,
  memoryType: 'Unknown'
};

const emptyOs: OsInfo = {
  name: 'Unknown',
  edition: 'Unknown',
  version: 'Unknown',
  buildNumber: 'Unknown',
  architecture: 'Unknown',
  locale: 'Unknown'
};

export const SystemInfo: React.FC = () => {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Manual Override Dialog State
  const [overrideModalOpen, setOverrideModalOpen] = useState<boolean>(false);
  const [overrideField, setOverrideField] = useState<string>('');
  const [overrideLabel, setOverrideLabel] = useState<string>('');
  const [detectedValue, setDetectedValue] = useState<string>('');
  const [overrideInput, setOverrideInput] = useState<string>('');
  const [isSubmittingOverride, setIsSubmittingOverride] = useState<boolean>(false);

  const fetchProfile = useCallback(async () => {
    try {
      setLoading(true);
      const data = await systemService.getHardwareProfile();
      setProfile(data);
    } catch (err) {
      console.error('Failed to load hardware profile:', err);
      addToast('error', 'Failed to load hardware profile');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const handleReanalyze = async () => {
    try {
      setIsAnalyzing(true);
      const newProfile = await systemService.analyzeSystem();
      setProfile(newProfile);
      addToast('success', 'Hardware re-analysis complete');
    } catch (err) {
      console.error('Re-analyze failed:', err);
      addToast('error', 'Failed to re-analyze hardware');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const openOverrideModal = (fieldPath: string, label: string, currentVal: unknown) => {
    setOverrideField(fieldPath);
    setOverrideLabel(label);
    const valStr = currentVal !== null && currentVal !== undefined ? String(currentVal) : '';
    setDetectedValue(valStr);
    
    const existingOverride = profile?.overrides?.[fieldPath];
    setOverrideInput(existingOverride !== undefined && existingOverride !== null ? String(existingOverride) : valStr);
    setOverrideModalOpen(true);
  };

  const handleSaveOverride = async () => {
    if (!overrideField) return;
    try {
      setIsSubmittingOverride(true);
      let parsedValue: unknown = overrideInput.trim();
      if (!isNaN(Number(parsedValue)) && parsedValue !== '') {
        parsedValue = Number(parsedValue);
      } else if (parsedValue === 'true') {
        parsedValue = true;
      } else if (parsedValue === 'false') {
        parsedValue = false;
      }

      const updatedProfile = await systemService.overrideHardwareValue(overrideField, parsedValue);
      setProfile(updatedProfile);
      addToast('success', `Updated override for ${overrideLabel}`);
      setOverrideModalOpen(false);
    } catch (err) {
      console.error('Save override error:', err);
      addToast('error', 'Failed to apply override');
    } finally {
      setIsSubmittingOverride(false);
    }
  };

  const handleRevertOverride = async () => {
    if (!overrideField) return;
    try {
      setIsSubmittingOverride(true);
      const updatedProfile = await systemService.revertHardwareOverride(overrideField);
      setProfile(updatedProfile);
      addToast('info', `Reverted override for ${overrideLabel}`);
      setOverrideModalOpen(false);
    } catch (err) {
      console.error('Revert override error:', err);
      addToast('error', 'Failed to revert override');
    } finally {
      setIsSubmittingOverride(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column', gap: '16px' }}>
        <Spinner size="lg" color="var(--accent)" />
        <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Analyzing Hardware & System Architecture...</span>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className={styles.container}>
        <Card>
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <AlertTriangle size={36} color="var(--warning)" style={{ marginBottom: '12px' }} />
            <p>No hardware profile detected.</p>
            <Button onClick={fetchProfile} style={{ marginTop: '16px' }}>Retry Scan</Button>
          </div>
        </Card>
      </div>
    );
  }

  // Safely unwrap OverrideValue<T> wrappers sent by Rust backend
  const cpu: CpuInfo = unwrapOverride(profile.cpu, emptyCpu);
  const rawGpus = unwrapOverride(profile.gpus, []);
  const gpus: GpuInfo[] = Array.isArray(rawGpus) ? rawGpus : [];
  const memory: MemoryInfo = unwrapOverride(profile.memory, emptyMemory);
  const rawStorage = unwrapOverride(profile.storage, []);
  const storage: StorageInfo[] = Array.isArray(rawStorage) ? rawStorage : [];
  const os: OsInfo = unwrapOverride(profile.os, emptyOs);
  const software: SoftwareEnvironment = unwrapOverride(profile.software, {} as SoftwareEnvironment);
  const rawRuntimes = unwrapOverride(profile.aiRuntimes, []);
  const aiRuntimes: AIRuntimeInfo[] = Array.isArray(rawRuntimes) ? rawRuntimes : [];

  const aiCapabilities = profile.aiCapabilities || { recommendedQuantizations: [], multiModelCapable: false, visionReady: false, embeddingReady: false };
  const validation = profile.validation || { isReadyForAi: false, score: 0, warnings: [], errors: [], recommendations: [] };
  const overrides = profile.overrides || {};

  // A scan that found nothing leaves every field at zero. Rendering that as
  // "0 Bytes" reads as broken hardware rather than as an absent scan, so the
  // readout is told which of the two it is looking at.
  const hardwareDetected = Boolean(
    (memory.totalBytes || 0) > 0 || (cpu.logicalProcessors || 0) > 0 || gpus.length > 0,
  );

  const memoryUsedPercentage = formatPercentage(memory.usedBytes || 0, memory.totalBytes || 1);
  const primaryGpu = gpus[0];
  const gpuVramUsedBytes = primaryGpu ? (primaryGpu.vramTotalBytes || 0) - (primaryGpu.vramFreeBytes || 0) : 0;
  const gpuVramPercentage = primaryGpu ? formatPercentage(gpuVramUsedBytes, primaryGpu.vramTotalBytes || 1) : 0;

  const isFieldOverridden = (fieldPath: string) => Boolean(overrides && overrides[fieldPath] !== undefined);

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.header}>
        <div className={styles.headerInfo}>
          <div className={styles.titleRow}>
            <h1 className={styles.headerTitle}>This machine</h1>
          </div>
          <p className={styles.headerSubtitle}>
            What ARJUN can run here, and the hardware it read to decide.
          </p>
        </div>

        <div className={styles.headerActions}>
          <Button
            variant="primary"
            icon={<Sparkles size={16} />}
            // Storage, not Discover. There is no Discover: this build reaches
            // no model catalogue, so models arrive by a reviewed offline
            // transfer into the model directory and are picked up there.
            onClick={() => navigate('/models')}
          >
            Models On This Machine
          </Button>
          <Button
            variant="secondary"
            icon={<RotateCcw size={16} className={isAnalyzing ? styles.spinningIcon : ''} />}
            onClick={handleReanalyze}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? 'Scanning Hardware...' : 'Re-analyze System'}
          </Button>
        </div>
      </div>

      {/* The question ARJUN actually asks of a host, answered before the
        * hardware inventory that produced the answer. */}
      <CapabilityReadout capabilities={aiCapabilities} detected={hardwareDetected} />

      {/* Main Grid */}
      <div className={styles.grid}>
        {/* CPU Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Cpu size={20} /></div>
                <span className={styles.cardTitle}>Processor (CPU)</span>
              </div>
              <button
                className={styles.overrideTriggerBtn}
                onClick={() => openOverrideModal('cpu.model', 'CPU Model', cpu.model)}
                title="Override CPU Spec"
              >
                <Sliders size={13} />
                <span>Override</span>
              </button>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Model Name</span>
              <span className={styles.specValue}>
                {cpu.model || 'Unknown'}
                {isFieldOverridden('cpu.model') && <span className={styles.overrideBadge}>Overridden</span>}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Architecture</span>
              <span className={styles.specValue}>{cpu.architecture || 'Unknown'} ({cpu.manufacturer || 'Unknown'})</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Cores / Threads</span>
              <span className={styles.specValue}>{cpu.physicalCores || 0} Cores / {cpu.logicalProcessors || 0} Threads</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Base / Boost Frequency</span>
              <span className={styles.specValue}>
                {formatFrequency(cpu.baseFrequencyMhz || 0)} / {formatFrequency(cpu.boostFrequencyMhz || 0)}
              </span>
            </div>
            {cpu.cacheL3Kb && (
              <div className={styles.specRow}>
                <span className={styles.specLabel}>L3 Cache Size</span>
                <span className={styles.specValue}>{formatBytes(cpu.cacheL3Kb * 1024)}</span>
              </div>
            )}
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Hardware Virtualization</span>
              <span className={styles.specValue}>
                {cpu.virtualizationSupported ? (
                  <Badge variant="success">Supported</Badge>
                ) : (
                  <Badge variant="warning">Disabled / Unsupported</Badge>
                )}
              </span>
            </div>

            <div style={{ marginTop: '8px' }}>
              <span className={styles.specLabel} style={{ marginBottom: '6px' }}>Instruction Sets & SIMD:</span>
              <div className={styles.pillsGroup}>
                {(cpu.simdCapabilities || []).map(cap => (
                  <span key={cap} className={styles.badgeTag}>{cap}</span>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* GPU Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Zap size={20} /></div>
                <span className={styles.cardTitle}>Graphics & AI Acceleration (GPU)</span>
              </div>
              {primaryGpu && (
                <button
                  className={styles.overrideTriggerBtn}
                  onClick={() => openOverrideModal('gpus.0.vramTotalBytes', 'GPU VRAM Total Bytes', primaryGpu.vramTotalBytes)}
                  title="Override GPU VRAM Spec"
                >
                  <Sliders size={13} />
                  <span>Override VRAM</span>
                </button>
              )}
            </div>
          }
        >
          {gpus.length === 0 ? (
            <div style={{ padding: '16px', color: 'var(--text-secondary)', fontSize: '13px' }}>
              No dedicated graphics accelerators detected. CPU-only inference backend will be active.
            </div>
          ) : (
            gpus.map((gpu, idx) => {
              const isIntegrated = gpu.gpuType === 'Integrated' || (!gpu.isDedicated && gpu.gpuType !== 'Dedicated');
              const gpuVramAllocated = (gpu.vramTotalBytes || 0) - (gpu.vramFreeBytes || 0);
              const gpuPct = formatPercentage(gpuVramAllocated, gpu.vramTotalBytes || 1);
              return (
                <div key={idx} className={styles.specList} style={{ marginBottom: idx < gpus.length - 1 ? '20px' : '0' }}>
                  <div className={styles.specRow}>
                    <span className={styles.specLabel}>Device Model #{idx + 1}</span>
                    <span className={styles.specValue}>
                      {gpu.model || 'Unknown'} ({gpu.vendor || 'Unknown'})
                      <Badge variant={isIntegrated ? 'default' : 'info'}>
                        {gpu.gpuType || (isIntegrated ? 'Integrated' : 'Dedicated')} GPU
                      </Badge>
                    </span>
                  </div>

                  {isIntegrated ? (
                    <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: '6px', margin: '8px 0', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <div className={styles.specRow}>
                        <span className={styles.specLabel}>Dedicated / Reserved VRAM</span>
                        <span className={styles.specValue}>{formatBytes(gpu.dedicatedVideoMemoryBytes || 536870912)}</span>
                      </div>
                      <div className={styles.specRow}>
                        <span className={styles.specLabel}>Shared System Memory</span>
                        <span className={styles.specValue}>{formatBytes(gpu.sharedSystemMemoryBytes)}</span>
                      </div>
                      <div className={styles.specRow}>
                        <span className={styles.specLabel}>Total Available Graphics Memory</span>
                        <span className={styles.specValue} style={{ color: 'var(--accent)', fontWeight: 600 }}>
                          {formatBytes(gpu.totalAvailableGraphicsMemoryBytes || (gpu.dedicatedVideoMemoryBytes + gpu.sharedSystemMemoryBytes))}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.progressSection}>
                      <div className={styles.progressHeader}>
                        <span>Dedicated VRAM Allocation</span>
                        <span>
                          {formatBytes(gpuVramAllocated)} used of {formatBytes(gpu.dedicatedVideoMemoryBytes || gpu.vramTotalBytes || 0)} ({gpuPct}%)
                        </span>
                      </div>
                      <div className={styles.progressBarTrack}>
                        <div className={styles.progressBarFill} style={{ width: `${gpuPct}%` }} />
                      </div>
                    </div>
                  )}

                  <div className={styles.specRow}>
                    <span className={styles.specLabel}>Driver Version</span>
                    <span className={styles.specValue}>
                      {gpu.driverVersion || 'N/A'} {gpu.vendorId ? `(Vendor ID: 0x${gpu.vendorId.toString(16).toUpperCase()})` : ''}
                    </span>
                  </div>
                  {gpu.computeCapability && (
                    <div className={styles.specRow}>
                      <span className={styles.specLabel}>Compute Capability</span>
                      <span className={styles.specValue}>v{gpu.computeCapability}</span>
                    </div>
                  )}
                  {gpu.detectionSource && (
                    <div className={styles.specRow}>
                      <span className={styles.specLabel}>Telemetry Source</span>
                      <span className={styles.specValue}>
                        {gpu.detectionSource} &bull; Confidence: {gpu.confidence || 'High'}
                      </span>
                    </div>
                  )}

                  <div style={{ marginTop: '8px' }}>
                    <span className={styles.specLabel} style={{ marginBottom: '6px' }}>Acceleration API Support:</span>
                    <div className={styles.pillsGroup}>
                      <Badge variant={gpu.cudaSupported ? 'success' : 'default'}>
                        CUDA: {gpu.cudaSupported ? 'Available' : 'Not Supported'}
                      </Badge>
                      <Badge variant={gpu.vulkanSupported ? 'success' : 'default'}>
                        Vulkan: {gpu.vulkanSupported ? 'Available' : 'Not Supported'}
                      </Badge>
                      <Badge variant={gpu.directxSupported ? 'success' : 'default'}>
                        DirectX: {gpu.directxSupported ? 'Available' : 'Not Supported'}
                      </Badge>
                      <Badge variant={gpu.openclSupported ? 'info' : 'default'}>
                        OpenCL: {gpu.openclSupported ? 'Available' : 'Not Supported'}
                      </Badge>
                      <Badge variant={gpu.rocmSupported ? 'success' : 'default'}>
                        ROCm: {gpu.rocmSupported ? 'Available' : 'Not Supported'}
                      </Badge>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </Card>

        {/* System Memory Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Database size={20} /></div>
                <span className={styles.cardTitle}>System Memory (RAM)</span>
              </div>
              <button
                className={styles.overrideTriggerBtn}
                onClick={() => openOverrideModal('memory.totalBytes', 'System RAM Total Bytes', memory.totalBytes)}
                title="Override RAM Spec"
              >
                <Sliders size={13} />
                <span>Override</span>
              </button>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.progressSection}>
              <div className={styles.progressHeader}>
                <span>RAM Usage</span>
                <span>
                  {formatBytes(memory.usedBytes || 0)} used of {formatBytes(memory.totalBytes || 0)} ({memoryUsedPercentage}%)
                  {isFieldOverridden('memory.totalBytes') && <span className={styles.overrideBadge} style={{ marginLeft: '6px' }}>Overridden</span>}
                </span>
              </div>
              <div className={styles.progressBarTrack}>
                <div className={styles.progressBarFill} style={{ width: `${memoryUsedPercentage}%` }} />
              </div>
            </div>

            <div className={styles.specRow}>
              <span className={styles.specLabel}>Available Memory</span>
              <span className={styles.specValue}>{formatBytes(memory.availableBytes || 0)}</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Memory Type & Speed</span>
              <span className={styles.specValue}>
                {memory.memoryType || 'Unknown'} {memory.speedMts ? `@ ${formatNumber(memory.speedMts)} MT/s` : ''}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Memory Slots</span>
              <span className={styles.specValue}>
                {memory.populatedSlots ?? 'Unknown'} / {memory.totalSlots ?? 'Unknown'} Slots Populated
              </span>
            </div>
          </div>
        </Card>

        {/* Storage Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><HardDrive size={20} /></div>
                <span className={styles.cardTitle}>Storage & Model Drives</span>
              </div>
            </div>
          }
        >
          <div className={styles.specList}>
            {storage.length === 0 ? (
              <div style={{ padding: '16px', color: 'var(--text-secondary)', fontSize: '13px' }}>
                No storage volumes detected.
              </div>
            ) : (
              storage.map((drive, idx) => {
                const driveUsedBytes = (drive.totalBytes || 0) - (drive.freeBytes || 0);
                const drivePct = formatPercentage(driveUsedBytes, drive.totalBytes || 1);
                return (
                  <div key={idx} style={{ marginBottom: idx < storage.length - 1 ? '16px' : '0' }}>
                    <div className={styles.specRow}>
                      <span className={styles.specLabel}>{drive.driveName || 'Volume'} ({drive.mountPoint || ''})</span>
                      <span className={styles.specValue}>
                        {drive.driveType || 'Disk'} ({drive.fileSystem || 'NTFS'})
                        {drive.isAiStorageReady && <Badge variant="success">AI Model Storage Ready</Badge>}
                      </span>
                    </div>

                    <div className={styles.progressSection}>
                      <div className={styles.progressHeader}>
                        <span>Capacity Allocation</span>
                        <span>
                          {formatBytes(drive.freeBytes || 0)} free of {formatBytes(drive.totalBytes || 0)} ({100 - drivePct}% available)
                        </span>
                      </div>
                      <div className={styles.progressBarTrack}>
                        <div className={styles.progressBarFill} style={{ width: `${drivePct}%` }} />
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </Card>

        {/* Operating System & Environment */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Server size={20} /></div>
                <span className={styles.cardTitle}>Operating System & Software Dependencies</span>
              </div>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Operating System</span>
              <span className={styles.specValue}>
                {os.name || 'Unknown'} {os.edition || ''} ({os.architecture || 'x86_64'})
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Kernel / Build Version</span>
              <span className={styles.specValue}>
                v{os.version || 'Unknown'} (Build {os.buildNumber || 'Unknown'}) &bull; Locale: {os.locale || 'en-US'}
              </span>
            </div>

            <div style={{ marginTop: '8px' }}>
              <span className={styles.specLabel} style={{ marginBottom: '6px' }}>Detected Software Runtimes:</span>
              <div className={styles.pillsGroup}>
                {Object.entries(software).map(([key, sw]) => {
                  if (!sw || typeof sw !== 'object' || !('name' in sw)) return null;
                  const item = sw as any;
                  return (
                    <Badge key={key} variant={item.installed ? 'success' : 'default'}>
                      {item.name}: {item.installed ? (item.version || 'Installed') : 'Not Installed'}
                    </Badge>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        {/* AI Capability Profile */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Sparkles size={20} /></div>
                <span className={styles.cardTitle}>AI Capability Profile & Inferred Limits</span>
                <Badge variant="info">Sarathi-Calculated Capability Profile</Badge>
              </div>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Max Recommended Model Memory</span>
              <span className={styles.specValue} style={{ color: 'var(--accent)', fontWeight: 600 }}>
                {aiCapabilities.maxRecommendedModelSizeBytes ? formatBytes(aiCapabilities.maxRecommendedModelSizeBytes) : 'Unknown'}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Optimal Inference Backend</span>
              <span className={styles.specValue}>{aiCapabilities.preferredInferenceBackend || 'Unknown'}</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Recommended Context Window</span>
              <span className={styles.specValue}>{aiCapabilities.recommendedContextLength ? `${formatNumber(aiCapabilities.recommendedContextLength)} Tokens` : 'Unknown'}</span>
            </div>

            <div className={styles.specRow}>
              <span className={styles.specLabel}>Recommended Quantizations</span>
              <div className={styles.pillsGroup}>
                {(aiCapabilities.recommendedQuantizations || []).map(q => (
                  <span key={q} className={styles.badgeTag}>{q}</span>
                ))}
              </div>
            </div>

            <div style={{ marginTop: '8px' }}>
              <span className={styles.specLabel} style={{ marginBottom: '8px' }}>Module & Feature Readiness:</span>
              <div className={styles.pillsGroup}>
                <Badge variant={aiCapabilities.visionReady ? 'success' : 'default'}>
                  {aiCapabilities.visionReady ? '✓ Multimodal Vision Ready' : 'Vision Limited'}
                </Badge>
                <Badge variant={aiCapabilities.embeddingReady ? 'success' : 'default'}>
                  {aiCapabilities.embeddingReady ? '✓ RAG Vector Embeddings' : 'Embedding Limited'}
                </Badge>
                <Badge variant={aiCapabilities.multiModelCapable ? 'success' : 'default'}>
                  {aiCapabilities.multiModelCapable ? '✓ Multi-Model Parallel' : 'Single Model Only'}
                </Badge>
              </div>
            </div>
          </div>
        </Card>

        {/* System Validation & Diagnostics */}
        <Card
          className={styles.fullWidth}
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><ShieldCheck size={20} /></div>
                <span className={styles.cardTitle}>System Validation & Diagnostic Assessment</span>
              </div>
            </div>
          }
        >
          {(validation.warnings || []).length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.warningBox)}>
              <div className={styles.diagTitle}>
                <AlertTriangle size={16} />
                <span>System Warnings ({(validation.warnings || []).length})</span>
              </div>
              <ul className={styles.diagList}>
                {(validation.warnings || []).map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {(validation.errors || []).length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.errorBox)}>
              <div className={styles.diagTitle}>
                <XCircle size={16} />
                <span>System Blockers ({(validation.errors || []).length})</span>
              </div>
              <ul className={styles.diagList}>
                {(validation.errors || []).map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {(validation.recommendations || []).length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.recBox)}>
              <div className={styles.diagTitle}>
                <Info size={16} />
                <span>System Optimization Recommendations ({(validation.recommendations || []).length})</span>
              </div>
              <ul className={styles.diagList}>
                {(validation.recommendations || []).map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      {/* Bottom Primary Action Banner */}
      <div className={styles.findModelsBanner}>
        <div className={styles.bannerContent}>
          <div className={styles.bannerInfo}>
            <div className={styles.bannerTitleRow}>
              <Sparkles size={22} color="var(--accent)" />
              <h3>Ready to Explore Local AI Models for Your PC</h3>
            </div>
            <p>
              Based on your detected hardware profile, Sarathi&apos;s Recommendation Engine
              calculates the optimal LLMs your machine can run comfortably. Copy the weights
              into the model directory and press <em>Detect models</em> &mdash; this build
              reaches no catalogue, by design.
            </p>
          </div>
          <Button
            variant="primary"
            onClick={() => navigate('/models')}
            className={styles.bannerButton}
          >
            Models On This Machine
            <ArrowRight size={18} style={{ marginLeft: 8 }} />
          </Button>
        </div>
      </div>

      {/* Manual Override Controls Dialog */}
      <Dialog
        isOpen={overrideModalOpen}
        onClose={() => setOverrideModalOpen(false)}
        title={`Manual Spec Override — ${overrideLabel}`}
        actions={
          <div className={styles.dialogActions}>
            <Button
              variant="danger"
              onClick={handleRevertOverride}
              loading={isSubmittingOverride}
              icon={<RotateCcw size={14} />}
            >
              Revert to Detected
            </Button>
            <div style={{ display: 'flex', gap: '8px' }}>
              <Button
                variant="ghost"
                onClick={() => setOverrideModalOpen(false)}
                disabled={isSubmittingOverride}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleSaveOverride}
                loading={isSubmittingOverride}
                icon={<Check size={14} />}
              >
                Save Override
              </Button>
            </div>
          </div>
        }
      >
        <div className={styles.dialogForm}>
          <div className={styles.dialogRow}>
            <label className={styles.dialogLabel}>Target Spec Field Path</label>
            <div className={styles.detectedValueBox}>{overrideField}</div>
          </div>

          <div className={styles.dialogRow}>
            <label className={styles.dialogLabel}>Original Detected Value</label>
            <div className={styles.detectedValueBox}>{detectedValue || '(Empty / Unset)'}</div>
          </div>

          <div className={styles.dialogRow}>
            <Input
              label="New Override Value"
              value={overrideInput}
              onChange={e => setOverrideInput(e.target.value)}
              placeholder="Enter custom override value"
            />
            <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
              Overrides will persist in local Sarathi configuration and adjust AI capability profiling.
            </span>
          </div>
        </div>
      </Dialog>
    </div>
  );
};
