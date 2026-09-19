import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowLeft, Box, Check, ChevronDown, ChevronRight,
  CircleStop, Clock3, Code2, Cpu, Database, FileJson, FolderOpen, Gauge,
  GitBranch, GitCompareArrows, HardDrive, Layers3, Moon, MoreHorizontal,
  Package, Play, Plus, RefreshCw, Rocket, Search, Settings, SlidersHorizontal,
  Sparkles, Sun, TerminalSquare, Workflow, X, Zap,
} from 'lucide-react'
import './App.css'

type Stage = 'stage0' | 'stage-a' | 'stage-b'
type RunStatus = 'created' | 'running' | 'stopping' | 'completed' | 'failed' | 'stopped' | 'unknown'
type Primitive = string | number | boolean | null
type Page = 'overview' | 'new' | 'runs' | 'assets' | 'hardware' | 'settings'
type AssetKind = 'dataset' | 'manifest' | 'checkpoint' | 'model' | 'output'

interface RunCreate {
  name: string
  stage: Stage
  model: string
  data: string
  output_dir: string
  validation_data?: string | null
  world_size: number
  device: 'auto' | 'cuda' | 'npu' | 'cpu'
  dtype: 'bf16' | 'fp32'
  resume?: string | null
  init_cid_checkpoint?: string | null
  visible_devices?: number[] | null
  parameters: Record<string, Primitive>
  extra_args: string[]
}
interface CommandPreview { argv: string[]; display: string; cwd: string; environment: Record<string, string>; warnings: string[] }
interface RunRecord {
  id: string; name: string; stage: Stage; status: RunStatus; created_at: number
  started_at?: number | null; finished_at?: number | null; pid?: number | null; exit_code?: number | null
  execution_host?: string | null
  output_dir: string; log_path: string; repo_head?: string | null
  request: RunCreate; command: CommandPreview
}
interface HardwareDevice {
  index: number; name: string; kind: 'cuda' | 'npu'
  memory_total_mb?: number | null; memory_used_mb?: number | null
  utilization_percent?: number | null; temperature_c?: number | null; power_w?: number | null
}
interface RepositoryInfo {
  available: boolean; path: string; branch?: string | null; head?: string | null
  dirty?: boolean | null; remote?: string | null
}
interface RuntimeInfo {
  available: boolean; python: string; python_version?: string | null
  torch_version?: string | null; transformers_version?: string | null
  cuda_available?: boolean | null; cuda_device_count?: number | null
  npu_available?: boolean | null; error?: string | null
}
interface AssetRecord {
  kind: AssetKind; name: string; path: string; root: string
  modified_at?: number | null; size_bytes?: number | null; is_symlink: boolean
  details: Record<string, unknown>
}
interface AssetRoot { path: string; label: string; available: boolean }
interface RunComparisonEntry {
  id: string; name: string; stage: Stage; status: RunStatus; model: string
  world_size: number; device: string; created_at: number; repo_head?: string | null
  parameters: Record<string, Primitive>; latest_step?: number | null
  latest_loss?: number | null; latest_raw_loss?: number | null
  latest_learning_rate?: number | null; elapsed_seconds?: number | null
  progress_fraction?: number | null; validation_loss?: number | null
  best_validation_loss?: number | null; metrics: Record<string, unknown>[]
}
interface StageSurface {
  label: string; description: string; accent: string
  defaults: { model: string; world_size: number; device: RunCreate['device']; dtype: RunCreate['dtype']; parameters: Record<string, Primitive> }
}
type StageSurfaces = Record<Stage, StageSurface>

const fallbackSurfaces: StageSurfaces = {
  stage0: {
    label: 'Stage 0', description: 'Convert an autoregressive base into a diffusion-native backbone.', accent: 'violet',
    defaults: { model: 'openbmb/MiniCPM5-2B-Base', world_size: 4, device: 'cuda', dtype: 'bf16',
      parameters: { steps: 10000, micro_batch_size: 1, target_global_batch_size: 96, learning_rate: 0.00002, checkpoint_every: 250, cpu_offload: true } },
  },
  'stage-a': {
    label: 'Stage A', description: 'Train CID modules while keeping the diffusion backbone frozen.', accent: 'cyan',
    defaults: { model: 'GSAI-ML/iLLaDA-8B-Base', world_size: 4, device: 'cuda', dtype: 'bf16',
      parameters: { epochs: 3, learning_rate: 0.0001, micro_batch_size: 1, target_global_batch_size: 96, thought_capacity: 128, checkpoint_every_steps: 5000 } },
  },
  'stage-b': {
    label: 'Stage B', description: 'Continue with full-parameter FSDP training after Stage A.', accent: 'amber',
    defaults: { model: 'GSAI-ML/iLLaDA-8B-Base', world_size: 4, device: 'cuda', dtype: 'bf16',
      parameters: { epochs: 1, learning_rate: 0.00002, backbone_lr_scale: 0.25, target_global_batch_size: 32, checkpoint_every_steps: 2500 } },
  },
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) }, ...init })
  if (!response.ok) {
    let message = response.status + ' ' + response.statusText
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch { /* keep HTTP status */ }
    throw new Error(message)
  }
  return (await response.json()) as T
}
function humanBytes(mb?: number | null) {
  if (mb == null) return '—'
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB'
}
function humanFileBytes(bytes?: number | null) {
  if (bytes == null) return '—'
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1) + ' GB'
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + ' MB'
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return bytes + ' B'
}
function formatTime(unix?: number | null) {
  if (!unix) return '—'
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(unix * 1000))
}
function stageLabel(stage: Stage) { return fallbackSurfaces[stage].label }
function statusTone(status: RunStatus) {
  if (status === 'running') return 'success'
  if (status === 'completed') return 'neutral'
  if (status === 'failed') return 'danger'
  if (status === 'stopping' || status === 'stopped') return 'warning'
  return 'muted'
}

function StatusPill({ status }: { status: RunStatus }) {
  return <span className={'status-pill ' + statusTone(status)}><span className="status-dot" />{status}</span>
}
function EmptyState({ icon, title, description, action }: { icon: React.ReactNode; title: string; description: string; action?: React.ReactNode }) {
  return <div className="empty-state"><div className="empty-icon">{icon}</div><h3>{title}</h3><p>{description}</p>{action}</div>
}

function Sidebar({ page, setPage, onNewRun }: { page: Page; setPage: (page: Page) => void; onNewRun: () => void }) {
  const nav = [
    { id: 'overview' as Page, label: 'Overview', icon: Gauge },
    { id: 'runs' as Page, label: 'Runs', icon: Activity },
    { id: 'assets' as Page, label: 'Assets', icon: FolderOpen },
    { id: 'hardware' as Page, label: 'Hardware', icon: Cpu },
  ]
  return <aside className="sidebar">
    <button className="brand" onClick={() => setPage('overview')}>
      <span className="brand-mark"><Sparkles size={17} /></span>
      <span className="brand-copy"><strong>CID Factory</strong><small>Training Console</small></span>
    </button>
    <button className="new-run-button" onClick={onNewRun}><Plus size={17} />New run<span className="shortcut">N</span></button>
    <nav className="nav-section"><span className="nav-label">Workspace</span>
      {nav.map(({ id, label, icon: Icon }) =>
        <button key={id} className={'nav-item ' + (page === id ? 'active' : '')} onClick={() => setPage(id)}>
          <Icon size={17} /><span>{label}</span>
        </button>
      )}
    </nav>
    <div className="sidebar-spacer" />
    <nav className="nav-section">
      <button className={'nav-item ' + (page === 'settings' ? 'active' : '')} onClick={() => setPage('settings')}>
        <Settings size={17} /><span>Settings</span>
      </button>
    </nav>
    <div className="sidebar-footer">
      <div className="cid-orbit"><span /><span /><span /></div>
      <div><strong>Continuous Interaction Diffusion</strong><small>Frontend to the main CID repository</small></div>
    </div>
  </aside>
}

function MobileNav({ page, setPage, onNewRun }: { page: Page; setPage: (page: Page) => void; onNewRun: () => void }) {
  return <nav className="mobile-nav">
    <button className={page === 'overview' ? 'active' : ''} onClick={() => setPage('overview')}><Gauge size={18} /><span>Overview</span></button>
    <button className={page === 'runs' ? 'active' : ''} onClick={() => setPage('runs')}><Activity size={18} /><span>Runs</span></button>
    <button className="mobile-new" onClick={onNewRun}><Plus size={21} /><span>New</span></button>
    <button className={page === 'assets' ? 'active' : ''} onClick={() => setPage('assets')}><FolderOpen size={18} /><span>Assets</span></button>
    <button className={page === 'hardware' ? 'active' : ''} onClick={() => setPage('hardware')}><Cpu size={18} /><span>Hardware</span></button>
    <button className={page === 'settings' ? 'active' : ''} onClick={() => setPage('settings')}><Settings size={18} /><span>Settings</span></button>
  </nav>
}

function Topbar({ theme, toggleTheme, repository, refresh }: {
  theme: 'light' | 'dark'; toggleTheme: () => void; repository: RepositoryInfo | null; refresh: () => void
}) {
  return <header className="topbar">
    <div className="repo-chip">
      <span className={'repo-status ' + (repository?.available ? 'online' : '')} /><GitBranch size={14} />
      <span>{repository?.branch || 'repository'}</span><span className="mono dim">{repository?.head || 'not connected'}</span>
      {repository?.dirty && <span className="dirty-mark">dirty</span>}
    </div>
    <div className="topbar-actions">
      <button className="icon-button" title="Refresh" onClick={refresh}><RefreshCw size={16} /></button>
      <button className="icon-button" title="Toggle theme" onClick={toggleTheme}>{theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}</button>
    </div>
  </header>
}

function StatCard({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return <div className="stat-card surface"><div className="stat-icon">{icon}</div><div>
    <span className="eyebrow">{label}</span><div className="stat-value">{value}</div><small>{detail}</small>
  </div></div>
}

function GpuMini({ device }: { device: HardwareDevice }) {
  const memoryPercent = device.memory_total_mb && device.memory_used_mb != null ? Math.min(100, device.memory_used_mb / device.memory_total_mb * 100) : 0
  return <div className="gpu-mini">
    <div className="gpu-mini-head">
      <span className="gpu-index">{device.index}</span>
      <div className="gpu-name"><strong>{device.name}</strong><small>{humanBytes(device.memory_used_mb)} / {humanBytes(device.memory_total_mb)}</small></div>
      <span className="gpu-util">{Math.round(device.utilization_percent || 0)}%</span>
    </div>
    <div className="meter"><span style={{ width: memoryPercent + '%' }} /></div>
  </div>
}

function Overview({ runs, devices, repository, surfaces, onNewRun, onRun }: {
  runs: RunRecord[]; devices: HardwareDevice[]; repository: RepositoryInfo | null; surfaces: StageSurfaces
  onNewRun: (stage?: Stage) => void; onRun: (run: RunRecord) => void
}) {
  const active = runs.filter((run) => run.status === 'running')
  const usedMemory = devices.reduce((sum, gpu) => sum + (gpu.memory_used_mb || 0), 0)
  const totalMemory = devices.reduce((sum, gpu) => sum + (gpu.memory_total_mb || 0), 0)
  return <div className="page">
    <section className="page-heading overview-heading">
      <div><span className="overline">CONTROL CENTER</span><h1>Build CID models without fighting the training stack.</h1>
        <p>Launch and observe the main CID repository from one focused workspace. Factory never owns model or training semantics.</p></div>
      <button className="primary-button" onClick={() => onNewRun()}><Rocket size={17} />Launch a run</button>
    </section>
    <div className="stats-grid">
      <StatCard icon={<Activity size={19} />} label="Active runs" value={String(active.length)} detail={runs.length ? runs.length + ' total recorded' : 'No run history yet'} />
      <StatCard icon={<Cpu size={19} />} label="Accelerators" value={String(devices.length)} detail={devices.length ? devices[0].name : 'No CUDA devices detected'} />
      <StatCard icon={<HardDrive size={19} />} label="VRAM" value={totalMemory ? humanBytes(totalMemory) : '—'} detail={totalMemory ? humanBytes(usedMemory) + ' currently used' : 'Unavailable'} />
      <StatCard icon={<GitBranch size={19} />} label="CID source" value={repository?.branch || 'Offline'} detail={repository?.head ? 'Commit ' + repository.head : 'Repository not connected'} />
    </div>
    <div className="overview-grid">
      <section className="surface panel">
        <div className="section-heading"><div><span className="eyebrow">LIVE</span><h2>Active runs</h2></div>
          <button className="text-button" onClick={() => onNewRun()}>New run <ChevronRight size={15} /></button></div>
        {active.length ? <div className="run-stack">{active.slice(0, 3).map((run) =>
          <button className="run-card" key={run.id} onClick={() => onRun(run)}>
            <div className="run-stage-mark" data-stage={run.stage}><Workflow size={16} /></div>
            <div className="run-card-main"><div className="run-card-title"><strong>{run.name}</strong><StatusPill status={run.status} /></div>
              <span>{stageLabel(run.stage)} · {run.request.model}</span><div className="run-progress-row"><div className="skeleton-progress"><span /></div><span>Live metrics</span></div>
            </div><ChevronRight size={17} className="dim" />
          </button>)}</div> :
          <EmptyState icon={<Activity size={24} />} title="Nothing is training" description="Start from Stage 0, Stage A, or continue an existing checkpoint into Stage B."
            action={<button className="secondary-button" onClick={() => onNewRun()}>Configure run</button>} />}
      </section>
      <section className="surface panel">
        <div className="section-heading"><div><span className="eyebrow">SYSTEM</span><h2>Accelerators</h2></div><span className="section-count">{devices.length} devices</span></div>
        <div className="gpu-mini-list">{devices.length ? devices.slice(0, 4).map((device) => <GpuMini key={device.index} device={device} />) :
          <div className="hardware-empty"><Cpu size={22} /><span>No CUDA devices detected on the Factory host.</span></div>}</div>
      </section>
    </div>
    <section className="workflow-section">
      <div className="section-heading"><div><span className="eyebrow">PIPELINE</span><h2>Start where your model is</h2></div><span className="section-note">Each stage maps directly to the CID repository</span></div>
      <div className="stage-grid">{(Object.entries(surfaces) as [Stage, StageSurface][]).map(([stage, surface], index) =>
        <button className="stage-card surface" key={stage} onClick={() => onNewRun(stage)}>
          <div className="stage-card-top"><span className="stage-number">0{index + 1}</span><span className={'stage-accent ' + surface.accent} /></div>
          <div className="stage-icon" data-stage={stage}>{stage === 'stage0' ? <Box size={20} /> : stage === 'stage-a' ? <Layers3 size={20} /> : <Zap size={20} />}</div>
          <h3>{surface.label}</h3><p>{surface.description}</p><span className="stage-cta">Configure <ChevronRight size={14} /></span>
        </button>)}</div>
    </section>
  </div>
}

function RunsPage({ runs, onRun, onNewRun, onCompare }: {
  runs: RunRecord[]
  onRun: (run: RunRecord) => void
  onNewRun: () => void
  onCompare: (runIds: string[]) => void
}) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const filtered = runs.filter((run) =>
    (run.name + ' ' + run.stage + ' ' + run.request.model + ' ' + run.status)
      .toLowerCase()
      .includes(query.toLowerCase()),
  )
  function toggle(runId: string) {
    setSelected((current) =>
      current.includes(runId)
        ? current.filter((item) => item !== runId)
        : current.length < 4 ? [...current, runId] : current,
    )
  }
  return <div className="page"><section className="page-heading compact-heading"><div>
    <span className="overline">EXPERIMENTS</span><h1>Runs</h1><p>Every launch records the exact main-repository command and source commit.</p>
  </div><button className="primary-button" onClick={onNewRun}><Plus size={17} />New run</button></section>
    <section className="surface data-panel"><div className="table-toolbar">
      <div className="search-field"><Search size={16} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search runs" /></div>
      <div className="table-actions">
        {selected.length > 0 && <span className="selection-count">{selected.length}/4 selected</span>}
        <button className="ghost-button" disabled={selected.length < 2} onClick={() => onCompare(selected)}>
          <GitCompareArrows size={15} />Compare
        </button>
      </div>
    </div>
      {filtered.length ? <div className="runs-table-wrap"><table className="runs-table"><thead><tr>
        <th className="select-column" /><th>Run</th><th>Stage</th><th>Status</th><th>Model</th><th>Started</th><th>Commit</th><th />
      </tr></thead><tbody>{filtered.map((run) => <tr key={run.id} className={selected.includes(run.id) ? 'selected-row' : ''} onClick={() => onRun(run)}>
        <td className="select-column" onClick={(event) => event.stopPropagation()}>
          <input type="checkbox" checked={selected.includes(run.id)} onChange={() => toggle(run.id)} aria-label={'Select ' + run.name} />
        </td>
        <td><strong>{run.name}</strong><small className="mono">{run.id}</small></td>
        <td><span className="stage-table-pill" data-stage={run.stage}>{stageLabel(run.stage)}</span></td><td><StatusPill status={run.status} /></td>
        <td className="truncate-cell">{run.request.model}</td><td>{formatTime(run.started_at || run.created_at)}</td><td className="mono">{run.repo_head || '—'}</td><td><MoreHorizontal size={17} /></td>
      </tr>)}</tbody></table></div> :
        <EmptyState icon={<Database size={24} />} title={runs.length ? 'No matching runs' : 'No runs yet'} description={runs.length ? 'Try a different search.' : 'Your training history will appear here.'} />}
    </section>
  </div>
}

function assetIcon(kind: AssetKind) {
  if (kind === 'dataset') return <Database size={17} />
  if (kind === 'manifest') return <FileJson size={17} />
  if (kind === 'checkpoint') return <Package size={17} />
  if (kind === 'model') return <Box size={17} />
  return <FolderOpen size={17} />
}

function assetDetail(asset: AssetRecord) {
  const details = asset.details
  if (asset.kind === 'dataset') {
    const examples = details.examples
    const transitions = details.transitions
    if (examples != null || transitions != null) {
      return [examples != null ? String(examples) + ' examples' : null, transitions != null ? String(transitions) + ' transitions' : null].filter(Boolean).join(' · ')
    }
    return 'JSONL dataset'
  }
  if (asset.kind === 'checkpoint') {
    return details.stage ? String(details.stage) : 'checkpoint'
  }
  if (asset.kind === 'model') {
    return details.model_type ? String(details.model_type) : 'local model'
  }
  if (asset.kind === 'manifest') {
    return details.format ? String(details.format) : 'manifest'
  }
  return 'CID run output'
}

function AssetsPage() {
  const [assets, setAssets] = useState<AssetRecord[]>([])
  const [roots, setRoots] = useState<AssetRoot[]>([])
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState<AssetKind | 'all'>('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams()
        if (kind !== 'all') params.set('kind', kind)
        if (query) params.set('query', query)
        params.set('limit', '400')
        const [assetData, rootData] = await Promise.all([
          api<AssetRecord[]>('/api/assets?' + params.toString()),
          api<AssetRoot[]>('/api/assets/roots'),
        ])
        if (!cancelled) {
          setAssets(assetData)
          setRoots(rootData)
          setLoading(false)
        }
      } catch {
        if (!cancelled) setLoading(false)
      }
    }, 180)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [kind, query])

  const kinds: { id: AssetKind | 'all'; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'dataset', label: 'Datasets' },
    { id: 'manifest', label: 'Manifests' },
    { id: 'checkpoint', label: 'Checkpoints' },
    { id: 'model', label: 'Models' },
    { id: 'output', label: 'Outputs' },
  ]

  return <div className="page">
    <section className="page-heading compact-heading"><div>
      <span className="overline">LOCAL CATALOG</span><h1>Assets</h1>
      <p>Read-only discovery of data, checkpoints, models, and outputs visible to the Factory host.</p>
    </div></section>

    <div className="asset-root-strip">
      {roots.map((root) => <div className={'asset-root-chip ' + (root.available ? '' : 'missing')} key={root.path}>
        <FolderOpen size={13} /><span>{root.path}</span>
      </div>)}
    </div>

    <section className="surface data-panel">
      <div className="asset-toolbar">
        <div className="search-field"><Search size={16} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search paths and asset names" /></div>
        <div className="asset-kind-tabs">{kinds.map((item) =>
          <button key={item.id} className={kind === item.id ? 'active' : ''} onClick={() => setKind(item.id)}>{item.label}</button>
        )}</div>
      </div>

      {loading ? <div className="asset-loading"><RefreshCw className="spin" size={18} />Scanning configured roots…</div> :
        assets.length ? <div className="asset-list">{assets.map((asset) =>
          <div className="asset-row" key={asset.kind + ':' + asset.path}>
            <div className="asset-type-icon" data-kind={asset.kind}>{assetIcon(asset.kind)}</div>
            <div className="asset-main"><div className="asset-title"><strong>{asset.name}</strong><span>{asset.kind}</span>{asset.is_symlink && <span>alias</span>}</div>
              <div className="mono asset-path">{asset.path}</div><small>{assetDetail(asset)}</small></div>
            <div className="asset-meta"><strong>{humanFileBytes(asset.size_bytes)}</strong><span>{formatTime(asset.modified_at)}</span></div>
          </div>
        )}</div> :
          <EmptyState icon={<FolderOpen size={25} />} title="No matching assets" description="Adjust the filter, or add roots with CID_FACTORY_ASSET_ROOTS." />}
    </section>
  </div>
}

function HardwarePage({ devices }: { devices: HardwareDevice[] }) {
  return <div className="page"><section className="page-heading compact-heading"><div><span className="overline">SYSTEM</span><h1>Hardware</h1><p>Live accelerator utilization from the host running CID Factory.</p></div></section>
    {devices.length ? <div className="hardware-grid">{devices.map((device) => {
      const memoryPercent = device.memory_total_mb && device.memory_used_mb != null ? device.memory_used_mb / device.memory_total_mb * 100 : 0
      return <article className="surface gpu-card" key={device.index}>
        <div className="gpu-card-head"><div className="gpu-id"><span>GPU {device.index}</span><strong>{device.name}</strong></div><span className="available-chip">online</span></div>
        <div className="gpu-visual"><div className="radial" style={{ '--value': (device.utilization_percent || 0) + '%' } as React.CSSProperties}>
          <strong>{Math.round(device.utilization_percent || 0)}%</strong><small>compute</small></div>
          <div className="gpu-vitals"><div><span>VRAM</span><strong>{humanBytes(device.memory_used_mb)}</strong></div>
            <div><span>Temperature</span><strong>{device.temperature_c != null ? Math.round(device.temperature_c) + '°C' : '—'}</strong></div>
            <div><span>Power</span><strong>{device.power_w != null ? Math.round(device.power_w) + ' W' : '—'}</strong></div></div>
        </div>
        <div className="memory-row"><span>Memory</span><span>{humanBytes(device.memory_used_mb)} / {humanBytes(device.memory_total_mb)}</span></div>
        <div className="meter large"><span style={{ width: Math.min(100, memoryPercent) + '%' }} /></div>
      </article>
    })}</div> : <section className="surface panel"><EmptyState icon={<Cpu size={28} />} title="No CUDA accelerators detected"
      description="Factory still works for CPU launches. Accelerator telemetry appears automatically when nvidia-smi is available." /></section>}
  </div>
}

const parameterMeta: Record<string, { label: string; type?: 'number' | 'boolean' | 'select'; options?: string[] }> = {
  steps: { label: 'Training steps', type: 'number' }, epochs: { label: 'Epochs', type: 'number' },
  learning_rate: { label: 'CID learning rate', type: 'number' }, backbone_lr_scale: { label: 'Backbone LR scale', type: 'number' },
  embedding_lr_scale: { label: 'Embedding LR scale', type: 'number' }, weight_decay: { label: 'Weight decay', type: 'number' },
  micro_batch_size: { label: 'Micro batch', type: 'number' }, physical_micro_batch_size: { label: 'Physical micro batch', type: 'number' },
  target_global_batch_size: { label: 'Target global batch', type: 'number' }, gradient_accumulation_steps: { label: 'Gradient accumulation', type: 'number' },
  max_grad_norm: { label: 'Max grad norm', type: 'number' }, warmup_ratio: { label: 'Warmup ratio', type: 'number' },
  warmup_steps: { label: 'Warmup steps', type: 'number' }, lr_schedule: { label: 'LR schedule', type: 'select', options: ['constant', 'cosine', 'wsd-linear'] },
  wsd_decay_ratio: { label: 'WSD decay ratio', type: 'number' }, min_learning_rate_ratio: { label: 'Minimum LR ratio', type: 'number' },
  rollout_horizon: { label: 'Rollout horizon', type: 'number' }, thought_capacity: { label: 'Thought capacity', type: 'number' },
  max_display_tokens: { label: 'Max display tokens', type: 'number' }, display_canvas_tokens: { label: 'Display canvas', type: 'number' },
  checkpoint_every: { label: 'Checkpoint interval', type: 'number' }, checkpoint_every_steps: { label: 'Checkpoint interval', type: 'number' },
  keep_checkpoints: { label: 'Keep checkpoints', type: 'number' }, log_every_steps: { label: 'Log interval', type: 'number' },
  cpu_offload: { label: 'CPU offload', type: 'boolean' }, fsdp_cpu_offload: { label: 'FSDP CPU offload', type: 'boolean' },
  gradient_checkpointing: { label: 'Gradient checkpointing', type: 'boolean' }, fsdp_no_sync_accumulation: { label: 'FSDP no-sync accumulation', type: 'boolean' },
  fsdp_aggressive_prefetch: { label: 'Aggressive FSDP prefetch', type: 'boolean' }, aggressive_prefetch: { label: 'Aggressive FSDP prefetch', type: 'boolean' },
  export_hf: { label: 'Export Hugging Face weights', type: 'boolean' },
}
function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <label className="field"><span className="field-label">{label}{hint && <small>{hint}</small>}</span>{children}</label>
}
function Toggle({ checked, onChange }: { checked: boolean; onChange: (value: boolean) => void }) {
  return <button type="button" className={'toggle ' + (checked ? 'on' : '')} onClick={() => onChange(!checked)} aria-pressed={checked}><span /></button>
}

type PickerTarget = 'model' | 'data' | 'validation' | 'resume' | 'init'

function AssetInput({ value, placeholder, onChange, onBrowse }: {
  value: string
  placeholder?: string
  onChange: (value: string) => void
  onBrowse: () => void
}) {
  return <div className="asset-input-control">
    <input className="mono-input" placeholder={placeholder} value={value} onChange={(e) => onChange(e.target.value)} />
    <button type="button" onClick={onBrowse}><FolderOpen size={14} />Browse</button>
  </div>
}

function AssetPicker({ kind, checkpointStage, onSelect, onClose }: {
  kind: AssetKind
  checkpointStage?: Stage
  onSelect: (asset: AssetRecord) => void
  onClose: () => void
}) {
  const [assets, setAssets] = useState<AssetRecord[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ kind, limit: '300' })
        if (query) params.set('query', query)
        const found = await api<AssetRecord[]>('/api/assets?' + params.toString())
        if (!cancelled) {
          setAssets(found)
          setLoading(false)
        }
      } catch {
        if (!cancelled) {
          setAssets([])
          setLoading(false)
        }
      }
    }, 150)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [kind, query])

  const visible = checkpointStage
    ? assets.filter((asset) => asset.details.stage === checkpointStage)
    : assets

  return <div className="modal-backdrop" onMouseDown={onClose}>
    <section className="asset-picker surface" onMouseDown={(event) => event.stopPropagation()}>
      <div className="asset-picker-head"><div>
        <span className="eyebrow">ASSET PICKER</span><h2>Select {kind}</h2>
      </div><button className="icon-button" onClick={onClose}><X size={17} /></button></div>
      <div className="search-field picker-search"><Search size={16} /><input autoFocus value={query} onChange={(e) => setQuery(e.target.value)} placeholder={'Search ' + kind + ' assets'} /></div>
      <div className="asset-picker-list">
        {loading ? <div className="asset-loading"><RefreshCw className="spin" size={17} />Scanning…</div> :
          visible.length ? visible.map((asset) =>
            <button className="asset-picker-row" key={asset.path} onClick={() => onSelect(asset)}>
              <div className="asset-type-icon" data-kind={asset.kind}>{assetIcon(asset.kind)}</div>
              <div><strong>{asset.name}</strong><span className="mono">{asset.path}</span><small>{assetDetail(asset)}</small></div>
              <ChevronRight size={16} />
            </button>
          ) : <EmptyState icon={<FolderOpen size={24} />} title="No matching assets" description="Manual paths and Hugging Face model IDs remain available in the form." />}
      </div>
    </section>
  </div>
}

function NewRunPage({ surfaces, devices, initialStage, onCreated }: {
  surfaces: StageSurfaces; devices: HardwareDevice[]; initialStage: Stage; onCreated: (run: RunRecord) => void
}) {
  const buildDefault = useCallback((stage: Stage): RunCreate => {
    const surface = surfaces[stage] || fallbackSurfaces[stage]
    return {
      name: surface.label.toLowerCase().replace(' ', '-') + '-' + new Date().toISOString().slice(0, 10),
      stage, model: surface.defaults.model, data: '', output_dir: '/workspace/runs/' + stage, validation_data: '',
      world_size: surface.defaults.world_size, device: surface.defaults.device, dtype: surface.defaults.dtype,
      resume: '', init_cid_checkpoint: '', visible_devices: null, parameters: { ...surface.defaults.parameters }, extra_args: [],
    }
  }, [surfaces])
  const [form, setForm] = useState<RunCreate>(() => buildDefault(initialStage))
  const [preview, setPreview] = useState<CommandPreview | null>(null)
  const [advanced, setAdvanced] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pickerTarget, setPickerTarget] = useState<PickerTarget | null>(null)
  const visibleDevices = useMemo(() => devices.slice(0, form.world_size).map((device) => device.index), [devices, form.world_size])
  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!form.data || !form.output_dir || !form.model) { setPreview(null); return }
      const request = form.device === 'cuda' && devices.length ? { ...form, visible_devices: visibleDevices } : form
      api<CommandPreview>('/api/runs/preview', { method: 'POST', body: JSON.stringify(request) }).then(setPreview).catch(() => setPreview(null))
    }, 250)
    return () => window.clearTimeout(timer)
  }, [form, devices.length, visibleDevices])
  const update = <K extends keyof RunCreate>(key: K, value: RunCreate[K]) => setForm((current) => ({ ...current, [key]: value }))
  const setStage = (stage: Stage) => setForm(buildDefault(stage))
  const updateParam = (key: string, value: Primitive) => setForm((current) => ({ ...current, parameters: { ...current.parameters, [key]: value } }))
  async function launch() {
    if (!form.data || !form.output_dir || !form.model) { setError('Model, training data, and output directory are required.'); return }
    setLaunching(true); setError(null)
    const request = form.device === 'cuda' && devices.length ? { ...form, visible_devices: visibleDevices } : form
    try { onCreated(await api<RunRecord>('/api/runs?launch=true', { method: 'POST', body: JSON.stringify(request) })) }
    catch (err) { setError(err instanceof Error ? err.message : 'Failed to launch run') }
    finally { setLaunching(false) }
  }
  function selectAsset(asset: AssetRecord) {
    if (pickerTarget === 'model') update('model', asset.path)
    if (pickerTarget === 'data') update('data', asset.path)
    if (pickerTarget === 'validation') update('validation_data', asset.path)
    if (pickerTarget === 'resume') update('resume', asset.path)
    if (pickerTarget === 'init') update('init_cid_checkpoint', asset.path)
    setPickerTarget(null)
  }
  const pickerKind: AssetKind =
    pickerTarget === 'model' ? 'model' :
      pickerTarget === 'data' && form.stage === 'stage0' ? 'manifest' :
        pickerTarget === 'data' || pickerTarget === 'validation' ? 'dataset' : 'checkpoint'
  const pickerStage =
    pickerTarget === 'init' ? 'stage-a' as Stage :
      pickerTarget === 'resume' ? form.stage : undefined

  const commonKeys = form.stage === 'stage0' ? ['steps', 'learning_rate', 'micro_batch_size', 'target_global_batch_size', 'checkpoint_every'] :
    form.stage === 'stage-a' ? ['epochs', 'learning_rate', 'micro_batch_size', 'target_global_batch_size', 'checkpoint_every_steps'] :
      ['epochs', 'learning_rate', 'backbone_lr_scale', 'target_global_batch_size', 'checkpoint_every_steps']
  const advancedKeys = Object.keys(form.parameters).filter((key) => !commonKeys.includes(key))
  return <div className="page new-run-page">
    <section className="page-heading compact-heading"><div><span className="overline">LAUNCHER</span><h1>New training run</h1>
      <p>Configure the main CID repository through a safer, inspectable interface.</p></div></section>
    <div className="launch-layout"><div className="launch-form">
      <section className="surface config-section"><div className="config-section-heading"><span className="step-badge">1</span><div><h2>Training stage</h2><p>Select the main-repository entry point.</p></div></div>
        <div className="stage-selector">{(Object.entries(surfaces) as [Stage, StageSurface][]).map(([stage, surface]) =>
          <button key={stage} className={'stage-option ' + (form.stage === stage ? 'selected' : '')} onClick={() => setStage(stage)}>
            <span className="radio"><span /></span><div><strong>{surface.label}</strong><small>{surface.description}</small></div>
          </button>)}</div>
      </section>
      <section className="surface config-section"><div className="config-section-heading"><span className="step-badge">2</span><div><h2>Model & data</h2><p>Point Factory at the assets consumed by CID.</p></div></div>
        <div className="form-grid">
          <Field label="Run name"><input value={form.name} onChange={(e) => update('name', e.target.value)} /></Field>
          <Field label="Base model"><AssetInput value={form.model} onChange={(value) => update('model', value)} onBrowse={() => setPickerTarget('model')} /></Field>
          <Field label={form.stage === 'stage0' ? 'Data manifest' : 'Training data'} hint={form.stage === 'stage0' ? 'Prepared diffusion stream manifest' : 'Trajectory JSONL'}>
            <AssetInput placeholder={form.stage === 'stage0' ? '/data/manifest.json' : '/data/train.jsonl'} value={form.data} onChange={(value) => update('data', value)} onBrowse={() => setPickerTarget('data')} />
          </Field>
          {form.stage !== 'stage0' && <Field label="Validation data" hint="Optional"><AssetInput placeholder="/data/validation.jsonl" value={form.validation_data || ''} onChange={(value) => update('validation_data', value)} onBrowse={() => setPickerTarget('validation')} /></Field>}
          <Field label="Output directory"><input className="mono-input" value={form.output_dir} onChange={(e) => update('output_dir', e.target.value)} /></Field>
          {form.stage === 'stage-b' && <Field label="Stage A checkpoint" hint="Required for a fresh Stage B"><AssetInput placeholder="/runs/stage-a/stage-a-latest.pt" value={form.init_cid_checkpoint || ''} onChange={(value) => update('init_cid_checkpoint', value)} onBrowse={() => setPickerTarget('init')} /></Field>}
          <Field label="Resume checkpoint" hint="Optional"><AssetInput placeholder="Leave empty for a fresh run" value={form.resume || ''} onChange={(value) => update('resume', value)} onBrowse={() => setPickerTarget('resume')} /></Field>
        </div>
      </section>
      <section className="surface config-section"><div className="config-section-heading"><span className="step-badge">3</span><div><h2>Compute</h2><p>Choose the device and distributed world size.</p></div></div>
        <div className="compute-row">
          <Field label="Device"><div className="segmented">{(['cuda', 'npu', 'cpu', 'auto'] as const).map((device) =>
            <button key={device} className={form.device === device ? 'active' : ''} onClick={() => update('device', device)}>{device.toUpperCase()}</button>)}</div></Field>
          <Field label="World size"><div className="number-control"><button onClick={() => update('world_size', Math.max(1, form.world_size - 1))}>−</button>
            <input type="number" min={1} max={64} value={form.world_size} onChange={(e) => update('world_size', Math.max(1, Number(e.target.value) || 1))} />
            <button onClick={() => update('world_size', form.world_size + 1)}>+</button></div></Field>
          {form.stage !== 'stage0' && <Field label="Precision"><div className="segmented"><button className={form.dtype === 'bf16' ? 'active' : ''} onClick={() => update('dtype', 'bf16')}>BF16</button>
            {form.stage !== 'stage-b' && <button className={form.dtype === 'fp32' ? 'active' : ''} onClick={() => update('dtype', 'fp32')}>FP32</button>}</div></Field>}
        </div>
        {form.device === 'cuda' && devices.length > 0 && <div className="selected-hardware"><Cpu size={15} /><span>Factory will expose GPUs {visibleDevices.join(', ')} to this run.</span><small>{visibleDevices.length}/{devices.length} selected automatically</small></div>}
      </section>
      <section className="surface config-section"><div className="config-section-heading"><span className="step-badge">4</span><div><h2>Training</h2><p>Common parameters stay visible; low-level controls stay out of the way.</p></div></div>
        <div className="parameter-grid">{commonKeys.filter((key) => key in form.parameters).map((key) => {
          const meta = parameterMeta[key] || { label: key }; const value = form.parameters[key]
          return <Field key={key} label={meta.label}><input type="number" step="any" value={typeof value === 'number' ? value : ''} onChange={(e) => updateParam(key, Number(e.target.value))} /></Field>
        })}</div>
        <button className="advanced-toggle" onClick={() => setAdvanced((value) => !value)}><SlidersHorizontal size={16} />Advanced controls<span>{advancedKeys.length} parameters</span>{advanced ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</button>
        {advanced && <div className="advanced-panel">{advancedKeys.map((key) => {
          const meta = parameterMeta[key] || { label: key.replaceAll('_', ' ') }; const value = form.parameters[key]
          if (typeof value === 'boolean') return <div className="toggle-row" key={key}><div><strong>{meta.label}</strong></div><Toggle checked={value} onChange={(next) => updateParam(key, next)} /></div>
          if (meta.type === 'select' && meta.options) return <Field key={key} label={meta.label}><select value={String(value ?? '')} onChange={(e) => updateParam(key, e.target.value)}>{meta.options.map((option) => <option key={option}>{option}</option>)}</select></Field>
          return <Field key={key} label={meta.label}><input type={typeof value === 'number' ? 'number' : 'text'} step="any" value={value == null ? '' : String(value)}
            onChange={(e) => updateParam(key, typeof value === 'number' ? Number(e.target.value) : e.target.value)} /></Field>
        })}</div>}
      </section>
    </div>
    <aside className="launch-summary"><div className="surface summary-card">
      <div className="summary-title"><div><span className="eyebrow">REVIEW</span><h2>Ready to launch</h2></div><span className="stage-table-pill" data-stage={form.stage}>{stageLabel(form.stage)}</span></div>
      <div className="summary-list"><div><span>Model</span><strong>{form.model}</strong></div><div><span>Compute</span><strong>{form.world_size} × {form.device.toUpperCase()}</strong></div><div><span>Output</span><strong>{form.output_dir}</strong></div></div>
      {preview?.warnings.map((warning) => <div className="warning-box" key={warning}><AlertTriangle size={16} /><span>{warning}</span></div>)}
      <div className="command-preview"><div className="command-head"><span><TerminalSquare size={14} />Main-repo command</span><span className="mono">{preview?.cwd || 'Waiting for fields'}</span></div>
        <pre>{preview?.display || 'Complete the model, data, and output fields to preview the exact command.'}</pre></div>
      {error && <div className="error-box">{error}</div>}
      <button className="launch-button" onClick={launch} disabled={launching}>{launching ? <RefreshCw className="spin" size={17} /> : <Play size={17} />}{launching ? 'Launching…' : 'Launch training'}</button>
      <p className="launch-note">Factory starts this command in the configured CID repository and reads metrics written by CID itself.</p>
    </div></aside></div>
    {pickerTarget && <AssetPicker kind={pickerKind} checkpointStage={pickerStage} onSelect={selectAsset} onClose={() => setPickerTarget(null)} />}
  </div>
}

function LossChart({ data }: { data: { step: number; loss: number }[] }) {
  const width = 820
  const height = 270
  const padX = 48
  const padY = 28
  const values = data.map((item) => item.loss).filter(Number.isFinite)
  const minLoss = Math.min(...values)
  const maxLoss = Math.max(...values)
  const span = Math.max(maxLoss - minLoss, Math.max(Math.abs(maxLoss), 1) * 0.04)
  const low = minLoss - span * 0.12
  const high = maxLoss + span * 0.12
  const x = (index: number) => padX + index / Math.max(1, data.length - 1) * (width - padX * 2)
  const y = (value: number) => padY + (high - value) / Math.max(1e-12, high - low) * (height - padY * 2)
  const points = data.map((item, index) => x(index).toFixed(1) + ',' + y(item.loss).toFixed(1)).join(' ')
  const area = padX + ',' + (height - padY) + ' ' + points + ' ' + (width - padX) + ',' + (height - padY)
  const last = data[data.length - 1]
  return <div className="native-chart">
    <svg viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label="Training loss chart">
      <defs>
        <linearGradient id="lossArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const gy = padY + ratio * (height - padY * 2)
        const label = high - ratio * (high - low)
        return <g key={ratio}>
          <line x1={padX} x2={width - padX} y1={gy} y2={gy} className="chart-grid-line" />
          <text x={padX - 10} y={gy + 3} textAnchor="end" className="chart-axis-label">{label.toFixed(3)}</text>
        </g>
      })}
      <polygon points={area} fill="url(#lossArea)" />
      <polyline points={points} className="chart-loss-line" />
      <circle cx={x(data.length - 1)} cy={y(last.loss)} r="4" className="chart-last-dot" />
      <text x={padX} y={height - 5} textAnchor="start" className="chart-axis-label">{data[0].step}</text>
      <text x={width - padX} y={height - 5} textAnchor="end" className="chart-axis-label">{last.step}</text>
    </svg>
  </div>
}

function ComparisonChart({ runs }: { runs: RunComparisonEntry[] }) {
  const width = 900
  const height = 300
  const padX = 52
  const padY = 30
  const series = runs.map((run) => run.metrics.map((item, index) => ({
    step: Number(item.optimizer_steps ?? item.step ?? index),
    loss: Number(item.mean_loss ?? item.loss ?? Number.NaN),
  })).filter((item) => Number.isFinite(item.step) && Number.isFinite(item.loss)))
  const all = series.flat()
  if (all.length < 2) return <EmptyState icon={<Activity size={24} />} title="Not enough metrics" description="At least two metric points are needed for comparison." />

  const minStep = Math.min(...all.map((item) => item.step))
  const maxStep = Math.max(...all.map((item) => item.step))
  const minLoss = Math.min(...all.map((item) => item.loss))
  const maxLoss = Math.max(...all.map((item) => item.loss))
  const lossSpan = Math.max(maxLoss - minLoss, Math.max(Math.abs(maxLoss), 1) * 0.04)
  const low = minLoss - lossSpan * 0.12
  const high = maxLoss + lossSpan * 0.12
  const x = (step: number) => padX + (step - minStep) / Math.max(1, maxStep - minStep) * (width - padX * 2)
  const y = (loss: number) => padY + (high - loss) / Math.max(1e-12, high - low) * (height - padY * 2)

  return <div className="native-chart comparison-chart">
    <svg viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label="Run loss comparison">
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const gy = padY + ratio * (height - padY * 2)
        const label = high - ratio * (high - low)
        return <g key={ratio}>
          <line x1={padX} x2={width - padX} y1={gy} y2={gy} className="chart-grid-line" />
          <text x={padX - 10} y={gy + 3} textAnchor="end" className="chart-axis-label">{label.toFixed(3)}</text>
        </g>
      })}
      {series.map((items, index) =>
        <polyline
          key={runs[index].id}
          points={items.map((item) => x(item.step).toFixed(1) + ',' + y(item.loss).toFixed(1)).join(' ')}
          className={'comparison-line series-' + index}
        />
      )}
      <text x={padX} y={height - 6} textAnchor="start" className="chart-axis-label">{minStep}</text>
      <text x={width - padX} y={height - 6} textAnchor="end" className="chart-axis-label">{maxStep}</text>
    </svg>
  </div>
}

function ComparePage({ runIds, onBack }: { runIds: string[]; onBack: () => void }) {
  const [entries, setEntries] = useState<RunComparisonEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const params = new URLSearchParams()
    runIds.forEach((runId) => params.append('run_id', runId))
    api<RunComparisonEntry[]>('/api/runs/compare?' + params.toString())
      .then((data) => { setEntries(data); setError(null); setLoading(false) })
      .catch((err) => { setError(err instanceof Error ? err.message : 'Comparison failed'); setLoading(false) })
  }, [runIds])

  const configRows = [
    ['Stage', (entry: RunComparisonEntry) => stageLabel(entry.stage)],
    ['Model', (entry: RunComparisonEntry) => entry.model],
    ['Device', (entry: RunComparisonEntry) => entry.world_size + ' × ' + entry.device.toUpperCase()],
    ['Learning rate', (entry: RunComparisonEntry) => String(entry.parameters.learning_rate ?? '—')],
    ['Global batch', (entry: RunComparisonEntry) => String(entry.parameters.target_global_batch_size ?? '—')],
    ['CID commit', (entry: RunComparisonEntry) => entry.repo_head || '—'],
  ] as const

  return <div className="page">
    <button className="back-button" onClick={onBack}><ArrowLeft size={16} />Back to runs</button>
    <section className="page-heading compact-heading"><div>
      <span className="overline">EXPERIMENT ANALYSIS</span><h1>Compare runs</h1>
      <p>Native CID metrics and the exact recorded launch configuration, aligned side by side.</p>
    </div></section>

    {loading ? <section className="surface panel"><div className="asset-loading"><RefreshCw className="spin" size={18} />Loading run metrics…</div></section> :
      error ? <section className="surface panel"><div className="error-box">{error}</div></section> :
        <>
          <div className="comparison-card-grid">{entries.map((entry, index) =>
            <article className="surface comparison-card" key={entry.id}>
              <div className="comparison-card-head"><span className={'series-dot series-' + index} /><div><strong>{entry.name}</strong><small>{stageLabel(entry.stage)} · {entry.world_size} × {entry.device.toUpperCase()}</small></div><StatusPill status={entry.status} /></div>
              <div className="comparison-metrics">
                <div><span>Latest loss</span><strong>{entry.latest_loss != null ? entry.latest_loss.toFixed(4) : '—'}</strong></div>
                <div><span>Step</span><strong>{entry.latest_step?.toLocaleString() || '—'}</strong></div>
                <div><span>Validation</span><strong>{entry.validation_loss != null ? entry.validation_loss.toFixed(4) : '—'}</strong></div>
                <div><span>Progress</span><strong>{entry.progress_fraction != null ? Math.round(entry.progress_fraction * 100) + '%' : '—'}</strong></div>
              </div>
            </article>
          )}</div>

          <section className="surface chart-panel comparison-panel">
            <div className="section-heading"><div><span className="eyebrow">TRAINING</span><h2>Loss by optimizer step</h2></div><span className="metric-source">Native CID metrics</span></div>
            <div className="chart-wrap comparison-chart-wrap"><ComparisonChart runs={entries} /></div>
            <div className="comparison-legend">{entries.map((entry, index) =>
              <span key={entry.id}><i className={'series-dot series-' + index} />{entry.name}</span>
            )}</div>
          </section>

          <section className="surface data-panel comparison-table-panel">
            <div className="section-heading"><div><span className="eyebrow">CONFIGURATION</span><h2>Recorded launch settings</h2></div></div>
            <div className="runs-table-wrap"><table className="comparison-table"><thead><tr><th>Setting</th>{entries.map((entry) => <th key={entry.id}>{entry.name}</th>)}</tr></thead>
              <tbody>{configRows.map(([label, value]) => <tr key={label}><td>{label}</td>{entries.map((entry) => <td key={entry.id} className={label === 'CID commit' ? 'mono' : ''}>{value(entry)}</td>)}</tr>)}</tbody>
            </table></div>
          </section>
        </>}
  </div>
}

function RunDetail({ run, onBack, onStop }: { run: RunRecord; onBack: () => void; onStop: (run: RunRecord) => void }) {
  const [metrics, setMetrics] = useState<Record<string, unknown>[]>([])
  const [validation, setValidation] = useState<Record<string, unknown>[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [tab, setTab] = useState<'metrics' | 'logs' | 'command'>('metrics')
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const result = await api<{ training: { records: Record<string, unknown>[] }; validation: { records: Record<string, unknown>[] } }>('/api/runs/' + run.id + '/metrics')
        const logResult = await api<{ lines: string[] }>('/api/runs/' + run.id + '/logs?lines=500')
        if (!cancelled) { setMetrics(result.training.records); setValidation(result.validation.records); setLogs(logResult.lines) }
      } catch { /* just-launched runs may not have files yet */ }
    }
    load(); const timer = window.setInterval(load, run.status === 'running' ? 2500 : 10000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [run.id, run.status])
  const chartData = metrics.map((item, index) => ({ step: Number(item.optimizer_steps ?? item.step ?? index), loss: Number(item.mean_loss ?? item.loss ?? 0), lr: Number(item.learning_rate ?? item.lr ?? 0) }))
  const latest = metrics.at(-1); const latestStep = Number(latest?.optimizer_steps ?? latest?.step ?? 0); const latestLoss = Number(latest?.mean_loss ?? latest?.loss ?? NaN); const latestLr = Number(latest?.learning_rate ?? latest?.lr ?? NaN)
  return <div className="page"><button className="back-button" onClick={onBack}><ArrowLeft size={16} />Back to runs</button>
    <section className="run-detail-head"><div className="run-detail-title"><div className="run-stage-mark large" data-stage={run.stage}><Workflow size={20} /></div>
      <div><div className="title-row"><h1>{run.name}</h1><StatusPill status={run.status} /></div><p>{stageLabel(run.stage)} · <span className="mono">{run.request.model}</span></p></div></div>
      {run.status === 'running' && <button className="danger-button" onClick={() => onStop(run)}><CircleStop size={16} />Stop run</button>}</section>
    <div className="run-stats-grid">
      <StatCard icon={<Activity size={18} />} label="Optimizer step" value={latestStep ? latestStep.toLocaleString() : '—'} detail={metrics.length + ' metric records'} />
      <StatCard icon={<Gauge size={18} />} label="Loss" value={Number.isFinite(latestLoss) ? latestLoss.toFixed(4) : '—'} detail="Latest training metric" />
      <StatCard icon={<Zap size={18} />} label="Learning rate" value={Number.isFinite(latestLr) ? latestLr.toExponential(2) : '—'} detail="Current schedule value" />
      <StatCard icon={<Clock3 size={18} />} label="Started" value={formatTime(run.started_at)} detail={run.repo_head ? 'CID ' + run.repo_head : 'Source commit unavailable'} />
    </div>
    <div className="detail-tabs"><button className={tab === 'metrics' ? 'active' : ''} onClick={() => setTab('metrics')}>Metrics</button>
      <button className={tab === 'logs' ? 'active' : ''} onClick={() => setTab('logs')}>Logs</button><button className={tab === 'command' ? 'active' : ''} onClick={() => setTab('command')}>Command</button></div>
    {tab === 'metrics' && <div className="metrics-layout"><section className="surface chart-panel"><div className="section-heading"><div><span className="eyebrow">TRAINING</span><h2>Loss</h2></div><span className="metric-source">{metrics.length ? 'Native CID metrics' : 'Waiting for metrics'}</span></div>
      {chartData.length > 1 ? <div className="chart-wrap"><LossChart data={chartData} /></div> :
        <EmptyState icon={<Activity size={24} />} title="Waiting for metrics" description="The chart populates directly from the CID repository train_metrics JSONL." />}</section>
      <section className="surface panel validation-panel"><div className="section-heading"><div><span className="eyebrow">VALIDATION</span><h2>Epoch checks</h2></div><span className="section-count">{validation.length}</span></div>
        {validation.length ? <div className="validation-list">{validation.slice(-6).reverse().map((record, index) => <div key={index}><span>Epoch {String(record.epoch ?? '—')}</span>
          <strong>{Number.isFinite(Number(record.mean_loss)) ? Number(record.mean_loss).toFixed(4) : Number.isFinite(Number(record.loss)) ? Number(record.loss).toFixed(4) : 'recorded'}</strong></div>)}</div> :
          <p className="muted-copy">No validation metrics have been written yet.</p>}</section></div>}
    {tab === 'logs' && <section className="surface terminal-panel"><div className="terminal-head"><span><TerminalSquare size={15} />{run.log_path}</span><span>{logs.length} lines</span></div><pre>{logs.length ? logs.join('\n') : 'Waiting for process output…'}</pre></section>}
    {tab === 'command' && <section className="surface command-detail"><div className="command-detail-grid"><div><span>CID repository</span><strong className="mono">{run.command.cwd}</strong></div>
      <div><span>Source commit</span><strong className="mono">{run.repo_head || '—'}</strong></div><div><span>Output directory</span><strong className="mono">{run.output_dir}</strong></div><div><span>PID</span><strong className="mono">{run.pid || '—'}</strong></div><div><span>Execution host</span><strong className="mono">{run.execution_host || '—'}</strong></div></div>
      <div className="code-block"><div><Code2 size={15} />Exact command</div><pre>{run.command.display}</pre></div></section>}
  </div>
}

function SettingsPage({ repository, runtime }: { repository: RepositoryInfo | null; runtime: RuntimeInfo | null }) {
  return <div className="page narrow-page"><section className="page-heading compact-heading"><div><span className="overline">CONFIGURATION</span><h1>Settings</h1>
    <p>Factory is intentionally thin. The main CID repository remains the source of training behavior.</p></div></section>
    <section className="surface settings-section"><div className="settings-heading"><div className="settings-icon"><GitBranch size={18} /></div><div><h2>CID repository</h2><p>Training commands are executed from this checkout.</p></div></div>
      <div className="settings-grid"><div><span>Path</span><strong className="mono">{repository?.path || '—'}</strong></div><div><span>Branch</span><strong>{repository?.branch || '—'}</strong></div>
        <div><span>Commit</span><strong className="mono">{repository?.head || '—'}</strong></div><div><span>Working tree</span><strong>{repository?.dirty ? 'Modified' : 'Clean'}</strong></div>
        <div className="span-two"><span>Remote</span><strong className="mono">{repository?.remote || '—'}</strong></div></div>
      <div className="info-banner"><Check size={16} /><span>Factory does not contain CID model, optimizer, dataset, or runtime implementations. It launches and observes the repository above.</span></div>
    </section>
    <section className="surface settings-section"><div className="settings-heading"><div className="settings-icon"><TerminalSquare size={18} /></div><div><h2>CID training runtime</h2><p>Factory stays lightweight and launches the main repository with this Python environment.</p></div></div>
      <div className="settings-grid">
        <div className="span-two"><span>Python</span><strong className="mono">{runtime?.python || '—'}</strong></div>
        <div><span>Python version</span><strong>{runtime?.python_version || '—'}</strong></div>
        <div><span>PyTorch</span><strong>{runtime?.torch_version || 'not detected'}</strong></div>
        <div><span>Transformers</span><strong>{runtime?.transformers_version || 'not detected'}</strong></div>
        <div><span>CUDA</span><strong>{runtime?.cuda_available ? (runtime.cuda_device_count || 0) + ' device(s)' : 'not available'}</strong></div>
      </div>
      {!runtime?.available && runtime?.error && <div className="warning-box"><AlertTriangle size={16} /><span>{runtime.error}</span></div>}
    </section>
    <section className="surface settings-section"><div className="settings-heading"><div className="settings-icon"><Settings size={18} /></div><div><h2>Environment overrides</h2><p>Configure Factory without editing its source.</p></div></div>
      <div className="env-list"><div><code>CID_FACTORY_CID_REPO</code><span>Path to the main CID checkout</span></div><div><code>CID_FACTORY_CID_PYTHON</code><span>Python environment used to execute CID training</span></div><div><code>CID_FACTORY_ASSET_ROOTS</code><span>Path-separated roots scanned by the read-only asset catalog</span></div><div><code>CID_FACTORY_STATE_DIR</code><span>Factory run metadata and process logs</span></div></div>
    </section>
  </div>
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (localStorage.getItem('cid-factory-theme') as 'light' | 'dark') || 'dark')
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [devices, setDevices] = useState<HardwareDevice[]>([])
  const [repository, setRepository] = useState<RepositoryInfo | null>(null)
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null)
  const [surfaces, setSurfaces] = useState<StageSurfaces>(fallbackSurfaces)
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null)
  const [comparisonRunIds, setComparisonRunIds] = useState<string[]>([])
  const [newRunStage, setNewRunStage] = useState<Stage>('stage-a')
  const [connected, setConnected] = useState(true)

  const load = useCallback(async () => {
    try {
      const [runData, hardwareData, repoData, runtimeData, surfaceData] = await Promise.all([
        api<RunRecord[]>('/api/runs'), api<{ devices: HardwareDevice[] }>('/api/hardware'),
        api<RepositoryInfo>('/api/repository'), api<RuntimeInfo>('/api/runtime'),
        api<StageSurfaces>('/api/surfaces'),
      ])
      setRuns(runData); setDevices(hardwareData.devices); setRepository(repoData); setRuntime(runtimeData); setSurfaces(surfaceData); setConnected(true)
      setSelectedRun((current) => current ? runData.find((run) => run.id === current.id) || current : null)
    } catch { setConnected(false) }
  }, [])
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('cid-factory-theme', theme) }, [theme])
  useEffect(() => {
    const initial = window.setTimeout(load, 0)
    const timer = window.setInterval(load, 5000)
    return () => { window.clearTimeout(initial); window.clearInterval(timer) }
  }, [load])
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === 'n' && !['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target as HTMLElement).tagName)) {
        setSelectedRun(null); setNewRunStage('stage-a'); setPage('new')
      }
    }
    window.addEventListener('keydown', keydown); return () => window.removeEventListener('keydown', keydown)
  }, [])
  function openNewRun(stage: Stage = 'stage-a') { setSelectedRun(null); setComparisonRunIds([]); setNewRunStage(stage); setPage('new') }
  function changePage(next: Page) { setSelectedRun(null); setComparisonRunIds([]); setPage(next) }
  async function stopRun(run: RunRecord) {
    try { setSelectedRun(await api<RunRecord>('/api/runs/' + run.id + '/stop', { method: 'POST' })); load() } catch { /* preserve state */ }
  }
  return <div className="app-shell"><Sidebar page={page} setPage={changePage} onNewRun={() => openNewRun()} />
    <MobileNav page={page} setPage={changePage} onNewRun={() => openNewRun()} />
    <div className="main-column"><Topbar theme={theme} toggleTheme={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')} repository={repository} refresh={load} />
      {!connected && <div className="connection-banner"><AlertTriangle size={15} />Factory API is unavailable. The interface will reconnect automatically.</div>}
      <main>{selectedRun ? <RunDetail run={selectedRun} onBack={() => setSelectedRun(null)} onStop={stopRun} /> :
        comparisonRunIds.length ? <ComparePage runIds={comparisonRunIds} onBack={() => setComparisonRunIds([])} /> :
          page === 'overview' ? <Overview runs={runs} devices={devices} repository={repository} surfaces={surfaces} onNewRun={openNewRun} onRun={(run) => { setComparisonRunIds([]); setSelectedRun(run) }} /> :
            page === 'new' ? <NewRunPage key={newRunStage} surfaces={surfaces} devices={devices} initialStage={newRunStage} onCreated={(run) => { setSelectedRun(run); load() }} /> :
              page === 'runs' ? <RunsPage runs={runs} onRun={(run) => { setComparisonRunIds([]); setSelectedRun(run) }} onNewRun={() => openNewRun()} onCompare={setComparisonRunIds} /> :
                page === 'assets' ? <AssetsPage /> :
                  page === 'hardware' ? <HardwarePage devices={devices} /> : <SettingsPage repository={repository} runtime={runtime} />}</main>
    </div>
  </div>
}
export default App
