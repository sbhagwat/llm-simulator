import { useState, useEffect, useRef, useCallback } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { Play, Pause, RotateCcw, HardDrive, Cpu, Zap, Network, Activity, Database } from 'lucide-react'

// ─── constants ───────────────────────────────────────────────────────────────
const BLOCK_SIZES = { '4K': 4096, '16K': 16384, '64K': 65536, '128K': 131072 }
const MAX_QUEUE = 32
const NAND_PAGES = 128          // visual grid cells
const TICK_MS = 280

// ─── helpers ─────────────────────────────────────────────────────────────────
let _cmdId = 1
function newCmdId() { return _cmdId++ }

function randBetween(lo, hi) { return lo + Math.floor(Math.random() * (hi - lo + 1)) }

function fmtBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB/s'
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB/s'
  return (b / 1e3).toFixed(0) + ' KB/s'
}

// Realistic latency model (µs)
function simulateLatency(blockBytes, isWrite, queueDepth) {
  const queueUs   = queueDepth * randBetween(2, 6)          // waiting in SQ
  const pcieUs    = Math.ceil(blockBytes / (16 * 1024))      // PCIe DMA (16 GB/s)
  const nandUs    = isWrite
    ? randBetween(50, 120)                                    // NAND write program
    : randBetween(20, 60)                                     // NAND read
  const networkUs = randBetween(80, 300)                     // EBS network hop
  const total     = queueUs + pcieUs + nandUs + networkUs
  return { queueUs, pcieUs, nandUs, networkUs, total }
}

// ─── sub-components ──────────────────────────────────────────────────────────
function Card({ title, icon: Icon, children, style }) {
  return (
    <div style={{ background: '#13131a', border: '1px solid #1e1e2e', borderRadius: 12, padding: 16, ...style }}>
      {title && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          {Icon && <Icon size={14} color="#7c3aed" />}
          <span style={{ fontSize: 11, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            {title}
          </span>
        </div>
      )}
      {children}
    </div>
  )
}

function StatBox({ label, value, sub, color = '#f1f5f9' }) {
  return (
    <div style={{ textAlign: 'center', background: '#0a0a0f', padding: '10px 16px', borderRadius: 8, border: '1px solid #1e1e2e', minWidth: 90 }}>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: '#7c3aed', marginTop: 1 }}>{sub}</div>}
      <div style={{ fontSize: 10, color: '#6b7280', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
    </div>
  )
}

// Pipeline arrow bus
function PipelineArrow({ label, color, active, pct }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
      <div style={{ fontSize: 9, color: '#4b5563', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ position: 'relative', width: 64, height: 12, background: '#1a1a2e', borderRadius: 6, overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${pct}%`,
          background: color,
          borderRadius: 6,
          transition: 'width 0.3s ease',
          boxShadow: active ? `0 0 8px ${color}88` : 'none'
        }} />
      </div>
      <div style={{ fontSize: 9, color: active ? color : '#374151' }}>{pct}%</div>
    </div>
  )
}

// Queue visualizer
function QueueViz({ slots, label, color }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: '#6b7280', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {slots.map((s, i) => (
          <div key={i} style={{
            width: 16, height: 16, borderRadius: 3,
            background: s.used ? (s.type === 'write' ? '#7c3aed' : '#06b6d4') : '#1a1a2e',
            border: `1px solid ${s.used ? color : '#2d2d3e'}`,
            transition: 'background 0.2s',
            boxShadow: s.used ? `0 0 4px ${color}66` : 'none',
          }} title={s.used ? `cmd#${s.id} ${s.type}` : 'empty'} />
        ))}
      </div>
    </div>
  )
}

// NAND flash block map
function NandMap({ pages }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
      {pages.map((p, i) => {
        const bg =
          p.state === 'written'   ? '#7c3aed' :
          p.state === 'active'    ? '#f59e0b' :
          p.state === 'read'      ? '#06b6d4' :
          p.state === 'erased'    ? '#1a2e1a' :
                                    '#1a1a2e'
        return (
          <div key={i} style={{
            width: 14, height: 14, borderRadius: 2,
            background: bg,
            border: '1px solid #0a0a0f',
            transition: 'background 0.3s',
          }} title={`page ${i}: ${p.state}`} />
        )
      })}
    </div>
  )
}

// Pipeline stage box
function Stage({ label, icon: Icon, color, active, sublabel }) {
  return (
    <div style={{
      background: active ? `${color}22` : '#13131a',
      border: `1px solid ${active ? color : '#2d2d3e'}`,
      borderRadius: 10, padding: '10px 14px',
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
      transition: 'all 0.25s',
      boxShadow: active ? `0 0 16px ${color}44` : 'none',
      minWidth: 90
    }}>
      <Icon size={18} color={active ? color : '#374151'} />
      <span style={{ fontSize: 11, fontWeight: 600, color: active ? color : '#4b5563', textAlign: 'center' }}>{label}</span>
      {sublabel && <span style={{ fontSize: 9, color: active ? '#9ca3af' : '#374151', textAlign: 'center' }}>{sublabel}</span>}
    </div>
  )
}

function Arrow({ active, color = '#7c3aed' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', padding: '0 2px' }}>
      <div style={{
        width: 28, height: 2,
        background: active ? color : '#1e1e2e',
        position: 'relative',
        transition: 'background 0.2s',
        boxShadow: active ? `0 0 6px ${color}` : 'none'
      }}>
        <div style={{
          position: 'absolute', right: -4, top: -4,
          borderLeft: `8px solid ${active ? color : '#1e1e2e'}`,
          borderTop: '5px solid transparent',
          borderBottom: '5px solid transparent',
          transition: 'border-color 0.2s'
        }} />
      </div>
    </div>
  )
}

// ─── main component ───────────────────────────────────────────────────────────
export default function EBSSimulator() {
  const [running, setRunning]         = useState(false)
  const [queueDepth, setQueueDepth]   = useState(8)
  const [blockSize, setBlockSize]     = useState('4K')
  const [writeRatio, setWriteRatio]   = useState(70)

  // Metrics
  const [iops, setIops]               = useState({ read: 0, write: 0 })
  const [totalOps, setTotalOps]       = useState(0)
  const [bw, setBw]                   = useState(0)          // bytes/sec
  const [avgLatency, setAvgLatency]   = useState(0)          // µs

  // Pipeline active stages (set of stage names)
  const [activeStages, setActiveStages] = useState(new Set())

  // SQ / CQ slots
  const [sqSlots, setSqSlots]         = useState(Array.from({ length: MAX_QUEUE }, () => ({ used: false })))
  const [cqSlots, setCqSlots]         = useState(Array.from({ length: MAX_QUEUE }, () => ({ used: false })))

  // NAND pages
  const [nandPages, setNandPages]     = useState(Array.from({ length: NAND_PAGES }, () => ({ state: 'free' })))

  // Charts
  const [iopsHistory, setIopsHistory] = useState([])
  const [latHistory, setLatHistory]   = useState([])

  // Request log
  const [reqLog, setReqLog]           = useState([])

  // PCIe utilization
  const [pciePct, setPciePct]         = useState(0)
  const [netPct, setNetPct]           = useState(0)

  // Refs for accumulating per-second stats
  const secCounterRef  = useRef({ reads: 0, writes: 0, bytes: 0, latSum: 0, latCount: 0 })
  const tickRef        = useRef(null)
  const secRef         = useRef(null)
  const stepRef        = useRef(0)

  const flashStages = useCallback((stages, durationMs = 200) => {
    setActiveStages(new Set(stages))
    setTimeout(() => setActiveStages(new Set()), durationMs)
  }, [])

  const doTick = useCallback(() => {
    const isWrite   = Math.random() * 100 < writeRatio
    const bytes     = BLOCK_SIZES[blockSize]
    const { queueUs, pcieUs, nandUs, networkUs, total } = simulateLatency(bytes, isWrite, queueDepth)
    const cmdId     = newCmdId()
    const type      = isWrite ? 'write' : 'read'

    // How many commands to inject this tick (based on queue depth)
    const concurrency = Math.max(1, Math.floor(queueDepth / 4))

    // Update SQ
    setSqSlots(prev => {
      const next = [...prev]
      let filled = 0
      for (let i = 0; i < next.length && filled < concurrency; i++) {
        if (!next[i].used) { next[i] = { used: true, id: cmdId, type }; filled++ }
      }
      return next
    })

    // Animate pipeline stages in sequence
    const stages = ['app', 'kernel', 'nvme-driver', 'sq', 'pcie', 'controller', 'network', 'nand']
    stages.forEach((s, i) => {
      setTimeout(() => setActiveStages(new Set([s])), i * 30)
    })
    setTimeout(() => setActiveStages(new Set()), stages.length * 30 + 60)

    // PCIe / network utilization
    setPciePct(Math.min(99, 20 + queueDepth * 2 + randBetween(0, 15)))
    setNetPct(Math.min(99, 10 + queueDepth + randBetween(0, 20)))

    // Drain SQ → CQ after latency sim
    setTimeout(() => {
      setSqSlots(prev => {
        const next = [...prev]
        let cleared = 0
        for (let i = 0; i < next.length && cleared < concurrency; i++) {
          if (next[i].used) { next[i] = { used: false }; cleared++ }
        }
        return next
      })
      setCqSlots(prev => {
        const next = [...prev]
        let filled = 0
        for (let i = 0; i < next.length && filled < concurrency; i++) {
          if (!next[i].used) { next[i] = { used: true, id: cmdId, type }; filled++ }
        }
        return next
      })
      // Clear CQ shortly after
      setTimeout(() => {
        setCqSlots(prev => prev.map(s => ({ used: false })))
      }, 150)
    }, 220)

    // Update NAND pages
    setNandPages(prev => {
      const next = [...prev]
      const count = Math.max(1, Math.floor(bytes / 4096))
      const candidates = []
      for (let i = 0; i < next.length; i++) {
        if (isWrite ? next[i].state !== 'active' : next[i].state === 'written') candidates.push(i)
      }
      for (let k = 0; k < count && candidates.length; k++) {
        const idx = candidates[randBetween(0, candidates.length - 1)]
        next[idx] = { state: isWrite ? 'active' : 'read' }
      }
      return next
    })
    // Settle NAND state
    setTimeout(() => {
      setNandPages(prev => prev.map(p =>
        p.state === 'active' ? { state: 'written' } :
        p.state === 'read'   ? { state: 'written' } : p
      ))
    }, 300)

    // Accumulate per-second stats
    const sc = secCounterRef.current
    if (isWrite) sc.writes += concurrency; else sc.reads += concurrency
    sc.bytes += bytes * concurrency
    sc.latSum += total; sc.latCount++

    // Update log
    setReqLog(prev => [{
      id: cmdId, type, size: blockSize,
      queueUs, pcieUs, nandUs, networkUs, total,
      ts: Date.now()
    }, ...prev].slice(0, 12))

    setTotalOps(n => n + concurrency)
    stepRef.current++
  }, [writeRatio, blockSize, queueDepth])

  // Per-second aggregation
  useEffect(() => {
    secRef.current = setInterval(() => {
      const sc = secCounterRef.current
      setIops({ read: sc.reads, write: sc.writes })
      setBw(sc.bytes)
      setAvgLatency(sc.latCount ? Math.round(sc.latSum / sc.latCount) : 0)
      setIopsHistory(h => [...h.slice(-40), { t: stepRef.current, read: sc.reads, write: sc.writes }])
      setLatHistory(h => [...h.slice(-40), { t: stepRef.current, lat: sc.latCount ? Math.round(sc.latSum / sc.latCount) : 0 }])
      secCounterRef.current = { reads: 0, writes: 0, bytes: 0, latSum: 0, latCount: 0 }
    }, 1000)
    return () => clearInterval(secRef.current)
  }, [])

  useEffect(() => {
    if (running) {
      tickRef.current = setInterval(doTick, TICK_MS)
    } else {
      clearInterval(tickRef.current)
      setActiveStages(new Set())
    }
    return () => clearInterval(tickRef.current)
  }, [running, doTick])

  const reset = () => {
    setRunning(false)
    setIops({ read: 0, write: 0 }); setTotalOps(0); setBw(0); setAvgLatency(0)
    setActiveStages(new Set())
    setSqSlots(Array.from({ length: MAX_QUEUE }, () => ({ used: false })))
    setCqSlots(Array.from({ length: MAX_QUEUE }, () => ({ used: false })))
    setNandPages(Array.from({ length: NAND_PAGES }, () => ({ state: 'free' })))
    setIopsHistory([]); setLatHistory([]); setReqLog([])
    setPciePct(0); setNetPct(0)
    secCounterRef.current = { reads: 0, writes: 0, bytes: 0, latSum: 0, latCount: 0 }
    stepRef.current = 0; _cmdId = 1
  }

  const as = activeStages

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f', padding: '20px 24px' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          <span style={{ color: '#7c3aed' }}>NVMe / EBS</span> I/O Simulator
        </h1>
        <p style={{ color: '#6b7280', fontSize: 13 }}>
          Simulate how a CPU submits NVMe commands over PCIe to write data through a network-attached SSD — like AWS EBS
        </p>
      </div>

      {/* Controls */}
      <div style={{ background: '#13131a', border: '1px solid #1e1e2e', borderRadius: 12, padding: 16, marginBottom: 14, display: 'flex', alignItems: 'flex-end', gap: 20, flexWrap: 'wrap' }}>
        {/* Queue depth */}
        <div>
          <label style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
            Queue Depth: <span style={{ color: '#a78bfa' }}>{queueDepth}</span>
          </label>
          <input type="range" min={1} max={32} value={queueDepth}
            onChange={e => setQueueDepth(+e.target.value)} disabled={running}
            style={{ width: 140, accentColor: '#7c3aed' }} />
        </div>
        {/* Block size */}
        <div>
          <label style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>Block Size</label>
          <div style={{ display: 'flex', gap: 6 }}>
            {Object.keys(BLOCK_SIZES).map(s => (
              <button key={s} onClick={() => setBlockSize(s)} disabled={running}
                style={{
                  padding: '4px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: running ? 'not-allowed' : 'pointer',
                  background: blockSize === s ? '#7c3aed' : '#1e1e2e',
                  color: blockSize === s ? '#fff' : '#6b7280',
                  border: `1px solid ${blockSize === s ? '#7c3aed' : '#2d2d3e'}`
                }}>{s}</button>
            ))}
          </div>
        </div>
        {/* Write ratio */}
        <div>
          <label style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
            Write Ratio: <span style={{ color: '#a78bfa' }}>{writeRatio}%</span>
          </label>
          <input type="range" min={0} max={100} value={writeRatio}
            onChange={e => setWriteRatio(+e.target.value)} disabled={running}
            style={{ width: 140, accentColor: '#7c3aed' }} />
        </div>
        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, marginLeft: 'auto' }}>
          <button onClick={() => setRunning(r => !r)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px',
              background: running ? '#1e1e2e' : '#7c3aed', color: running ? '#f59e0b' : '#fff',
              border: `1px solid ${running ? '#f59e0b44' : 'transparent'}`,
              borderRadius: 8, cursor: 'pointer', fontWeight: 600, fontSize: 13,
              boxShadow: running ? 'none' : '0 0 18px rgba(124,58,237,0.4)'
            }}>
            {running ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Start I/O</>}
          </button>
          <button onClick={reset}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
              background: '#1e1e2e', color: '#9ca3af', border: '1px solid #2d2d3e',
              borderRadius: 8, cursor: 'pointer', fontWeight: 600, fontSize: 13
            }}>
            <RotateCcw size={14} /> Reset
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <StatBox label="Write IOPS" value={iops.write.toLocaleString()} color="#a78bfa" />
        <StatBox label="Read IOPS"  value={iops.read.toLocaleString()}  color="#06b6d4" />
        <StatBox label="Throughput" value={fmtBytes(bw)}                color="#10b981" />
        <StatBox label="Avg Latency" value={avgLatency ? `${avgLatency}µs` : '—'} color="#f59e0b" />
        <StatBox label="Total Ops"  value={totalOps.toLocaleString()}   color="#f1f5f9" />
      </div>

      {/* Pipeline diagram */}
      <Card title="I/O Pipeline" icon={Activity} style={{ marginBottom: 14 }}>
        <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
          {/* Local path row */}
          <div style={{ marginBottom: 6 }}>
            <div style={{ fontSize: 10, color: '#4b5563', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Host path
            </div>
            <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'nowrap' }}>
              <Stage label="Application" icon={Cpu}      color="#f59e0b" active={as.has('app')}         sublabel="syscall write()" />
              <Arrow active={as.has('app') || as.has('kernel')} color="#f59e0b" />
              <Stage label="OS Kernel"   icon={Database} color="#8b5cf6" active={as.has('kernel')}      sublabel="VFS → Block Layer" />
              <Arrow active={as.has('kernel') || as.has('nvme-driver')} color="#8b5cf6" />
              <Stage label="NVMe Driver" icon={Zap}      color="#06b6d4" active={as.has('nvme-driver')} sublabel="build NVMe cmd" />
              <Arrow active={as.has('nvme-driver') || as.has('sq')} color="#06b6d4" />
              <Stage label="Submit Queue" icon={Activity} color="#7c3aed" active={as.has('sq')}         sublabel="SQ tail + doorbell" />
              <Arrow active={as.has('sq') || as.has('pcie')} color="#7c3aed" />
              <Stage label="PCIe DMA"   icon={Zap}       color="#10b981" active={as.has('pcie')}        sublabel="Gen4 ×4 lanes" />
              <Arrow active={as.has('pcie') || as.has('controller')} color="#10b981" />
              <Stage label="NVMe Ctrl"  icon={HardDrive} color="#f97316" active={as.has('controller')}  sublabel="FTL + scheduler" />
            </div>
          </div>
          {/* EBS network + remote NAND */}
          <div style={{ display: 'flex', alignItems: 'center', marginLeft: 360 }}>
            <div style={{ width: 2, height: 28, background: as.has('controller') ? '#f97316' : '#1e1e2e', transition: 'background 0.2s', marginLeft: 44 }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginLeft: 360 }}>
            <Arrow active={as.has('controller') || as.has('network')} color="#ef4444" />
            <Stage label="EBS Network" icon={Network} color="#ef4444" active={as.has('network')} sublabel="SR-IOV / RDMA" />
            <Arrow active={as.has('network') || as.has('nand')} color="#ef4444" />
            <Stage label="NAND Flash"  icon={HardDrive} color="#22c55e" active={as.has('nand')} sublabel="TLC / QLC pages" />
          </div>
          {/* Bus utilization bars */}
          <div style={{ display: 'flex', gap: 24, marginTop: 16 }}>
            <PipelineArrow label="PCIe Bus" color="#10b981" active={as.has('pcie')} pct={pciePct} />
            <PipelineArrow label="EBS Net"  color="#ef4444" active={as.has('network')} pct={netPct} />
          </div>
        </div>
      </Card>

      {/* Middle grid: queues + NAND */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, marginBottom: 14 }}>

        {/* NVMe Queues */}
        <Card title="NVMe Queue Pair" icon={Activity}>
          <div style={{ display: 'flex', flex: 1, gap: 20, flexWrap: 'wrap' }}>
            <QueueViz slots={sqSlots} label={`Submission Queue (depth ${queueDepth})`} color="#7c3aed" />
            <QueueViz slots={cqSlots} label="Completion Queue" color="#22c55e" />
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: '#6b7280', lineHeight: 1.6 }}>
            <span style={{ color: '#7c3aed' }}>■</span> Write &nbsp;
            <span style={{ color: '#06b6d4' }}>■</span> Read &nbsp;
            <span style={{ color: '#2d2d3e' }}>■</span> Empty slot<br />
            CPU writes command → SQ tail, rings doorbell register → controller DMA-fetches → writes CQ entry
          </div>
        </Card>

        {/* NAND Flash map */}
        <Card title="NAND Flash Page Map" icon={HardDrive}>
          <div style={{ marginBottom: 8, fontSize: 11, color: '#6b7280' }}>
            {NAND_PAGES} pages × 4KB = {(NAND_PAGES * 4)}KB simulated storage
          </div>
          <NandMap pages={nandPages} />
          <div style={{ marginTop: 10, display: 'flex', gap: 12, fontSize: 11, color: '#6b7280', flexWrap: 'wrap' }}>
            {[
              { color: '#1a1a2e', label: 'Free' },
              { color: '#f59e0b', label: 'Active write' },
              { color: '#7c3aed', label: 'Written' },
              { color: '#06b6d4', label: 'Read' },
            ].map(({ color, label }) => (
              <span key={label}><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: color, marginRight: 4 }} />{label}</span>
            ))}
          </div>
        </Card>

        {/* Latency breakdown */}
        <Card title="Latency Breakdown (last op)" icon={Activity}>
          {reqLog.length > 0 ? (() => {
            const r = reqLog[0]
            const bars = [
              { label: 'Queue wait', value: r.queueUs,   color: '#7c3aed', pct: r.queueUs   / r.total },
              { label: 'PCIe DMA',   value: r.pcieUs,    color: '#10b981', pct: r.pcieUs    / r.total },
              { label: 'NAND I/O',   value: r.nandUs,    color: '#f59e0b', pct: r.nandUs    / r.total },
              { label: 'EBS Network',value: r.networkUs, color: '#ef4444', pct: r.networkUs / r.total },
            ]
            return (
              <div>
                {bars.map(b => (
                  <div key={b.label} style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                      <span style={{ color: '#9ca3af' }}>{b.label}</span>
                      <span style={{ color: b.color, fontFamily: 'monospace' }}>{b.value}µs</span>
                    </div>
                    <div style={{ height: 7, background: '#1e1e2e', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${b.pct * 100}%`, background: b.color, borderRadius: 4, transition: 'width 0.4s' }} />
                    </div>
                  </div>
                ))}
                <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid #1e1e2e', display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span style={{ color: '#6b7280' }}>Total end-to-end</span>
                  <span style={{ color: '#f1f5f9', fontWeight: 700, fontFamily: 'monospace' }}>{r.total}µs</span>
                </div>
              </div>
            )
          })() : (
            <div style={{ color: '#374151', fontSize: 12 }}>Start I/O to see latency breakdown</div>
          )}
        </Card>

      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
        <Card title="IOPS Over Time" icon={Activity}>
          {iopsHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={iopsHistory} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#6b7280' }} />
                <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={32} />
                <Tooltip contentStyle={{ background: '#13131a', border: '1px solid #2d2d3e', borderRadius: 6, fontSize: 11 }} />
                <Bar dataKey="write" name="Write" fill="#7c3aed" stackId="a" radius={[0,0,0,0]} />
                <Bar dataKey="read"  name="Read"  fill="#06b6d4" stackId="a" radius={[3,3,0,0]} />
                <Legend wrapperStyle={{ fontSize: 10, color: '#6b7280' }} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: 12 }}>Start I/O to see IOPS</div>
          )}
        </Card>

        <Card title="Avg Latency Over Time (µs)" icon={Activity}>
          {latHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={latHistory} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#6b7280' }} />
                <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={36} unit="µs" />
                <Tooltip contentStyle={{ background: '#13131a', border: '1px solid #2d2d3e', borderRadius: 6, fontSize: 11 }} />
                <Line type="monotone" dataKey="lat" name="Latency" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: 12 }}>Start I/O to see latency</div>
          )}
        </Card>
      </div>

      {/* Request log */}
      <Card title="I/O Request Log" icon={Database}>
        {reqLog.length === 0 ? (
          <div style={{ color: '#374151', fontSize: 12 }}>No requests yet</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr>
                  {['Cmd ID', 'Type', 'Size', 'Queue (µs)', 'PCIe (µs)', 'NAND (µs)', 'Network (µs)', 'Total (µs)'].map(h => (
                    <th key={h} style={{ textAlign: 'left', color: '#4b5563', padding: '4px 10px', borderBottom: '1px solid #1e1e2e', fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reqLog.map((r, i) => (
                  <tr key={r.id} style={{ opacity: i === 0 ? 1 : 0.7 - i * 0.04 }}>
                    <td style={{ padding: '4px 10px', color: '#6b7280', fontFamily: 'monospace' }}>#{r.id}</td>
                    <td style={{ padding: '4px 10px', color: r.type === 'write' ? '#a78bfa' : '#06b6d4', fontWeight: 600 }}>{r.type.toUpperCase()}</td>
                    <td style={{ padding: '4px 10px', color: '#9ca3af', fontFamily: 'monospace' }}>{r.size}</td>
                    <td style={{ padding: '4px 10px', color: '#9ca3af', fontFamily: 'monospace' }}>{r.queueUs}</td>
                    <td style={{ padding: '4px 10px', color: '#9ca3af', fontFamily: 'monospace' }}>{r.pcieUs}</td>
                    <td style={{ padding: '4px 10px', color: '#9ca3af', fontFamily: 'monospace' }}>{r.nandUs}</td>
                    <td style={{ padding: '4px 10px', color: '#ef4444', fontFamily: 'monospace' }}>{r.networkUs}</td>
                    <td style={{ padding: '4px 10px', color: '#f59e0b', fontFamily: 'monospace', fontWeight: 700 }}>{r.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Concept explainer */}
      <div style={{ marginTop: 14, background: '#13131a', border: '1px solid #1e1e2e', borderRadius: 12, padding: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
          How it works (real NVMe / EBS)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 12, color: '#6b7280', lineHeight: 1.7 }}>
          {[
            { title: 'NVMe Queue Pair', body: 'Each CPU core gets its own Submission Queue (SQ) and Completion Queue (CQ) — eliminating lock contention. The CPU writes a 64-byte NVMe command to SQ, then writes to the doorbell MMIO register to notify the controller.' },
            { title: 'PCIe DMA', body: 'The NVMe controller DMA-fetches the command over PCIe, then DMA-reads or DMA-writes the data buffer directly from host memory — no CPU involvement. PCIe Gen4 ×4 gives ~7 GB/s.' },
            { title: 'FTL (Flash Translation Layer)', body: 'The controller\'s FTL maps logical block addresses (LBAs) to physical NAND pages. NAND can\'t overwrite in place — it erases in blocks (~256KB) and writes pages (~4KB). The FTL handles wear leveling and garbage collection.' },
            { title: 'AWS EBS Architecture', body: 'EBS volumes are not local — I/O crosses a high-speed network (AWS Nitro / SR-IOV) to a remote storage server. The EBS agent on the Nitro card intercepts NVMe commands and routes them over the network, adding ~100–300µs latency.' },
          ].map(({ title, body }) => (
            <div key={title}>
              <div style={{ color: '#a78bfa', fontWeight: 600, marginBottom: 4 }}>{title}</div>
              <div>{body}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 14, textAlign: 'center', color: '#374151', fontSize: 11 }}>
        Latency values are modeled approximations. Real EBS gp3 delivers up to 16,000 IOPS and 1,000 MB/s with ~1ms p99 latency.
      </div>
    </div>
  )
}
