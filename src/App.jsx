import { useState, useEffect, useRef, useCallback } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Play, Pause, RotateCcw, ChevronRight, Cpu, Brain, BarChart2, Zap } from 'lucide-react'
import { buildVocab, encode, TinyTransformer } from './transformer'
import EBSSimulator from './EBSSimulator'

const DEFAULT_TEXT = `the cat sat on the mat
the dog ran in the park
a big red fox jumped over
the lazy brown dog slept`

const BLOCK_SIZE = 8

const STAGE_INFO = {
  idle:      { label: 'Ready',         color: '#6b7280', icon: '⏸' },
  tokenize:  { label: 'Tokenizing',    color: '#f59e0b', icon: '✂' },
  embed:     { label: 'Embedding',     color: '#8b5cf6', icon: '🔢' },
  attention: { label: 'Attention',     color: '#06b6d4', icon: '👁' },
  ffn:       { label: 'Feed-Forward',  color: '#10b981', icon: '⚡' },
  loss:      { label: 'Loss',          color: '#ef4444', icon: '📉' },
  backprop:  { label: 'Backprop',      color: '#f97316', icon: '↩' },
  update:    { label: 'Weight Update', color: '#22c55e', icon: '✅' },
}

function Card({ title, icon: Icon, children, style }) {
  return (
    <div style={{
      background: '#13131a', border: '1px solid #1e1e2e',
      borderRadius: 12, padding: '16px', ...style
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        {Icon && <Icon size={15} color="#7c3aed" />}
        <span style={{ fontSize: 12, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  )
}

function TokenChip({ char, index, isActive }) {
  return (
    <div style={{
      display: 'inline-flex', flexDirection: 'column', alignItems: 'center',
      gap: 3, margin: '2px', transition: 'transform 0.2s',
      transform: isActive ? 'scale(1.2)' : 'scale(1)'
    }}>
      <div style={{
        background: isActive ? '#7c3aed' : '#1e1e2e',
        border: `1px solid ${isActive ? '#a855f7' : '#2d2d3e'}`,
        borderRadius: 6, padding: '4px 8px',
        fontSize: 13, fontFamily: 'monospace', fontWeight: 600,
        color: isActive ? '#fff' : '#94a3b8',
        minWidth: 24, textAlign: 'center',
        boxShadow: isActive ? '0 0 10px rgba(124,58,237,0.4)' : 'none',
      }}>
        {char === ' ' ? '␣' : char === '\n' ? '↵' : char}
      </div>
      <div style={{ fontSize: 9, color: '#4b5563' }}>{index}</div>
    </div>
  )
}

function EmbeddingViz({ embeddings }) {
  if (!embeddings || embeddings.length === 0)
    return <div style={{ color: '#4b5563', fontSize: 12 }}>Waiting for training…</div>
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
      {embeddings.slice(0, 12).map((row, i) => (
        <div key={i} style={{ display: 'flex', gap: 1 }}>
          {row.slice(0, 8).map((v, j) => {
            const norm = Math.max(-1, Math.min(1, v / 0.3))
            const r = norm > 0 ? Math.floor(norm * 200) : 0
            const b = norm < 0 ? Math.floor(-norm * 200) : 0
            return (
              <div key={j} style={{
                width: 9, height: 22, borderRadius: 2,
                background: `rgb(${r},${Math.floor(Math.abs(norm) * 80)},${b})`,
              }} title={v.toFixed(3)} />
            )
          })}
        </div>
      ))}
    </div>
  )
}

function AttentionMap({ weights, tokens }) {
  if (!weights || weights.length === 0)
    return <div style={{ color: '#4b5563', fontSize: 12 }}>Waiting for training…</div>
  const size = Math.min(8, weights.length)
  const cellSize = 28
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'inline-block' }}>
        {weights.slice(0, size).map((row, i) => (
          <div key={i} style={{ display: 'flex' }}>
            {row.slice(0, size).map((w, j) => (
              <div key={j} style={{
                width: cellSize, height: cellSize,
                background: `rgba(124,58,237,${w.toFixed(2)})`,
                border: '1px solid #1a1a2e',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 8, color: w > 0.4 ? '#fff' : '#555',
              }} title={`[${i}→${j}]: ${w.toFixed(3)}`}>
                {w > 0.15 ? w.toFixed(2) : ''}
              </div>
            ))}
          </div>
        ))}
        <div style={{ display: 'flex', marginTop: 4 }}>
          {(tokens || []).slice(0, size).map((t, i) => (
            <div key={i} style={{ width: cellSize, textAlign: 'center', fontSize: 9, color: '#6b7280', fontFamily: 'monospace' }}>
              {t === ' ' ? '␣' : t === '\n' ? '↵' : t}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function PredictionBar({ token, prob, rank }) {
  const colors = ['#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95', '#3b0764']
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 12 }}>
        <span style={{ fontFamily: 'monospace', color: '#e2e8f0', background: '#1e1e2e', padding: '1px 6px', borderRadius: 4 }}>
          {token === ' ' ? '␣' : token === '\n' ? '↵' : token || '?'}
        </span>
        <span style={{ color: '#9ca3af' }}>{(prob * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: '#1e1e2e', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${prob * 100}%`,
          background: colors[rank] || '#4c1d95',
          borderRadius: 3, transition: 'width 0.4s ease'
        }} />
      </div>
    </div>
  )
}

function StageIndicator({ currentStage }) {
  const stages = ['tokenize', 'embed', 'attention', 'ffn', 'loss', 'backprop', 'update']
  const currentIdx = stages.indexOf(currentStage)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, overflowX: 'auto', paddingBottom: 4 }}>
      {stages.map((s, i) => {
        const info = STAGE_INFO[s]
        const isActive = s === currentStage
        const isDone = currentIdx > i
        return (
          <div key={s} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
            <div style={{
              padding: '5px 12px', borderRadius: 20, fontSize: 11, fontWeight: 600,
              background: isActive ? info.color : isDone ? '#1a2e1a' : '#1a1a2e',
              color: isActive ? '#fff' : isDone ? '#22c55e' : '#4b5563',
              border: `1px solid ${isActive ? info.color : isDone ? '#22c55e33' : '#2d2d3e'}`,
              transition: 'all 0.3s',
              boxShadow: isActive ? `0 0 14px ${info.color}66` : 'none',
              whiteSpace: 'nowrap'
            }}>
              {info.icon} {info.label}
            </div>
            {i < stages.length - 1 && (
              <div style={{ width: 16, height: 1, background: isDone ? '#22c55e44' : '#1e1e2e', flexShrink: 0 }} />
            )}
          </div>
        )
      })}
    </div>
  )
}

function LLMSimulator() {
  const [text, setText] = useState(DEFAULT_TEXT)
  const [vocab, setVocab] = useState(null)
  const [running, setRunning] = useState(false)
  const [step, setStep] = useState(0)
  const [lossHistory, setLossHistory] = useState([])
  const [currentLoss, setCurrentLoss] = useState(null)
  const [stage, setStage] = useState('idle')
  const [tokenView, setTokenView] = useState([])
  const [activeTokenIdx, setActiveTokenIdx] = useState(0)
  const [attnWeights, setAttnWeights] = useState([])
  const [embeddings, setEmbeddings] = useState([])
  const [predictions, setPredictions] = useState([])
  const [currentContext, setCurrentContext] = useState('')
  const [totalParams, setTotalParams] = useState(0)
  const [initialized, setInitialized] = useState(false)

  const intervalRef = useRef(null)
  const modelRef = useRef(null)
  const vocabRef = useRef(null)
  const encodedRef = useRef([])

  const init = useCallback((autoStart = false) => {
    const v = buildVocab(text)
    vocabRef.current = v
    setVocab(v)

    const m = new TinyTransformer({ vocabSize: v.vocabSize, embedDim: 16, blockSize: BLOCK_SIZE, numHeads: 2, ffDim: 32 })
    modelRef.current = m
    encodedRef.current = encode(text, v.stoi)
    setTokenView(text.slice(0, 36).split(''))
    setStep(0)
    setLossHistory([])
    setCurrentLoss(null)
    setStage('idle')
    setAttnWeights([])
    setEmbeddings([])
    setPredictions([])
    setCurrentContext('')
    setInitialized(true)

    const vSize = v.vocabSize, e = 16, f = 32, h = 2, hd = 8
    setTotalParams(vSize * e + BLOCK_SIZE * e + h * (e * hd * 3 + hd * e) + e * f + f * e + e * vSize)

    if (autoStart) setTimeout(() => setRunning(true), 50)
  }, [text])

  const doStep = useCallback(() => {
    const m = modelRef.current
    const v = vocabRef.current
    const encoded = encodedRef.current
    if (!m || !v || encoded.length < 2) return

    const maxStart = Math.max(0, encoded.length - BLOCK_SIZE - 1)
    const start = Math.floor(Math.random() * maxStart)
    const tokens = encoded.slice(start, start + BLOCK_SIZE)
    const targets = encoded.slice(start + 1, start + BLOCK_SIZE + 1)
    if (tokens.length < 2) return

    const delays = [0, 80, 160, 240, 320, 400, 480]
    const actions = [
      () => { setStage('tokenize'); setActiveTokenIdx(start % Math.min(36, text.length)) },
      () => { setStage('embed'); setEmbeddings([...m.tokenEmbed.slice(0, 12)]) },
      () => { setStage('attention'); m.forward(tokens); if (m.lastAttentionWeights) setAttnWeights([...m.lastAttentionWeights]) },
      () => { setStage('ffn') },
      () => {
        setStage('loss')
        const loss = m.trainStep(tokens, targets, 0.08)
        setCurrentLoss(loss)
        setStep(s => {
          const ns = s + 1
          setLossHistory(h => [...h.slice(-80), { step: ns, loss: parseFloat(loss.toFixed(4)) }])
          return ns
        })
      },
      () => { setStage('backprop') },
      () => {
        setStage('update')
        const ctxTokens = tokens.slice(-4)
        setPredictions(m.predict(ctxTokens, v.itos))
        setCurrentContext(ctxTokens.map(t => v.itos[t]).join(''))
      },
    ]
    actions.forEach((fn, i) => setTimeout(fn, delays[i]))
  }, [text])

  useEffect(() => {
    if (running) {
      doStep()
      intervalRef.current = setInterval(doStep, 620)
    } else {
      clearInterval(intervalRef.current)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, doStep])

  const avgRecentLoss = lossHistory.length >= 5
    ? (lossHistory.slice(-5).reduce((s, d) => s + d.loss, 0) / 5).toFixed(4)
    : '—'

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f', padding: '20px 24px' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          <span style={{ color: '#7c3aed' }}>LLM</span> Training Simulator
        </h1>
        <p style={{ color: '#6b7280', fontSize: 13 }}>
          Watch a tiny transformer learn to predict the next character — every step visualized
        </p>
      </div>

      {/* Controls row */}
      <div style={{
        background: '#13131a', border: '1px solid #1e1e2e', borderRadius: 12,
        padding: '16px', marginBottom: 14, display: 'flex',
        alignItems: 'flex-start', gap: 16, flexWrap: 'wrap'
      }}>
        <div style={{ flex: 1, minWidth: 220 }}>
          <label style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
            Training Text
          </label>
          <textarea
            value={text} onChange={e => setText(e.target.value)}
            disabled={running} rows={3}
            style={{
              width: '100%', background: '#0a0a0f', border: '1px solid #2d2d3e',
              color: '#e2e8f0', borderRadius: 8, padding: '8px 10px', fontSize: 12,
              fontFamily: 'monospace', resize: 'vertical', outline: 'none',
              opacity: running ? 0.5 : 1
            }}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 22 }}>
          <button
            onClick={() => init(true)}
            disabled={running}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px',
              background: running ? '#1a1a2e' : '#7c3aed', color: '#fff',
              border: 'none', borderRadius: 8, cursor: running ? 'not-allowed' : 'pointer',
              fontWeight: 600, fontSize: 13,
              boxShadow: running ? 'none' : '0 0 20px rgba(124,58,237,0.35)',
              opacity: running ? 0.5 : 1
            }}
          >
            <Play size={14} /> Start Training
          </button>
          <button
            onClick={() => setRunning(r => !r)}
            disabled={!initialized}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px',
              background: '#1e1e2e', color: running ? '#f59e0b' : '#22c55e',
              border: `1px solid ${running ? '#f59e0b44' : '#22c55e44'}`,
              borderRadius: 8, cursor: initialized ? 'pointer' : 'not-allowed',
              fontWeight: 600, fontSize: 13, opacity: initialized ? 1 : 0.4
            }}
          >
            {running ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Resume</>}
          </button>
          <button onClick={() => { setRunning(false); setTimeout(init, 50) }}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px',
              background: '#1e1e2e', color: '#9ca3af', border: '1px solid #2d2d3e',
              borderRadius: 8, cursor: 'pointer', fontWeight: 600, fontSize: 13
            }}
          >
            <RotateCcw size={14} /> Reset
          </button>
        </div>
        {/* Stats */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', paddingTop: 22 }}>
          {[
            { label: 'Step',        value: step.toLocaleString() },
            { label: 'Loss',        value: currentLoss ? currentLoss.toFixed(4) : '—' },
            { label: 'Avg (5)',     value: avgRecentLoss },
            { label: 'Vocab',       value: vocab?.vocabSize ?? '—' },
            { label: 'Params',      value: totalParams ? `~${totalParams}` : '—' },
          ].map(({ label, value }) => (
            <div key={label} style={{
              textAlign: 'center', background: '#0a0a0f',
              padding: '8px 14px', borderRadius: 8, border: '1px solid #1e1e2e'
            }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f1f5f9', fontVariantNumeric: 'tabular-nums' }}>{value}</div>
              <div style={{ fontSize: 10, color: '#6b7280', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Stage pipeline */}
      <div style={{ background: '#13131a', border: '1px solid #1e1e2e', borderRadius: 12, padding: '12px 16px', marginBottom: 14 }}>
        <StageIndicator currentStage={stage} />
      </div>

      {/* Main grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>

        <Card title="1. Tokenization" icon={Cpu}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>
            Text → character IDs. Vocab: {vocab?.vocabSize ?? '…'} unique chars
          </div>
          <div style={{ maxHeight: 110, overflowY: 'auto', marginBottom: 8 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {tokenView.map((c, i) => (
                <TokenChip key={i} char={c} index={i} isActive={i === activeTokenIdx} />
              ))}
            </div>
          </div>
          {vocab && (
            <div style={{ fontSize: 11, color: '#6b7280' }}>
              <span style={{ color: '#9ca3af' }}>Chars: </span>
              <span style={{ fontFamily: 'monospace', color: '#a78bfa' }}>
                {vocab.chars.slice(0, 24).map(c => c === ' ' ? '␣' : c === '\n' ? '↵' : c).join(' ')}
                {vocab.chars.length > 24 ? ` +${vocab.chars.length - 24}` : ''}
              </span>
            </div>
          )}
        </Card>

        <Card title="2. Token Embeddings" icon={Brain}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>
            Each token ID → learned 16-dim vector (red=positive, blue=negative)
          </div>
          <EmbeddingViz embeddings={embeddings} />
          <div style={{ marginTop: 8, display: 'flex', gap: 6, fontSize: 10, color: '#6b7280', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: 2 }}>
              {[0.9, 0.5, 0.1, -0.1, -0.5, -0.9].map((v, i) => {
                const r = v > 0 ? Math.floor(v * 200) : 0
                const b = v < 0 ? Math.floor(-v * 200) : 0
                return <div key={i} style={{ width: 12, height: 12, borderRadius: 2, background: `rgb(${r},${Math.floor(Math.abs(v) * 80)},${b})` }} />
              })}
            </div>
            <span>high → low value</span>
          </div>
        </Card>

        <Card title="3. Self-Attention" icon={Zap}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>
            Each token attends to previous tokens (causal masking)
          </div>
          <AttentionMap weights={attnWeights} tokens={tokenView} />
        </Card>

        <Card title="4. Training Loss" icon={BarChart2}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>
            Cross-entropy loss — decreases as model learns
          </div>
          {lossHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={lossHistory} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#6b7280' }} />
                <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} domain={['auto', 'auto']} width={35} />
                <Tooltip
                  contentStyle={{ background: '#13131a', border: '1px solid #2d2d3e', borderRadius: 6, fontSize: 11 }}
                  labelStyle={{ color: '#9ca3af' }}
                  itemStyle={{ color: '#a78bfa' }}
                />
                <Line type="monotone" dataKey="loss" stroke="#7c3aed" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: 12 }}>
              Start training to see the loss curve
            </div>
          )}
        </Card>

        <Card title="5. Next Token Predictions" icon={ChevronRight}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 10 }}>
            {currentContext
              ? <>Given <span style={{ fontFamily: 'monospace', color: '#a78bfa', background: '#1e1e2e', padding: '1px 6px', borderRadius: 4 }}>
                  {currentContext.split('').map(c => c === ' ' ? '␣' : c === '\n' ? '↵' : c).join('')}
                </span> predict next character</>
              : 'Start training to see predictions'}
          </div>
          {predictions.length > 0
            ? predictions.map((p, i) => <PredictionBar key={i} token={p.token} prob={p.prob} rank={i} />)
            : <div style={{ color: '#374151', fontSize: 12 }}>No predictions yet</div>
          }
        </Card>

        <Card title="Model Architecture">
          <div style={{ fontSize: 12 }}>
            {[
              { label: 'Type',            value: 'Character-level Transformer' },
              { label: 'Embedding dim',   value: '16' },
              { label: 'Attention heads', value: '2  (head dim: 8)' },
              { label: 'FF hidden dim',   value: '32' },
              { label: 'Context window',  value: `${BLOCK_SIZE} tokens` },
              { label: 'Total params',    value: `~${totalParams}` },
              { label: 'Loss fn',         value: 'Cross-entropy' },
              { label: 'Learning rate',   value: '0.08' },
            ].map(({ label, value }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid #1a1a2e' }}>
                <span style={{ color: '#6b7280' }}>{label}</span>
                <span style={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 11 }}>{value}</span>
              </div>
            ))}
          </div>
        </Card>

      </div>

      <div style={{ marginTop: 16, textAlign: 'center', color: '#374151', fontSize: 11 }}>
        Educational simulation. Real LLMs use the same core principles at massive scale — billions of parameters, terabytes of data.
      </div>
    </div>
  )
}

const TABS = [
  { id: 'llm', label: '🧠 LLM Training' },
  { id: 'ebs', label: '💾 NVMe / EBS I/O' },
]

export default function App() {
  const [tab, setTab] = useState('llm')
  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f' }}>
      <div style={{ background: '#0d0d14', borderBottom: '1px solid #1e1e2e', padding: '0 24px', display: 'flex' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '14px 20px', background: 'transparent', border: 'none',
              borderBottom: `2px solid ${tab === t.id ? '#7c3aed' : 'transparent'}`,
              color: tab === t.id ? '#f1f5f9' : '#6b7280',
              cursor: 'pointer', fontWeight: 600, fontSize: 13,
              transition: 'all 0.2s'
            }}>
            {t.label}
          </button>
        ))}
      </div>
      {tab === 'llm' ? <LLMSimulator /> : <EBSSimulator />}
    </div>
  )
}
