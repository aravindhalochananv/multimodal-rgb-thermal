import React from 'react';
import { Columns3, Play, Clock, Users, ArrowRight, ShieldCheck } from 'lucide-react';

export default function TripleComparison({
  onRunCompare,
  loading,
  compareData,
  error,
  selectedSample,
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header Banner */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14 }}>
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Columns3 size={18} color="#a855f7" />
            Triple Comparison: RGB vs Thermal vs Multimodal Fusion
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Executes all three 10-epoch Colab models simultaneously on the paired inputs to evaluate detection differentials.
          </p>
        </div>

        <button
          onClick={onRunCompare}
          disabled={loading}
          className="btn btn-primary"
          style={{
            padding: '12px 28px',
            fontSize: 14,
            background: 'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)',
          }}
        >
          {loading ? 'Evaluating All 3 Models...' : 'Run 3-Way Comparison'}
        </button>
      </div>

      {error && (
        <div style={{ padding: 12, borderRadius: 8, background: 'rgba(239, 68, 68, 0.1)', color: '#f87171', fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* 3 Panels Side-by-Side */}
      {compareData ? (
        <>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: 16,
          }}>
            {/* Column 1: RGB */}
            <div className="glass-card" style={{ borderColor: 'rgba(56, 189, 248, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <span className="badge badge-rgb" style={{ fontSize: 12 }}>RGB-Only Model</span>
                <span className="badge badge-rgb" style={{ fontSize: 11 }}>
                  <Clock size={11} /> {compareData.rgb.latency_ms} ms
                </span>
              </div>
              <div style={{
                borderRadius: 8,
                overflow: 'hidden',
                background: '#050814',
                minHeight: 260,
                border: '1px solid var(--border-color)',
                marginBottom: 12,
              }}>
                <img
                  src={compareData.rgb.image_base64}
                  alt="RGB Detection"
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderRadius: 6 }}>
                <span style={{ color: 'var(--text-muted)' }}>Detected Persons:</span>
                <span style={{ fontWeight: 700, color: '#38bdf8', fontSize: 15 }}>
                  {compareData.rgb.count}
                </span>
              </div>
            </div>

            {/* Column 2: Thermal */}
            <div className="glass-card" style={{ borderColor: 'rgba(251, 146, 60, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <span className="badge badge-thermal" style={{ fontSize: 12 }}>Thermal-Only Model</span>
                <span className="badge badge-thermal" style={{ fontSize: 11 }}>
                  <Clock size={11} /> {compareData.thermal.latency_ms} ms
                </span>
              </div>
              <div style={{
                borderRadius: 8,
                overflow: 'hidden',
                background: '#050814',
                minHeight: 260,
                border: '1px solid var(--border-color)',
                marginBottom: 12,
              }}>
                <img
                  src={compareData.thermal.image_base64}
                  alt="Thermal Detection"
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderRadius: 6 }}>
                <span style={{ color: 'var(--text-muted)' }}>Detected Persons:</span>
                <span style={{ fontWeight: 700, color: '#fb923c', fontSize: 15 }}>
                  {compareData.thermal.count}
                </span>
              </div>
            </div>

            {/* Column 3: Fusion */}
            <div className="glass-card" style={{ borderColor: 'rgba(168, 85, 247, 0.5)', background: 'rgba(24, 34, 52, 0.8)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <span className="badge badge-fusion" style={{ fontSize: 12 }}>50:50 Multimodal Fusion</span>
                <span className="badge badge-fusion" style={{ fontSize: 11 }}>
                  <Clock size={11} /> {compareData.multimodal.latency_ms} ms
                </span>
              </div>
              <div style={{
                borderRadius: 8,
                overflow: 'hidden',
                background: '#050814',
                minHeight: 260,
                border: '1px solid var(--border-color)',
                marginBottom: 12,
              }}>
                <img
                  src={compareData.multimodal.image_base64}
                  alt="Fusion Detection"
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderRadius: 6 }}>
                <span style={{ color: 'var(--text-muted)' }}>Detected Persons:</span>
                <span style={{ fontWeight: 700, color: '#c084fc', fontSize: 15 }}>
                  {compareData.multimodal.count}
                </span>
              </div>
            </div>
          </div>

          {/* Comparative Summary Insight Card */}
          <div className="glass-card" style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(168, 85, 247, 0.3)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <ShieldCheck size={18} color="#34d399" />
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>Comparative Detection Insight</h3>
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Total Multi-Model Execution Time: <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{compareData.total_latency_ms} ms</strong>.
              {compareData.thermal.count > compareData.rgb.count ? (
                <> In this scene, Thermal detected <strong>+{compareData.thermal.count - compareData.rgb.count} more persons</strong> than RGB because infrared thermal radiation penetrates the dark background regions where visible pixels are underexposed.</>
              ) : (
                <> Both models detected comparable numbers of subjects, with Multimodal Fusion maintaining high confidence across all targets.</>
              )}
            </p>
          </div>
        </>
      ) : (
        <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-dim)' }}>
          <Columns3 size={40} color="var(--border-color)" style={{ margin: '0 auto 12px' }} />
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-muted)' }}>No Comparison Run Yet</div>
          <p style={{ fontSize: 12, marginTop: 4 }}>
            Click <strong>Run 3-Way Comparison</strong> above to execute RGB, Thermal, and Fusion models simultaneously.
          </p>
        </div>
      )}
    </div>
  );
}
