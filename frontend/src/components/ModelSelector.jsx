import React from 'react';
import { Eye, Flame, Layers, Sparkles, CheckCircle2 } from 'lucide-react';

export default function ModelSelector({ models, selectedModel, onSelectModel }) {
  const modelIcons = {
    rgb: <Eye size={18} color="#38bdf8" />,
    thermal: <Flame size={18} color="#fb923c" />,
    multimodal: <Layers size={18} color="#a855f7" />,
  };

  const badgeClass = {
    rgb: 'badge-rgb',
    thermal: 'badge-thermal',
    multimodal: 'badge-fusion',
  };

  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sparkles size={18} color="#38bdf8" />
            Select Colab-Trained Model
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Choose which 10-epoch GPU model from Google Colab runs inference
          </p>
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: 16,
      }}>
        {models.map((m) => {
          const isSelected = selectedModel === m.id;
          return (
            <div
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              className="glass-card"
              style={{
                cursor: 'pointer',
                borderColor: isSelected ? 'var(--border-bright)' : 'var(--border-color)',
                background: isSelected ? 'rgba(30, 41, 59, 0.9)' : 'rgba(24, 34, 52, 0.6)',
                boxShadow: isSelected ? '0 0 20px rgba(59, 130, 246, 0.25)' : 'none',
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              {isSelected && (
                <div style={{
                  position: 'absolute',
                  top: 12,
                  right: 12,
                  color: '#38bdf8',
                }}>
                  <CheckCircle2 size={18} />
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <div style={{
                  padding: 8,
                  borderRadius: 10,
                  background: 'rgba(255, 255, 255, 0.05)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  {modelIcons[m.id] || <Layers size={18} />}
                </div>
                <div>
                  <h3 style={{ fontSize: 14, fontWeight: 700 }}>{m.name}</h3>
                  <span className={`badge ${badgeClass[m.id] || 'badge-rgb'}`} style={{ fontSize: 10, marginTop: 2 }}>
                    Colab 10-Epochs
                  </span>
                </div>
              </div>

              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14, minHeight: 36 }}>
                {m.description}
              </p>

              {/* Metrics Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 8,
                padding: '10px 12px',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: 8,
                border: '1px solid rgba(255, 255, 255, 0.05)',
              }}>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>mAP@50</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                    {(m.mAP50 * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Precision</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#34d399', fontFamily: 'var(--font-mono)' }}>
                    {(m.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Recall</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#fb923c', fontFamily: 'var(--font-mono)' }}>
                    {(m.recall * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
