import React, { useState, useEffect } from 'react';
import { Eye, Image as ImageIcon, ZoomIn, Info } from 'lucide-react';

export default function QualitativeGallery() {
  const [panels, setPanels] = useState([]);
  const [selectedPanel, setSelectedPanel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/qualitative-panels')
      .then((res) => res.json())
      .then((data) => {
        setPanels(data);
        if (data.length > 0) setSelectedPanel(data[0]);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load qualitative panels:', err);
        setLoading(false);
      });
  }, []);

  const sceneDescriptions = {
    '190001_comparison': 'Pedestrians walking along unlit sidewalk. RGB model misses shaded subject, while Thermal and Fusion detect all pedestrians with high confidence.',
    '190002_comparison': 'Pedestrian crossing near vehicle headlights. Severe glare degrades RGB confidence; Thermal emits clear silhouette boundary.',
    '190003_comparison': 'Distant pedestrians partially occluded in deep shadows. Thermal contrast enables accurate bounding box localization.',
    '190004_comparison': 'Group of pedestrians adjacent to parked vehicles. Fusion provides crisp separation of overlapping silhouettes.',
    '190005_comparison': 'Low-contrast urban road scene. Thermal and Multimodal maintain zero false alarms on cool street objects.',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div className="glass-card">
        <h2 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <ImageIcon size={18} color="#38bdf8" />
          Qualitative Verification Gallery (Phase 9)
        </h2>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Side-by-side detection comparisons (RGB-Only | Thermal-Only | 50:50 Multimodal Fusion) across challenging nighttime surveillance scenes.
        </p>
      </div>

      {/* Main Selected Panel Showcase */}
      {selectedPanel && (
        <div className="glass-card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
              Scene #{selectedPanel.id.replace('_comparison', '')}
            </span>
            <span className="badge badge-fusion">3-Way Verification Panel</span>
          </div>

          <div style={{
            background: '#050814',
            borderRadius: 10,
            overflow: 'hidden',
            border: '1px solid var(--border-color)',
            marginBottom: 14,
          }}>
            <img
              src={selectedPanel.url}
              alt="Qualitative Comparison Panel"
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
            padding: 12,
            background: 'rgba(56, 189, 248, 0.06)',
            borderRadius: 8,
            border: '1px solid rgba(56, 189, 248, 0.2)',
          }}>
            <Info size={16} color="#38bdf8" style={{ marginTop: 2, flexShrink: 0 }} />
            <div style={{ fontSize: 13, color: 'var(--text-main)', lineHeight: 1.5 }}>
              {sceneDescriptions[selectedPanel.id] || 'Comparison panel showing detections across RGB, Thermal, and Fusion modalities.'}
            </div>
          </div>
        </div>
      )}

      {/* Thumbnails Picker */}
      <div className="glass-card">
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12 }}>
          Select Verification Scene:
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 12,
        }}>
          {panels.map((p) => {
            const isSel = selectedPanel?.id === p.id;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedPanel(p)}
                style={{
                  cursor: 'pointer',
                  borderRadius: 8,
                  overflow: 'hidden',
                  border: isSel ? '2px solid #38bdf8' : '1px solid var(--border-color)',
                  background: '#050814',
                  boxShadow: isSel ? '0 0 12px rgba(56, 189, 248, 0.3)' : 'none',
                  transition: 'all 0.15s ease',
                }}
              >
                <img
                  src={p.url}
                  alt={p.filename}
                  style={{ width: '100%', height: 90, objectFit: 'cover', display: 'block' }}
                />
                <div style={{
                  padding: '6px 10px',
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 600,
                  color: isSel ? '#38bdf8' : 'var(--text-muted)',
                  textAlign: 'center',
                }}>
                  {p.id.replace('_comparison', '')}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
