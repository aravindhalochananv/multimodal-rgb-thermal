import React, { useRef } from 'react';
import { Sliders, Upload, Image as ImageIcon, RefreshCw } from 'lucide-react';

export default function Controls({
  conf,
  setConf,
  iou,
  setIou,
  alpha,
  setAlpha,
  samples,
  selectedSample,
  setSelectedSample,
  selectedModel,
  onUploadRgb,
  onUploadThermal,
  rgbPreview,
  thermalPreview,
  onReset,
}) {
  const rgbInputRef = useRef(null);
  const thermalInputRef = useRef(null);

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: 12 }}>
        <h3 style={{ fontSize: 15, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Sliders size={16} color="#38bdf8" />
          Input & Detection Settings
        </h3>
        <button
          onClick={onReset}
          className="btn btn-secondary"
          style={{ padding: '6px 10px', fontSize: 11 }}
          title="Reset to defaults"
        >
          <RefreshCw size={12} /> Reset
        </button>
      </div>

      {/* 1. Quick Select LLVIP Test Samples */}
      <div>
        <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>
          1. Choose Paired Test Sample (LLVIP)
        </label>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: 6,
          maxHeight: 120,
          overflowY: 'auto',
          padding: 4,
          background: 'rgba(15, 23, 42, 0.4)',
          borderRadius: 8,
          border: '1px solid var(--border-color)',
        }}>
          {samples.map((s) => {
            const isSel = selectedSample === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setSelectedSample(s.id)}
                style={{
                  padding: '6px 4px',
                  borderRadius: 6,
                  border: isSel ? '1px solid #38bdf8' : '1px solid transparent',
                  background: isSel ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                  color: isSel ? '#38bdf8' : 'var(--text-muted)',
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  fontWeight: isSel ? 700 : 500,
                  transition: 'all 0.15s ease',
                }}
              >
                {s.id}
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Custom Upload Section */}
      <div>
        <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 8 }}>
          Or Upload Custom Files:
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {/* RGB Upload */}
          <div
            onClick={() => rgbInputRef.current?.click()}
            style={{
              padding: 12,
              borderRadius: 8,
              border: '1px dashed var(--border-color)',
              background: 'rgba(15, 23, 42, 0.4)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.2s',
            }}
          >
            <input
              type="file"
              ref={rgbInputRef}
              onChange={onUploadRgb}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <Upload size={16} color="#38bdf8" style={{ margin: '0 auto 4px' }} />
            <div style={{ fontSize: 11, fontWeight: 600, color: '#38bdf8' }}>Upload RGB</div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
              {rgbPreview ? 'Custom RGB loaded' : 'Visible spectrum'}
            </div>
          </div>

          {/* Thermal Upload */}
          <div
            onClick={() => thermalInputRef.current?.click()}
            style={{
              padding: 12,
              borderRadius: 8,
              border: '1px dashed var(--border-color)',
              background: 'rgba(15, 23, 42, 0.4)',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.2s',
            }}
          >
            <input
              type="file"
              ref={thermalInputRef}
              onChange={onUploadThermal}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <Upload size={16} color="#fb923c" style={{ margin: '0 auto 4px' }} />
            <div style={{ fontSize: 11, fontWeight: 600, color: '#fb923c' }}>Upload Thermal</div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
              {thermalPreview ? 'Custom Thermal loaded' : 'Infrared spectrum'}
            </div>
          </div>
        </div>
      </div>

      {/* 3. Sliders */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Confidence Threshold */}
        <div className="slider-group">
          <div className="slider-header">
            <span>Confidence Threshold</span>
            <span className="slider-val">{(conf * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
          />
        </div>

        {/* IoU NMS Threshold */}
        <div className="slider-group">
          <div className="slider-header">
            <span>IoU (NMS Suppression)</span>
            <span className="slider-val">{iou.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={iou}
            onChange={(e) => setIou(parseFloat(e.target.value))}
          />
        </div>

        {/* Fusion Alpha Slider (Interactive Blend) */}
        {(selectedModel === 'multimodal' || true) && (
          <div className="slider-group" style={{
            background: 'rgba(168, 85, 247, 0.08)',
            padding: 10,
            borderRadius: 8,
            border: '1px solid rgba(168, 85, 247, 0.2)',
          }}>
            <div className="slider-header">
              <span style={{ color: '#c084fc', fontWeight: 600 }}>Fusion Blend Ratio (α)</span>
              <span className="slider-val" style={{ color: '#c084fc' }}>
                {(alpha * 100).toFixed(0)}% RGB / {((1 - alpha) * 100).toFixed(0)}% Thermal
              </span>
            </div>
            <input
              type="range"
              min="0.00"
              max="1.00"
              step="0.05"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              style={{ accentColor: '#a855f7' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
              <span>Pure Thermal (0.0)</span>
              <span>Balanced (0.5)</span>
              <span>Pure RGB (1.0)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
