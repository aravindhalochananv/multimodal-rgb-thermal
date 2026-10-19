import React from 'react';
import { Play, Clock, Users, Zap, CheckCircle2, AlertCircle } from 'lucide-react';

export default function LiveInference({
  selectedModel,
  onRunInference,
  loading,
  result,
  error,
  rgbPreview,
  thermalPreview,
  selectedSample,
}) {
  const modelNames = {
    rgb: 'RGB-Only Baseline',
    thermal: 'Thermal-Only Baseline',
    multimodal: '50:50 Multimodal Fusion',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Action Banner */}
      <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>Target Model:</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8' }}>
              {modelNames[selectedModel]}
            </span>
            <span className="badge badge-success" style={{ fontSize: 10 }}>Colab 10-Epoch GPU</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            Selected Scene: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{selectedSample ? `LLVIP #${selectedSample}` : 'Custom Upload'}</strong>
          </div>
        </div>

        <button
          onClick={onRunInference}
          disabled={loading}
          className="btn btn-primary"
          style={{ padding: '12px 28px', fontSize: 14 }}
        >
          {loading ? (
            <>
              <div style={{
                width: 16,
                height: 16,
                border: '2px solid rgba(255,255,255,0.3)',
                borderTopColor: '#fff',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
              <span>Running Inference...</span>
            </>
          ) : (
            <>
              <Play size={16} fill="currentColor" />
              <span>Run Detection</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div style={{
          padding: '12px 16px',
          borderRadius: 8,
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#f87171',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          fontSize: 13,
        }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Detection Result & Inputs Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: result ? '1.4fr 1fr' : '1fr 1fr', gap: 20 }}>
        {/* Output Detected Image */}
        {result ? (
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <CheckCircle2 size={16} color="#34d399" />
                <h3 style={{ fontSize: 15, fontWeight: 700 }}>Model Detections</h3>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <span className="badge badge-success">
                  <Users size={12} /> {result.count} {result.count === 1 ? 'Person' : 'Persons'}
                </span>
                <span className="badge badge-rgb">
                  <Clock size={12} /> {result.latency_ms} ms ({result.fps} FPS)
                </span>
              </div>
            </div>

            <div style={{
              borderRadius: 10,
              overflow: 'hidden',
              background: '#0f172a',
              border: '1px solid var(--border-color)',
              minHeight: 320,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <img
                src={result.image_base64}
                alt="Detection Output"
                style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }}
              />
            </div>

            {/* Detections List */}
            {result.detections?.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8 }}>
                  Detected Bounding Boxes:
                </div>
                <div style={{ maxHeight: 160, overflowY: 'auto' }}>
                  <table className="custom-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Bounding Box [x1, y1, x2, y2]</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((d) => (
                        <tr key={d.id}>
                          <td>{d.id}</td>
                          <td><span className="badge badge-success">{d.class}</span></td>
                          <td style={{ color: '#38bdf8', fontWeight: 600 }}>{(d.confidence * 100).toFixed(1)}%</td>
                          <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                            [{d.bbox.join(', ')}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 360, textAlign: 'center', color: 'var(--text-dim)' }}>
            <div>
              <Zap size={36} color="var(--border-color)" style={{ margin: '0 auto 12px' }} />
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-muted)' }}>Ready for Detection</div>
              <p style={{ fontSize: 12, maxWidth: 300, marginTop: 4 }}>
                Choose a scene or upload images on the left, then click <strong>Run Detection</strong>.
              </p>
            </div>
          </div>
        )}

        {/* Input Previews */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* RGB Input */}
          <div className="glass-card" style={{ padding: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#38bdf8', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              Visible RGB Input
            </div>
            <div style={{
              borderRadius: 8,
              overflow: 'hidden',
              background: '#050814',
              height: 180,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(56, 189, 248, 0.2)',
            }}>
              {rgbPreview ? (
                <img src={rgbPreview} alt="RGB Input" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>No RGB loaded</span>
              )}
            </div>
          </div>

          {/* Thermal Input */}
          <div className="glass-card" style={{ padding: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#fb923c', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              Thermal Infrared Input
            </div>
            <div style={{
              borderRadius: 8,
              overflow: 'hidden',
              background: '#050814',
              height: 180,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(251, 146, 60, 0.2)',
            }}>
              {thermalPreview ? (
                <img src={thermalPreview} alt="Thermal Input" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>No Thermal loaded</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
