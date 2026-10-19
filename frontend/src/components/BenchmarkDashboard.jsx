import React, { useState } from 'react';
import { BarChart3, TrendingUp, CheckCircle, Award, FileText } from 'lucide-react';

export default function BenchmarkDashboard({ metrics }) {
  const [activeChart, setActiveChart] = useState('metrics_comparison_bar.png');

  const chartTabs = [
    { id: 'metrics_comparison_bar.png', label: 'All Metrics Bar Chart' },
    { id: 'map_comparison.png', label: 'mAP@50 & mAP@50:95 Chart' },
    { id: 'training_curves_comparison.png', label: '10-Epoch Training Curves' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Top Delta Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: 16,
      }}>
        <div className="glass-card" style={{ borderLeft: '4px solid #38bdf8' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
            RGB Baseline mAP@50
          </div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#38bdf8', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            85.96%
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Visible spectrum, degrades in darkness
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #fb923c' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
            Thermal Baseline mAP@50
          </div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#fb923c', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            92.22%
          </div>
          <div style={{ fontSize: 12, color: '#34d399', fontWeight: 600 }}>
            +6.26% over RGB baseline
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #a855f7' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
            Multimodal Fusion mAP@50
          </div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#c084fc', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            91.50%
          </div>
          <div style={{ fontSize: 12, color: '#34d399', fontWeight: 600 }}>
            +5.54% over RGB baseline
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #34d399' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
            Fusion Inference Throughput
          </div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#34d399', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            26.2 FPS
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            38.19 ms per image (CPU)
          </div>
        </div>
      </div>

      {/* Official Benchmark Comparison Table */}
      <div className="glass-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Award size={18} color="#38bdf8" />
              Official LLVIP Test Set Benchmark (3,463 Image Pairs)
            </h3>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Evaluated on the unseen LLVIP test set with YOLOv8n models trained for 10 epochs on Colab GPU
            </p>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="custom-table">
            <thead>
              <tr>
                <th>Modality / Experiment</th>
                <th>Model</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>mAP@50</th>
                <th>mAP@50:95</th>
                <th>Latency (ms)</th>
                <th>FPS (CPU)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="badge badge-rgb">RGB-Only</span></td>
                <td>yolov8n</td>
                <td>89.04%</td>
                <td>78.70%</td>
                <td>0.8356</td>
                <td style={{ color: '#38bdf8', fontWeight: 700 }}>85.96%</td>
                <td>46.92%</td>
                <td>54.07 ms</td>
                <td>18.5 FPS</td>
              </tr>
              <tr>
                <td><span className="badge badge-thermal">Thermal-Only</span></td>
                <td>yolov8n</td>
                <td>92.67%</td>
                <td>84.18%</td>
                <td>0.8822</td>
                <td style={{ color: '#fb923c', fontWeight: 700 }}>92.22%</td>
                <td>56.58%</td>
                <td>52.15 ms</td>
                <td>19.2 FPS</td>
              </tr>
              <tr style={{ background: 'rgba(168, 85, 247, 0.08)' }}>
                <td><span className="badge badge-fusion">50:50 Multimodal Fusion</span></td>
                <td>yolov8n</td>
                <td>92.18%</td>
                <td>82.95%</td>
                <td>0.8732</td>
                <td style={{ color: '#c084fc', fontWeight: 700 }}>91.50%</td>
                <td>55.05%</td>
                <td style={{ color: '#34d399', fontWeight: 700 }}>38.19 ms</td>
                <td style={{ color: '#34d399', fontWeight: 700 }}>26.2 FPS</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Embedded High-Resolution Visualization Charts */}
      <div className="glass-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={18} color="#38bdf8" />
            Evaluation Visualizations & Curves
          </h3>
          <div style={{ display: 'flex', gap: 8 }}>
            {chartTabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveChart(t.id)}
                className={`btn ${activeChart === t.id ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '6px 12px', fontSize: 12 }}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div style={{
          background: '#050814',
          borderRadius: 10,
          border: '1px solid var(--border-color)',
          padding: 16,
          textAlign: 'center',
          minHeight: 400,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <img
            src={`/api/static-plot/${activeChart}`}
            alt="Evaluation Chart"
            style={{ maxWidth: '100%', height: 'auto', maxHeight: 520, borderRadius: 6 }}
            onError={(e) => {
              e.target.style.display = 'none';
            }}
          />
        </div>
      </div>
    </div>
  );
}
