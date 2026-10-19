import React from 'react';
import { Eye, Flame, Layers, Activity, Cpu } from 'lucide-react';

export default function Header({ status }) {
  return (
    <header style={{
      background: 'rgba(17, 24, 39, 0.85)',
      borderBottom: '1px solid var(--border-color)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '16px 24px',
    }}>
      <div style={{
        maxWidth: 1440,
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 16
      }}>
        {/* Title & Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 42,
            height: 42,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #0284c7 0%, #a855f7 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(168, 85, 247, 0.4)',
          }}>
            <Layers size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8 }}>
              Multimodal RGB-Thermal Studio
              <span className="badge badge-fusion" style={{ fontSize: 10 }}>Colab YOLOv8n</span>
            </h1>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Human Detection Under Challenging Low-Light Visibility (LLVIP Benchmark)
            </p>
          </div>
        </div>

        {/* Server & Device Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 14px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: 20,
            fontSize: 12,
          }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: status?.status === 'online' ? '#10b981' : '#f87171',
              boxShadow: status?.status === 'online' ? '0 0 8px #10b981' : 'none',
            }}></span>
            <span style={{ color: 'var(--text-muted)' }}>API Status:</span>
            <span style={{ fontWeight: 600, color: status?.status === 'online' ? '#34d399' : '#f87171' }}>
              {status?.status === 'online' ? 'Connected' : 'Offline'}
            </span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 14px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: 20,
            fontSize: 12,
          }}>
            <Cpu size={14} color="#38bdf8" />
            <span style={{ color: 'var(--text-muted)' }}>Device:</span>
            <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
              {status?.device ? status.device.toUpperCase() : 'CPU'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
