import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ModelSelector from './components/ModelSelector';
import Controls from './components/Controls';
import LiveInference from './components/LiveInference';
import TripleComparison from './components/TripleComparison';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import QualitativeGallery from './components/QualitativeGallery';
import { Play, Columns3, BarChart3, Image as ImageIcon, Sparkles } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('inference'); // 'inference' | 'comparison' | 'benchmarks' | 'gallery'
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('multimodal');
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState('190001');

  // Sliders
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.50);
  const [alpha, setAlpha] = useState(0.50);

  // Upload files & Previews
  const [rgbFile, setRgbFile] = useState(null);
  const [thermalFile, setThermalFile] = useState(null);
  const [rgbPreview, setRgbPreview] = useState('/api/sample-image/rgb/190001.jpg');
  const [thermalPreview, setThermalPreview] = useState('/api/sample-image/thermal/190001.jpg');

  // Inference state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);

  // Initial Data Fetching
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then(setStatus)
      .catch((err) => console.error('Status fetch failed:', err));

    fetch('/api/models')
      .then((res) => res.json())
      .then((data) => {
        setModels(data);
        if (data.length > 0) setSelectedModel(data[2]?.id || data[0].id);
      })
      .catch((err) => console.error('Models fetch failed:', err));

    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        setSamples(data);
        if (data.length > 0) {
          setSelectedSample(data[0].id);
          setRgbPreview(data[0].rgb_url);
          setThermalPreview(data[0].thermal_url);
        }
      })
      .catch((err) => console.error('Samples fetch failed:', err));
  }, []);

  // Update previews when selectedSample changes
  useEffect(() => {
    if (selectedSample && !rgbFile && !thermalFile) {
      setRgbPreview(`/api/sample-image/rgb/${selectedSample}.jpg`);
      setThermalPreview(`/api/sample-image/thermal/${selectedSample}.jpg`);
      setSingleResult(null);
      setCompareResult(null);
    }
  }, [selectedSample]);

  // Upload Handlers
  const handleUploadRgb = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setRgbFile(file);
      setRgbPreview(URL.createObjectURL(file));
      setSelectedSample(null);
    }
  };

  const handleUploadThermal = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setThermalFile(file);
      setThermalPreview(URL.createObjectURL(file));
      setSelectedSample(null);
    }
  };

  const handleReset = () => {
    setConf(0.25);
    setIou(0.50);
    setAlpha(0.50);
    setRgbFile(null);
    setThermalFile(null);
    if (samples.length > 0) {
      setSelectedSample(samples[0].id);
      setRgbPreview(samples[0].rgb_url);
      setThermalPreview(samples[0].thermal_url);
    }
    setSingleResult(null);
    setCompareResult(null);
    setError(null);
  };

  // Run Single Inference
  const handleRunInference = async () => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('modality', selectedModel);
      formData.append('conf', conf.toString());
      formData.append('iou', iou.toString());
      formData.append('alpha', alpha.toString());

      if (selectedSample) {
        formData.append('sample_id', selectedSample);
      }
      if (rgbFile) {
        formData.append('rgb_file', rgbFile);
      }
      if (thermalFile) {
        formData.append('thermal_file', thermalFile);
      }

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Inference failed');
      }

      const data = await res.json();
      setSingleResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run Triple Comparison
  const handleRunCompare = async () => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('conf', conf.toString());
      formData.append('iou', iou.toString());

      if (selectedSample) {
        formData.append('sample_id', selectedSample);
      }
      if (rgbFile) {
        formData.append('rgb_file', rgbFile);
      }
      if (thermalFile) {
        formData.append('thermal_file', thermalFile);
      }

      const res = await fetch('/api/compare', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Comparison failed');
      }

      const data = await res.json();
      setCompareResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header status={status} />

      <main className="main-content">
        {/* Navigation Tabs */}
        <div className="nav-tabs">
          <button
            onClick={() => setActiveTab('inference')}
            className={`nav-tab ${activeTab === 'inference' ? 'active' : ''}`}
          >
            <Play size={16} />
            Live Detection
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`nav-tab ${activeTab === 'comparison' ? 'active' : ''}`}
          >
            <Columns3 size={16} />
            Triple Comparison (RGB vs Thermal vs Fusion)
          </button>
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`nav-tab ${activeTab === 'benchmarks' ? 'active' : ''}`}
          >
            <BarChart3 size={16} />
            LLVIP Benchmark Analytics
          </button>
          <button
            onClick={() => setActiveTab('gallery')}
            className={`nav-tab ${activeTab === 'gallery' ? 'active' : ''}`}
          >
            <ImageIcon size={16} />
            Qualitative Verification Gallery
          </button>
        </div>

        {/* Tab 1: Live Detection */}
        {activeTab === 'inference' && (
          <div>
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />

            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 24, alignItems: 'start' }}>
              <Controls
                conf={conf}
                setConf={setConf}
                iou={iou}
                setIou={setIou}
                alpha={alpha}
                setAlpha={setAlpha}
                samples={samples}
                selectedSample={selectedSample}
                setSelectedSample={setSelectedSample}
                selectedModel={selectedModel}
                onUploadRgb={handleUploadRgb}
                onUploadThermal={handleUploadThermal}
                rgbPreview={rgbPreview}
                thermalPreview={thermalPreview}
                onReset={handleReset}
              />

              <LiveInference
                selectedModel={selectedModel}
                onRunInference={handleRunInference}
                loading={loading}
                result={singleResult}
                error={error}
                rgbPreview={rgbPreview}
                thermalPreview={thermalPreview}
                selectedSample={selectedSample}
              />
            </div>
          </div>
        )}

        {/* Tab 2: Triple Comparison */}
        {activeTab === 'comparison' && (
          <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 24, alignItems: 'start' }}>
            <Controls
              conf={conf}
              setConf={setConf}
              iou={iou}
              setIou={setIou}
              alpha={alpha}
              setAlpha={setAlpha}
              samples={samples}
              selectedSample={selectedSample}
              setSelectedSample={setSelectedSample}
              selectedModel="comparison"
              onUploadRgb={handleUploadRgb}
              onUploadThermal={handleUploadThermal}
              rgbPreview={rgbPreview}
              thermalPreview={thermalPreview}
              onReset={handleReset}
            />

            <TripleComparison
              onRunCompare={handleRunCompare}
              loading={loading}
              compareData={compareResult}
              error={error}
              selectedSample={selectedSample}
            />
          </div>
        )}

        {/* Tab 3: Benchmark Analytics */}
        {activeTab === 'benchmarks' && <BenchmarkDashboard />}

        {/* Tab 4: Qualitative Gallery */}
        {activeTab === 'gallery' && <QualitativeGallery />}
      </main>

      <footer style={{
        marginTop: 40,
        padding: '20px 24px',
        textAlign: 'center',
        borderTop: '1px solid var(--border-color)',
        color: 'var(--text-dim)',
        fontSize: 12,
      }}>
        Master's Machine Learning Project • Multimodal RGB-Thermal Pedestrian Detection (LLVIP Benchmark) • Trained on Google Colab GPU
      </footer>
    </div>
  );
}
