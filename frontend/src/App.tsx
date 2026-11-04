/** Main App component */

import { useState, useCallback, useEffect } from 'react';
import { Brain, Shield, Info, Github, Loader2, RefreshCw } from 'lucide-react';
import { ImageUploader, ResultDisplay, HeatmapOverlay, ConfidenceMeter } from './components';
import { api } from './services/api';
import type { DetectionResponse, HealthResponse } from './types/detection';
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResponse | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // GitHub repo URL from environment variable
  const repoUrl = import.meta.env.VITE_REPO_URL || 'https://github.com/yourusername/deepfake-detector';

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthData = await api.healthCheck();
        setHealth(healthData);
      } catch (err) {
        console.warn('Health check failed:', err);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }

    setDetectionResult(null);
    setHeatmapUrl(null);
    setError(null);
  }, [previewUrl]);

  const runDetection = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      const result = await api.detectDeepfake(file);
      setDetectionResult(result);
      setHeatmapUrl(result.result.heatmap_url ?? null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Detection failed';
      setError(message);
      console.error('Detection error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDetect = useCallback(async () => {
    if (!selectedFile) return;
    await runDetection(selectedFile);
  }, [selectedFile, runDetection]);


  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (heatmapUrl) URL.revokeObjectURL(heatmapUrl);
    };
  }, [previewUrl, heatmapUrl]);

  const displayResult = detectionResult?.result!;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-content">
          <div className="brand-wrap">
            <div className="brand-icon">
              <Brain size={28} />
            </div>
            <div>
              <p className="eyebrow">AI security lab</p>
              <h1>Deepfake Detector</h1>
            </div>
          </div>

          <div className="header-actions">
            <div className={`status-pill ${health?.status ?? 'degraded'}`}>
              <span className="status-dot" />
              {health?.status ?? 'Waiting'}
            </div>
            <a href={repoUrl} target="_blank" rel="noreferrer" className="icon-link" aria-label="GitHub Repository">
              <Github size={18} />
            </a>
          </div>
        </div>
        <p className="header-tagline">Explainable AI pipeline for identifying synthetic or manipulated imagery.</p>
      </header>

      <main className="app-main">
        <div className="container">
          <section className="panel upload-panel">
            <div className="panel-topline">
              <div className="title-group">
                <Shield size={20} />
                <h2>Image analysis</h2>
              </div>
            </div>

            <p className="panel-copy">Upload a portrait or generated image to inspect whether it contains manipulation patterns.</p>

            <ImageUploader onFileSelect={handleFileSelect} disabled={loading} />

            <div className="action-row">
              <button type="button" className="primary-btn" onClick={handleDetect} disabled={!selectedFile || loading}>
                {loading ? <Loader2 size={18} className="spin" /> : <Brain size={18} />}
                {loading ? 'Analyzing...' : 'Analyze image'}
              </button>

              {selectedFile && (
                <button type="button" className="secondary-btn" onClick={() => handleFileSelect(null)}>
                  <RefreshCw size={16} />
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="alert-box" role="alert">
                <Info size={18} />
                <span>{error}</span>
              </div>
            )}
          </section>

          {loading && (
            <section className="panel loading-panel">
              <Loader2 size={28} className="spin" />
              <div>
                <h3>Running deepfake assessment</h3>
                <p>Checking facial artifact consistency, texture patterns, and model confidence.</p>
              </div>
            </section>
          )}

          {detectionResult && (
            <section className="results-layout">
              <div className="panel result-panel">
                <ResultDisplay result={displayResult} />
              </div>

              <aside className="panel sidebar-panel">
                <div className="side-card">
                  <h3>Confidence meter</h3>
                  <div className="meter-wrap">
                    <ConfidenceMeter value={displayResult.confidence} prediction={displayResult.prediction} size={150} strokeWidth={12} />
                  </div>
                  <div className="stat-grid">
                    <div className="mini-stat">
                      <span className="mini-label">Latency</span>
                      <strong>{displayResult.processing_time_ms.toFixed(1)} ms</strong>
                    </div>
                    <div className="mini-stat">
                      <span className="mini-label">Model</span>
                      <strong>{displayResult.model_used}</strong>
                    </div>
                    <div className="mini-stat">
                      <span className="mini-label">Face</span>
                      <strong>{displayResult.face_detected ? 'Detected' : 'Not detected'}</strong>
                    </div>
                  </div>
                </div>
              </aside>
            </section>
          )}

          {detectionResult && (
            <section className="panel heatmap-panel">
              <div className="panel-topline slim">
                <div className="title-group">
                  <Brain size={20} />
                  <h2>Attention map</h2>
                </div>
              </div>

              <HeatmapOverlay
                imageUrl={previewUrl}
                heatmapUrl={heatmapUrl}
                prediction={displayResult.prediction}
                confidence={displayResult.confidence}
              />
            </section>
          )}

          {!selectedFile && !detectionResult && (
            <section className="panel empty-panel">
              <div className="empty-graphic">
                <Brain size={64} />
              </div>
              <h2>Ready for your first scan</h2>
              <p>Choose an image and let the detector assess whether it looks authentic or synthetic.</p>
              <div className="feature-grid">
                <div className="feature-item">
                  <Shield size={22} />
                  <div>
                    <strong>Ensemble model</strong>
                    <span>EfficientNet + MesoNet</span>
                  </div>
                </div>
                <div className="feature-item">
                  <Info size={22} />
                  <div>
                    <strong>Explainability</strong>
                    <span>Heatmap overlays</span>
                  </div>
                </div>
                <div className="feature-item">
                  <Brain size={22} />
                  <div>
                    <strong>Face-aware</strong>
                    <span>Crop + detection</span>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Deepfake Detector v1.0.0 • IPCV Project</p>
      </footer>
    </div>
  );
}

export default App;