/** Heatmap overlay component for Grad-CAM visualization */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Eye, Download, Expand, Loader2, AlertCircle } from 'lucide-react';

interface HeatmapOverlayProps {
  /** Original image URL (blob URL or data URL) */
  imageUrl: string | null;
  /** Heatmap image URL (relative API path) */
  heatmapUrl: string | null;
  /** Prediction label for accessibility */
  prediction?: 'real' | 'fake' | 'unknown';
  /** Confidence score */
  confidence?: number;
}

export function HeatmapOverlay({
  imageUrl,
  heatmapUrl,
  prediction,
  confidence,
}: HeatmapOverlayProps) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.5);
  const [heatmapBlobUrl, setHeatmapBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch heatmap as blob and create blob URL
  const fetchHeatmap = useCallback(async () => {
    if (!heatmapUrl) return;

    setIsLoadingHeatmap(true);
    setError(null);

    try {
      const response = await fetch(heatmapUrl);
      if (!response.ok) {
        throw new Error(`Failed to load heatmap: ${response.status}`);
      }
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      setHeatmapBlobUrl(blobUrl);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load heatmap';
      setError(message);
      console.error('Heatmap load error:', err);
    } finally {
      setIsLoadingHeatmap(false);
    }
  }, [heatmapUrl]);

  // Fetch heatmap when URL changes
  useEffect(() => {
    if (heatmapUrl) {
      fetchHeatmap();
    } else {
      setHeatmapBlobUrl(null);
    }

    return () => {
      // Cleanup blob URL on unmount or URL change
      if (heatmapBlobUrl) {
        URL.revokeObjectURL(heatmapBlobUrl);
      }
    };
  }, [heatmapUrl, fetchHeatmap, heatmapBlobUrl]);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (heatmapBlobUrl) {
        URL.revokeObjectURL(heatmapBlobUrl);
      }
    };
  }, [heatmapBlobUrl]);

  // Draw combined image on canvas
  useEffect(() => {
    if (!canvasRef.current || !imageUrl || !heatmapBlobUrl) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const originalImg = new Image();
    const heatmapImg = new Image();

    let originalLoaded = false;
    let heatmapLoaded = false;

    const draw = () => {
      if (!originalLoaded || !heatmapLoaded) return;

      // Set canvas size to match original image
      canvas.width = originalImg.naturalWidth;
      canvas.height = originalImg.naturalHeight;

      // Draw original image
      ctx.drawImage(originalImg, 0, 0);

      // Draw heatmap with opacity
      ctx.globalAlpha = showHeatmap ? opacity : 0;
      ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    };

    originalImg.onload = () => {
      originalLoaded = true;
      draw();
    };
    originalImg.onerror = () => {
      console.error('Failed to load original image');
    };
    heatmapImg.onload = () => {
      heatmapLoaded = true;
      draw();
    };
    heatmapImg.onerror = () => {
      console.error('Failed to load heatmap image');
      setError('Failed to load heatmap image');
    };

    originalImg.src = imageUrl;
    heatmapImg.src = heatmapBlobUrl;
  }, [imageUrl, heatmapBlobUrl, showHeatmap, opacity]);

  // Download combined image
  const handleDownload = () => {
    if (!canvasRef.current) return;

    const link = document.createElement('a');
    link.download = `deepfake-analysis-${prediction || 'result'}-${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  // Open in new tab
  const handleExpand = () => {
    if (!canvasRef.current) return;

    const dataUrl = canvasRef.current.toDataURL('image/png');
    window.open(dataUrl, '_blank');
  };

  if (!imageUrl || !heatmapUrl) {
    return (
      <div className="heatmap-overlay empty">
        <div className="empty-state">
          <Eye size={32} className="empty-icon" />
          <p>No heatmap available</p>
          <span className="text-sm text-muted">Grad-CAM visualization will appear here after detection</span>
        </div>
      </div>
    );
  }

  if (isLoadingHeatmap) {
    return (
      <div className="heatmap-overlay loading">
        <div className="loading-state">
          <Loader2 size={32} className="spin" />
          <p>Loading heatmap...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="heatmap-overlay error">
        <div className="error-state">
          <AlertCircle size={32} className="error-icon" />
          <p>Failed to load heatmap</p>
          <span className="text-sm text-muted">{error}</span>
        </div>
      </div>
    );
  }

  if (!heatmapBlobUrl) {
    return (
      <div className="heatmap-overlay empty">
        <div className="empty-state">
          <Eye size={32} className="empty-icon" />
          <p>No heatmap available</p>
          <span className="text-sm text-muted">Grad-CAM visualization will appear here after detection</span>
        </div>
      </div>
    );
  }

  return (
    <div className="heatmap-overlay">
      <div className="heatmap-container">
        {/* Canvas for combined view */}
        <canvas
          ref={canvasRef}
          className="heatmap-canvas"
          aria-label={`Grad-CAM heatmap overlay showing ${prediction} regions`}
        />

        {/* Controls */}
        <div className="heatmap-controls">
          <div className="control-group">
            <label className="control-label">
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
                className="toggle-input"
              />
              <span className="toggle-label">Show Heatmap</span>
            </label>

            <label className="control-label">
              <span className="toggle-label">Opacity: {Math.round(opacity * 100)}%</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="opacity-slider"
                disabled={!showHeatmap}
              />
            </label>
          </div>

          <div className="control-actions">
            <button
              type="button"
              className="control-btn"
              onClick={handleDownload}
              aria-label="Download combined image"
              title="Download"
            >
              <Download size={16} />
            </button>
            <button
              type="button"
              className="control-btn"
              onClick={handleExpand}
              aria-label="View full size"
              title="View Full Size"
            >
              <Expand size={16} />
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="heatmap-legend">
          <div className="legend-gradient" />
          <div className="legend-labels">
            <span>Low Attention</span>
            <span>High Attention</span>
          </div>
        </div>
      </div>

      {/* Info bar */}
      <div className="heatmap-info">
        <div className="info-item">
          <span className="info-label">Prediction:</span>
          <span className={`info-value prediction-${prediction}`}>
            {prediction?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Confidence:</span>
          <span className="info-value">{confidence ? `${Math.round(confidence * 100)}%` : 'N/A'}</span>
        </div>
        <p className="heatmap-description">
          Red/yellow regions indicate areas the model focused on for its decision.
          {prediction === 'fake' ? ' These regions show manipulation artifacts.' : ' These regions confirm authentic features.'}
        </p>
      </div>
    </div>
  );
}