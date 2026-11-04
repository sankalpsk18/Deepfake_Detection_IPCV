/** Result display component showing detection verdict and details */

import type { FC, SVGProps } from 'react';
import { ShieldCheck, AlertTriangle, HelpCircle, Clock, Eye, Brain } from 'lucide-react';
import type { DetectionResult } from '../types/detection';

interface ResultDisplayProps {
  result: DetectionResult | null;
  className?: string;
}

// Lucide icons accept size as a prop but it's not in the default SVGProps
// Use a more permissive type to avoid prop type conflicts
type LucideIcon = FC<SVGProps<SVGSVGElement> & { size?: number | string }>;

function getVerdictConfig(prediction: DetectionResult['prediction']) {
  switch (prediction) {
    case 'fake':
      return {
        icon: AlertTriangle as LucideIcon,
        label: 'DEEPFAKE DETECTED',
        description: 'This image appears to be AI-generated or manipulated',
        color: 'fake',
        bgColor: 'bg-red-50 border-red-200',
        iconColor: 'text-red-600',
        textColor: 'text-red-800',
      };
    case 'real':
      return {
        icon: ShieldCheck as LucideIcon,
        label: 'LIKELY AUTHENTIC',
        description: 'This image appears to be a genuine photograph',
        color: 'real',
        bgColor: 'bg-green-50 border-green-200',
        iconColor: 'text-green-600',
        textColor: 'text-green-800',
      };
    default:
      return {
        icon: HelpCircle as LucideIcon,
        label: 'UNCERTAIN',
        description: 'Unable to determine with confidence',
        color: 'unknown',
        bgColor: 'bg-yellow-50 border-yellow-200',
        iconColor: 'text-yellow-600',
        textColor: 'text-yellow-800',
      };
  }
}

function ProbabilityBar({ label, value, color }: { label: string; value: number; color: 'real' | 'fake' }) {
  const percentage = Math.round(value * 100);
  const isFake = color === 'fake';

  return (
    <div className="probability-bar">
      <div className="probability-label">
        <span className={isFake ? 'text-red-600' : 'text-green-600'}>{label}</span>
        <span className="font-mono font-medium">{percentage}%</span>
      </div>
      <div className="probability-track" role="progressbar" aria-valuenow={percentage} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={`probability-fill ${isFake ? 'fake' : 'real'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function DetailRow({ icon: Icon, label, value, tooltip }: { icon: LucideIcon; label: string; value: string | number; tooltip?: string }) {
  return (
    <div className="detail-row" title={tooltip}>
      <Icon size={18} className="detail-icon" />
      <span className="detail-label">{label}</span>
      <span className="detail-value font-mono">{value}</span>
    </div>
  );
}

export function ResultDisplay({ result, className = '' }: ResultDisplayProps) {
  if (!result) {
    return (
      <div className={`result-display empty ${className}`}>
        <div className="empty-state">
          <Brain size={48} className="empty-icon" />
          <h3>No Results Yet</h3>
          <p>Upload an image to start deepfake detection</p>
        </div>
      </div>
    );
  }

  const verdict = getVerdictConfig(result.prediction);
  const Icon = verdict.icon;

  return (
    <div className={`result-display ${className}`}>
      {/* Main Verdict Card */}
      <div className={`verdict-card ${verdict.bgColor} border rounded-xl p-6`}>
        <div className="verdict-header">
          <div className="verdict-icon-wrapper">
            <Icon size={32} className={verdict.iconColor} />
          </div>
          <div className="verdict-text">
            <h2 className={`verdict-label ${verdict.textColor}`}>{verdict.label}</h2>
            <p className={`verdict-description ${verdict.textColor}/80`}>{verdict.description}</p>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="confidence-section">
          <div className="confidence-circle-wrapper">
            <svg className="confidence-circle" viewBox="0 0 100 100">
              <circle
                className="confidence-bg"
                cx="50"
                cy="50"
                r="45"
                fill="none"
                strokeWidth="8"
              />
              <circle
                className={`confidence-progress ${verdict.color}`}
                cx="50"
                cy="50"
                r="45"
                fill="none"
                strokeWidth="8"
                strokeDasharray={283}
                strokeDashoffset={283 - (283 * result.confidence)}
                strokeLinecap="round"
                style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
              />
              <text
                x="50"
                y="55"
                textAnchor="middle"
                dominantBaseline="middle"
                className="confidence-text"
                fill={verdict.color === 'fake' ? '#dc2626' : verdict.color === 'real' ? '#16a34a' : '#ca8a04'}
              >
                {Math.round(result.confidence * 100)}%
              </text>
            </svg>
          </div>
          <p className="confidence-label">Confidence Score</p>
        </div>
      </div>

      {/* Probability Breakdown */}
      <div className="probabilities-section">
        <h3 className="section-title">Probability Breakdown</h3>
        <div className="probabilities-grid">
          <ProbabilityBar label="Real" value={result.probabilities.real} color="real" />
          <ProbabilityBar label="Fake" value={result.probabilities.fake} color="fake" />
        </div>
      </div>

      {/* Technical Details */}
      <details className="details-section">
        <summary className="details-summary">
          <Brain size={18} />
          <span>Technical Details</span>
        </summary>
        <div className="details-grid">
          <DetailRow
            icon={ShieldCheck}
            label="Model Used"
            value={result.model_used}
            tooltip="Ensemble of EfficientNet-B0 and MesoNet"
          />
          <DetailRow
            icon={Clock}
            label="Processing Time"
            value={`${result.processing_time_ms.toFixed(1)} ms`}
            tooltip="Total inference time including preprocessing"
          />
          <DetailRow
            icon={Eye}
            label="Face Detected"
            value={result.face_detected ? 'Yes' : 'No'}
            tooltip="Whether a face was detected and cropped for analysis"
          />
          {result.heatmap_url && (
            <DetailRow
              icon={Brain}
              label="Heatmap Available"
              value="Yes"
              tooltip="Grad-CAM visualization highlights suspicious regions"
            />
          )}
        </div>
      </details>
    </div>
  );
}