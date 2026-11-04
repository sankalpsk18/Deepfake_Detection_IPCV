/** Confidence meter component with animated gauge */

import { useEffect, useRef } from 'react';

interface ConfidenceMeterProps {
  /** Confidence value (0-1) */
  value: number;
  /** Prediction type for color coding */
  prediction: 'real' | 'fake' | 'unknown';
  /** Size in pixels */
  size?: number;
  /** Stroke width */
  strokeWidth?: number;
  /** Show percentage label */
  showLabel?: boolean;
  /** Animation duration in ms */
  animationDuration?: number;
}

export function ConfidenceMeter({
  value,
  prediction,
  size = 120,
  strokeWidth = 8,
  showLabel = true,
  animationDuration = 1000,
}: ConfidenceMeterProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const circleRef = useRef<SVGCircleElement>(null);
  const prevValueRef = useRef(value);

  // Clamp value
  const clampedValue = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clampedValue * 100);

  // Color based on prediction
  const getColor = () => {
    switch (prediction) {
      case 'fake':
        return '#dc2626'; // red-600
      case 'real':
        return '#16a34a'; // green-600
      default:
        return '#ca8a04'; // yellow-600
    }
  };

  const color = getColor();
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // Animate on value change
  useEffect(() => {
    const circle = circleRef.current;
    if (!circle) return;

    const startValue = prevValueRef.current;
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / animationDuration, 1);
      // Easing function (ease-out)
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (clampedValue - startValue) * easedProgress;
      const currentOffset = circumference - (currentValue * circumference);

      circle.style.strokeDashoffset = String(currentOffset);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        prevValueRef.current = clampedValue;
      }
    };

    requestAnimationFrame(animate);
  }, [clampedValue, circumference, animationDuration]);

  // Initialize on mount
  useEffect(() => {
    if (circleRef.current) {
      circleRef.current.style.strokeDashoffset = String(circumference - (clampedValue * circumference));
      prevValueRef.current = clampedValue;
    }
  }, []); // Only run once on mount

  return (
    <div className="confidence-meter" role="img" aria-label={`Confidence: ${percentage}%, Prediction: ${prediction}`}>
      <svg
        ref={svgRef}
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="confidence-meter-svg"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb" // gray-200
          strokeWidth={strokeWidth}
        />

        {/* Progress circle */}
        <circle
          ref={circleRef}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
          strokeLinecap="round"
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: `${size / 2}px ${size / 2}px`,
            transition: `stroke-dashoffset ${animationDuration}ms cubic-bezier(0.25, 0.46, 0.45, 0.94)`,
          }}
        />

        {/* Center dot for unknown */}
        {prediction === 'unknown' && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius * 0.3}
            fill={color}
            opacity={0.3}
          />
        )}
      </svg>

      {showLabel && (
        <div className="confidence-meter-label">
          <span className="confidence-value" style={{ color }}>
            {percentage}%
          </span>
          <span className="confidence-prediction">{prediction.toUpperCase()}</span>
        </div>
      )}
    </div>
  );
}