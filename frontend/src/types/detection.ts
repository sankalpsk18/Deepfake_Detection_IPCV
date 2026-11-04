/** TypeScript types for detection API */

export type Prediction = 'real' | 'fake' | 'unknown';

export interface Probabilities {
  real: number;
  fake: number;
}

export interface DetectionResult {
  prediction: Prediction;
  confidence: number;
  probabilities: Probabilities;
  model_used: string;
  processing_time_ms: number;
  face_detected: boolean;
  heatmap_url?: string;
}

export interface DetectionResponse {
  request_id: string;
  timestamp: string;
  result: DetectionResult;
  filename: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  models_loaded: {
    efficientnet: boolean;
    mesonet: boolean;
  };
  device: string;
}

export interface ModelInfoResponse {
  model_name: string;
  version: string;
  architecture: string;
  input_size: [number, number];
  num_classes: number;
  classes: string[];
  accuracy?: number;
  description: string;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  request_id?: string;
}

export interface UploadState {
  file: File | null;
  preview: string | null;
  error: string | null;
}

export interface DetectionState {
  loading: boolean;
  result: DetectionResponse | null;
  error: string | null;
}

export interface AppState {
  upload: UploadState;
  detection: DetectionState;
  health: HealthResponse | null;
}