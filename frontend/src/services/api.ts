/** API client for deepfake detection backend */

import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type {
  DetectionResponse,
  HealthResponse,
  ModelInfoResponse,
  ErrorResponse,
} from '../types/detection';

// API base URL - uses Vite proxy in dev, relative in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000, // 30 second timeout for inference
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ErrorResponse>) => {
        if (error.response) {
          // Server responded with error status
          const message = error.response.data?.message || error.message;
          console.error(`[API Error] ${error.response.status}: ${message}`);
          return Promise.reject(new Error(message));
        } else if (error.request) {
          // Network error
          console.error('[API Error] Network error:', error.message);
          return Promise.reject(new Error('Unable to connect to server. Is the backend running?'));
        } else {
          // Other error
          console.error('[API Error]:', error.message);
          return Promise.reject(error);
        }
      }
    );
  }

  /** Health check endpoint */
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  /** Get model information */
  async getModelInfo(): Promise<ModelInfoResponse> {
    const response = await this.client.get<ModelInfoResponse>('/model/info');
    return response.data;
  }

  /** Detect deepfake from image file */
  async detectDeepfake(file: File): Promise<DetectionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<DetectionResponse>('/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  /** Get heatmap for a request (not implemented in backend yet) */
  async getHeatmap(requestId: string): Promise<Blob> {
    const response = await this.client.get(`/heatmap/${requestId}`, {
      responseType: 'blob',
    });
    return response.data;
  }
}

// Export singleton instance
export const api = new ApiClient();

// Export class for testing
export { ApiClient };