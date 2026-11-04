/** Image uploader component with drag & drop support */

import { useCallback, useState, type ChangeEvent } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { Upload, X, Image, AlertCircle, CheckCircle } from 'lucide-react';
import type { UploadState } from '../types/detection';

interface ImageUploaderProps {
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
  maxSize?: number; // in bytes
  acceptedTypes?: string[];
}

const DEFAULT_MAX_SIZE = 10 * 1024 * 1024; // 10MB
const DEFAULT_ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function ImageUploader({
  onFileSelect,
  disabled = false,
  maxSize = DEFAULT_MAX_SIZE,
  acceptedTypes = DEFAULT_ACCEPTED_TYPES,
}: ImageUploaderProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    preview: null,
    error: null,
  });

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const validateFile = useCallback((file: File): string | null => {
    // Check file type
    if (!acceptedTypes.includes(file.type)) {
      return `Unsupported file type. Please use: ${acceptedTypes.map(t => t.split('/')[1].toUpperCase()).join(', ')}`;
    }

    // Check file size
    if (file.size > maxSize) {
      return `File too large. Maximum size: ${formatFileSize(maxSize)}`;
    }

    return null;
  }, [acceptedTypes, maxSize]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const error = rejectedFiles[0].errors[0]?.message || 'File rejected';
      setUploadState(prev => ({ ...prev, error, file: null, preview: null }));
      onFileSelect(null);
      return;
    }

    // Handle accepted file
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const error = validateFile(file);

      if (error) {
        setUploadState(prev => ({ ...prev, error, file: null, preview: null }));
        onFileSelect(null);
        return;
      }

      // Create preview
      const preview = URL.createObjectURL(file);
      setUploadState({ file, preview, error: null });
      onFileSelect(file);
    }
  }, [validateFile, onFileSelect]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
    },
    maxSize,
    disabled,
    noClick: false,
    noKeyboard: false,
  });

  const handleRemove = () => {
    if (uploadState.preview) {
      URL.revokeObjectURL(uploadState.preview);
    }
    setUploadState({ file: null, preview: null, error: null });
    onFileSelect(null);
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      const error = validateFile(file);
      if (error) {
        setUploadState(prev => ({ ...prev, error, file: null, preview: null }));
        onFileSelect(null);
        return;
      }
      const preview = URL.createObjectURL(file);
      setUploadState({ file, preview, error: null });
      onFileSelect(file);
    }
  };

  // Cleanup preview on unmount
  // Note: In React 18+, useEffect cleanup runs after component unmounts
  // For simplicity, we rely on browser to clean up object URLs

  return (
    <div className="image-uploader">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`drop-zone ${isDragActive ? 'drag-active' : ''} ${isDragReject ? 'drag-reject' : ''} ${disabled ? 'disabled' : ''} ${uploadState.file ? 'has-file' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="Drop zone for image upload"
      >
        <input {...getInputProps()} onChange={handleFileInputChange} />

        {uploadState.file && uploadState.preview ? (
          // Preview mode
          <div className="preview-container">
            <img src={uploadState.preview} alt="Preview" className="preview-image" />
            <button
              type="button"
              className="remove-btn"
              onClick={(e) => { e.stopPropagation(); handleRemove(); }}
              aria-label="Remove image"
            >
              <X size={20} />
            </button>
          </div>
        ) : (
          // Upload prompt
          <div className="upload-prompt">
            <Upload className="upload-icon" size={48} />
            <p className="upload-text">
              {isDragActive ? 'Drop image here' : 'Drag & drop an image here, or click to browse'}
            </p>
            <p className="upload-hint">
              Supports: JPG, PNG, WebP • Max: {formatFileSize(maxSize)}
            </p>
          </div>
        )}

        {/* Error message */}
        {uploadState.error && (
          <div className="error-message" role="alert">
            <AlertCircle size={16} />
            <span>{uploadState.error}</span>
          </div>
        )}
      </div>

      {/* File info when selected */}
      {uploadState.file && !uploadState.error && (
        <div className="file-info">
          <div className="file-info-row">
            <Image size={16} />
            <span>{uploadState.file.name}</span>
          </div>
          <div className="file-info-row">
            <CheckCircle size={16} className="text-green" />
            <span>{formatFileSize(uploadState.file.size)} • {uploadState.file.type}</span>
          </div>
        </div>
      )}
    </div>
  );
}