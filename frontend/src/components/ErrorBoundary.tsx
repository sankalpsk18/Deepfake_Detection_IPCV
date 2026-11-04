/** Error Boundary component for catching React render errors */

import { Component, ErrorInfo, ReactNode } from 'react';
import { RefreshCw, AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ error, errorInfo });
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary" role="alert">
          <div className="error-content">
            <div className="error-icon">
              <AlertTriangle size={48} />
            </div>
            <h2>Something went wrong</h2>
            <p className="error-message">
              We caught an unexpected error. The development team has been notified.
            </p>
            {this.state.error && (
              <details className="error-details">
                <summary>Error details (for debugging)</summary>
                <pre>{this.state.error?.message}</pre>
                {this.state.errorInfo?.componentStack && (
                  <pre>{this.state.errorInfo.componentStack}</pre>
                )}
              </details>
            )}
            <button
              type="button"
              className="reload-button"
              onClick={this.handleReload}
              aria-label="Reload the application"
            >
              <RefreshCw size={18} className="spin" />
              Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;