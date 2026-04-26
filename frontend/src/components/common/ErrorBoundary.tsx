/**
 * ErrorBoundary — React error boundary that catches render-time exceptions
 * in its subtree and shows a tab-scoped fallback UI instead of letting the
 * whole app crash.
 *
 * Each tab in App.tsx is wrapped in its own boundary so a Plotly / vtk.js /
 * deck.gl exception in (say) the Budgets tab leaves the other tabs usable.
 *
 * React provides no hook equivalent; class components are still the only way
 * to implement getDerivedStateFromError + componentDidCatch.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import RefreshIcon from '@mui/icons-material/Refresh';

interface ErrorBoundaryProps {
  /** Subtree to guard. */
  children: ReactNode;
  /** Optional human label shown in the fallback ("Budgets tab", "3D mesh", …). */
  scope?: string;
  /** Optional override of the default fallback UI. Receives the caught error
   *  and a reset callback. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[ErrorBoundary${this.props.scope ? ` ${this.props.scope}` : ''}]`, error, info);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }
      return (
        <Box sx={{ p: 3, maxWidth: 720, mx: 'auto' }}>
          <Alert
            severity="error"
            action={
              <Button
                size="small"
                color="inherit"
                startIcon={<RefreshIcon />}
                onClick={this.reset}
              >
                Try again
              </Button>
            }
          >
            <AlertTitle>
              {this.props.scope
                ? `Something went wrong in the ${this.props.scope}`
                : 'Something went wrong'}
            </AlertTitle>
            <Box sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', mt: 1 }}>
              {this.state.error.message || String(this.state.error)}
            </Box>
            <Box sx={{ mt: 1, fontSize: '0.875rem' }}>
              Other tabs are still usable. Switching tabs and back, or clicking
              <em> Try again</em>, will retry rendering.
            </Box>
          </Alert>
        </Box>
      );
    }
    return this.props.children;
  }
}
