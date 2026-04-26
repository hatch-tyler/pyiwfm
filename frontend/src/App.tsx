/**
 * Main application component with tab navigation.
 */

import { useEffect } from 'react';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';

import { ModelOverview } from './components/Overview/ModelOverview';
import { Viewer3D } from './components/Viewer3D';
import { ControlPanel } from './components/Controls';
import { InfoPanel } from './components/InfoPanel';
import { CrossSectionPanel } from './components/CrossSection';
import { ResultsMapView } from './components/ResultsMap/ResultsMapView';
import { BudgetView } from './components/BudgetDashboard/BudgetView';
import { ZBudgetView } from './components/ZBudgetDashboard';
import { DiagnosticsView } from './components/Diagnostics';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { useViewerStore } from './stores/viewerStore';
import { fetchModelInfo, fetchBounds, fetchResultsInfo, fetchProperties, fetchObservations } from './api/client';

// Hash ↔ tab index mapping
const TAB_HASHES: Record<string, number> = {
  '#overview': 0,
  '#3d': 1,
  '#results': 2,
  '#budgets': 3,
  '#zbudgets': 4,
  '#diagnostics': 5,
};
const HASH_FOR_TAB = ['#overview', '#3d', '#results', '#budgets', '#zbudgets', '#diagnostics'];

export default function App() {
  const {
    modelInfo,
    error,
    activeTab,
    setModelInfo,
    setBounds,
    setLoading,
    setError,
    setActiveTab,
    setResultsInfo,
    setProperties,
    setObservations,
  } = useViewerStore();

  // Load model info on mount
  useEffect(() => {
    const loadModelData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [info, bounds] = await Promise.all([
          fetchModelInfo(),
          fetchBounds(),
        ]);

        setModelInfo(info);
        setBounds(bounds);

        // Also load results info and properties
        try {
          const resultsInfo = await fetchResultsInfo();
          setResultsInfo(resultsInfo);
        } catch {
          // Results not available — not an error
        }

        try {
          const props = await fetchProperties();
          setProperties(props);
        } catch {
          // Properties not available — not an error
        }

        // Load uploaded observations
        fetchObservations().then(setObservations).catch(() => {});

        setLoading(false);
      } catch (err) {
        console.error('Failed to load model:', err);
        setError(err instanceof Error ? err.message : 'Failed to load model');
        setLoading(false);
      }
    };

    loadModelData();
  }, [setModelInfo, setBounds, setLoading, setError, setResultsInfo, setProperties, setObservations]);

  // Read hash on mount to set initial tab
  useEffect(() => {
    const hash = window.location.hash.toLowerCase();
    if (hash && TAB_HASHES[hash] !== undefined) {
      setActiveTab(TAB_HASHES[hash]);
    }
  }, [setActiveTab]);

  // Write hash when activeTab changes
  useEffect(() => {
    const newHash = HASH_FOR_TAB[activeTab] ?? '';
    if (newHash && window.location.hash !== newHash) {
      window.location.hash = newHash;
    }
  }, [activeTab]);

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.toLowerCase();
      if (hash && TAB_HASHES[hash] !== undefined) {
        const target = TAB_HASHES[hash];
        if (target !== useViewerStore.getState().activeTab) {
          setActiveTab(target);
        }
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setActiveTab]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header with Tabs */}
      <AppBar position="static" elevation={1}>
        <Toolbar variant="dense" sx={{ minHeight: 48 }}>
          <Typography variant="h6" component="h1" sx={{ mr: 4, whiteSpace: 'nowrap' }}>
            {modelInfo?.name ?? 'IWFM Viewer'}
          </Typography>
          <Tabs
            value={activeTab}
            onChange={(_, v) => setActiveTab(v)}
            textColor="inherit"
            indicatorColor="secondary"
            sx={{ minHeight: 48 }}
          >
            <Tab label="Overview" sx={{ minHeight: 48 }} />
            <Tab label="3D Mesh" sx={{ minHeight: 48 }} />
            <Tab label="Results Map" sx={{ minHeight: 48 }} />
            <Tab label="Budgets" sx={{ minHeight: 48 }} />
            <Tab label="Z-Budgets" sx={{ minHeight: 48 }} />
            <Tab label="Diagnostics" sx={{ minHeight: 48 }} />
          </Tabs>
        </Toolbar>
      </AppBar>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ borderRadius: 0 }}>
          {error}
        </Alert>
      )}

      {/* Tab Panels — each tab is wrapped in its own ErrorBoundary so a
          render-time crash (Plotly, vtk.js, deck.gl) is contained to that
          tab and the rest of the app stays usable. */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex' }}>
        {/* Tab 0: Overview */}
        {activeTab === 0 && (
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            <ErrorBoundary scope="Overview tab">
              <ModelOverview />
            </ErrorBoundary>
          </Box>
        )}

        {/* Tab 1: 3D Mesh */}
        {activeTab === 1 && (
          <ErrorBoundary scope="3D Mesh tab">
            <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
              <Box sx={{ flexGrow: 1, position: 'relative' }}>
                <Viewer3D />
                <Box
                  sx={{
                    position: 'absolute',
                    top: 16,
                    left: 16,
                    maxWidth: 350,
                  }}
                >
                  <InfoPanel />
                </Box>
                <CrossSectionPanel />
              </Box>
              <ControlPanel />
            </Box>
          </ErrorBoundary>
        )}

        {/* Tab 2: Results Map */}
        {activeTab === 2 && (
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            <ErrorBoundary scope="Results Map tab">
              <ResultsMapView />
            </ErrorBoundary>
          </Box>
        )}

        {/* Tab 3: Budgets */}
        {activeTab === 3 && (
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            <ErrorBoundary scope="Budgets tab">
              <BudgetView />
            </ErrorBoundary>
          </Box>
        )}

        {/* Tab 4: Z-Budgets */}
        {activeTab === 4 && (
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            <ErrorBoundary scope="Z-Budgets tab">
              <ZBudgetView />
            </ErrorBoundary>
          </Box>
        )}

        {/* Tab 5: Diagnostics */}
        {activeTab === 5 && (
          <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
            <ErrorBoundary scope="Diagnostics tab">
              <DiagnosticsView />
            </ErrorBoundary>
          </Box>
        )}
      </Box>
    </Box>
  );
}
