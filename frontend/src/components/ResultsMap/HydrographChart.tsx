/**
 * Plotly time series chart for hydrograph data.
 * Shows simulated line and optional observed scatter overlay.
 *
 * Features:
 * - Stream hydrograph Flow / Stage / Both toggle (Issue 4)
 * - GW hydrograph all-layers selector (Issue 3)
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { HydrographData, ObservationData, GWAllLayersData } from '../../api/client';
import { fetchGWHydrographAllLayers } from '../../api/client';

type StreamViewMode = 'flow' | 'stage' | 'both';

interface HydrographChartProps {
  data: HydrographData;
  observation?: ObservationData | null;
  onClose: () => void;
}

export function HydrographChart({ data, observation, onClose }: HydrographChartProps) {
  const hasStreamDualAxis = data.type === 'stream' && data.flow_values && data.stage_values;
  const isGW = data.type === 'gw';

  // Stream view mode: flow | stage | both
  const [streamView, setStreamView] = useState<StreamViewMode>('flow');

  // GW all-layers state
  const [allLayersData, setAllLayersData] = useState<GWAllLayersData | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(data.layer ?? 1);
  const [loadingLayers, setLoadingLayers] = useState(false);

  // Fetch all-layers data when a GW hydrograph is opened
  useEffect(() => {
    if (!isGW) {
      setAllLayersData(null);
      return;
    }
    // Clear stale data immediately when switching locations
    setAllLayersData(null);
    setLoadingLayers(true);
    fetchGWHydrographAllLayers(data.location_id)
      .then((resp) => {
        setAllLayersData(resp);
        setSelectedLayer(data.layer ?? 1);
      })
      .catch(() => {
        setAllLayersData(null);
      })
      .finally(() => setLoadingLayers(false));
  }, [isGW, data.location_id, data.layer]);

  // Build traces
  const traces: Plotly.Data[] = [];

  if (hasStreamDualAxis) {
    if (streamView === 'flow' || streamView === 'both') {
      traces.push({
        x: data.times,
        y: data.flow_values,
        type: 'scatter',
        mode: 'lines',
        name: `Flow (${data.flow_units || 'cfs'})`,
        line: { color: '#1976d2', width: 2 },
        yaxis: 'y',
      });
    }
    if (streamView === 'stage' || streamView === 'both') {
      traces.push({
        x: data.times,
        y: data.stage_values,
        type: 'scatter',
        mode: 'lines',
        name: `Stage (${data.stage_units || 'ft'})`,
        line: { color: '#2e7d32', width: 2 },
        yaxis: streamView === 'both' ? 'y2' : 'y',
      });
    }
  } else if (isGW && allLayersData) {
    // Use data for the selected layer from all-layers response
    const layerEntry = allLayersData.layers.find(l => l.layer === selectedLayer);
    if (layerEntry) {
      traces.push({
        x: allLayersData.times,
        y: layerEntry.values,
        type: 'scatter',
        mode: 'lines',
        name: `Layer ${selectedLayer}`,
        line: { color: '#1976d2', width: 2 },
      });
    }
  } else {
    // Single axis fallback (GW head or stream flow-only)
    traces.push({
      x: data.times,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: 'Simulated',
      line: { color: '#1976d2', width: 2 },
    });
  }

  if (observation && observation.times.length > 0) {
    traces.push({
      x: observation.times,
      y: observation.values,
      type: 'scatter',
      mode: 'markers',
      name: 'Observed',
      marker: { color: '#d32f2f', size: 5 },
    });
  }

  // Determine y-axis label
  let yLabel = data.units || 'Value';
  if (hasStreamDualAxis) {
    if (streamView === 'flow') yLabel = data.flow_units || 'cfs';
    else if (streamView === 'stage') yLabel = data.stage_units || 'ft';
    else yLabel = data.flow_units || 'cfs';
  }

  const showDualAxis = hasStreamDualAxis && streamView === 'both';

  const layout: Partial<Plotly.Layout> = {
    margin: { l: 60, r: showDualAxis ? 60 : 20, t: 10, b: 40 },
    xaxis: { title: { text: 'Date' }, type: 'date' },
    yaxis: { title: { text: yLabel } },
    legend: { orientation: 'h', y: 1.02, x: 0 },
    autosize: true,
  };

  if (showDualAxis) {
    layout.yaxis2 = {
      title: { text: data.stage_units || 'ft' },
      overlaying: 'y',
      side: 'right',
    };
  }

  // Title text
  const displayLayer = isGW && allLayersData ? selectedLayer : data.layer;
  const titleText = `${data.name} — ${data.type.toUpperCase()} Hydrograph${displayLayer ? ` (Layer ${displayLayer})` : ''}`;

  return (
    <Box
      sx={{
        height: 280,
        bgcolor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 0.5, gap: 1 }}>
        <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
          {titleText}
          {loadingLayers && ' (loading layers...)'}
        </Typography>

        {/* GW layer selector */}
        {isGW && allLayersData && allLayersData.n_layers > 1 && (
          <FormControl size="small" sx={{ minWidth: 90 }}>
            <Select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(e.target.value as number)}
              size="small"
              sx={{ fontSize: 12, height: 28 }}
            >
              {allLayersData.layers.map((l) => (
                <MenuItem key={l.layer} value={l.layer} sx={{ fontSize: 12 }}>
                  Layer {l.layer}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        {/* Stream Flow/Stage/Both toggle */}
        {hasStreamDualAxis && (
          <ToggleButtonGroup
            value={streamView}
            exclusive
            onChange={(_, v) => { if (v) setStreamView(v as StreamViewMode); }}
            size="small"
          >
            <ToggleButton value="flow" sx={{ fontSize: 11, py: 0.25, px: 1 }}>
              Flow
            </ToggleButton>
            <ToggleButton value="stage" sx={{ fontSize: 11, py: 0.25, px: 1 }}>
              Stage
            </ToggleButton>
            <ToggleButton value="both" sx={{ fontSize: 11, py: 0.25, px: 1 }}>
              Both
            </ToggleButton>
          </ToggleButtonGroup>
        )}

        <IconButton size="small" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
      <Box sx={{ flexGrow: 1, px: 1, pb: 1 }}>
        <Plot
          data={traces}
          layout={layout}
          config={{ responsive: true, displaylogo: false }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </Box>
    </Box>
  );
}
