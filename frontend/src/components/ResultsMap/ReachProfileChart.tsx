/**
 * Stream reach longitudinal profile chart.
 * Shows bed elevation and ground surface elevation along a reach.
 * Clicking a bed elevation node with a rating table shows a
 * stage-discharge popup chart.
 */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Plot from 'react-plotly.js';
import type { ReachProfileData, StreamNodeRatingData } from '../../api/client';
import { fetchStreamNodeRating } from '../../api/client';

interface ReachProfileChartProps {
  data: ReachProfileData;
  onClose: () => void;
}

export function ReachProfileChart({ data, onClose }: ReachProfileChartProps) {
  const [ratingData, setRatingData] = useState<StreamNodeRatingData | null>(null);
  const [ratingLoading, setRatingLoading] = useState(false);

  const distances = data.nodes.map(n => n.distance);
  const bedElevs = data.nodes.map(n => n.bed_elev);
  const gsElevs = data.nodes.map(n => n.ground_surface_elev);

  // Color markers by rating availability
  const markerColors = data.nodes.map(n =>
    n.has_rating ? '#1f77b4' : '#aaaaaa'
  );

  // Custom text for hover
  const hoverText = data.nodes.map(n =>
    `Node ${n.stream_node_id}<br>Bed: ${n.bed_elev} ft` +
    (n.has_rating ? '<br><b>Click for rating table</b>' : '')
  );

  const traces: Plotly.Data[] = [
    {
      x: distances,
      y: gsElevs,
      type: 'scatter',
      mode: 'lines',
      name: 'Ground Surface',
      line: { color: '#8B4513', width: 2 },
      fill: 'tonexty',
      fillcolor: 'rgba(139, 69, 19, 0.1)',
    },
    {
      x: distances,
      y: bedElevs,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Bed Elevation',
      line: { color: '#1f77b4', width: 2 },
      marker: { size: 5, color: markerColors },
      fill: 'tozeroy',
      fillcolor: 'rgba(31, 119, 180, 0.15)',
      text: hoverText,
      hoverinfo: 'text',
    },
  ];

  const handlePlotClick = (event: Plotly.PlotMouseEvent) => {
    // Only respond to clicks on the bed elevation trace (curveNumber === 1)
    const point = event.points[0];
    if (!point || point.curveNumber !== 1) return;

    const nodeIdx = point.pointIndex;
    const node = data.nodes[nodeIdx];
    if (!node || !node.has_rating) return;

    setRatingLoading(true);
    fetchStreamNodeRating(node.stream_node_id)
      .then((rd) => setRatingData(rd))
      .catch(() => setRatingData(null))
      .finally(() => setRatingLoading(false));
  };

  return (
    <Paper
      elevation={3}
      sx={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 280,
        height: ratingData ? 350 : 280,
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', px: 2, pt: 1 }}>
        <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
          {data.name} &mdash; Profile ({data.total_length.toFixed(0)} ft, {data.n_nodes} nodes)
          {ratingLoading && ' (loading rating...)'}
        </Typography>
        <IconButton size="small" onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Box sx={{ display: 'flex', flexGrow: 1, px: 1, pb: 1, gap: 1 }}>
        {/* Profile chart */}
        <Box sx={{ flexGrow: 1, flexBasis: ratingData ? '55%' : '100%' }}>
          <Plot
            data={traces}
            layout={{
              margin: { l: 60, r: 20, t: 10, b: 40 },
              xaxis: { title: { text: 'Distance from Upstream (ft)' } },
              yaxis: { title: { text: 'Elevation (ft)' } },
              showlegend: true,
              legend: { orientation: 'h', y: 1.12 },
              autosize: true,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            onClick={handlePlotClick}
          />
        </Box>

        {/* Rating table chart (stage-discharge curve) */}
        {ratingData && (
          <Box sx={{ flexBasis: '45%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', px: 1 }}>
              <Typography variant="caption" sx={{ flexGrow: 1 }}>
                Node {ratingData.stream_node_id} — Rating (bed: {ratingData.bottom_elev} ft)
              </Typography>
              <IconButton size="small" onClick={() => setRatingData(null)}>
                <CloseIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Box>
            <Box sx={{ flexGrow: 1 }}>
              <Plot
                data={[
                  {
                    x: ratingData.flows,
                    y: ratingData.stages,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Rating',
                    line: { color: '#2e7d32', width: 2 },
                    marker: { size: 4, color: '#2e7d32' },
                  },
                ]}
                layout={{
                  margin: { l: 50, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: 'Flow (cfs)' } },
                  yaxis: { title: { text: 'Stage (ft)' } },
                  showlegend: false,
                  autosize: true,
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
}
