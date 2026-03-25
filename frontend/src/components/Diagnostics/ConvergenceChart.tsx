/**
 * Convergence iteration count chart (timestep vs iterations).
 * Supports fullscreen expansion, click-to-select, and average-line overlay.
 */

import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import Plot from 'react-plotly.js';
import type { ConvergenceRecord } from '../../api/client';

/** Convert IWFM date "MM/DD/YYYY_HH:MM" to ISO "YYYY-MM-DD" for Plotly. */
function toISODate(iwfmDate: string): string {
  // Expected: "10/31/1973_24:00"
  const parts = iwfmDate.split('_')[0]?.split('/');
  if (parts && parts.length === 3) {
    const [mm, dd, yyyy] = parts;
    return `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`;
  }
  return iwfmDate;
}

interface ConvergenceChartProps {
  records: ConvergenceRecord[];
  avgIterations?: number;
  onExpand?: () => void;
  onClickTimestep?: (index: number) => void;
  selectedTimestep?: number | null;
}

export function ConvergenceChart({
  records,
  avgIterations,
  onExpand,
  onClickTimestep,
  selectedTimestep,
}: ConvergenceChartProps) {
  const x = records.map((r) => r.date ? toISODate(r.date) : String(r.timestep_index));
  const y = records.map((r) => r.iteration_count);
  const colors = records.map((r) => {
    if (selectedTimestep !== undefined && selectedTimestep !== null && r.timestep_index === selectedTimestep) {
      return '#ff9800'; // highlight selected
    }
    return r.convergence_achieved ? '#1976d2' : '#d32f2f';
  });

  const customdata = records.map((r) => [
    r.bottleneck_variable || 'N/A',
    r.supply_adj_count || 0,
    r.date || '',
  ]);

  const shapes: Plotly.Shape[] = [];
  if (avgIterations && avgIterations > 0) {
    shapes.push({
      type: 'line',
      x0: 0,
      x1: 1,
      xref: 'paper',
      y0: avgIterations,
      y1: avgIterations,
      line: { color: '#ed6c02', width: 1.5, dash: 'dash' },
    } as Plotly.Shape);
  }

  return (
    <Box sx={{ height: '100%', p: 1, position: 'relative' }}>
      {onExpand && (
        <IconButton
          size="small"
          onClick={onExpand}
          title="Expand chart"
          sx={{ position: 'absolute', top: 4, right: 4, zIndex: 5, opacity: 0.6, '&:hover': { opacity: 1 } }}
        >
          <OpenInFullIcon fontSize="small" />
        </IconButton>
      )}
      <Plot
        data={[
          {
            x,
            y,
            type: 'bar',
            marker: { color: colors },
            name: 'Iterations',
            customdata,
            hovertemplate:
              '<b>%{customdata[2]}</b><br>' +
              'Iterations: %{y}<br>' +
              'Bottleneck: %{customdata[0]}<br>' +
              'Supply Adj: %{customdata[1]}' +
              '<extra></extra>',
          },
        ]}
        layout={{
          title: { text: 'Convergence Iterations per Timestep', font: { size: 14 } },
          xaxis: {
            title: { text: 'Date' },
            type: 'date',
            tickformat: '%Y',
            dtick: 'M12',
            tickangle: 0,
          },
          yaxis: { title: { text: 'Iteration Count' }, rangemode: 'tozero' },
          margin: { t: 40, b: 60, l: 70, r: 30 },
          bargap: 0.05,
          shapes,
          annotations: avgIterations ? [
            {
              x: 1,
              y: avgIterations,
              xref: 'paper',
              yref: 'y',
              text: `avg: ${Math.round(avgIterations)}`,
              showarrow: false,
              font: { size: 11, color: '#ed6c02' },
              xanchor: 'left',
              xshift: 4,
            },
          ] : [],
        }}
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
        onClick={(event) => {
          if (onClickTimestep && event.points && event.points.length > 0) {
            const idx = event.points[0].pointIndex;
            if (idx >= 0 && idx < records.length) {
              onClickTimestep(records[idx].timestep_index);
            }
          }
        }}
      />
    </Box>
  );
}
