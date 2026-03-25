/**
 * Convergence hotspots table — shows which variables are most frequently
 * the convergence bottleneck across all timesteps.
 */

import { useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import type { ConvergenceHotspot } from '../../api/client';

function EntityChip({ type }: { type: string }) {
  const color = type === 'groundwater' ? 'primary'
    : type === 'stream' ? 'success'
    : type === 'lake' ? 'info' : 'default';
  const label = type === 'groundwater' ? 'GW'
    : type === 'stream' ? 'ST'
    : type === 'lake' ? 'LK' : type.toUpperCase().slice(0, 3);
  return <Chip label={label} color={color} size="small" variant="outlined" />;
}

type SortKey = 'occurrence_count' | 'avg_iterations' | 'worst_timestep_date';

interface HotspotsTableProps {
  hotspots: ConvergenceHotspot[];
  maxRows?: number;
  onClickTimestep?: (timestepIndex: number) => void;
  onClickVariable?: (variable: string, worstTimestepIndex: number) => void;
}

export function HotspotsTable({ hotspots, maxRows = 20, onClickTimestep, onClickVariable }: HotspotsTableProps) {
  const [sortBy, setSortBy] = useState<SortKey>('occurrence_count');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const handleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(key);
      setSortDir('desc');
    }
  };

  const sorted = [...hotspots].sort((a, b) => {
    const mul = sortDir === 'asc' ? 1 : -1;
    if (sortBy === 'occurrence_count') return mul * (a.occurrence_count - b.occurrence_count);
    if (sortBy === 'avg_iterations') return mul * (a.avg_iterations - b.avg_iterations);
    return mul * a.worst_timestep_date.localeCompare(b.worst_timestep_date);
  }).slice(0, maxRows);

  if (hotspots.length === 0) {
    return <Typography color="text.secondary">No convergence hotspot data available.</Typography>;
  }

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        Convergence Hotspots
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        Variables most frequently responsible for limiting convergence. Click a row to investigate its location and stresses.
      </Typography>
      <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell width={160}>Variable</TableCell>
              <TableCell width={60}>Type</TableCell>
              <TableCell width={50}>ID</TableCell>
              <TableCell width={50}>Layer</TableCell>
              <TableCell width={100} sortDirection={sortBy === 'occurrence_count' ? sortDir : false}>
                <TableSortLabel
                  active={sortBy === 'occurrence_count'}
                  direction={sortBy === 'occurrence_count' ? sortDir : 'desc'}
                  onClick={() => handleSort('occurrence_count')}
                >
                  Count
                </TableSortLabel>
              </TableCell>
              <TableCell width={100} sortDirection={sortBy === 'avg_iterations' ? sortDir : false}>
                <TableSortLabel
                  active={sortBy === 'avg_iterations'}
                  direction={sortBy === 'avg_iterations' ? sortDir : 'desc'}
                  onClick={() => handleSort('avg_iterations')}
                >
                  Avg Iters
                </TableSortLabel>
              </TableCell>
              <TableCell width={120}>Worst Timestep</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sorted.map((h) => (
              <TableRow
                key={h.variable}
                hover
                sx={{ cursor: (onClickVariable || onClickTimestep) ? 'pointer' : 'default' }}
                onClick={() => {
                  onClickVariable?.(h.variable, h.worst_timestep_index);
                  onClickTimestep?.(h.worst_timestep_index);
                }}
              >
                <TableCell>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                    {h.variable}
                  </Typography>
                </TableCell>
                <TableCell><EntityChip type={h.entity_type} /></TableCell>
                <TableCell>{h.entity_id}</TableCell>
                <TableCell>{h.layer ?? '-'}</TableCell>
                <TableCell>{h.occurrence_count}</TableCell>
                <TableCell>{h.avg_iterations.toFixed(0)}</TableCell>
                <TableCell>
                  <Typography variant="caption">{h.worst_timestep_date}</Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
