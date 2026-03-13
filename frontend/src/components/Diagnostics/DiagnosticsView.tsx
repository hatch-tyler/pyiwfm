/**
 * Diagnostics tab — simulation messages, convergence tracking, mass balance.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Pagination from '@mui/material/Pagination';
import {
  fetchDiagnosticsSummary,
  fetchDiagnosticsMessages,
  fetchConvergenceData,
  fetchMassBalanceData,
} from '../../api/client';
import type {
  DiagnosticsSummary,
  DiagnosticMessage,
  ConvergenceRecord,
  MassBalanceRecord,
} from '../../api/client';
import { ConvergenceChart } from './ConvergenceChart';
import { MassBalanceChart } from './MassBalanceChart';

function SeverityChip({ severity }: { severity: string }) {
  const color = severity === 'FATAL' ? 'error'
    : severity === 'WARN' ? 'warning'
    : severity === 'INFO' ? 'info' : 'default';
  return <Chip label={severity} color={color} size="small" />;
}

function formatRuntime(seconds: number | null): string {
  if (seconds === null) return 'N/A';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// ---- Summary Cards ----

function SummaryCards({ summary }: { summary: DiagnosticsSummary }) {
  const cards = [
    { label: 'Total Messages', value: summary.message_count, color: '#1976d2' },
    { label: 'Warnings', value: summary.warning_count, color: '#ed6c02' },
    { label: 'Errors', value: summary.error_count, color: '#d32f2f' },
    { label: 'Runtime', value: formatRuntime(summary.total_runtime_seconds), color: '#2e7d32' },
    { label: 'Max Iterations', value: summary.max_iterations || 'N/A', color: '#9c27b0' },
    {
      label: 'Avg Iterations',
      value: summary.avg_iterations ? summary.avg_iterations.toFixed(1) : 'N/A',
      color: '#0288d1',
    },
  ];

  return (
    <Grid container spacing={2} sx={{ mb: 3 }}>
      {cards.map((c) => (
        <Grid item xs={6} sm={4} md={2} key={c.label}>
          <Card variant="outlined">
            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">
                {c.label}
              </Typography>
              <Typography variant="h5" sx={{ color: c.color, fontWeight: 600 }}>
                {c.value}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

// ---- Messages Table ----

function MessagesPanel() {
  const [messages, setMessages] = useState<DiagnosticMessage[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [severity, setSeverity] = useState<string>('');
  const [page, setPage] = useState(1);
  const pageSize = 50;

  useEffect(() => {
    setLoading(true);
    fetchDiagnosticsMessages(severity || undefined, pageSize, (page - 1) * pageSize)
      .then((res) => {
        setMessages(res.messages);
        setTotal(res.total);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [severity, page]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Severity</InputLabel>
          <Select value={severity} label="Severity" onChange={(e) => { setSeverity(e.target.value); setPage(1); }}>
            <MenuItem value="">All</MenuItem>
            <MenuItem value="FATAL">Fatal</MenuItem>
            <MenuItem value="WARN">Warning</MenuItem>
            <MenuItem value="INFO">Info</MenuItem>
          </Select>
        </FormControl>
        <Typography variant="body2" color="text.secondary">
          {total} message{total !== 1 ? 's' : ''}
        </Typography>
      </Box>

      {loading ? (
        <CircularProgress size={24} />
      ) : messages.length === 0 ? (
        <Typography color="text.secondary">No messages found.</Typography>
      ) : (
        <>
          <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 500 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell width={80}>Severity</TableCell>
                  <TableCell width={100}>Procedure</TableCell>
                  <TableCell>Message</TableCell>
                  <TableCell width={100}>Nodes</TableCell>
                  <TableCell width={100}>Elements</TableCell>
                  <TableCell width={60}>Line</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {messages.map((msg, i) => (
                  <TableRow key={i} hover>
                    <TableCell><SeverityChip severity={msg.severity} /></TableCell>
                    <TableCell>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {msg.procedure || '-'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {msg.text}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {msg.node_ids.length > 0 ? msg.node_ids.join(', ') : '-'}
                    </TableCell>
                    <TableCell>
                      {msg.element_ids.length > 0 ? msg.element_ids.join(', ') : '-'}
                    </TableCell>
                    <TableCell>{msg.line_number}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Pagination count={totalPages} page={page} onChange={(_, v) => setPage(v)} />
            </Box>
          )}
        </>
      )}
    </Box>
  );
}

// ---- Convergence Panel ----

function ConvergencePanel() {
  const [records, setRecords] = useState<ConvergenceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ max: 0, avg: 0, total: 0 });

  useEffect(() => {
    fetchConvergenceData()
      .then((res) => {
        setRecords(res.records);
        setStats({ max: res.max_iterations, avg: res.avg_iterations, total: res.total_timesteps });
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <CircularProgress size={24} />;
  if (records.length === 0) {
    return <Typography color="text.secondary">No convergence data available.</Typography>;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
        <Typography variant="body2">
          Total Timesteps: <strong>{stats.total}</strong>
        </Typography>
        <Typography variant="body2">
          Max Iterations: <strong>{stats.max}</strong>
        </Typography>
        <Typography variant="body2">
          Avg Iterations: <strong>{stats.avg.toFixed(1)}</strong>
        </Typography>
      </Box>
      <ConvergenceChart records={records} />
    </Box>
  );
}

// ---- Mass Balance Panel ----

function MassBalancePanel() {
  const [records, setRecords] = useState<MassBalanceRecord[]>([]);
  const [components, setComponents] = useState<string[]>([]);
  const [selectedComponent, setSelectedComponent] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMassBalanceData(selectedComponent || undefined)
      .then((res) => {
        setRecords(res.records);
        if (res.components.length > 0 && components.length === 0) {
          setComponents(res.components);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedComponent]);

  if (loading) return <CircularProgress size={24} />;
  if (records.length === 0) {
    return <Typography color="text.secondary">No mass balance data available.</Typography>;
  }

  return (
    <Box>
      {components.length > 1 && (
        <FormControl size="small" sx={{ minWidth: 150, mb: 2 }}>
          <InputLabel>Component</InputLabel>
          <Select value={selectedComponent} label="Component" onChange={(e) => setSelectedComponent(e.target.value)}>
            <MenuItem value="">All</MenuItem>
            {components.map((c) => (
              <MenuItem key={c} value={c}>{c}</MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      <MassBalanceChart records={records} />
    </Box>
  );
}

// ---- Main Component ----

export function DiagnosticsView() {
  const [summary, setSummary] = useState<DiagnosticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [subTab, setSubTab] = useState(0);

  useEffect(() => {
    fetchDiagnosticsSummary()
      .then((data) => {
        setSummary(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading diagnostics...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          No simulation diagnostics available. Run a simulation to generate SimulationMessages.out.
        </Alert>
      </Box>
    );
  }

  if (!summary || !summary.has_diagnostics) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          No simulation diagnostics found. SimulationMessages.out was not found in the model directory.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, overflowY: 'auto', height: '100%' }}>
      <Typography variant="h5" sx={{ mb: 2 }}>Simulation Diagnostics</Typography>

      <SummaryCards summary={summary} />

      <Tabs value={subTab} onChange={(_, v) => setSubTab(v)} sx={{ mb: 2 }}>
        <Tab label="Messages" />
        <Tab label="Convergence" />
        <Tab label="Mass Balance" />
      </Tabs>

      {subTab === 0 && <MessagesPanel />}
      {subTab === 1 && <ConvergencePanel />}
      {subTab === 2 && <MassBalancePanel />}
    </Box>
  );
}
