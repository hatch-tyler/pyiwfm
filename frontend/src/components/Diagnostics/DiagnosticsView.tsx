/**
 * Diagnostics dashboard — simulation convergence analysis, hotspots, and messages.
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
import Collapse from '@mui/material/Collapse';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import {
  fetchDiagnosticsSummary,
  fetchDiagnosticsMessages,
  fetchConvergenceData,
  fetchConvergenceHotspots,
  fetchHotspotContext,
  fetchElementBudget,
  fetchStreamNodeBudget,
} from '../../api/client';
import type {
  DiagnosticsSummary,
  DiagnosticMessage,
  ConvergenceRecord,
  ConvergenceHotspot,
  HotspotContext,
  ElementBudgetData,
  StreamNodeBudgetData,
} from '../../api/client';
import { ConvergenceChart } from './ConvergenceChart';
import { HotspotsTable } from './HotspotsTable';
import { HotspotMap } from './HotspotMap';
import { StressContextChart } from './StressContextChart';

function formatRuntime(seconds: number | null): string {
  if (seconds === null) return 'N/A';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function SeverityChip({ severity }: { severity: string }) {
  const color = severity === 'FATAL' ? 'error'
    : severity === 'WARN' ? 'warning'
    : severity === 'INFO' ? 'info' : 'default';
  return <Chip label={severity} color={color} size="small" />;
}

// ---- Summary Cards ----

function SummaryCards({ summary }: { summary: DiagnosticsSummary }) {
  const cards = [
    { label: 'Runtime', value: formatRuntime(summary.total_runtime_seconds), color: '#2e7d32' },
    { label: 'Timesteps', value: summary.total_timesteps || 'N/A', color: '#1976d2' },
    { label: 'Max Iterations', value: summary.max_iterations || 'N/A', color: '#9c27b0' },
    { label: 'Avg Iterations', value: summary.avg_iterations ? summary.avg_iterations.toFixed(1) : 'N/A', color: '#0288d1' },
    { label: 'Warnings', value: summary.warning_count, color: '#ed6c02' },
    { label: 'Errors', value: summary.error_count, color: '#d32f2f' },
  ];

  return (
    <Grid container spacing={2} sx={{ mb: 2 }}>
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

// ---- Messages Alert ----

function MessagesAlert({ messageCount, warningCount, errorCount }: {
  messageCount: number;
  warningCount: number;
  errorCount: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [messages, setMessages] = useState<DiagnosticMessage[]>([]);
  const [loading, setLoading] = useState(false);

  if (messageCount === 0) return null;

  const severity = errorCount > 0 ? 'error' : warningCount > 0 ? 'warning' : 'info';
  const summary = [
    errorCount > 0 ? `${errorCount} error${errorCount !== 1 ? 's' : ''}` : '',
    warningCount > 0 ? `${warningCount} warning${warningCount !== 1 ? 's' : ''}` : '',
    messageCount - errorCount - warningCount > 0 ? `${messageCount - errorCount - warningCount} info` : '',
  ].filter(Boolean).join(', ');

  const handleToggle = () => {
    if (!expanded && messages.length === 0) {
      setLoading(true);
      fetchDiagnosticsMessages(undefined, 50, 0)
        .then((res) => { setMessages(res.messages); setLoading(false); })
        .catch(() => setLoading(false));
    }
    setExpanded(!expanded);
  };

  return (
    <Box sx={{ mb: 2 }}>
      <Alert
        severity={severity}
        action={
          <IconButton size="small" onClick={handleToggle}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        }
      >
        Simulation messages found: {summary}
      </Alert>
      <Collapse in={expanded}>
        {loading ? (
          <Box sx={{ p: 2, textAlign: 'center' }}><CircularProgress size={20} /></Box>
        ) : messages.length > 0 ? (
          <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300, mt: 1 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell width={80}>Severity</TableCell>
                  <TableCell width={100}>Procedure</TableCell>
                  <TableCell>Message</TableCell>
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
                    <TableCell>{msg.line_number}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : null}
      </Collapse>
    </Box>
  );
}

// ---- Timestep Detail Panel ----

function TimestepDetail({ record }: { record: ConvergenceRecord }) {
  return (
    <Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
        Timestep {record.timestep_index} — {record.date}
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={3}>
          <Typography variant="caption" color="text.secondary">Total Iterations</Typography>
          <Typography variant="h6">{record.iteration_count}</Typography>
        </Grid>
        <Grid item xs={3}>
          <Typography variant="caption" color="text.secondary">Supply Adjustments</Typography>
          <Typography variant="h6">{record.supply_adj_count || 'N/A'}</Typography>
        </Grid>
        <Grid item xs={3}>
          <Typography variant="caption" color="text.secondary">Final Residual</Typography>
          <Typography variant="h6">
            {record.max_residual !== null ? record.max_residual.toExponential(3) : 'N/A'}
          </Typography>
        </Grid>
        <Grid item xs={3}>
          <Typography variant="caption" color="text.secondary">Bottleneck</Typography>
          <Typography variant="h6" sx={{ fontFamily: 'monospace', fontSize: '1rem' }}>
            {record.bottleneck_variable || 'N/A'}
          </Typography>
        </Grid>
      </Grid>
    </Box>
  );
}

// ---- Main Component ----

export function DiagnosticsView() {
  const [summary, setSummary] = useState<DiagnosticsSummary | null>(null);
  const [convergenceRecords, setConvergenceRecords] = useState<ConvergenceRecord[]>([]);
  const [hotspots, setHotspots] = useState<ConvergenceHotspot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimestep, setSelectedTimestep] = useState<number | null>(null);
  const [chartExpanded, setChartExpanded] = useState(false);
  const [avgIterations, setAvgIterations] = useState(0);

  // Hotspot investigation state
  const [hotspotContext, setHotspotContext] = useState<HotspotContext | null>(null);
  const [hotspotLoading, setHotspotLoading] = useState(false);
  const [selectedElementId, setSelectedElementId] = useState<number | null>(null);
  const [budgetData, setBudgetData] = useState<ElementBudgetData | StreamNodeBudgetData | null>(null);
  const [budgetLoading, setBudgetLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      fetchDiagnosticsSummary(),
      fetchConvergenceData(),
      fetchConvergenceHotspots(),
    ])
      .then(([sum, conv, spots]) => {
        setSummary(sum);
        setConvergenceRecords(conv.records);
        setAvgIterations(conv.avg_iterations);
        setHotspots(spots);
        setLoading(false);
        // Auto-select the top hotspot so the investigation panel is visible
        if (spots.length > 0) {
          const top = spots[0];
          setHotspotLoading(true);
          fetchHotspotContext(top.variable, top.worst_timestep_index)
            .then((ctx) => { setHotspotContext(ctx); setHotspotLoading(false); })
            .catch(() => setHotspotLoading(false));
        }
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const selectedRecord = selectedTimestep !== null
    ? convergenceRecords.find((r) => r.timestep_index === selectedTimestep)
    : undefined;

  // Handle timestep click from chart or iteration history — update both timestep and hotspot
  const handleClickTimestep = (timestepIndex: number) => {
    setSelectedTimestep(timestepIndex);
    const rec = convergenceRecords.find((r) => r.timestep_index === timestepIndex);
    if (rec?.bottleneck_variable && rec.bottleneck_variable !== hotspotContext?.variable) {
      handleSelectHotspot(rec.bottleneck_variable, timestepIndex);
    }
  };

  // Load hotspot context when a hotspot variable is selected
  const handleSelectHotspot = (variable: string, worstTimestepIndex?: number) => {
    setHotspotLoading(true);
    setBudgetData(null);
    setSelectedElementId(null);
    const tsIdx = worstTimestepIndex ?? selectedTimestep ?? undefined;
    fetchHotspotContext(variable, tsIdx)
      .then((ctx) => {
        setHotspotContext(ctx);
        setHotspotLoading(false);
        // For stream nodes, auto-load the stream node budget
        if (ctx.entity_type === 'stream') {
          setBudgetLoading(true);
          fetchStreamNodeBudget(ctx.entity_id, tsIdx)
            .then((data) => { setBudgetData(data); setBudgetLoading(false); })
            .catch(() => setBudgetLoading(false));
        }
      })
      .catch(() => setHotspotLoading(false));
  };

  // Load element budget when an element is clicked on the map
  const handleClickElement = (elementId: number) => {
    setSelectedElementId(elementId);
    setBudgetLoading(true);
    const layer = hotspotContext?.layer ?? 1;
    const tsIdx = selectedTimestep ?? hotspotContext?.iteration_history[0]?.timestep_index;
    fetchElementBudget(elementId, tsIdx, layer)
      .then((data) => { setBudgetData(data); setBudgetLoading(false); })
      .catch(() => setBudgetLoading(false));
  };

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

  // Determine stress chart title and highlight timestep
  const budgetTitle = budgetData
    ? 'element_id' in budgetData
      ? `Element ${(budgetData as ElementBudgetData).element_id} — GW ZBudget`
      : `Stream Node ${(budgetData as StreamNodeBudgetData).stream_node_id} — Stream Budget`
    : '';
  const highlightTs = selectedTimestep && budgetData?.times
    ? budgetData.times[Math.min(5, budgetData.times.length - 1)] // center of window
    : undefined;

  return (
    <Box sx={{ p: 3, overflowY: 'auto', height: '100%' }}>
      <Typography variant="h5" sx={{ mb: 2 }}>Simulation Diagnostics</Typography>

      <SummaryCards summary={summary} />

      <MessagesAlert
        messageCount={summary.message_count}
        warningCount={summary.warning_count}
        errorCount={summary.error_count}
      />

      {/* Convergence Chart */}
      {convergenceRecords.length > 0 && (
        <Paper variant="outlined" sx={{ mb: 2, height: 420, position: 'relative' }}>
          <ConvergenceChart
            records={convergenceRecords}
            avgIterations={avgIterations}
            onExpand={() => setChartExpanded(true)}
            onClickTimestep={handleClickTimestep}
            selectedTimestep={selectedTimestep}
          />
        </Paper>
      )}

      {/* Investigation Panel — between chart and hotspots table */}
      {(selectedRecord || hotspotContext || hotspotLoading) && (
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          {/* Timestep Detail row */}
          {selectedRecord && (
            <Box sx={{ mb: hotspotContext ? 2 : 0 }}>
              <TimestepDetail record={selectedRecord} />
            </Box>
          )}

          {/* Hotspot spatial + budget investigation */}
          {hotspotLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Loading hotspot context...</Typography>
            </Box>
          ) : hotspotContext ? (
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                Investigating: {hotspotContext.variable}
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <HotspotMap
                    context={hotspotContext}
                    selectedElementId={selectedElementId}
                    onClickElement={handleClickElement}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  {/* Iteration history for this variable */}
                  {hotspotContext.iteration_history.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                        Iteration History ({hotspotContext.iteration_history.length} timesteps)
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Timesteps where this variable was the convergence bottleneck
                      </Typography>
                      <Box sx={{ maxHeight: 280, overflowY: 'auto', mt: 1 }}>
                        {[...hotspotContext.iteration_history]
                          .sort((a, b) => b.iteration_count - a.iteration_count)
                          .slice(0, 15)
                          .map((h) => (
                            <Box
                              key={h.timestep_index}
                              sx={{
                                display: 'flex', justifyContent: 'space-between',
                                py: 0.3, px: 1, cursor: 'pointer',
                                '&:hover': { bgcolor: 'action.hover' },
                                bgcolor: h.timestep_index === selectedTimestep ? 'action.selected' : 'transparent',
                                borderRadius: 0.5,
                              }}
                              onClick={() => handleClickTimestep(h.timestep_index)}
                            >
                              <Typography variant="caption">{h.date}</Typography>
                              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                {h.iteration_count} iters
                              </Typography>
                            </Box>
                          ))}
                      </Box>
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} md={4}>
                  {/* Stress Context Chart */}
                  {(budgetData || budgetLoading) ? (
                    <StressContextChart
                      title={budgetTitle}
                      times={budgetData?.times ?? []}
                      columns={budgetData?.columns ?? []}
                      loading={budgetLoading}
                      highlightTimestep={highlightTs}
                    />
                  ) : (
                    <Box sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        {hotspotContext.entity_type === 'groundwater'
                          ? 'Click an element on the map to view its GW ZBudget'
                          : 'Loading budget data...'}
                      </Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>
            </Box>
          ) : null}
        </Paper>
      )}

      {/* Hotspots Table */}
      {hotspots.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <HotspotsTable
            hotspots={hotspots}
            onClickTimestep={handleClickTimestep}
            onClickVariable={handleSelectHotspot}
          />
        </Box>
      )}

      {/* Fullscreen chart dialog */}
      <Dialog
        fullScreen
        open={chartExpanded}
        onClose={() => setChartExpanded(false)}
      >
        <DialogContent sx={{ p: 0, position: 'relative', height: '100vh' }}>
          <IconButton
            onClick={() => setChartExpanded(false)}
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}
          >
            <CloseIcon />
          </IconButton>
          <ConvergenceChart
            records={convergenceRecords}
            avgIterations={avgIterations}
            onClickTimestep={setSelectedTimestep}
            selectedTimestep={selectedTimestep}
          />
        </DialogContent>
      </Dialog>
    </Box>
  );
}
