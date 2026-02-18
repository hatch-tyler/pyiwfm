/**
 * Slide-in detail panel showing all data for a clicked mesh element.
 * Uses MUI Accordion for collapsible sections and improved visual hierarchy.
 */

import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Drawer from '@mui/material/Drawer';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import CircularProgress from '@mui/material/CircularProgress';
import Slider from '@mui/material/Slider';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Plot from 'react-plotly.js';
import {
  fetchElementLandUseTimeseries,
  fetchElementDetail,
  fetchLandUseDates,
} from '../../api/client';
import type { ElementLandUseTimeseries, LandUseTimesteps } from '../../api/client';

/** Top-N threshold for pie chart aggregation */
const PIE_TOP_N = 8;

/** Qualitative color palette for pie chart slices */
const PIE_PALETTE = [
  '#2563eb', '#16a34a', '#ea580c', '#9333ea',
  '#0891b2', '#e11d48', '#ca8a04', '#4f46e5',
];

/** Stacked area chart showing land-use area over time for one element. */
function LandUseTimeseriesChart({ elementId }: { elementId: number }) {
  const [tsData, setTsData] = useState<ElementLandUseTimeseries | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchElementLandUseTimeseries(elementId)
      .then((data: ElementLandUseTimeseries) => { if (!cancelled) setTsData(data); })
      .catch((e: Error) => { if (!cancelled) setError(e.message ?? 'Failed to load'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [elementId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }
  if (error) return null; // Silently skip if timeseries unavailable
  if (!tsData || !tsData.dates || tsData.dates.length === 0) return null;

  const colors: Record<string, string> = {
    nonponded: '#4caf50',
    ponded: '#66bb6a',
    urban: '#9e9e9e',
    native: '#8d6e63',
  };
  const labels: Record<string, string> = {
    nonponded: 'Non-ponded Ag',
    ponded: 'Ponded Ag',
    urban: 'Urban',
    native: 'Native/Riparian',
  };

  const traces: Plotly.Data[] = [];
  for (const key of ['nonponded', 'ponded', 'urban', 'native'] as const) {
    const series = tsData[key];
    if (!series || !series.areas) continue;
    // Sum all columns per timestep to get total area for this land-use type
    const totals = series.areas.map((row: number[]) =>
      row.reduce((a: number, b: number) => a + b, 0)
    );
    traces.push({
      type: 'scatter',
      mode: 'lines',
      stackgroup: 'one',
      name: labels[key],
      x: tsData.dates,
      y: totals,
      line: { width: 0 },
      fillcolor: colors[key],
    });
  }

  if (traces.length === 0) return null;

  return (
    <Box sx={{ mt: 1.5 }}>
      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5, display: 'block' }}>
        Land Use Area Over Time
      </Typography>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 220,
          margin: { t: 10, b: 40, l: 50, r: 10 },
          showlegend: true,
          legend: { orientation: 'h', y: -0.3, font: { size: 10 } },
          xaxis: { type: 'date' },
          yaxis: { title: { text: 'Area (acres)', font: { size: 10 } } },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
        }}
        useResizeHandler
        style={{ width: '100%' }}
        config={{ displayModeBar: false }}
      />
    </Box>
  );
}

interface ElementDetailPanelProps {
  detail: Record<string, unknown> | null;
  onClose: () => void;
}

export function ElementDetailPanel({ detail, onClose }: ElementDetailPanelProps) {
  // Timestep state for pie chart
  const [luDates, setLuDates] = useState<string[]>([]);
  const [luTimestep, setLuTimestep] = useState<number>(-1); // -1 = last available
  const [luDetail, setLuDetail] = useState<Record<string, unknown> | null>(null);
  const [luLoading, setLuLoading] = useState(false);

  const currentDetail = luDetail ?? detail;

  // Load available dates on mount
  useEffect(() => {
    fetchLandUseDates()
      .then((resp: LandUseTimesteps) => {
        setLuDates(resp.dates);
        // Default to last timestep
        if (resp.dates.length > 0) {
          setLuTimestep(resp.dates.length - 1);
        }
      })
      .catch(() => {});
  }, []);

  // Re-fetch element detail when timestep changes
  useEffect(() => {
    if (!detail || luTimestep < 0) return;
    const elemId = detail.element_id as number;
    if (!elemId) return;

    setLuLoading(true);
    fetchElementDetail(elemId, luTimestep)
      .then(setLuDetail)
      .catch(() => setLuDetail(null))
      .finally(() => setLuLoading(false));
  }, [luTimestep, detail]);

  if (!currentDetail) return null;

  const elemId = currentDetail.element_id as number;
  const subregion = currentDetail.subregion as { id: number; name: string };
  const vertices = currentDetail.vertices as Array<{
    node_id: number; x: number; y: number; lng: number; lat: number;
  }>;
  const area = currentDetail.area as number;
  const layerProps = currentDetail.layer_properties as Array<Record<string, number>>;
  const wells = currentDetail.wells as Array<{
    id: number; name: string; pump_rate: number; layers: number[];
  }>;
  const landUse = currentDetail.land_use as {
    fractions: Record<string, number>;
    total_area: number;
    categories?: Array<{ category: string; name: string; area: number; crop_id?: number }>;
    units?: string;
    crops?: Array<{ crop_id: number; name: string; fraction: number; area: number }>;
  } | null;

  // Format date for display
  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
      });
    } catch {
      return iso;
    }
  };

  const selectedDateLabel = luTimestep >= 0 && luTimestep < luDates.length
    ? formatDate(luDates[luTimestep])
    : '';

  /** Common table sx for all data tables */
  const tableSx = {
    '& td, & th': { px: 1, py: 0.5, fontSize: 12 },
    '& thead th': {
      fontSize: 11,
      fontWeight: 700,
      textTransform: 'uppercase',
      bgcolor: 'grey.50',
      borderBottomWidth: 2,
    },
    '& tbody tr:nth-of-type(even)': { bgcolor: 'rgba(0,0,0,0.02)' },
    '& tbody tr:hover': { bgcolor: 'rgba(0,0,0,0.04)' },
  };

  /** Common accordion summary sx */
  const accordionSummarySx = {
    minHeight: 48,
    '& .MuiAccordionSummary-content': { my: 1 },
    '&:hover': { bgcolor: 'grey.50' },
  };

  return (
    <Drawer
      anchor="right"
      open={true}
      onClose={onClose}
      variant="persistent"
      sx={{ '& .MuiDrawer-paper': { width: 480 } }}
    >
      {/* Header */}
      <Box sx={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        px: 2.5, py: 1.5, bgcolor: 'primary.main', color: 'primary.contrastText',
        boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
      }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
          Element {elemId}
        </Typography>
        <IconButton size="small" onClick={onClose} sx={{ color: 'inherit' }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Subregion + stats strip */}
      <Box sx={{ px: 2.5, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {subregion.name}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Subregion {subregion.id}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
          <Chip label={`Area: ${area.toLocaleString()} acres`} size="small" variant="outlined" />
          <Chip label={`${vertices.length} vertices`} size="small" variant="outlined" />
          {wells.length > 0 && (
            <Chip label={`${wells.length} well${wells.length > 1 ? 's' : ''}`} size="small" variant="outlined" color="error" />
          )}
        </Box>
      </Box>

      {/* Accordion sections */}
      <Box sx={{ overflowY: 'auto', flex: 1 }}>
        {/* Layer Properties (default expanded) */}
        <Accordion defaultExpanded disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' }, borderBottom: 1, borderColor: 'divider' }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={accordionSummarySx}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, letterSpacing: '0.02em' }}>
              Layer Properties
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={tableSx}>
              <TableHead>
                <TableRow>
                  <TableCell>Lyr</TableCell>
                  <TableCell align="right">Top (ft)</TableCell>
                  <TableCell align="right">Bot (ft)</TableCell>
                  <TableCell align="right">Thk (ft)</TableCell>
                  <TableCell align="right">Kh</TableCell>
                  <TableCell align="right">Sy</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {layerProps.map((lp) => (
                  <TableRow key={lp.layer}>
                    <TableCell>{lp.layer}</TableCell>
                    <TableCell align="right">{lp.top_elev?.toFixed(0) ?? '-'}</TableCell>
                    <TableCell align="right">{lp.bottom_elev?.toFixed(0) ?? '-'}</TableCell>
                    <TableCell align="right">{lp.thickness?.toFixed(0) ?? '-'}</TableCell>
                    <TableCell align="right">{lp.kh?.toFixed(2) ?? '-'}</TableCell>
                    <TableCell align="right">{lp.sy?.toFixed(3) ?? '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>

        {/* Land Use (always shown; displays data or diagnostic message) */}
        <Accordion defaultExpanded={!!landUse} disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' }, borderBottom: 1, borderColor: 'divider' }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={accordionSummarySx}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, letterSpacing: '0.02em' }}>
              Land Use
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ px: 2.5, py: 1 }}>
            {/* Timestep selector */}
            {luDates.length > 1 && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    Timestep: {selectedDateLabel}
                    {luLoading && <CircularProgress size={12} />}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {(luTimestep >= 0 ? luTimestep : luDates.length - 1) + 1} / {luDates.length}
                  </Typography>
                </Box>
                <Slider
                  value={luTimestep >= 0 ? luTimestep : luDates.length - 1}
                  min={0}
                  max={Math.max(0, luDates.length - 1)}
                  step={1}
                  onChange={(_, v) => setLuTimestep(v as number)}
                  size="small"
                  sx={{ mt: 0.5 }}
                />
              </Box>
            )}

            {landUse ? (
              <>
                {(() => {
                  // Use per-category breakdown if available, else fall back to aggregated fractions
                  const cats = landUse.categories?.filter((c) => c.area > 0) ?? [];
                  const areaUnits = landUse.units ?? 'acres';
                  if (cats.length > 0) {
                    // Sort by area descending for "Top N + Other" aggregation
                    const sorted = [...cats].sort((a, b) => b.area - a.area);

                    const pieLabels: string[] = [];
                    const pieValues: number[] = [];
                    const pieColors: string[] = [];

                    if (sorted.length <= PIE_TOP_N) {
                      for (let i = 0; i < sorted.length; i++) {
                        pieLabels.push(sorted[i].name || sorted[i].category);
                        pieValues.push(sorted[i].area);
                        pieColors.push(PIE_PALETTE[i % PIE_PALETTE.length]);
                      }
                    } else {
                      for (let i = 0; i < PIE_TOP_N; i++) {
                        pieLabels.push(sorted[i].name || sorted[i].category);
                        pieValues.push(sorted[i].area);
                        pieColors.push(PIE_PALETTE[i]);
                      }
                      const restCount = sorted.length - PIE_TOP_N;
                      const restArea = sorted.slice(PIE_TOP_N).reduce((s, c) => s + c.area, 0);
                      pieLabels.push(`Other (${restCount})`);
                      pieValues.push(restArea);
                      pieColors.push('#9e9e9e');
                    }

                    // Color lookup for table rows (all categories, sorted order)
                    const getTableColor = (idx: number) =>
                      idx < PIE_TOP_N ? PIE_PALETTE[idx % PIE_PALETTE.length] : '#9e9e9e';

                    return (
                      <>
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                          {selectedDateLabel ? `Land Use — ${selectedDateLabel}` : 'Land Use Breakdown'}
                        </Typography>
                        <Plot
                          data={[{
                            type: 'pie',
                            labels: pieLabels,
                            values: pieValues,
                            marker: {
                              colors: pieColors,
                              line: { color: '#ffffff', width: 2 },
                            },
                            textinfo: 'percent',
                            textposition: 'inside',
                            hovertemplate: '%{label}<br>%{value:.1f} ' + areaUnits + '<br>%{percent}<extra></extra>',
                            hole: 0.45,
                            sort: false,
                            direction: 'clockwise',
                            rotation: 90,
                            domain: { x: [0.0, 0.55] },
                          } as Plotly.Data]}
                          layout={{
                            autosize: true,
                            height: 280,
                            margin: { t: 10, b: 10, l: 0, r: 0 },
                            showlegend: true,
                            legend: {
                              x: 0.6,
                              y: 0.5,
                              yanchor: 'middle',
                              font: { size: 11 },
                            },
                            annotations: [{
                              text: `<b>${landUse.total_area.toLocaleString()}</b><br>${areaUnits}`,
                              showarrow: false,
                              font: { size: 12, color: '#666' },
                              x: 0.275,
                              y: 0.5,
                              xanchor: 'center',
                              yanchor: 'middle',
                            }],
                            paper_bgcolor: 'transparent',
                            plot_bgcolor: 'transparent',
                          }}
                          useResizeHandler
                          style={{ width: '100%' }}
                          config={{ displayModeBar: false }}
                        />
                        <Table size="small" sx={{ mt: 1, ...tableSx }}>
                          <TableHead>
                            <TableRow>
                              <TableCell>Category</TableCell>
                              <TableCell align="right">Area ({areaUnits})</TableCell>
                              <TableCell align="right">%</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {sorted.map((c, i) => (
                              <TableRow key={i}>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <Box sx={{
                                      width: 10, height: 10, borderRadius: '50%',
                                      bgcolor: getTableColor(i), flexShrink: 0,
                                    }} />
                                    {c.name || c.category}
                                  </Box>
                                </TableCell>
                                <TableCell align="right">{c.area.toFixed(1)}</TableCell>
                                <TableCell align="right">
                                  {landUse.total_area > 0
                                    ? ((c.area / landUse.total_area) * 100).toFixed(1)
                                    : '0.0'}%
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </>
                    );
                  }
                  // Fallback: aggregated fractions only
                  return (
                    <>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                        {selectedDateLabel ? `Land Use — ${selectedDateLabel}` : 'Land Use Breakdown'}
                      </Typography>
                      <Plot
                        data={[{
                          type: 'pie',
                          labels: ['Agricultural', 'Urban', 'Native/Riparian'],
                          values: [
                            landUse.fractions.agricultural,
                            landUse.fractions.urban,
                            landUse.fractions.native_riparian,
                          ],
                          marker: {
                            colors: ['#4caf50', '#9e9e9e', '#8d6e63'],
                            line: { color: '#ffffff', width: 2 },
                          },
                          textinfo: 'percent',
                          textposition: 'inside',
                          hoverinfo: 'label+percent+name',
                          hole: 0.45,
                        } as Plotly.Data]}
                        layout={{
                          autosize: true,
                          height: 280,
                          margin: { t: 10, b: 10, l: 30, r: 30 },
                          showlegend: true,
                          legend: {
                            orientation: 'h',
                            y: -0.1,
                            x: 0.5,
                            xanchor: 'center',
                            font: { size: 11 },
                          },
                          annotations: [{
                            text: `<b>${landUse.total_area.toLocaleString()}</b><br>${areaUnits}`,
                            showarrow: false,
                            font: { size: 12, color: '#666' },
                            x: 0.5,
                            y: 0.5,
                            xanchor: 'center',
                            yanchor: 'middle',
                          }],
                          paper_bgcolor: 'transparent',
                          plot_bgcolor: 'transparent',
                        }}
                        useResizeHandler
                        style={{ width: '100%' }}
                        config={{ displayModeBar: false }}
                      />
                    </>
                  );
                })()}
                <LandUseTimeseriesChart elementId={elemId} />
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No land use data available for this element. Check that root zone
                area files are correctly parsed and wired.
              </Typography>
            )}
          </AccordionDetails>
        </Accordion>

        {/* Wells (expanded if any, collapsed if none) */}
        {wells.length > 0 && (
          <Accordion defaultExpanded disableGutters elevation={0}
            sx={{ '&:before': { display: 'none' }, borderBottom: 1, borderColor: 'divider' }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={accordionSummarySx}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, letterSpacing: '0.02em' }}>
                Wells ({wells.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 2.5, py: 1 }}>
              {wells.map((w) => (
                <Box key={w.id} sx={{
                  mb: 1.5, p: 1.5, borderRadius: 1.5, bgcolor: 'grey.50',
                  border: 1, borderColor: 'divider',
                }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {w.name}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                    <Chip
                      label={`Rate: ${w.pump_rate.toFixed(1)}`}
                      size="small"
                      color={w.pump_rate <= 0 ? 'error' : 'info'}
                      variant="outlined"
                    />
                    <Chip
                      label={`Layers: ${w.layers.join(', ') || 'N/A'}`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>
        )}

        {/* Vertices (collapsed by default) */}
        <Accordion disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' }, borderBottom: 1, borderColor: 'divider' }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={accordionSummarySx}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, letterSpacing: '0.02em' }}>
              Vertices ({vertices.length})
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={tableSx}>
              <TableHead>
                <TableRow>
                  <TableCell>Node</TableCell>
                  <TableCell align="right">Lng</TableCell>
                  <TableCell align="right">Lat</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {vertices.map((v) => (
                  <TableRow key={v.node_id}>
                    <TableCell>{v.node_id}</TableCell>
                    <TableCell align="right">{v.lng.toFixed(5)}</TableCell>
                    <TableCell align="right">{v.lat.toFixed(5)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>
      </Box>
    </Drawer>
  );
}
