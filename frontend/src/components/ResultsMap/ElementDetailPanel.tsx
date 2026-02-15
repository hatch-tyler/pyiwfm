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
    <Box sx={{ mt: 1 }}>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
        Land Use Area Over Time
      </Typography>
      <Plot
        data={traces}
        layout={{
          width: 340,
          height: 200,
          margin: { t: 10, b: 40, l: 50, r: 10 },
          showlegend: true,
          legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
          xaxis: { type: 'date' },
          yaxis: { title: { text: 'Area (acres)', font: { size: 10 } } },
        }}
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

  return (
    <Drawer
      anchor="right"
      open={true}
      onClose={onClose}
      variant="persistent"
      sx={{ '& .MuiDrawer-paper': { width: 420, pt: 1 } }}
    >
      {/* Header */}
      <Box sx={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        px: 2, py: 1, bgcolor: 'primary.main', color: 'primary.contrastText',
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            label={`Element ${elemId}`}
            size="small"
            sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'inherit', fontWeight: 700 }}
          />
        </Box>
        <IconButton size="small" onClick={onClose} sx={{ color: 'inherit' }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Subregion + stats strip */}
      <Box sx={{ px: 2, py: 1.5, bgcolor: 'grey.50', borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="body2" color="text.secondary">
          {subregion.name} (SR {subregion.id})
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mt: 0.5, flexWrap: 'wrap' }}>
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
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">Layer Properties</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={{
              '& td, & th': { px: 1, py: 0.25, fontSize: 11 },
              '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
            }}>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.200' }}>
                  <TableCell sx={{ fontWeight: 700 }}>Lyr</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Top (ft)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Bot (ft)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Thk (ft)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Kh</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Sy</TableCell>
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
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">Land Use</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 1 }}>
            {/* Timestep selector */}
            {luDates.length > 1 && (
              <Box sx={{ mb: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Timestep: {selectedDateLabel}
                  {luLoading && ' (loading...)'}
                </Typography>
                <Slider
                  value={luTimestep >= 0 ? luTimestep : luDates.length - 1}
                  min={0}
                  max={Math.max(0, luDates.length - 1)}
                  step={1}
                  onChange={(_, v) => setLuTimestep(v as number)}
                  size="small"
                  sx={{ mt: 0.5 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {(luTimestep >= 0 ? luTimestep : luDates.length - 1) + 1} / {luDates.length}
                </Typography>
              </Box>
            )}

            {landUse ? (
              <>
                {(() => {
                  // Use per-category breakdown if available, else fall back to aggregated fractions
                  const cats = landUse.categories?.filter((c) => c.area > 0) ?? [];
                  const areaUnits = landUse.units ?? 'acres';
                  if (cats.length > 0) {
                    // Color map by category type
                    const catColors: Record<string, string[]> = {
                      nonponded: ['#2e7d32','#388e3c','#43a047','#4caf50','#66bb6a','#81c784','#a5d6a7','#c8e6c9','#1b5e20','#33691e','#558b2f','#689f38','#7cb342','#8bc34a','#9ccc65','#aed581','#c5e1a5','#dcedc8','#f1f8e9','#e8f5e9'],
                      ponded: ['#0d47a1','#1565c0','#1976d2','#1e88e5','#42a5f5'],
                      urban: ['#757575'],
                      native: ['#5d4037','#8d6e63'],
                    };
                    const labels: string[] = [];
                    const values: number[] = [];
                    const colors: string[] = [];
                    for (const c of cats) {
                      labels.push(c.name || c.category);
                      values.push(c.area);
                      const palette = catColors[c.category] ?? ['#bdbdbd'];
                      const idx = cats.filter((x) => x.category === c.category).indexOf(c);
                      colors.push(palette[idx % palette.length]);
                    }
                    return (
                      <>
                        <Plot
                          data={[{
                            type: 'pie',
                            labels,
                            values,
                            marker: { colors },
                            textinfo: 'label+percent',
                            textposition: 'auto',
                            hovertemplate: '%{label}<br>%{value:.1f} ' + areaUnits + '<br>%{percent}<extra></extra>',
                            hole: 0.3,
                          }]}
                          layout={{
                            width: 400,
                            height: 360,
                            margin: { t: 30, b: 60, l: 30, r: 30 },
                            title: selectedDateLabel
                              ? { text: `Land Use — ${selectedDateLabel}`, font: { size: 12 } }
                              : undefined,
                            showlegend: true,
                            legend: {
                              orientation: 'h',
                              y: -0.15,
                              x: 0.5,
                              xanchor: 'center',
                              font: { size: 9 },
                            },
                          }}
                          config={{ displayModeBar: false }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          Total area: {landUse.total_area.toLocaleString()} {areaUnits}
                        </Typography>
                        <Table size="small" sx={{
                          mt: 1,
                          '& td, & th': { px: 1, py: 0.25, fontSize: 11 },
                          '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
                        }}>
                          <TableHead>
                            <TableRow sx={{ bgcolor: 'grey.200' }}>
                              <TableCell sx={{ fontWeight: 700 }}>Category</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 700 }}>Area ({areaUnits})</TableCell>
                              <TableCell align="right" sx={{ fontWeight: 700 }}>%</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {cats.map((c, i) => (
                              <TableRow key={i}>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <Box sx={{
                                      width: 10, height: 10, borderRadius: '50%',
                                      bgcolor: colors[i], flexShrink: 0,
                                    }} />
                                    {c.name}
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
                      <Plot
                        data={[{
                          type: 'pie',
                          labels: ['Agricultural', 'Urban', 'Native/Riparian'],
                          values: [
                            landUse.fractions.agricultural,
                            landUse.fractions.urban,
                            landUse.fractions.native_riparian,
                          ],
                          marker: { colors: ['#4caf50', '#9e9e9e', '#8d6e63'] },
                          textinfo: 'label+percent',
                          textposition: 'auto',
                          hoverinfo: 'label+percent+name',
                          hole: 0.3,
                        }]}
                        layout={{
                          width: 400,
                          height: 320,
                          margin: { t: 30, b: 60, l: 30, r: 30 },
                          title: selectedDateLabel
                            ? { text: `Land Use — ${selectedDateLabel}`, font: { size: 12 } }
                            : undefined,
                          showlegend: true,
                          legend: {
                            orientation: 'h',
                            y: -0.15,
                            x: 0.5,
                            xanchor: 'center',
                            font: { size: 10 },
                          },
                        }}
                        config={{ displayModeBar: false }}
                      />
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                        Total area: {landUse.total_area.toLocaleString()} {areaUnits}
                      </Typography>
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
            sx={{ '&:before': { display: 'none' } }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}
              sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
            >
              <Typography variant="subtitle2">Wells ({wells.length})</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 1 }}>
              {wells.map((w) => (
                <Box key={w.id} sx={{
                  mb: 1, p: 1, borderRadius: 1, bgcolor: 'grey.50',
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
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">Vertices ({vertices.length})</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={{
              '& td, & th': { px: 1, py: 0.25, fontSize: 11 },
              '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
            }}>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.200' }}>
                  <TableCell sx={{ fontWeight: 700 }}>Node</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Lng</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Lat</TableCell>
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
