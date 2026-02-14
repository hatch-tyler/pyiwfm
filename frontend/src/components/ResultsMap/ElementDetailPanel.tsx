/**
 * Slide-in detail panel showing all data for a clicked mesh element.
 * Uses MUI Accordion for collapsible sections and improved visual hierarchy.
 */

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
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface ElementDetailPanelProps {
  detail: Record<string, unknown> | null;
  onClose: () => void;
}

export function ElementDetailPanel({ detail, onClose }: ElementDetailPanelProps) {
  if (!detail) return null;

  const elemId = detail.element_id as number;
  const subregion = detail.subregion as { id: number; name: string };
  const vertices = detail.vertices as Array<{
    node_id: number; x: number; y: number; lng: number; lat: number;
  }>;
  const area = detail.area as number;
  const layerProps = detail.layer_properties as Array<Record<string, number>>;
  const wells = detail.wells as Array<{
    id: number; name: string; pump_rate: number; layers: number[];
  }>;

  return (
    <Drawer
      anchor="right"
      open={true}
      onClose={onClose}
      variant="persistent"
      sx={{ '& .MuiDrawer-paper': { width: 380, pt: 1 } }}
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
          <Chip label={`Area: ${area.toLocaleString()} sq ft`} size="small" variant="outlined" />
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
