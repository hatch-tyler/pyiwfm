/**
 * Slide-in detail panel for a clicked small watershed marker.
 * Shows summary chips, root zone parameters, aquifer parameters,
 * and a table of GW routing nodes with qmaxwb values.
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

import type { SmallWatershed } from '../../api/client';

interface WatershedDetailPanelProps {
  watershed: SmallWatershed;
  onClose: () => void;
}

export function WatershedDetailPanel({ watershed, onClose }: WatershedDetailPanelProps) {
  const ws = watershed;

  return (
    <Drawer
      anchor="right"
      open={true}
      onClose={onClose}
      variant="persistent"
      sx={{ '& .MuiDrawer-paper': { width: 380, pt: 1 } }}
    >
      {/* Header — forest green */}
      <Box sx={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        px: 2, py: 1, bgcolor: '#1b7a3d', color: '#fff',
      }}>
        <Chip
          label={`Watershed ${ws.id}`}
          size="small"
          sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'inherit', fontWeight: 700 }}
        />
        <IconButton size="small" onClick={onClose} sx={{ color: 'inherit' }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Summary strip */}
      <Box sx={{ px: 2, py: 1.5, bgcolor: 'grey.50', borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip label={`Area: ${ws.area.toLocaleString()}`} size="small" variant="outlined" />
          <Chip label={`CN: ${ws.curve_number.toFixed(1)}`} size="small" variant="outlined" />
          <Chip label={`${ws.n_gw_nodes} GW nodes`} size="small" variant="outlined" />
          {ws.dest_stream_node > 0 && (
            <Chip
              label={`Dest: Strm Node ${ws.dest_stream_node}`}
              size="small"
              variant="outlined"
              color="info"
            />
          )}
        </Box>
      </Box>

      {/* Accordion sections */}
      <Box sx={{ overflowY: 'auto', flex: 1 }}>
        {/* Root Zone Parameters */}
        <Accordion defaultExpanded disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">Root Zone Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={{
              '& td, & th': { px: 1.5, py: 0.4, fontSize: 12 },
              '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
            }}>
              <TableBody>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Wilting Point</TableCell>
                  <TableCell align="right">{ws.wilting_point.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Field Capacity</TableCell>
                  <TableCell align="right">{ws.field_capacity.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Total Porosity</TableCell>
                  <TableCell align="right">{ws.total_porosity.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Lambda</TableCell>
                  <TableCell align="right">{ws.lambda_param.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Root Depth</TableCell>
                  <TableCell align="right">{ws.root_depth.toFixed(2)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Hydraulic Cond.</TableCell>
                  <TableCell align="right">{ws.hydraulic_cond.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>K_unsat Method</TableCell>
                  <TableCell align="right">{ws.kunsat_method}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Curve Number</TableCell>
                  <TableCell align="right">{ws.curve_number.toFixed(1)}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>

        {/* Aquifer Parameters */}
        <Accordion defaultExpanded disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">Aquifer Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={{
              '& td, & th': { px: 1.5, py: 0.4, fontSize: 12 },
              '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
            }}>
              <TableBody>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>GW Threshold</TableCell>
                  <TableCell align="right">{ws.gw_threshold.toFixed(2)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Max GW Storage</TableCell>
                  <TableCell align="right">{ws.max_gw_storage.toFixed(2)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Surface Flow Coeff.</TableCell>
                  <TableCell align="right">{ws.surface_flow_coeff.toFixed(4)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Baseflow Coeff.</TableCell>
                  <TableCell align="right">{ws.baseflow_coeff.toFixed(4)}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>

        {/* GW Routing Nodes */}
        <Accordion defaultExpanded disableGutters elevation={0}
          sx={{ '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}
            sx={{ bgcolor: 'grey.100', minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}
          >
            <Typography variant="subtitle2">GW Routing Nodes ({ws.gw_nodes.length})</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <Table size="small" sx={{
              '& td, & th': { px: 1, py: 0.25, fontSize: 11 },
              '& tbody tr:nth-of-type(even)': { bgcolor: 'grey.50' },
            }}>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.200' }}>
                  <TableCell sx={{ fontWeight: 700 }}>Node</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>qmaxwb</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Type</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {ws.gw_nodes.map((gwn) => (
                  <TableRow key={gwn.node_id}>
                    <TableCell>{gwn.node_id}</TableCell>
                    <TableCell align="right">{gwn.raw_qmaxwb.toFixed(2)}</TableCell>
                    <TableCell>
                      <Chip
                        label={gwn.is_baseflow ? `Baseflow L${gwn.layer}` : 'Perc'}
                        size="small"
                        color={gwn.is_baseflow ? 'secondary' : 'warning'}
                        variant="outlined"
                        sx={{ height: 20, fontSize: 10 }}
                      />
                    </TableCell>
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
