/**
 * Slide-in panel showing diversion list and detail.
 * Uses MUI Drawer with scrollable list and expandable detail section.
 */

import { useRef, useEffect, useCallback } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Drawer from '@mui/material/Drawer';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CircleIcon from '@mui/icons-material/Circle';
import { useViewerStore } from '../../stores/viewerStore';
import { fetchDiversionDetail } from '../../api/client';
import type { DiversionArc, DiversionDetail } from '../../api/client';
import { DiversionTimeseriesChart } from './DiversionTimeseriesChart';

interface DiversionPanelProps {
  diversions: DiversionArc[];
  onClose: () => void;
}

export function DiversionPanel({ diversions, onClose }: DiversionPanelProps) {
  const {
    selectedDiversionId, diversionDetail,
    setSelectedDiversionId, setDiversionDetail,
    setActiveBudgetType, setActiveBudgetLocation, setActiveTab,
  } = useViewerStore();

  const listRef = useRef<Record<string, HTMLDivElement | null>>({});

  // Auto-scroll to selected item when map dot is clicked
  useEffect(() => {
    if (selectedDiversionId !== null) {
      const el = listRef.current[String(selectedDiversionId)];
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedDiversionId]);

  const handleSelect = useCallback(async (divId: number) => {
    setSelectedDiversionId(divId);
    try {
      const detail = await fetchDiversionDetail(divId);
      setDiversionDetail(detail);
    } catch {
      setDiversionDetail(null);
    }
  }, [setSelectedDiversionId, setDiversionDetail]);

  const handleViewBudget = useCallback((detail: DiversionDetail) => {
    setActiveBudgetType('diversion');
    setActiveBudgetLocation(detail.name);
    setActiveTab(3); // Budgets tab
  }, [setActiveBudgetType, setActiveBudgetLocation, setActiveTab]);

  const handleClose = useCallback(() => {
    setSelectedDiversionId(null);
    setDiversionDetail(null);
    onClose();
  }, [setSelectedDiversionId, setDiversionDetail, onClose]);

  return (
    <Drawer
      anchor="right"
      open={true}
      onClose={handleClose}
      variant="persistent"
      sx={{ '& .MuiDrawer-paper': { width: 420, pt: 0, display: 'flex', flexDirection: 'column' } }}
    >
      {/* Header */}
      <Box sx={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        px: 2, py: 1, bgcolor: '#e65100', color: 'white', flexShrink: 0,
      }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
          Diversions ({diversions.length})
        </Typography>
        <IconButton size="small" onClick={handleClose} sx={{ color: 'inherit' }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Detail section (when a diversion is selected) */}
      {diversionDetail && (
        <Box sx={{ borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
          {/* Detail header */}
          <Box sx={{ px: 2, py: 1, bgcolor: 'grey.50' }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              {diversionDetail.name}
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
              <Chip
                label={`Source: ${diversionDetail.source_node > 0 ? `Node ${diversionDetail.source_node}` : 'Outside'}`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Dest: ${diversionDetail.destination_type}`}
                size="small"
                variant="outlined"
              />
              {diversionDetail.max_rate !== null && (
                <Chip
                  label={`Max: ${diversionDetail.max_rate.toFixed(1)}`}
                  size="small"
                  variant="outlined"
                />
              )}
              <Chip
                label={`Priority: ${diversionDetail.priority}`}
                size="small"
                variant="outlined"
              />
            </Box>
          </Box>

          {/* Delivery area accordion */}
          <Accordion disableGutters elevation={0} sx={{ '&:before': { display: 'none' } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}
              sx={{ bgcolor: 'grey.100', minHeight: 36, '& .MuiAccordionSummary-content': { my: 0.5 } }}
            >
              <Typography variant="body2">
                Delivery Area ({diversionDetail.delivery.element_ids.length} elements)
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Type: {diversionDetail.delivery.dest_type} (ID: {diversionDetail.delivery.dest_id})
              </Typography>
              {diversionDetail.delivery.element_ids.length > 0 && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  Elements highlighted on map in orange.
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>

          {/* Timeseries accordion */}
          <Accordion disableGutters elevation={0} sx={{ '&:before': { display: 'none' } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}
              sx={{ bgcolor: 'grey.100', minHeight: 36, '& .MuiAccordionSummary-content': { my: 0.5 } }}
            >
              <Typography variant="body2">Timeseries</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              {diversionDetail.timeseries ? (
                <DiversionTimeseriesChart
                  data={diversionDetail.timeseries}
                  name={diversionDetail.name}
                />
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
                  No timeseries data available.
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>

          {/* Navigate to budget tab */}
          <Box sx={{ px: 2, py: 1 }}>
            <Button
              variant="outlined"
              size="small"
              endIcon={<ArrowForwardIcon />}
              onClick={() => handleViewBudget(diversionDetail)}
              fullWidth
              sx={{ textTransform: 'none' }}
            >
              View Diversion Budget
            </Button>
          </Box>
        </Box>
      )}

      {/* Scrollable diversion list */}
      <Box sx={{ overflowY: 'auto', flex: 1 }}>
        <List dense disablePadding>
          {diversions.map((div) => (
            <ListItemButton
              key={div.id}
              ref={(el: HTMLDivElement | null) => { listRef.current[String(div.id)] = el; }}
              selected={selectedDiversionId === div.id}
              onClick={() => handleSelect(div.id)}
              sx={{
                borderBottom: '1px solid',
                borderColor: 'divider',
                py: 0.5,
              }}
            >
              <CircleIcon
                sx={{
                  fontSize: 10,
                  mr: 1,
                  color: div.source_node > 0 ? '#e65100' : '#9e9e9e',
                }}
              />
              <ListItemText
                primary={div.name}
                secondary={`Source: ${div.source_node > 0 ? `Node ${div.source_node}` : 'Outside'} → ${div.destination_type}`}
                primaryTypographyProps={{ variant: 'body2', noWrap: true }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItemButton>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}
