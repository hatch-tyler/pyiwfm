/**
 * Control panel for viewer settings.
 */

import { useEffect } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Checkbox from '@mui/material/Checkbox';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import { useViewerStore } from '../../stores/viewerStore';
import { fetchProperties } from '../../api/client';
import { LAYER_COLORS_HEX } from '../../constants/colors';

export default function ControlPanel() {
  const {
    modelInfo,
    properties,
    setProperties,
    activeProperty,
    setActiveProperty,
    opacity,
    setOpacity,
    showEdges,
    setShowEdges,
    zExaggeration,
    setZExaggeration,
    showMesh,
    setShowMesh,
    showStreams,
    setShowStreams,
    showCrossSection,
    setShowCrossSection,
    sliceAngle,
    setSliceAngle,
    slicePosition,
    setSlicePosition,
    visibleLayers,
    setLayerVisible,
    setAllLayersVisible,
  } = useViewerStore();

  // Load properties on mount
  useEffect(() => {
    const loadProperties = async () => {
      try {
        const props = await fetchProperties();
        setProperties(props);
      } catch (error) {
        console.error('Failed to load properties:', error);
      }
    };
    loadProperties();
  }, [setProperties]);

  const nLayers = modelInfo?.n_layers ?? 0;

  const angleLabel =
    sliceAngle === 0
      ? 'N-S'
      : sliceAngle === 90
        ? 'E-W'
        : `${sliceAngle}°`;

  return (
    <Paper
      sx={{
        width: 300,
        height: '100%',
        overflow: 'auto',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
      elevation={2}
    >
      <Typography variant="h6" component="h2">
        Controls
      </Typography>

      <Divider />

      {/* Visibility Toggles */}
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Visibility
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showMesh}
              onChange={(e) => setShowMesh(e.target.checked)}
              size="small"
            />
          }
          label="Show Mesh"
        />
        {modelInfo?.has_streams && (
          <FormControlLabel
            control={
              <Switch
                checked={showStreams}
                onChange={(e) => setShowStreams(e.target.checked)}
                size="small"
              />
            }
            label="Show Streams"
          />
        )}
      </Box>

      <Divider />

      {/* Property Selection */}
      <FormControl fullWidth size="small">
        <InputLabel>Property</InputLabel>
        <Select
          value={activeProperty}
          label="Property"
          onChange={(e) => setActiveProperty(e.target.value)}
        >
          {properties.map((prop) => (
            <MenuItem key={prop.id} value={prop.id}>
              {prop.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Per-Layer Visibility */}
      {nLayers > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Layers
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
            <Button
              size="small"
              variant="outlined"
              onClick={() => setAllLayersVisible(true)}
              sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 1 }}
            >
              All
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() => setAllLayersVisible(false)}
              sx={{ fontSize: '0.7rem', minWidth: 'auto', px: 1 }}
            >
              None
            </Button>
          </Box>
          {Array.from({ length: nLayers }, (_, i) => (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center' }}>
              <Checkbox
                checked={visibleLayers[i] ?? true}
                onChange={(e) => setLayerVisible(i, e.target.checked)}
                size="small"
                sx={{ py: 0.25 }}
              />
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  bgcolor: LAYER_COLORS_HEX[i % LAYER_COLORS_HEX.length],
                  mr: 1,
                  flexShrink: 0,
                }}
              />
              <Typography variant="body2">Layer {i + 1}</Typography>
            </Box>
          ))}
        </Box>
      )}

      <Divider />

      {/* Display Settings */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Display</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Opacity */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Opacity: {(opacity * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={opacity}
                onChange={(_, value) => setOpacity(value as number)}
                min={0.1}
                max={1}
                step={0.05}
                size="small"
              />
            </Box>

            {/* Z Exaggeration */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Z Exaggeration: {zExaggeration.toFixed(1)}x
              </Typography>
              <Slider
                value={zExaggeration}
                onChange={(_, value) => setZExaggeration(value as number)}
                min={1}
                max={50}
                step={1}
                size="small"
              />
            </Box>

            {/* Show Edges */}
            <FormControlLabel
              control={
                <Switch
                  checked={showEdges}
                  onChange={(e) => setShowEdges(e.target.checked)}
                  size="small"
                />
              }
              label="Show Edges"
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Cross-Section */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Cross-Section</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={showCrossSection}
                  onChange={(e) => setShowCrossSection(e.target.checked)}
                  size="small"
                />
              }
              label="Enable"
            />

            {/* Angle from E-W face */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Angle: {angleLabel} (0°=N-S, 90°=E-W)
              </Typography>
              <Slider
                value={sliceAngle}
                onChange={(_, value) => setSliceAngle(value as number)}
                min={0}
                max={180}
                step={5}
                marks={[
                  { value: 0, label: '0°' },
                  { value: 45, label: '45°' },
                  { value: 90, label: '90°' },
                  { value: 135, label: '135°' },
                  { value: 180, label: '180°' },
                ]}
                disabled={!showCrossSection}
                size="small"
              />
            </Box>

            {/* Position along domain */}
            <Box>
              <Typography variant="body2" gutterBottom>
                Position: {(slicePosition * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={slicePosition}
                onChange={(_, value) => setSlicePosition(value as number)}
                min={0}
                max={1}
                step={0.01}
                disabled={!showCrossSection}
                size="small"
              />
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
}
