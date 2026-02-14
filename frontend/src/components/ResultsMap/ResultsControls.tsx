/**
 * Right sidebar controls for the Results Map tab.
 * Time slider, layer select, color-by dropdown, overlay toggles,
 * basemap selector, head diff mode, cross-section tool, upload button.
 */

import { useRef, useCallback, useEffect } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Divider from '@mui/material/Divider';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import DownloadIcon from '@mui/icons-material/Download';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import DeleteIcon from '@mui/icons-material/Delete';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListIcon from '@mui/icons-material/List';
import { useViewerStore } from '../../stores/viewerStore';
import {
  uploadObservation, fetchObservations, deleteObservation,
  getExportHeadsCsvUrl, getExportMeshGeoJsonUrl,
} from '../../api/client';
import { BASEMAPS } from './mapStyles';

interface ResultsControlsProps {
  nLayers: number;
  onUploadComplete?: () => void;
}

export function ResultsControls({ nLayers, onUploadComplete }: ResultsControlsProps) {
  const {
    headTimestep, headTimes, headLayer,
    showGWLocations, showStreamLocations, showSubsidenceLocations,
    showSubregions, showStreamsOnMap, showWells, showNodes,
    showLakes, showBoundaryConditions,
    showSmallWatersheds, showDiversions,
    mapColorProperty, properties,
    isAnimating, resultsInfo, modelInfo,
    headDiffMode, headDiffTimestepA, headDiffTimestepB,
    crossSectionMode,
    compareMode, comparedLocationIds,
    animationSpeed, observations,
    selectedBasemap,
    setHeadTimestep, setHeadLayer,
    setAnimationSpeed,
    setShowGWLocations, setShowStreamLocations, setShowSubsidenceLocations,
    setShowSubregions, setShowStreamsOnMap, setShowWells, setShowNodes,
    setShowLakes, setShowBoundaryConditions,
    setShowSmallWatersheds, setShowDiversions, setDiversionListOpen,
    setMapColorProperty,
    setIsAnimating, setObservations,
    setHeadDiffMode, setHeadDiffTimestepA, setHeadDiffTimestepB,
    setCrossSectionMode, setCrossSectionPoints, setCrossSectionData,
    setCompareMode, setComparedLocationIds,
    setSelectedBasemap,
  } = useViewerStore();

  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const nTimesteps = headTimes.length;
  const currentTime = headTimes[headTimestep] || '';
  const hasHeads = resultsInfo ? resultsInfo.n_head_timesteps > 0 : false;
  const hasStreams = modelInfo?.has_streams ?? false;
  const hasLakes = (modelInfo?.n_lakes ?? 0) > 0;

  // Format datetime for display
  const formatTime = (iso: string) => {
    if (!iso) return '';
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
      });
    } catch {
      return iso;
    }
  };

  // Animation control
  const toggleAnimation = useCallback(() => {
    if (isAnimating) {
      if (animRef.current) clearInterval(animRef.current);
      animRef.current = null;
      setIsAnimating(false);
    } else {
      setIsAnimating(true);
      animRef.current = setInterval(() => {
        setHeadTimestep(
          useViewerStore.getState().headTimestep + 1 >= nTimesteps
            ? 0
            : useViewerStore.getState().headTimestep + 1
        );
      }, animationSpeed);
    }
  }, [isAnimating, nTimesteps, setIsAnimating, setHeadTimestep, animationSpeed]);

  // Restart animation interval when speed changes during playback
  useEffect(() => {
    if (!isAnimating) return;
    if (animRef.current) clearInterval(animRef.current);
    animRef.current = setInterval(() => {
      setHeadTimestep(
        useViewerStore.getState().headTimestep + 1 >= nTimesteps
          ? 0
          : useViewerStore.getState().headTimestep + 1
      );
    }, animationSpeed);
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, [animationSpeed, isAnimating, nTimesteps, setHeadTimestep]);

  useEffect(() => {
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, []);

  // File upload handler
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadObservation(file);
      const obs = await fetchObservations();
      setObservations(obs);
      onUploadComplete?.();
    } catch (err) {
      console.error('Upload failed:', err);
    }
    e.target.value = '';
  };

  const layerOptions = Array.from({ length: nLayers }, (_, i) => i + 1);

  // Build color-by options: "Head" + available properties
  const colorOptions: Array<{ id: string; label: string; disabled?: boolean }> = [
    { id: 'head', label: hasHeads ? 'Head' : 'Head (no data)', disabled: !hasHeads },
  ];
  for (const prop of properties) {
    if (prop.id !== 'layer') {
      colorOptions.push({ id: prop.id, label: prop.name });
    }
  }

  // Toggle cross-section tool
  const handleCrossSectionToggle = useCallback(() => {
    if (crossSectionMode) {
      setCrossSectionMode(false);
      setCrossSectionPoints([]);
      setCrossSectionData(null);
    } else {
      setCrossSectionMode(true);
      setCrossSectionPoints([]);
      setCrossSectionData(null);
    }
  }, [crossSectionMode, setCrossSectionMode, setCrossSectionPoints, setCrossSectionData]);

  return (
    <Paper
      elevation={2}
      sx={{ width: 280, p: 2, overflowY: 'auto', flexShrink: 0 }}
    >
      <Typography variant="subtitle2" gutterBottom>
        Results Controls
      </Typography>

      {/* Basemap selector */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        Basemap
      </Typography>
      <ButtonGroup size="small" fullWidth sx={{ mb: 1.5 }}>
        {Object.entries(BASEMAPS).map(([key, bm]) => (
          <Button
            key={key}
            variant={selectedBasemap === key ? 'contained' : 'outlined'}
            onClick={() => setSelectedBasemap(key)}
            sx={{ textTransform: 'none', fontSize: 11, px: 0.5 }}
          >
            {bm.name}
          </Button>
        ))}
      </ButtonGroup>

      {/* Color By dropdown */}
      <FormControl fullWidth size="small" sx={{ mb: 2 }}>
        <InputLabel>Color By</InputLabel>
        <Select
          value={headDiffMode ? 'head' : mapColorProperty}
          label="Color By"
          onChange={(e) => {
            setMapColorProperty(e.target.value as string);
            if (headDiffMode) setHeadDiffMode(false);
          }}
          disabled={headDiffMode}
        >
          {colorOptions.map((opt) => (
            <MenuItem key={opt.id} value={opt.id} disabled={opt.disabled}>
              {opt.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Time Slider */}
      {hasHeads && nTimesteps > 0 && mapColorProperty === 'head' && !headDiffMode && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Timestep: {formatTime(currentTime)}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton size="small" onClick={toggleAnimation}>
              {isAnimating ? <PauseIcon /> : <PlayArrowIcon />}
            </IconButton>
            <Slider
              value={headTimestep}
              min={0}
              max={Math.max(0, nTimesteps - 1)}
              step={1}
              onChange={(_, v) => setHeadTimestep(v as number)}
              size="small"
            />
          </Box>
          <Typography variant="caption" color="text.secondary">
            {headTimestep + 1} / {nTimesteps}
          </Typography>
          <Box sx={{ mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              Speed: {animationSpeed}ms
            </Typography>
            <Slider
              value={animationSpeed}
              min={100}
              max={2000}
              step={100}
              onChange={(_, v) => setAnimationSpeed(v as number)}
              size="small"
              track="inverted"
            />
          </Box>
        </Box>
      )}

      {/* Head Difference Mode */}
      {hasHeads && nTimesteps > 1 && (
        <>
          <FormControlLabel
            control={
              <Checkbox
                size="small"
                checked={headDiffMode}
                onChange={(_, c) => {
                  setHeadDiffMode(c);
                  if (c) setMapColorProperty('head');
                }}
              />
            }
            label={<Typography variant="body2">Head Difference</Typography>}
          />
          {headDiffMode && (
            <Box sx={{ pl: 1, mb: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Timestep A: {formatTime(headTimes[headDiffTimestepA] || '')}
              </Typography>
              <Slider
                value={headDiffTimestepA}
                min={0}
                max={Math.max(0, nTimesteps - 1)}
                step={1}
                onChange={(_, v) => setHeadDiffTimestepA(v as number)}
                size="small"
              />
              <Typography variant="caption" color="text.secondary">
                Timestep B: {formatTime(headTimes[headDiffTimestepB] || '')}
              </Typography>
              <Slider
                value={headDiffTimestepB}
                min={0}
                max={Math.max(0, nTimesteps - 1)}
                step={1}
                onChange={(_, v) => setHeadDiffTimestepB(v as number)}
                size="small"
              />
              <Typography variant="caption" color="text.secondary">
                Shows B - A (blue=rise, red=drawdown)
              </Typography>
            </Box>
          )}
        </>
      )}

      {/* Layer selector */}
      <FormControl fullWidth size="small" sx={{ mb: 2 }}>
        <InputLabel>Layer</InputLabel>
        <Select
          value={headLayer}
          label="Layer"
          onChange={(e) => setHeadLayer(e.target.value as number)}
        >
          {layerOptions.map((l) => (
            <MenuItem key={l} value={l}>Layer {l}</MenuItem>
          ))}
        </Select>
      </FormControl>

      <Divider sx={{ my: 1 }} />

      {/* Unified Layers section (merged Map Overlays + Show Locations) */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        Layers
      </Typography>
      <FormControlLabel
        control={<Checkbox size="small" checked={showSubregions} onChange={(_, c) => setShowSubregions(c)} />}
        label={<Typography variant="body2">Subregions</Typography>}
      />
      {hasStreams && (
        <FormControlLabel
          control={<Checkbox size="small" checked={showStreamsOnMap} onChange={(_, c) => setShowStreamsOnMap(c)} />}
          label={<Typography variant="body2">Streams</Typography>}
        />
      )}
      <FormControlLabel
        control={<Checkbox size="small" checked={showWells} onChange={(_, c) => setShowWells(c)} />}
        label={<Typography variant="body2">Wells</Typography>}
      />
      {hasLakes && (
        <FormControlLabel
          control={<Checkbox size="small" checked={showLakes} onChange={(_, c) => setShowLakes(c)} />}
          label={<Typography variant="body2">Lakes</Typography>}
        />
      )}
      <FormControlLabel
        control={<Checkbox size="small" checked={showBoundaryConditions} onChange={(_, c) => setShowBoundaryConditions(c)} />}
        label={<Typography variant="body2">Boundary Conditions</Typography>}
      />
      <FormControlLabel
        control={<Checkbox size="small" checked={showSmallWatersheds} onChange={(_, c) => setShowSmallWatersheds(c)} />}
        label={<Typography variant="body2">Small Watersheds</Typography>}
      />
      {hasStreams && (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <FormControlLabel
            control={<Checkbox size="small" checked={showDiversions} onChange={(_, c) => setShowDiversions(c)} />}
            label={<Typography variant="body2">Diversions</Typography>}
            sx={{ flex: 1 }}
          />
          {showDiversions && (
            <IconButton
              size="small"
              onClick={() => setDiversionListOpen(true)}
              title="Open diversion list"
            >
              <ListIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      )}
      <FormControlLabel
        control={<Checkbox size="small" checked={showNodes} onChange={(_, c) => setShowNodes(c)} />}
        label={<Typography variant="body2">GW Nodes</Typography>}
      />
      <FormControlLabel
        control={<Checkbox size="small" checked={showGWLocations} onChange={(_, c) => setShowGWLocations(c)} />}
        label={<Typography variant="body2">GW Hydrograph Locations</Typography>}
      />
      <FormControlLabel
        control={<Checkbox size="small" checked={showStreamLocations} onChange={(_, c) => setShowStreamLocations(c)} />}
        label={<Typography variant="body2">Stream Gage Locations</Typography>}
      />
      <FormControlLabel
        control={<Checkbox size="small" checked={showSubsidenceLocations} onChange={(_, c) => setShowSubsidenceLocations(c)} />}
        label={<Typography variant="body2">Subsidence Locations</Typography>}
      />

      <Divider sx={{ my: 1 }} />

      {/* Cross-section tool */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        Tools
      </Typography>
      <Button
        variant={crossSectionMode ? 'contained' : 'outlined'}
        size="small"
        color={crossSectionMode ? 'error' : 'primary'}
        onClick={handleCrossSectionToggle}
        fullWidth
        sx={{ mb: 1 }}
      >
        {crossSectionMode ? 'Cancel Cross-Section' : 'Draw Cross-Section'}
      </Button>

      {/* Compare mode toggle */}
      <Button
        variant={compareMode ? 'contained' : 'outlined'}
        size="small"
        color={compareMode ? 'warning' : 'primary'}
        startIcon={<CompareArrowsIcon />}
        onClick={() => {
          if (compareMode) {
            setCompareMode(false);
            setComparedLocationIds([]);
          } else {
            setCompareMode(true);
            setComparedLocationIds([]);
          }
        }}
        fullWidth
        sx={{ mb: 1 }}
      >
        {compareMode
          ? `Compare (${comparedLocationIds.length} selected)`
          : 'Compare Hydrographs'}
      </Button>

      <Divider sx={{ my: 1 }} />

      {/* Export buttons */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        Export
      </Typography>
      {hasHeads && (
        <Button
          variant="outlined"
          size="small"
          startIcon={<DownloadIcon />}
          component="a"
          href={getExportHeadsCsvUrl(headTimestep, headLayer)}
          download
          fullWidth
          sx={{ mb: 0.5, textTransform: 'none' }}
        >
          Heads CSV
        </Button>
      )}
      <Button
        variant="outlined"
        size="small"
        startIcon={<DownloadIcon />}
        component="a"
        href={getExportMeshGeoJsonUrl(headLayer)}
        download
        fullWidth
        sx={{ mb: 0.5, textTransform: 'none' }}
      >
        Mesh GeoJSON
      </Button>

      <Divider sx={{ my: 1 }} />

      {/* Upload button */}
      <Box>
        <Button
          variant="outlined"
          size="small"
          startIcon={<UploadFileIcon />}
          component="label"
          fullWidth
        >
          Upload Observation
          <input type="file" accept=".csv,.txt" hidden onChange={handleUpload} />
        </Button>
      </Box>

      {/* Observation list */}
      {observations.length > 0 && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            Uploaded Observations ({observations.length})
          </Typography>
          <List dense disablePadding>
            {observations.map((obs) => (
              <ListItem
                key={obs.id}
                disablePadding
                secondaryAction={
                  <IconButton
                    edge="end"
                    size="small"
                    onClick={async () => {
                      try {
                        await deleteObservation(obs.id);
                        const updated = await fetchObservations();
                        setObservations(updated);
                      } catch (err) {
                        console.error('Delete failed:', err);
                      }
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={obs.filename}
                  secondary={`${obs.n_records} records`}
                  primaryTypographyProps={{ variant: 'body2', noWrap: true }}
                  secondaryTypographyProps={{ variant: 'caption' }}
                />
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Paper>
  );
}
