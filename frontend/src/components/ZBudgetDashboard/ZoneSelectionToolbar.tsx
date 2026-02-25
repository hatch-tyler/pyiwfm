/**
 * ZoneSelectionToolbar: overlay toolbar for zone selection mode switching.
 * Sits at top-center of ZoneMap.
 */

import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Box from '@mui/material/Box';
import TouchAppIcon from '@mui/icons-material/TouchApp';
import CropSquareIcon from '@mui/icons-material/CropSquare';
import PolylineIcon from '@mui/icons-material/Polyline';
import FileUploadIcon from '@mui/icons-material/FileUpload';

export type SelectionMode = 'point' | 'rectangle' | 'polygon';

interface ZoneSelectionToolbarProps {
  mode: SelectionMode;
  onModeChange: (mode: SelectionMode) => void;
  onUploadClick: () => void;
  disabled?: boolean;
}

export function ZoneSelectionToolbar({ mode, onModeChange, onUploadClick, disabled }: ZoneSelectionToolbarProps) {
  return (
    <Box sx={{
      position: 'absolute',
      top: 8,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 10,
      bgcolor: 'rgba(255,255,255,0.95)',
      borderRadius: 1,
      px: 0.5,
      py: 0.25,
      display: 'flex',
      alignItems: 'center',
      gap: 0.5,
      boxShadow: 1,
    }}>
      <ToggleButtonGroup
        value={mode}
        exclusive
        onChange={(_, val) => { if (val) onModeChange(val); }}
        size="small"
      >
        <ToggleButton value="point">
          <Tooltip title="Point select (click)">
            <TouchAppIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="rectangle" disabled={disabled}>
          <Tooltip title={disabled ? 'Add a zone first' : 'Rectangle select (drag)'}>
            <CropSquareIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
        <ToggleButton value="polygon" disabled={disabled}>
          <Tooltip title={disabled ? 'Add a zone first' : 'Polygon select (click vertices, double-click to close)'}>
            <PolylineIcon fontSize="small" />
          </Tooltip>
        </ToggleButton>
      </ToggleButtonGroup>
      <Tooltip title="Upload shapefile or GeoJSON">
        <IconButton size="small" onClick={onUploadClick}>
          <FileUploadIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    </Box>
  );
}
