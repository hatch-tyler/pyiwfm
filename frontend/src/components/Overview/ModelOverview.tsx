/**
 * Model Overview tab — card-based summary of all model components.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import { fetchModelSummary } from '../../api/client';
import type { ModelSummary } from '../../api/client';
import { useViewerStore } from '../../stores/viewerStore';

function StatusChip({ loaded }: { loaded: boolean }) {
  return (
    <Chip
      label={loaded ? 'Loaded' : 'Not loaded'}
      color={loaded ? 'success' : 'default'}
      size="small"
      variant={loaded ? 'filled' : 'outlined'}
    />
  );
}

function StatLine({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (value === null || value === undefined) return null;
  return (
    <Typography variant="body2" color="text.secondary">
      {label}: <strong>{typeof value === 'number' ? value.toLocaleString() : value}</strong>
    </Typography>
  );
}

export function ModelOverview() {
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const setActiveTab = useViewerStore((s) => s.setActiveTab);

  useEffect(() => {
    fetchModelSummary()
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
        <Typography sx={{ ml: 2 }}>Loading model summary...</Typography>
      </Box>
    );
  }

  if (error || !summary) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">Failed to load model summary: {error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, overflowY: 'auto', height: '100%' }}>
      {summary.source && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Source: {summary.source}
        </Typography>
      )}

      <Grid container spacing={2}>
        {/* Mesh */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Mesh & Stratigraphy</Typography>
                <Chip label="Core" color="primary" size="small" />
              </Box>
              <StatLine label="Nodes" value={summary.mesh.n_nodes} />
              <StatLine label="Elements" value={summary.mesh.n_elements} />
              <StatLine label="Layers" value={summary.mesh.n_layers} />
              <StatLine label="Subregions" value={summary.mesh.n_subregions} />
              <Button
                size="small"
                sx={{ mt: 1 }}
                onClick={() => setActiveTab(1)}
              >
                View 3D Mesh
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Groundwater */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Groundwater</Typography>
                <StatusChip loaded={summary.groundwater.loaded} />
              </Box>
              {summary.groundwater.loaded ? (
                <>
                  <StatLine label="Wells" value={summary.groundwater.n_wells} />
                  <StatLine label="Hydrograph Locations" value={summary.groundwater.n_hydrograph_locations} />
                  <StatLine label="Boundary Conditions" value={summary.groundwater.n_boundary_conditions} />
                  <StatLine label="Tile Drains" value={summary.groundwater.n_tile_drains} />
                  <StatLine
                    label="Aquifer Parameters"
                    value={summary.groundwater.has_aquifer_params ? 'Yes' : 'No'}
                  />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Groundwater component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Streams */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Streams</Typography>
                <StatusChip loaded={summary.streams.loaded} />
              </Box>
              {summary.streams.loaded ? (
                <>
                  <StatLine label="Stream Nodes" value={summary.streams.n_nodes} />
                  <StatLine label="Reaches" value={summary.streams.n_reaches} />
                  <StatLine label="Diversions" value={summary.streams.n_diversions} />
                  <StatLine label="Bypasses" value={summary.streams.n_bypasses} />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Stream component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Lakes */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Lakes</Typography>
                <StatusChip loaded={summary.lakes.loaded} />
              </Box>
              {summary.lakes.loaded ? (
                <>
                  <StatLine label="Lakes" value={summary.lakes.n_lakes} />
                  <StatLine label="Lake Elements" value={summary.lakes.n_lake_elements} />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Lake component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Root Zone */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Root Zone</Typography>
                <StatusChip loaded={summary.rootzone.loaded} />
              </Box>
              {summary.rootzone.loaded ? (
                <>
                  <StatLine label="Crop Types" value={summary.rootzone.n_crop_types} />
                  <StatLine label="Land Use Types" value={
                    summary.rootzone.land_use_type_names
                      ? `${summary.rootzone.n_land_use_types} (${summary.rootzone.land_use_type_names.join(', ')})`
                      : summary.rootzone.n_land_use_types
                  } />
                  <StatLine label="Land Use Coverage" value={summary.rootzone.land_use_coverage} />
                  <StatLine label="Area Timesteps" value={summary.rootzone.n_area_timesteps} />
                  <StatLine label="Soil Parameter Sets" value={summary.rootzone.n_soil_parameter_sets} />
                  {summary.rootzone.n_missing_land_use != null && summary.rootzone.n_missing_land_use > 0 && (
                    <Typography variant="body2" color="warning.main" sx={{ mt: 0.5 }}>
                      {summary.rootzone.n_missing_land_use} elements missing land use data
                    </Typography>
                  )}
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Root zone component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Small Watersheds */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Small Watersheds</Typography>
                <StatusChip loaded={summary.small_watersheds.loaded} />
              </Box>
              {summary.small_watersheds.loaded ? (
                <StatLine label="Watersheds" value={summary.small_watersheds.n_watersheds} />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Small watershed component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Unsaturated Zone */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Unsaturated Zone</Typography>
                <StatusChip loaded={summary.unsaturated_zone.loaded} />
              </Box>
              {summary.unsaturated_zone.loaded ? (
                <>
                  <StatLine label="Layers" value={summary.unsaturated_zone.n_layers} />
                  <StatLine label="Elements" value={summary.unsaturated_zone.n_elements} />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Unsaturated zone component not loaded.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Available Results */}
        <Grid item xs={12} sm={6} md={4}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">Available Results</Typography>
                <Chip label="Data" color="info" size="small" />
              </Box>
              <StatLine
                label="Head Data"
                value={summary.available_results.has_head_data
                  ? `${summary.available_results.n_head_timesteps} timesteps`
                  : 'None'}
              />
              <StatLine
                label="GW Hydrographs"
                value={summary.available_results.has_gw_hydrographs ? 'Available' : 'None'}
              />
              <StatLine
                label="Stream Hydrographs"
                value={summary.available_results.has_stream_hydrographs ? 'Available' : 'None'}
              />
              <StatLine
                label="Budget Types"
                value={summary.available_results.n_budget_types > 0
                  ? summary.available_results.budget_types.join(', ')
                  : 'None'}
              />
              <StatLine
                label="Z-Budget Types"
                value={summary.available_results.n_zbudget_types > 0
                  ? summary.available_results.zbudget_types.join(', ')
                  : 'None'}
              />
              {summary.available_results.has_head_data && (
                <Button
                  size="small"
                  sx={{ mt: 1 }}
                  onClick={() => setActiveTab(2)}
                >
                  View Results Map
                </Button>
              )}
              {summary.available_results.n_budget_types > 0 && (
                <Button
                  size="small"
                  sx={{ mt: 1, ml: 1 }}
                  onClick={() => setActiveTab(3)}
                >
                  View Budgets
                </Button>
              )}
              {summary.available_results.n_zbudget_types > 0 && (
                <Button
                  size="small"
                  sx={{ mt: 1, ml: 1 }}
                  onClick={() => setActiveTab(4)}
                >
                  View Z-Budgets
                </Button>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
