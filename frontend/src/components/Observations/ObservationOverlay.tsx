/**
 * Observation overlay logic — finds matching observation for a selected
 * hydrograph location and returns the data for chart overlay.
 */

import { useState, useEffect } from 'react';
import { useViewerStore } from '../../stores/viewerStore';
import { fetchObservationData } from '../../api/client';
import type { ObservationData } from '../../api/client';

/**
 * Hook that returns observation data for the currently selected location.
 * Returns null if no matching observation is uploaded.
 */
export function useObservationOverlay(): ObservationData | null {
  const { selectedLocation, observations } = useViewerStore();
  const [obsData, setObsData] = useState<ObservationData | null>(null);

  useEffect(() => {
    if (!selectedLocation || observations.length === 0) {
      setObsData(null);
      return;
    }

    // Find observation that matches the selected location
    const match = observations.find(
      (o) => o.location_id === selectedLocation.id && o.type === selectedLocation.type
    );

    if (!match) {
      setObsData(null);
      return;
    }

    fetchObservationData(match.id)
      .then(setObsData)
      .catch(() => setObsData(null));
  }, [selectedLocation, observations]);

  return obsData;
}
