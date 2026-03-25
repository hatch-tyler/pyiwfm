/**
 * SVG mini-map showing hotspot location with entity-type-aware rendering.
 *
 * GW nodes: clickable 1st-ring elements + non-clickable 2nd-ring context + stream reference line
 * ST nodes: stream reach polyline + non-clickable context elements
 */

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import type { HotspotContext } from '../../api/client';

interface HotspotMapProps {
  context: HotspotContext;
  selectedElementId: number | null;
  onClickElement: (elementId: number) => void;
}

const COLORS = {
  context: '#f0f0f0',
  contextStroke: '#ccc',
  element: '#e3f2fd',
  elementStroke: '#1565c0',
  selected: '#ff9800',
  stream: '#1565c0',
  streamNode: '#42a5f5',
  hotspot: '#d32f2f',
};

export function HotspotMap({ context, selectedElementId, onClickElement }: HotspotMapProps) {
  const {
    surrounding_elements_geometry: elements,
    context_elements_geometry: contextElements,
    stream_reach_coords: streamCoords,
    x, y, entity_type,
  } = context;

  const hasGeometry = elements.length > 0 || contextElements.length > 0 || streamCoords.length > 0;
  if (!hasGeometry || x === null || y === null) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
        No spatial data available for this hotspot.
      </Typography>
    );
  }

  // Compute bounding box — GW: from elements only; ST: from elements + stream reach
  let xMin = x, xMax = x, yMin = y, yMax = y;
  const expandBounds = (px: number, py: number) => {
    if (px < xMin) xMin = px;
    if (px > xMax) xMax = px;
    if (py < yMin) yMin = py;
    if (py > yMax) yMax = py;
  };
  for (const elem of [...elements, ...contextElements]) {
    for (const [vx, vy] of elem.vertices) expandBounds(vx, vy);
  }
  // Only include stream reach in bounds for ST nodes — for GW it's just reference context
  if (entity_type !== 'groundwater') {
    for (const [sx, sy] of streamCoords) expandBounds(sx, sy);
  }

  // Add padding
  const dx = (xMax - xMin) || 1;
  const dy = (yMax - yMin) || 1;
  const pad = Math.max(dx, dy) * 0.12;
  xMin -= pad;
  xMax += pad;
  yMin -= pad;
  yMax += pad;

  const width = 350;
  const height = 350;
  const toSx = (mx: number) => ((mx - xMin) / (xMax - xMin)) * width;
  const toSy = (my: number) => ((yMax - my) / (yMax - yMin)) * height;

  const isGW = entity_type === 'groundwater';

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        {isGW
          ? `Node ${context.entity_id}${context.layer !== null ? ` (L${context.layer})` : ''}`
          : `Stream Node ${context.entity_id}${context.stream_reach_id !== null
              ? ` — Reach ${context.stream_reach_id}${context.stream_reach_name ? ` (${context.stream_reach_name})` : ''}`
              : ''}`}
      </Typography>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ border: '1px solid #e0e0e0', borderRadius: 4, background: '#fafafa' }}
      >
        {/* 2nd ring context elements — always non-clickable */}
        {contextElements.map((elem) => {
          const points = elem.vertices.map(([vx, vy]) => `${toSx(vx)},${toSy(vy)}`).join(' ');
          return (
            <polygon
              key={`ctx-${elem.id}`}
              points={points}
              fill={COLORS.context}
              stroke={COLORS.contextStroke}
              strokeWidth={0.5}
              opacity={0.7}
            />
          );
        })}

        {/* 1st ring elements */}
        {elements.map((elem) => {
          const points = elem.vertices.map(([vx, vy]) => `${toSx(vx)},${toSy(vy)}`).join(' ');
          const isSelected = elem.id === selectedElementId;

          if (isGW) {
            // GW: clickable elements with IDs
            return (
              <polygon
                key={elem.id}
                points={points}
                fill={isSelected ? COLORS.selected : COLORS.element}
                stroke={COLORS.elementStroke}
                strokeWidth={1.5}
                opacity={0.85}
                style={{ cursor: 'pointer' }}
                onClick={() => onClickElement(elem.id)}
              >
                <title>Element {elem.id} — click for budget</title>
              </polygon>
            );
          }
          // ST: non-clickable context elements
          return (
            <polygon
              key={elem.id}
              points={points}
              fill={COLORS.element}
              stroke={COLORS.elementStroke}
              strokeWidth={0.8}
              opacity={0.5}
            />
          );
        })}

        {/* Element ID labels — GW only */}
        {isGW && elements.map((elem) => {
          const cx = elem.vertices.reduce((s, [vx]) => s + vx, 0) / elem.vertices.length;
          const cy = elem.vertices.reduce((s, [, vy]) => s + vy, 0) / elem.vertices.length;
          return (
            <text
              key={`label-${elem.id}`}
              x={toSx(cx)}
              y={toSy(cy)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={10}
              fill="#333"
              style={{ pointerEvents: 'none' }}
            >
              {elem.id}
            </text>
          );
        })}

        {/* Stream reach polyline */}
        {streamCoords.length > 1 && (
          <polyline
            points={streamCoords.map(([sx, sy]) => `${toSx(sx)},${toSy(sy)}`).join(' ')}
            fill="none"
            stroke={COLORS.stream}
            strokeWidth={isGW ? 2 : 3.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={isGW ? 0.5 : 0.9}
          />
        )}

        {/* Stream node dots — ST mode only */}
        {!isGW && streamCoords.map(([sx, sy], i) => (
          <circle
            key={`sn-${i}`}
            cx={toSx(sx)}
            cy={toSy(sy)}
            r={3}
            fill={COLORS.streamNode}
            opacity={0.7}
          />
        ))}

        {/* Hotspot node marker */}
        <circle
          cx={toSx(x)}
          cy={toSy(y)}
          r={7}
          fill={COLORS.hotspot}
          stroke="#fff"
          strokeWidth={2}
        >
          <title>{context.variable}</title>
        </circle>
      </svg>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
        {isGW
          ? 'Click an element to view its GW ZBudget'
          : `Reach ${context.stream_reach_id ?? ''}${context.stream_reach_name ? ` — ${context.stream_reach_name}` : ''}`}
      </Typography>
    </Box>
  );
}
