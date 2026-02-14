/**
 * Side drawer showing budget term definitions for the active budget type.
 * Fetches definitions from GET /api/budgets/glossary (cached after first load).
 */

import { useState, useEffect } from 'react';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { fetchBudgetGlossary } from '../../api/client';
import type { BudgetGlossary as GlossaryType } from '../../api/client';

const BUDGET_LABELS: Record<string, string> = {
  gw: 'Groundwater',
  stream: 'Stream',
  lwu: 'Land & Water Use',
  rootzone: 'Root Zone',
  unsaturated: 'Unsaturated Zone',
  diversion: 'Diversion',
  lake: 'Lake',
};

interface BudgetGlossaryProps {
  open: boolean;
  onClose: () => void;
  budgetType: string;
}

let cachedGlossary: GlossaryType | null = null;

export function BudgetGlossary({ open, onClose, budgetType }: BudgetGlossaryProps) {
  const [glossary, setGlossary] = useState<GlossaryType | null>(cachedGlossary);

  useEffect(() => {
    if (!open || cachedGlossary) return;
    fetchBudgetGlossary()
      .then((data) => {
        cachedGlossary = data;
        setGlossary(data);
      })
      .catch(console.warn);
  }, [open]);

  // Detect category from budget type string
  const detectCategory = (bt: string): string => {
    const l = bt.toLowerCase();
    if (l.startsWith('gw') || l.includes('groundwater')) return 'gw';
    if (l === 'lwu' || l.includes('land')) return 'lwu';
    if (l.includes('root')) return 'rootzone';
    if (l.includes('unsat')) return 'unsaturated';
    if (l.includes('stream')) return 'stream';
    if (l.includes('diver')) return 'diversion';
    if (l.includes('lake')) return 'lake';
    return l;
  };

  const category = detectCategory(budgetType);
  const definitions = glossary?.[category] ?? {};
  const entries = Object.entries(definitions);
  const typeLabel = BUDGET_LABELS[category] || budgetType;

  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Box sx={{ width: 360, p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">{typeLabel} Glossary</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
        <Divider sx={{ mb: 1 }} />
        {entries.length === 0 ? (
          <Typography color="text.secondary" sx={{ mt: 2 }}>
            No glossary entries available for this budget type.
          </Typography>
        ) : (
          <List dense>
            {entries.map(([term, definition]) => (
              <ListItem key={term} sx={{ alignItems: 'flex-start', py: 0.5 }}>
                <ListItemText
                  primary={term}
                  secondary={definition}
                  primaryTypographyProps={{ variant: 'subtitle2', fontWeight: 600 }}
                  secondaryTypographyProps={{ variant: 'body2' }}
                />
              </ListItem>
            ))}
          </List>
        )}
      </Box>
    </Drawer>
  );
}
