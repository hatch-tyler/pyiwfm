/**
 * ZBudget glossary drawer — term definitions from IWFM Ch. 6.
 */

import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import CloseIcon from '@mui/icons-material/Close';
import { fetchZBudgetGlossary } from '../../api/client';

type GlossaryType = Record<string, Record<string, string>>;

const ZBUDGET_LABELS: Record<string, string> = {
  gw: 'Groundwater',
  rootzone: 'Root Zone',
  lwu: 'Land & Water Use',
};

let cachedGlossary: GlossaryType | null = null;

interface ZBudgetGlossaryProps {
  open: boolean;
  onClose: () => void;
  budgetType: string;
}

export function ZBudgetGlossary({ open, onClose, budgetType }: ZBudgetGlossaryProps) {
  const [glossary, setGlossary] = useState<GlossaryType | null>(cachedGlossary);

  useEffect(() => {
    if (!open || cachedGlossary) return;
    fetchZBudgetGlossary()
      .then((data) => {
        cachedGlossary = data;
        setGlossary(data);
      })
      .catch(console.warn);
  }, [open]);

  const detectCategory = (bt: string): string => {
    const l = bt.toLowerCase();
    if (l.startsWith('gw') || l.includes('groundwater')) return 'gw';
    if (l === 'lwu' || l.includes('land')) return 'lwu';
    if (l.includes('root')) return 'rootzone';
    return l;
  };

  const category = detectCategory(budgetType);
  const definitions = glossary?.[category] ?? {};
  const entries = Object.entries(definitions);

  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Box sx={{ width: 360, p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">{ZBUDGET_LABELS[category] || budgetType} ZBudget Glossary</Typography>
          <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
        </Box>
        <Divider sx={{ mb: 1 }} />
        {entries.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No glossary entries available for this type.
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
