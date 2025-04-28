import React from 'react';
import { Card, Typography, Divider, Box, Grid } from '@mui/material';

const VisualizationDisplay = ({ visualizations }) => {
  console.log('VisualizationDisplay received:', visualizations);
  if (!visualizations || !visualizations.data || !visualizations.types || visualizations.types.length === 0) {
    return (
      <Box p={3}>
        <Typography variant="h6" color="error">
          No visualizations available
        </Typography>
      </Box>
    );
  }

  const { data, types } = visualizations;

  return (
    <Box p={3}>
      <Typography variant="h5" gutterBottom>
        Dataset Visualizations
      </Typography>
      
      <Divider sx={{ my: 2 }} />
      
      <Grid container spacing={3}>
        {types.map((type) => (
          <Grid item xs={12} md={6} key={type}>
            <Card sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Typography>
              <Box
                component="img"
                src={`data:image/png;base64,${data[type]}`}
                alt={type}
                sx={{
                  width: '100%',
                  height: 'auto',
                  objectFit: 'contain'
                }}
              />
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default VisualizationDisplay; 