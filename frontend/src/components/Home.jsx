import React from 'react';
import { Box, Typography, Button, Paper, Grid, Container, useTheme } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BarChartIcon from '@mui/icons-material/BarChart';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import TimelineIcon from '@mui/icons-material/Timeline';
import InsightsIcon from '@mui/icons-material/Insights';

const FeatureItem = ({ icon, title, description }) => {
  const Icon = icon;
  return (
    <Box sx={{ display: 'flex', mb: 2, alignItems: 'flex-start' }}>
      <Box sx={{ mr: 2, mt: 0.5 }}>
        <Icon color="primary" />
      </Box>
      <Box>
        <Typography variant="subtitle1" fontWeight="500">
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </Box>
    </Box>
  );
};

const Home = () => {
  const navigate = useNavigate();
  const theme = useTheme();

  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Box sx={{ textAlign: 'center', mb: 8 }}>
        <Typography 
          variant="h2" 
          component="h1" 
          gutterBottom
          sx={{ 
            fontWeight: 700,
            background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2
          }}
        >
          Welcome to DataViz
        </Typography>
        <Typography 
          variant="h5" 
          color="text.secondary" 
          sx={{ maxWidth: 700, mx: 'auto', mb: 4 }}
        >
          Transform your raw data into meaningful insights with powerful visualizations
        </Typography>
      </Box>

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper
            elevation={4}
            sx={{
              p: 4,
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              borderRadius: 2,
              position: 'relative',
              overflow: 'hidden',
              transition: 'transform 0.3s, box-shadow 0.3s',
              cursor: 'pointer',
              '&:hover': {
                transform: 'translateY(-8px)',
                boxShadow: 8
              },
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '6px',
                background: `linear-gradient(to right, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
              }
            }}
            onClick={() => navigate('/upload')}
          >
            <Box 
              sx={{ 
                p: 2, 
                borderRadius: '50%', 
                bgcolor: 'primary.light', 
                color: 'primary.contrastText',
                mb: 3
              }}
            >
              <CloudUploadIcon sx={{ fontSize: 48 }} />
            </Box>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              Upload Dataset
            </Typography>
            <Typography color="text.secondary" align="center" sx={{ mb: 4 }}>
              Upload your CSV file to get started with automated data analysis and visualization
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<CloudUploadIcon />}
              sx={{ 
                px: 4, 
                py: 1.5, 
                borderRadius: 2,
                background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                '&:hover': {
                  boxShadow: '0 4px 8px 3px rgba(33, 203, 243, .4)'
                }
              }}
            >
              Upload Now
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper
            elevation={4}
            sx={{
              p: 4,
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 2,
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '6px',
                background: `linear-gradient(to right, ${theme.palette.secondary.main}, ${theme.palette.primary.main})`
              }
            }}
          >
            <Box 
              sx={{ 
                p: 2, 
                borderRadius: '50%', 
                bgcolor: 'secondary.light', 
                color: 'secondary.contrastText',
                alignSelf: 'center',
                mb: 3
              }}
            >
              <BarChartIcon sx={{ fontSize: 48 }} />
            </Box>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 3 }}>
              Advanced Features
            </Typography>
            
            <FeatureItem 
              icon={InsightsIcon} 
              title="Automated Data Cleaning" 
              description="Smart preprocessing with missing value handling and outlier detection"
            />
            
            <FeatureItem 
              icon={AnalyticsIcon} 
              title="Feature Importance Analysis" 
              description="Identify key variables driving your dataset patterns"
            />
            
            <FeatureItem 
              icon={TimelineIcon} 
              title="Interactive Visualizations" 
              description="Explore your data with 2D and 3D interactive charts"
            />
            
            <FeatureItem 
              icon={BarChartIcon} 
              title="Statistical Insights" 
              description="Comprehensive statistics to understand your data distribution"
            />
            
            <FeatureItem 
              icon={InsightsIcon} 
              title="Advanced Metrics" 
              description="Cross-entropy and correlation analysis for deeper understanding"
            />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Home;