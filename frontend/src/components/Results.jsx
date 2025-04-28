import React, { useState, useEffect } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import { 
  Box, 
  Typography, 
  CircularProgress, 
  Alert, 
  Paper, 
  Tabs, 
  Tab, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Container,
  Chip,
  Card,
  CardContent,
  Divider,
  IconButton,
  Tooltip,
  useTheme,
  Fade,
  Zoom,
  Button
} from '@mui/material';
import { firebaseService } from '../services/firebaseService';
import Plot from 'react-plotly.js';
import StorageIcon from '@mui/icons-material/Storage';
import InsightsIcon from '@mui/icons-material/Insights';
import TimelineIcon from '@mui/icons-material/Timeline';
import VisibilityIcon from '@mui/icons-material/Visibility';
import InfoIcon from '@mui/icons-material/Info';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import FilterListIcon from '@mui/icons-material/FilterList';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import VisualizationDisplay from './VisualizationDisplay';

const Results = () => {
  const { datasetId } = useParams();
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [data, setData] = useState({
    cleaningData: null,
    featureRankings: null,
    visualizations: null,
    datasetInfo: null
  });
  const theme = useTheme();

  useEffect(() => {
    // If backendData is passed via navigation, use it directly
    if (location.state && location.state.backendData) {
      const backendData = location.state.backendData;
      setData({
        cleaningData: backendData.cleaning_summary || {},
        featureRankings: backendData.feature_rankings || {},
        visualizations: backendData.visualizations
          ? {
              data: backendData.visualizations.data || backendData.visualizations,
              types: backendData.visualizations.types || Object.keys(backendData.visualizations.data || backendData.visualizations)
            }
          : null,
        datasetInfo: backendData.dataset_info || {}
      });
      setLoading(false);
      return;
    }
    // Otherwise, fetch from Firebase as before
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Load all data in parallel
        const [cleaningData, featureRankings, visualizations] = await Promise.all([
          firebaseService.getCleanedDataset(datasetId),
          firebaseService.getFeatureRankings(datasetId),
          firebaseService.getVisualizations(datasetId)
        ]);

        console.log('Loaded data:', { cleaningData, featureRankings, visualizations });

        setData({
          cleaningData,
          featureRankings: featureRankings?.rankings || {},
          visualizations
        });
      } catch (err) {
        console.error('Error loading results:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [datasetId, location.state]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress size={60} thickness={4} />
        <Typography variant="h6" color="text.secondary" sx={{ mt: 3 }}>
          Loading results...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md" sx={{ py: 8 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2, textAlign: 'center' }}>
          <ErrorOutlineIcon color="error" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h5" color="error" gutterBottom>
            Error Loading Results
          </Typography>
          <Alert severity="error" sx={{ mb: 3, justifyContent: 'center' }}>
            {error}
          </Alert>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => window.location.reload()}
          >
            Retry
          </Button>
        </Paper>
      </Container>
    );
  }

  const renderDatasetDetails = () => {
    const { null_values, data_types } = data.cleaningData || {};
    const { total_rows, total_columns } = data.datasetInfo || {};
    
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" gutterBottom fontWeight={600}>
            Dataset Summary
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6} lg={3}>
              <Card 
                elevation={2} 
                sx={{ 
                  borderRadius: 2,
                  height: '100%',
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 }
                }}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <StorageIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight="bold" color="text.primary">
                    {total_rows || 0}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    Total Rows
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6} lg={3}>
              <Card 
                elevation={2} 
                sx={{ 
                  borderRadius: 2,
                  height: '100%',
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 }
                }}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <DataUsageIcon sx={{ fontSize: 48, color: 'info.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight="bold" color="text.primary">
                    {total_columns || 0}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    Total Columns
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6} lg={3}>
              <Card 
                elevation={2} 
                sx={{ 
                  borderRadius: 2,
                  height: '100%',
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 }
                }}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <ErrorOutlineIcon sx={{ fontSize: 48, color: 'warning.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight="bold" color="text.primary">
                    {null_values ? Object.values(null_values).reduce((a, b) => a + b, 0) : 0}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    Total Missing Values
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6} lg={3}>
              <Card 
                elevation={2} 
                sx={{ 
                  borderRadius: 2,
                  height: '100%',
                  transition: 'transform 0.2s',
                  '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 }
                }}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <VisibilityIcon sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
                  <Typography variant="h4" fontWeight="bold" color="text.primary">
                    {data_types ? Object.values(data_types).filter(t => t === 'numeric').length : 0}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    Numeric Features
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
        
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={2} 
              sx={{ 
                p: 3, 
                borderRadius: 2,
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: theme.palette.primary.main
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <InsightsIcon sx={{ color: 'primary.main', mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                  Data Types
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Tooltip title="Column data types determined during preprocessing">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              
              <TableContainer sx={{ maxHeight: 350 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>Column</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data_types && Object.entries(data_types).map(([column, type]) => (
                      <TableRow key={column} hover>
                        <TableCell>{column}</TableCell>
                        <TableCell>
                          <Chip 
                            size="small" 
                            label={type} 
                            color={
                              type === 'numeric' ? 'primary' : 
                              type === 'categorical' ? 'secondary' : 
                              type === 'datetime' ? 'info' : 'default'
                            }
                            variant="outlined"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={2} 
              sx={{ 
                p: 3, 
                borderRadius: 2,
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: theme.palette.warning.main
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <ErrorOutlineIcon sx={{ color: 'warning.main', mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                  Missing Values
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Tooltip title="Columns with missing values that were handled during preprocessing">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              
              <TableContainer sx={{ maxHeight: 350 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>Column</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Missing Values</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>% of Total</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {null_values && Object.entries(null_values)
                      .sort((a, b) => b[1] - a[1]) // Sort by count descending
                      .map(([column, count]) => (
                        <TableRow key={column} hover>
                          <TableCell>{column}</TableCell>
                          <TableCell>{count}</TableCell>
                          <TableCell>
                            {total_rows ? (count / total_rows * 100).toFixed(2) + '%' : 'N/A'}
                          </TableCell>
                        </TableRow>
                    ))}
                    {(!null_values || Object.keys(null_values).length === 0) && (
                      <TableRow>
                        <TableCell colSpan={3} align="center">
                          <Typography variant="body2" color="text.secondary">
                            No missing values detected
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper 
              elevation={2} 
              sx={{ 
                p: 3, 
                borderRadius: 2,
                position: 'relative',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: theme.palette.success.main
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TimelineIcon sx={{ color: 'success.main', mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                  Feature Importance Rankings
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Tooltip title="Features ranked by their importance in explaining dataset variance">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>Rank</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Feature</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Importance Score</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Relative Importance</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.featureRankings && Object.entries(data.featureRankings)
                      .sort((a, b) => b[1] - a[1]) // Sort by importance descending
                      .map(([feature, score], index) => {
                        // Calculate relative importance (proportion of max score)
                        const maxScore = Math.max(...Object.values(data.featureRankings));
                        const relativeImportance = score / maxScore;
                        
                        return (
                          <TableRow key={feature} hover>
                            <TableCell>{index + 1}</TableCell>
                            <TableCell sx={{ fontWeight: index < 3 ? 600 : 400 }}>
                              {feature}
                              {index < 3 && (
                                <Chip 
                                  size="small" 
                                  label="Top Feature" 
                                  color="success" 
                                  variant="outlined" 
                                  sx={{ ml: 1 }}
                                />
                              )}
                            </TableCell>
                            <TableCell>{score.toFixed(4)}</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Box
                                  sx={{
                                    width: '100%',
                                    height: 8,
                                    bgcolor: 'grey.200',
                                    borderRadius: 4,
                                    mr: 1
                                  }}
                                >
                                  <Box
                                    sx={{
                                      width: `${relativeImportance * 100}%`,
                                      height: '100%',
                                      bgcolor: index < 3 ? 'success.main' : 'primary.main',
                                      borderRadius: 4
                                    }}
                                  />
                                </Box>
                                <Typography variant="body2">
                                  {(relativeImportance * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    );
  };

  return (
    <Box sx={{ bgcolor: 'background.default', minHeight: '100vh' }}>
      <Container maxWidth="xl" sx={{ pt: 4, pb: 8 }}>
        <Paper 
          elevation={0} 
          sx={{ 
            p: 2, 
            mb: 4, 
            borderRadius: 2, 
            bgcolor: theme.palette.background.paper,
            backgroundImage: `linear-gradient(to right, ${theme.palette.primary.light}20, ${theme.palette.secondary.light}20)`,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
            <Typography 
              variant="h4" 
              component="h1" 
              sx={{ 
                fontWeight: 700, 
                background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mr: 3
              }}
            >
              Analysis Results
            </Typography>
            
            <Chip 
              label={`Dataset ID: ${datasetId.substring(0, 8)}...`}
              variant="outlined" 
              color="primary"
              size="small"
            />
            
            <Box sx={{ flexGrow: 1 }} />
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                size="small"
                sx={{ borderRadius: 2 }}
              >
                Export Results
              </Button>
              
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                size="small"
                sx={{ 
                  borderRadius: 2,
                  background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`
                }}
              >
                New Dataset
              </Button>
            </Box>
          </Box>
        </Paper>
        
        <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'hidden' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            variant="fullWidth"
            textColor="primary"
            indicatorColor="primary"
            sx={{ 
              borderBottom: 1, 
              borderColor: 'divider',
              '& .MuiTab-root': {
                py: 2,
                fontSize: 16,
                fontWeight: 500
              }
            }}
          >
            <Tab 
              label="Dataset Details" 
              icon={<StorageIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="Visualizations" 
              icon={<TimelineIcon />} 
              iconPosition="start"
            />
          </Tabs>

          {activeTab === 0 && <Fade in={activeTab === 0}>{renderDatasetDetails()}</Fade>}
          {activeTab === 1 && (
            <Fade in={activeTab === 1}>
              <div>
                <VisualizationDisplay visualizations={data.visualizations} />
              </div>
            </Fade>
          )}
        </Paper>
      </Container>
    </Box>
  );
};

export default Results;