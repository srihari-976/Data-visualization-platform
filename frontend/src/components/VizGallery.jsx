import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  CircularProgress, 
  Alert, 
  Card, 
  CardContent, 
  CardMedia, 
  Skeleton, 
  Fade, 
  Chip, 
  Button, 
  Divider,
  IconButton,
  Tooltip,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { 
  DownloadOutlined, 
  FullscreenOutlined, 
  InfoOutlined,
  ArrowBackIosNew,
  ArrowForwardIos
} from '@mui/icons-material';
import { db } from '../firebase';
import { doc, getDoc } from '@firebase/firestore';

const VizGallery = () => {
  const [visualizations, setVisualizations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [metaData, setMetaData] = useState(null);
  
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const itemsPerPage = isMobile ? 1 : 4;

  useEffect(() => {
    const loadVisualizations = async () => {
      try {
        // Show loading state first
        setLoading(true);
        
        // Get the document ID from localStorage
        const docId = localStorage.getItem('current_viz_doc');
        if (!docId) {
          setError('No visualization data found. Please upload a dataset first.');
          setLoading(false);
          return;
        }

        // Fetch the data from Firestore
        const docRef = doc(db, 'visualizations', docId);
        const docSnap = await getDoc(docRef);

        if (!docSnap.exists()) {
          setError('Visualization data not found. Please upload a dataset again.');
          setLoading(false);
          return;
        }

        const data = docSnap.data();
        
        if (data.status === 'error') {
          setError(data.error || 'Error processing the dataset');
          setLoading(false);
          return;
        }

        if (!data.visualizations || Object.keys(data.visualizations).length === 0) {
          setError('No visualizations available in the dataset');
          setLoading(false);
          return;
        }

        // Extract metadata if available
        if (data.metadata) {
          setMetaData(data.metadata);
        }

        setVisualizations(data.visualizations);
        setLoading(false);
      } catch (err) {
        console.error('Error loading visualizations:', err);
        setError('Error loading visualizations. Please try uploading the dataset again.');
        setLoading(false);
      }
    };

    loadVisualizations();
  }, []);

  const handleFullscreen = (title, imageData) => {
    setFullscreenImage({ title, imageData });
  };

  const closeFullscreen = () => {
    setFullscreenImage(null);
  };

  const handleDownload = (title, imageData) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imageData}`;
    link.download = `${title.replace(/ /g, '_')}_visualization.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatTitle = (title) => {
    return title
      .replace(/_/g, ' ')
      .replace(/^dist |^bar /, '')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const handleNextPage = () => {
    if (visualizations && Object.keys(visualizations).length > currentPage * itemsPerPage) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  if (loading) {
    return (
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Generating Visualizations
        </Typography>
        <Box display="flex" justifyContent="center" alignItems="center" flexDirection="column" minHeight="300px">
          <CircularProgress size={60} thickness={4} />
          <Typography variant="body1" color="text.secondary" mt={2}>
            Processing your dataset and creating insights...
          </Typography>
        </Box>
        <Grid container spacing={3} mt={2}>
          {[1, 2, 3, 4].map((item) => (
            <Grid item xs={12} md={6} key={item}>
              <Card>
                <Skeleton variant="rectangular" height={250} animation="wave" />
                <CardContent>
                  <Skeleton variant="text" width="60%" height={30} />
                  <Skeleton variant="text" width="40%" height={20} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert 
          severity="error" 
          sx={{ 
            py: 2, 
            display: 'flex', 
            alignItems: 'center' 
          }}
        >
          <Box sx={{ ml: 1 }}>
            <Typography variant="h6" gutterBottom>
              Unable to Generate Visualizations
            </Typography>
            <Typography variant="body1">
              {error}
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              sx={{ mt: 2 }} 
              href="/upload"
            >
              Try Again
            </Button>
          </Box>
        </Alert>
      </Box>
    );
  }

  if (!visualizations || Object.keys(visualizations).length === 0) {
    return (
      <Box p={3}>
        <Alert severity="info" sx={{ py: 2 }}>
          <Typography variant="h6" gutterBottom>
            No Visualizations Available
          </Typography>
          <Typography variant="body1">
            Please upload a dataset with numeric columns for analysis.
          </Typography>
          <Button 
            variant="contained" 
            color="primary" 
            sx={{ mt: 2 }} 
            href="/upload"
          >
            Upload Dataset
          </Button>
        </Alert>
      </Box>
    );
  }

  // Paginate the visualizations
  const vizEntries = Object.entries(visualizations);
  const totalPages = Math.ceil(vizEntries.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentVizEntries = vizEntries.slice(startIndex, startIndex + itemsPerPage);

  return (
    <Fade in={!loading}>
      <Box sx={{ py: 3 }}>
        {/* Header with dataset info */}
        <Box mb={4}>
          <Typography variant="h4" gutterBottom fontWeight="500">
            Dataset Insights
          </Typography>
          
          {metaData && (
            <Box sx={{ mb: 2 }}>
              <Chip 
                label={`${metaData.rows || '?'} rows`} 
                color="primary" 
                size="small" 
                sx={{ mr: 1, mb: 1 }} 
              />
              <Chip 
                label={`${metaData.columns || '?'} columns`} 
                color="secondary" 
                size="small" 
                sx={{ mr: 1, mb: 1 }} 
              />
              {metaData.fileName && (
                <Chip 
                  label={metaData.fileName} 
                  variant="outlined" 
                  size="small"
                  sx={{ mb: 1 }} 
                />
              )}
            </Box>
          )}
          
          <Divider sx={{ mb: 3 }} />
        </Box>

        {/* Visualization Grid */}
        <Grid container spacing={3}>
          {currentVizEntries.map(([title, imageData]) => (
            <Grid item xs={12} md={6} key={title}>
              <Card 
                elevation={2} 
                sx={{ 
                  height: '100%',
                  transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 6,
                  }
                }}
              >
                <CardMedia
                  component="div"
                  sx={{
                    position: 'relative',
                    height: 280,
                    overflow: 'hidden',
                    backgroundColor: '#f5f5f5',
                  }}
                >
                  <Box
                    component="img"
                    src={`data:image/png;base64,${imageData}`}
                    alt={formatTitle(title)}
                    sx={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                    }}
                    loading="lazy"
                  />
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      display: 'flex',
                      gap: 1,
                      opacity: 0.8,
                      '&:hover': {
                        opacity: 1,
                      },
                    }}
                  >
                    <Tooltip title="View Fullscreen">
                      <IconButton
                        size="small"
                        onClick={() => handleFullscreen(title, imageData)}
                        sx={{ 
                          bgcolor: 'rgba(255, 255, 255, 0.9)',
                          '&:hover': { bgcolor: 'white' }
                        }}
                      >
                        <FullscreenOutlined fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download Image">
                      <IconButton
                        size="small"
                        onClick={() => handleDownload(title, imageData)}
                        sx={{ 
                          bgcolor: 'rgba(255, 255, 255, 0.9)',
                          '&:hover': { bgcolor: 'white' }
                        }}
                      >
                        <DownloadOutlined fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </CardMedia>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {formatTitle(title)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {title.includes('dist') ? 'Distribution Analysis' : 
                     title.includes('bar') ? 'Categorical Analysis' : 
                     title.includes('scatter') ? 'Correlation Analysis' : 
                     'Data Visualization'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              mt: 4 
            }}
          >
            <Button
              startIcon={<ArrowBackIosNew />}
              onClick={handlePrevPage}
              disabled={currentPage === 1}
              sx={{ mr: 2 }}
            >
              Previous
            </Button>
            <Typography variant="body1">
              {currentPage} of {totalPages}
            </Typography>
            <Button
              endIcon={<ArrowForwardIos />}
              onClick={handleNextPage}
              disabled={currentPage === totalPages}
              sx={{ ml: 2 }}
            >
              Next
            </Button>
          </Box>
        )}

        {/* Fullscreen Modal */}
        {fullscreenImage && (
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              bgcolor: 'rgba(0, 0, 0, 0.85)',
              zIndex: 9999,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onClick={closeFullscreen}
          >
            <Box sx={{ position: 'relative', width: '90%', height: '90%', maxWidth: 1200 }}>
              <Box
                component="img"
                src={`data:image/png;base64,${fullscreenImage.imageData}`}
                alt={formatTitle(fullscreenImage.title)}
                sx={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                }}
              />
              <Typography 
                variant="h6" 
                sx={{ 
                  position: 'absolute', 
                  bottom: -40, 
                  left: 0, 
                  color: 'white',
                  textAlign: 'center',
                  width: '100%'
                }}
              >
                {formatTitle(fullscreenImage.title)}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Fade>
  );
};

export default VizGallery;