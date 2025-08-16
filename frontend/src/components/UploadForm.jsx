import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Typography, 
  CircularProgress, 
  Alert, 
  LinearProgress, 
  Paper, 
  Container,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  useTheme,
  Fade
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { firebaseService } from '../services/firebaseService';

// Configure axios defaults
axios.defaults.withCredentials = true;
axios.defaults.baseURL = 'https://dataviz-production.up.railway.app/';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [activeStep, setActiveStep] = useState(0);
  const navigate = useNavigate();
  const theme = useTheme();

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
      setActiveStep(1);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv']
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024 // 500MB limit
  });
  

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    setStatus('Uploading dataset...');
    setActiveStep(2);

    try {
      // Step 1: Upload to Firebase
      setStatus('Storing dataset in Firebase...');
      const datasetId = await firebaseService.storeDataset(file, {
        originalName: file.name,
        size: file.size,
        type: file.type
      }).catch(err => {
        console.error('Firebase error:', err);
        throw new Error(`Firebase error: ${err.message}`);
      });
      setProgress(20);

      // Step 2: Send to backend for processing
      setStatus('Processing dataset...');
      
      // Create FormData and append the file
      const formData = new FormData();
      formData.append('file', file);
      formData.append('datasetId', datasetId);
      
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        withCredentials: true,
        crossDomain: true
      }).catch(err => {
        console.error('Backend error:', err);
        if (err.response) {
          throw new Error(`Backend error: ${err.response.data.error || err.message}`);
        } else if (err.request) {
          throw new Error('Could not connect to the backend server');
        } else {
          throw new Error(`Request error: ${err.message}`);
        }
      });
      setProgress(40);

      if (response.data.status === 'error') {
        throw new Error(response.data.error);
      }

      // Step 3: Store cleaned dataset
      setStatus('Storing cleaned dataset...');
      await firebaseService.storeCleanedDataset(datasetId, response.data.data.cleaning_summary)
        .catch(err => {
          console.error('Firebase error:', err);
          throw new Error(`Error storing cleaned dataset: ${err.message}`);
        });
      setProgress(60);

      // Step 4: Store feature rankings
      setStatus('Storing feature rankings...');
      const featureRankings = response.data.data.feature_rankings || {};
      await firebaseService.storeFeatureRankings(datasetId, {
        rankings: featureRankings,
        timestamp: new Date().toISOString()
      }).catch(err => {
        console.error('Firebase error:', err);
        throw new Error(`Error storing feature rankings: ${err.message}`);
      });
      setProgress(80);

      // Step 5: Store visualizations
      setStatus('Generating visualizations...');
      const visualizations = response.data.data.visualizations || {};
      await firebaseService.storeVisualizations(datasetId, {
        summary: visualizations.summary || {},
        plots: visualizations.plots || {},
        timestamp: new Date().toISOString()
      }).catch(err => {
        console.error('Firebase error:', err);
        throw new Error(`Error storing visualizations: ${err.message}`);
      });
      setProgress(100);
      setActiveStep(3);

      // Wait a moment to show completion before navigating
      setTimeout(() => {
        // Navigate to results page and pass backend data in state
        navigate(`/results/${datasetId}`, { state: { backendData: response.data.data } });
      }, 1500);
    } catch (err) {
      console.error('Processing error:', err);
      setError(err.message || 'Error processing dataset');
      setProgress(0);
      setActiveStep(1); // Go back to file selection step
    } finally {
      setLoading(false);
    }
  };

  const steps = [
    {
      label: 'Select File',
      description: 'Drag and drop or select a CSV file to upload',
    },
    {
      label: 'Review and Submit',
      description: 'Verify your file and submit for processing',
    },
    {
      label: 'Processing',
      description: 'Your file is being analyzed...',
    },
    {
      label: 'Complete',
      description: 'Analysis complete!',
    },
  ];

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(2) + ' MB';
  };

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Typography 
        variant="h3" 
        component="h1" 
        gutterBottom
        sx={{ 
          fontWeight: 700,
          textAlign: 'center',
          mb: 4,
          background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Upload Your Dataset
      </Typography>

      <Paper
        elevation={3}
        sx={{
          p: 4,
          borderRadius: 2,
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <Stepper activeStep={activeStep} orientation="vertical" sx={{ mb: 4 }}>
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel>
                <Typography variant="subtitle1" fontWeight={500}>
                  {step.label}
                </Typography>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {step.description}
                </Typography>
              </StepContent>
            </Step>
          ))}
        </Stepper>

        {activeStep === 0 && (
          <Fade in={activeStep === 0}>
            <Box>
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderRadius: 2,
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  p: 6,
                  textAlign: 'center',
                  cursor: 'pointer',
                  mb: 3,
                  transition: 'all 0.2s',
                  backgroundColor: isDragActive ? 'action.hover' : 'transparent',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'action.hover'
                  }
                }}
              >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive
                    ? 'Drop the CSV file here'
                    : 'Drag and drop a CSV file here'}
                </Typography>
                <Typography color="text.secondary" sx={{ mb: 2 }}>
                  Or click to browse files
                </Typography>
                <Chip 
                  label="CSV and TSV files only (max 500MB)" 
                  size="small" 
                  variant="outlined" 
                  color="primary" 
                />
              </Box>
            </Box>
          </Fade>
        )}

        {activeStep === 1 && file && (
          <Fade in={activeStep === 1}>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Box 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  p: 3,
                  mb: 3,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 2,
                  backgroundColor: 'background.paper',
                }}
              >
                <DescriptionIcon sx={{ fontSize: 40, color: 'primary.light', mr: 2 }} />
                <Box sx={{ textAlign: 'left' }}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {file.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {formatFileSize(file.size)}
                  </Typography>
                </Box>
              </Box>
              
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={loading}
                size="large"
                sx={{ 
                  px: 4, 
                  py: 1.5, 
                  borderRadius: 2,
                  background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.secondary.main} 90%)`,
                }}
              >
                Process Dataset
              </Button>
            </Box>
          </Fade>
        )}

        {activeStep === 2 && loading && (
          <Fade in={activeStep === 2}>
            <Box sx={{ width: '100%', mb: 3, textAlign: 'center' }}>
              <CircularProgress 
                size={80} 
                thickness={4} 
                sx={{ mb: 4 }} 
                variant="determinate" 
                value={progress} 
              />
              <Box sx={{ position: 'relative', display: 'inline-block' }}>
                <Typography 
                  variant="h6" 
                  sx={{ 
                    position: 'absolute', 
                    top: -56, 
                    left: '50%', 
                    transform: 'translateX(-50%)'
                  }}
                >
                  {`${Math.round(progress)}%`}
                </Typography>
              </Box>
              
              <Typography variant="h6" color="primary" gutterBottom>
                {status}
              </Typography>
              
              <LinearProgress 
                variant="determinate" 
                value={progress} 
                sx={{ 
                  height: 10, 
                  borderRadius: 5,
                  mb: 2
                }} 
              />
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Please don't close this window while processing
              </Typography>
            </Box>
          </Fade>
        )}

        {activeStep === 3 && (
          <Fade in={activeStep === 3}>
            <Box sx={{ textAlign: 'center', my: 3 }}>
              <CheckCircleIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom color="success.main" fontWeight={600}>
                Analysis Complete!
              </Typography>
              <Typography variant="body1" gutterBottom>
                Redirecting to results page...
              </Typography>
            </Box>
          </Fade>
        )}

        {error && (
          <Alert 
            severity="error" 
            sx={{ mt: 3 }}
            icon={<ErrorIcon fontSize="inherit" />}
          >
            <Typography variant="subtitle2">{error}</Typography>
          </Alert>
        )}
      </Paper>
    </Container>
  );
};

export default UploadForm;
