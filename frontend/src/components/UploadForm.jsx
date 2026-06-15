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
    StepLabel
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { firebaseService } from '../services/firebaseService';

axios.defaults.withCredentials = true;
axios.defaults.baseURL = '';

const UploadForm = () => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('');
    const [activeStep, setActiveStep] = useState(0);
    const navigate = useNavigate();

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0]);
            setError(null);
            setActiveStep(1);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'], 'text/tab-separated-values': ['.tsv'] },
        maxFiles: 1,
        maxSize: 500 * 1024 * 1024
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
            setStatus('Storing dataset in Firebase...');
            const datasetId = await firebaseService.storeDataset(file, {
                originalName: file.name,
                size: file.size,
                type: file.type
            });
            setProgress(20);

            setStatus('Processing dataset...');
            const formData = new FormData();
            formData.append('file', file);
            formData.append('datasetId', datasetId);

            const response = await axios.post('/api/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setProgress(40);

            if (response.data.status === 'error') {
                throw new Error(response.data.error);
            }

            setStatus('Storing cleaned dataset...');
            await firebaseService.storeCleanedDataset(datasetId, response.data.data.cleaning_summary);
            setProgress(60);

            setStatus('Storing feature rankings...');
            await firebaseService.storeFeatureRankings(datasetId, {
                rankings: response.data.data.feature_rankings || {},
                timestamp: new Date().toISOString()
            });
            setProgress(80);

            setStatus('Generating visualizations...');
            const vizData = response.data.data.visualizations || {};
            await firebaseService.storeVisualizations(datasetId, {
                summary: { count: vizData.count || 0, types: vizData.types || [] },
                plots: {},
                timestamp: new Date().toISOString()
            });
            setProgress(100);
            setActiveStep(3);

            localStorage.setItem('current_viz_doc', datasetId);

            setTimeout(() => {
                navigate(`/results/${datasetId}`, { state: { backendData: response.data.data } });
            }, 1500);
        } catch (err) {
            console.error('Processing error:', err);
            setError(err.message || 'Error processing dataset');
            setProgress(0);
            setActiveStep(1);
        } finally {
            setLoading(false);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(2) + ' MB';
    };

    const steps = ['Select File', 'Review', 'Processing', 'Complete'];

    return (
        <Box sx={{ minHeight: '100vh', bgcolor: '#0a0a0f', pt: 12, pb: 6 }}>
            <Container maxWidth="md">
                <Box sx={{ textAlign: 'center', mb: 6 }}>
                    <Typography variant="h3" sx={{
                        fontWeight: 800,
                        mb: 2,
                        background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                    }}>
                        Upload Dataset
                    </Typography>
                    <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.6)' }}>
                        Upload your CSV file and let AI analyze your data
                    </Typography>
                </Box>

                <Paper sx={{
                    p: 4,
                    borderRadius: 4,
                    bgcolor: 'rgba(26, 26, 36, 0.7)',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.1)'
                }}>
                    {/* Progress Steps */}
                    <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
                        {steps.map((label, index) => (
                            <Step key={label}>
                                <StepLabel
                                    StepIconProps={{
                                        sx: {
                                            color: index <= activeStep ? '#8b5cf6' : 'rgba(255,255,255,0.3)',
                                            '&.Mui-active': { color: '#8b5cf6' },
                                            '&.Mui-completed': { color: '#8b5cf6' }
                                        }
                                    }}
                                >
                                    <Typography sx={{ color: index <= activeStep ? 'white' : 'rgba(255,255,255,0.5)' }}>
                                        {label}
                                    </Typography>
                                </StepLabel>
                            </Step>
                        ))}
                    </Stepper>

                    {/* Step 0: File Selection */}
                    {activeStep === 0 && (
                        <Box
                            {...getRootProps()}
                            sx={{
                                border: '2px dashed',
                                borderColor: isDragActive ? '#8b5cf6' : 'rgba(255,255,255,0.2)',
                                borderRadius: 3,
                                p: 6,
                                textAlign: 'center',
                                cursor: 'pointer',
                                transition: 'all 0.3s',
                                bgcolor: isDragActive ? 'rgba(139, 92, 246, 0.1)' : 'transparent',
                                '&:hover': {
                                    borderColor: '#8b5cf6',
                                    bgcolor: 'rgba(139, 92, 246, 0.05)'
                                }
                            }}
                        >
                            <input {...getInputProps()} />
                            <CloudUploadIcon sx={{ fontSize: 80, color: isDragActive ? '#8b5cf6' : 'rgba(255,255,255,0.4)', mb: 2 }} />
                            <Typography variant="h5" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
                                {isDragActive ? 'Drop it here!' : 'Drag & drop your file'}
                            </Typography>
                            <Typography sx={{ color: 'rgba(255,255,255,0.5)', mb: 2 }}>
                                or click to browse
                            </Typography>
                            <Box sx={{
                                display: 'inline-flex',
                                px: 2,
                                py: 0.5,
                                borderRadius: 2,
                                bgcolor: 'rgba(139, 92, 246, 0.1)',
                                border: '1px solid rgba(139, 92, 246, 0.3)'
                            }}>
                                <Typography variant="body2" sx={{ color: '#c4b5fd' }}>
                                    CSV & TSV files • Max 500MB
                                </Typography>
                            </Box>
                        </Box>
                    )}

                    {/* Step 1: Review File */}
                    {activeStep === 1 && file && (
                        <Box sx={{ textAlign: 'center' }}>
                            <Paper sx={{
                                p: 3,
                                mb: 4,
                                borderRadius: 3,
                                bgcolor: 'rgba(139, 92, 246, 0.1)',
                                border: '1px solid rgba(139, 92, 246, 0.3)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: 2
                            }}>
                                <Box sx={{
                                    width: 64,
                                    height: 64,
                                    borderRadius: 2,
                                    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}>
                                    <DescriptionIcon sx={{ fontSize: 32, color: 'white' }} />
                                </Box>
                                <Box sx={{ textAlign: 'left' }}>
                                    <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                                        {file.name}
                                    </Typography>
                                    <Typography sx={{ color: 'rgba(255,255,255,0.5)' }}>
                                        {formatFileSize(file.size)}
                                    </Typography>
                                </Box>
                            </Paper>
                            <Button
                                variant="contained"
                                size="large"
                                onClick={handleSubmit}
                                disabled={loading}
                                startIcon={<CloudUploadIcon />}
                                sx={{
                                    px: 5,
                                    py: 1.5,
                                    borderRadius: 2,
                                    fontWeight: 600,
                                    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                                    boxShadow: '0 0 20px rgba(139, 92, 246, 0.4)',
                                    '&:hover': {
                                        boxShadow: '0 0 30px rgba(139, 92, 246, 0.6)'
                                    }
                                }}
                            >
                                Process Dataset
                            </Button>
                        </Box>
                    )}

                    {/* Step 2: Processing */}
                    {activeStep === 2 && loading && (
                        <Box sx={{ textAlign: 'center' }}>
                            <Box sx={{ position: 'relative', display: 'inline-block', mb: 3 }}>
                                <CircularProgress
                                    size={100}
                                    variant="determinate"
                                    value={progress}
                                    sx={{ color: '#8b5cf6' }}
                                />
                                <Box sx={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: 'translate(-50%, -50%)'
                                }}>
                                    <Typography variant="h5" sx={{ color: 'white', fontWeight: 700 }}>
                                        {Math.round(progress)}%
                                    </Typography>
                                </Box>
                            </Box>
                            <Typography variant="h6" sx={{
                                mb: 2,
                                background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent'
                            }}>
                                {status}
                            </Typography>
                            <LinearProgress
                                variant="determinate"
                                value={progress}
                                sx={{
                                    height: 8,
                                    borderRadius: 4,
                                    bgcolor: 'rgba(255,255,255,0.1)',
                                    '& .MuiLinearProgress-bar': {
                                        background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                                        borderRadius: 4
                                    }
                                }}
                            />
                            <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)', mt: 2 }}>
                                Please don't close this window
                            </Typography>
                        </Box>
                    )}

                    {/* Step 3: Complete */}
                    {activeStep === 3 && (
                        <Box sx={{ textAlign: 'center', py: 4 }}>
                            <Box sx={{
                                width: 80,
                                height: 80,
                                borderRadius: '50%',
                                bgcolor: 'rgba(34, 197, 94, 0.2)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                mx: 'auto',
                                mb: 3
                            }}>
                                <CheckCircleIcon sx={{ fontSize: 48, color: '#22c55e' }} />
                            </Box>
                            <Typography variant="h4" sx={{ color: '#22c55e', fontWeight: 700, mb: 1 }}>
                                Analysis Complete!
                            </Typography>
                            <Typography sx={{ color: 'rgba(255,255,255,0.5)' }}>
                                Redirecting to results...
                            </Typography>
                        </Box>
                    )}

                    {/* Error */}
                    {error && (
                        <Alert
                            severity="error"
                            icon={<ErrorIcon />}
                            sx={{
                                mt: 3,
                                bgcolor: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                color: '#ef4444'
                            }}
                        >
                            {error}
                        </Alert>
                    )}
                </Paper>
            </Container>
        </Box>
    );
};

export default UploadForm;
