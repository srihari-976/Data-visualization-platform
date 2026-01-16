import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Box,
    Typography,
    Button,
    Paper,
    Grid,
    Container,
    useTheme
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BarChartIcon from '@mui/icons-material/BarChart';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import PsychologyIcon from '@mui/icons-material/Psychology';
import DownloadIcon from '@mui/icons-material/Download';
import SpeedIcon from '@mui/icons-material/Speed';

const Home = () => {
    const navigate = useNavigate();
    const theme = useTheme();

    const steps = [
        {
            icon: <CloudUploadIcon sx={{ fontSize: 40 }} />,
            title: 'Upload Data',
            description: 'Drop your CSV file. We automatically analyze and prepare your dataset.',
            gradient: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)'
        },
        {
            icon: <PsychologyIcon sx={{ fontSize: 40 }} />,
            title: 'Ask Questions',
            description: 'Use natural language to ask about your data. Our AI understands your intent.',
            gradient: 'linear-gradient(135deg, #ec4899 0%, #f97316 100%)'
        },
        {
            icon: <AutoGraphIcon sx={{ fontSize: 40 }} />,
            title: 'Get Insights',
            description: 'Receive intelligent visualizations instantly. Download and share your insights.',
            gradient: 'linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%)'
        }
    ];

    const features = [
        {
            icon: <SpeedIcon />,
            title: 'Lightning Fast',
            description: 'Get visualizations in seconds with optimized processing.',
            gradient: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)',
            color: '#f59e0b'
        },
        {
            icon: <PsychologyIcon />,
            title: 'AI-Powered',
            description: 'Llama 3.1 3B understands your queries and selects visualizations.',
            gradient: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
            color: '#8b5cf6'
        },
        {
            icon: <AutoGraphIcon />,
            title: 'Smart Analysis',
            description: 'Automatic data cleaning, feature analysis, and statistics.',
            gradient: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
            color: '#10b981'
        },
        {
            icon: <DownloadIcon />,
            title: 'Export Anywhere',
            description: 'Download high-resolution visualizations in PNG format.',
            gradient: 'linear-gradient(135deg, #ec4899 0%, #f97316 100%)',
            color: '#ec4899'
        }
    ];

    return (
        <Box sx={{
            minHeight: '100vh',
            bgcolor: '#0a0a0f',
            pt: 10
        }}>
            {/* Hero Section */}
            <Container maxWidth="lg" sx={{ py: 10, textAlign: 'center', position: 'relative' }}>
                {/* Animated background */}
                <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: '25%',
                    width: 300,
                    height: 300,
                    borderRadius: '50%',
                    background: 'rgba(139, 92, 246, 0.15)',
                    filter: 'blur(80px)',
                    animation: 'pulse 4s ease-in-out infinite'
                }} />
                <Box sx={{
                    position: 'absolute',
                    top: '10%',
                    right: '20%',
                    width: 250,
                    height: 250,
                    borderRadius: '50%',
                    background: 'rgba(59, 130, 246, 0.15)',
                    filter: 'blur(80px)',
                    animation: 'pulse 4s ease-in-out infinite',
                    animationDelay: '1s'
                }} />

                <Box sx={{ position: 'relative', zIndex: 1 }}>
                    <Box sx={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 1,
                        px: 2,
                        py: 0.5,
                        mb: 3,
                        borderRadius: 10,
                        bgcolor: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.3)'
                    }}>
                        <PsychologyIcon sx={{ color: '#8b5cf6', fontSize: 18 }} />
                        <Typography variant="body2" sx={{ color: '#c4b5fd' }}>
                            AI-Powered Data Visualization
                        </Typography>
                    </Box>

                    <Typography
                        variant="h2"
                        className="animated-gradient-text"
                        sx={{
                            fontWeight: 800,
                            mb: 2,
                            fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4.5rem' },
                            lineHeight: 1.1
                        }}
                    >
                        Visualize Data
                    </Typography>
                    <Typography variant="h2" sx={{
                        fontWeight: 800,
                        mb: 3,
                        color: 'white',
                        fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4.5rem' },
                        lineHeight: 1.1
                    }}>
                        Effortlessly
                    </Typography>

                    <Typography
                        variant="subtitle1"
                        sx={{
                            color: 'rgba(255,255,255,0.7)',
                            maxWidth: 650,
                            mx: 'auto',
                            mb: 5,
                            fontSize: { xs: '1.1rem', md: '1.25rem' },
                            lineHeight: 1.7,
                            fontWeight: 400
                        }}
                    >
                        Upload your data and let our AI understand your questions. Get intelligent visualizations tailored to your queries in seconds.
                    </Typography>

                    <Box sx={{ display: 'flex', gap: 3, justifyContent: 'center', flexWrap: 'wrap' }}>
                        <Button
                            variant="contained"
                            size="large"
                            startIcon={<CloudUploadIcon />}
                            onClick={() => navigate('/upload')}
                            className="glow-button"
                            sx={{
                                px: 5,
                                py: 2,
                                borderRadius: 3,
                                fontWeight: 600,
                                fontSize: '1.1rem',
                                background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                                boxShadow: '0 0 25px rgba(139, 92, 246, 0.4)',
                                '&:hover': {
                                    boxShadow: '0 0 40px rgba(139, 92, 246, 0.6)',
                                    transform: 'translateY(-3px)'
                                },
                                transition: 'all 0.3s ease'
                            }}
                        >
                            Start Visualizing
                        </Button>
                        <Button
                            variant="outlined"
                            size="large"
                            startIcon={<BarChartIcon />}
                            onClick={() => navigate('/visualizations')}
                            sx={{
                                px: 5,
                                py: 2,
                                borderRadius: 3,
                                fontWeight: 600,
                                fontSize: '1.1rem',
                                color: 'white',
                                borderWidth: 2,
                                borderColor: 'rgba(139, 92, 246, 0.5)',
                                bgcolor: 'rgba(139, 92, 246, 0.1)',
                                backdropFilter: 'blur(8px)',
                                '&:hover': {
                                    borderColor: '#8b5cf6',
                                    bgcolor: 'rgba(139, 92, 246, 0.2)',
                                    boxShadow: '0 0 30px rgba(139, 92, 246, 0.3)',
                                    transform: 'translateY(-3px)',
                                    borderWidth: 2
                                },
                                transition: 'all 0.3s ease'
                            }}
                        >
                            View Gallery
                        </Button>
                    </Box>
                </Box>
            </Container>

            {/* Three Steps Section */}
            <Container maxWidth="lg" sx={{ py: 10 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, textAlign: 'center', mb: 1, color: 'white' }}>
                    Three steps to{' '}
                    <Box component="span" sx={{
                        background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                    }}>
                        insights
                    </Box>
                </Typography>
                <Typography sx={{ color: 'rgba(255,255,255,0.5)', textAlign: 'center', mb: 6 }}>
                    From raw data to actionable insights in minutes
                </Typography>

                <Grid container spacing={5} justifyContent="center">
                    {steps.map((step, index) => (
                        <Grid item xs={12} md={4} key={index}>
                            <Paper
                                className="card-glow"
                                sx={{
                                    p: 5,
                                    textAlign: 'center',
                                    borderRadius: 4,
                                    bgcolor: 'rgba(26, 26, 36, 0.8)',
                                    backdropFilter: 'blur(16px)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    position: 'relative',
                                    overflow: 'hidden',
                                    '&:hover': {
                                        borderColor: 'rgba(139, 92, 246, 0.4)',
                                        boxShadow: '0 25px 50px rgba(139, 92, 246, 0.15)'
                                    }
                                }}
                            >
                                {/* Step Number Badge */}
                                <Box sx={{
                                    position: 'absolute',
                                    top: 20,
                                    left: 20,
                                    width: 32,
                                    height: 32,
                                    borderRadius: '50%',
                                    background: 'rgba(139, 92, 246, 0.2)',
                                    border: '1px solid rgba(139, 92, 246, 0.3)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: '#8b5cf6',
                                    fontWeight: 700,
                                    fontSize: '0.875rem'
                                }}>
                                    {index + 1}
                                </Box>
                                <Box sx={{
                                    width: 90,
                                    height: 90,
                                    borderRadius: 4,
                                    background: step.gradient,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mx: 'auto',
                                    mb: 3,
                                    color: 'white',
                                    boxShadow: `0 15px 35px ${step.gradient.includes('8b5cf6') ? 'rgba(139, 92, 246, 0.3)' : step.gradient.includes('ec4899') ? 'rgba(236, 72, 153, 0.3)' : 'rgba(20, 184, 166, 0.3)'}`,
                                    transition: 'transform 0.3s ease',
                                    '&:hover': {
                                        transform: 'scale(1.05)'
                                    }
                                }}>
                                    {step.icon}
                                </Box>
                                <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, color: 'white' }}>
                                    {step.title}
                                </Typography>
                                <Typography sx={{ color: 'rgba(255,255,255,0.6)', lineHeight: 1.7 }}>
                                    {step.description}
                                </Typography>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>
            </Container>

            {/* Features Section */}
            <Container maxWidth="lg" sx={{ py: 10 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, textAlign: 'center', mb: 6, color: 'white' }}>
                    Powerful features,{' '}
                    <Box component="span" sx={{
                        background: 'linear-gradient(135deg, #ec4899 0%, #f97316 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                    }}>
                        zero complexity
                    </Box>
                </Typography>

                <Grid container spacing={4} justifyContent="center">
                    {features.map((feature, index) => (
                        <Grid item xs={12} sm={6} md={3} key={index}>
                            <Paper
                                className="card-glow"
                                sx={{
                                    p: 4,
                                    borderRadius: 4,
                                    bgcolor: 'rgba(19, 19, 26, 0.9)',
                                    border: '1px solid rgba(255,255,255,0.08)',
                                    position: 'relative',
                                    overflow: 'hidden',
                                    height: '100%',
                                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
                                    '&:hover': {
                                        boxShadow: `0 20px 40px rgba(0, 0, 0, 0.4), 0 0 30px ${feature.color}20`
                                    },
                                    '&::before': {
                                        content: '""',
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        right: 0,
                                        height: '3px',
                                        background: feature.gradient,
                                        borderRadius: '4px 4px 0 0'
                                    }
                                }}
                            >
                                <Box sx={{
                                    width: 56,
                                    height: 56,
                                    borderRadius: 3,
                                    background: feature.gradient,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mb: 3,
                                    color: 'white',
                                    boxShadow: `0 8px 20px ${feature.color}40`,
                                    transition: 'transform 0.3s ease',
                                    '&:hover': {
                                        transform: 'scale(1.1) rotate(5deg)'
                                    }
                                }}>
                                    {feature.icon}
                                </Box>
                                <Typography variant="h6" sx={{ fontWeight: 700, mb: 1.5, color: 'white' }}>
                                    {feature.title}
                                </Typography>
                                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.6)', lineHeight: 1.7 }}>
                                    {feature.description}
                                </Typography>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>
            </Container>

            {/* CTA Section */}
            <Container maxWidth="md" sx={{ py: 10 }}>
                <Paper sx={{
                    p: 6,
                    textAlign: 'center',
                    borderRadius: 4,
                    bgcolor: 'rgba(26, 26, 36, 0.7)',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.1)'
                }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, mb: 2, color: 'white' }}>
                        Ready to{' '}
                        <Box component="span" sx={{
                            background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent'
                        }}>
                            explore your data
                        </Box>
                        ?
                    </Typography>
                    <Typography sx={{ color: 'rgba(255,255,255,0.5)', mb: 4 }}>
                        Join data analysts using AI-powered visualization
                    </Typography>
                    <Button
                        variant="contained"
                        size="large"
                        startIcon={<CloudUploadIcon />}
                        onClick={() => navigate('/upload')}
                        className="glow-button"
                        sx={{
                            px: 5,
                            py: 2,
                            borderRadius: 3,
                            fontWeight: 600,
                            fontSize: '1.1rem',
                            background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                            boxShadow: '0 0 25px rgba(139, 92, 246, 0.4)',
                            '&:hover': {
                                boxShadow: '0 0 40px rgba(139, 92, 246, 0.6)',
                                transform: 'translateY(-3px)'
                            },
                            transition: 'all 0.3s ease'
                        }}
                    >
                        Upload Your Dataset
                    </Button>
                </Paper>
            </Container>
        </Box>
    );
};

export default Home;
