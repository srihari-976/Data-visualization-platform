import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  LinearProgress,
  Container,
  Typography,
  IconButton,
  Grid
} from '@mui/material';
import { Link } from 'react-router-dom';
import GitHubIcon from '@mui/icons-material/GitHub';
import TwitterIcon from '@mui/icons-material/Twitter';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import BarChartIcon from '@mui/icons-material/BarChart';
import Navbar from './components/Navbar';
import Home from './components/Home';
import UploadForm from './components/UploadForm';
import Results from './components/Results';
import VizGallery from './components/VizGallery';

// Dark theme with gradient accents
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#8b5cf6',
    },
    secondary: {
      main: '#3b82f6',
    },
    background: {
      default: '#0a0a0f',
      paper: '#13131a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontWeight: 800 },
    h2: { fontWeight: 800 },
    h3: { fontWeight: 700 },
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#0a0a0f',
          backgroundImage: 'radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.1) 0, transparent 50%), radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.1) 0, transparent 50%)',
          minHeight: '100vh',
        },
        '@keyframes pulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 }
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          backgroundColor: 'rgba(26, 26, 36, 0.7)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        }
      }
    }
  },
});

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
    }, 500);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          bgcolor: 'background.default'
        }}>
          <Navbar />

          {isLoading && (
            <LinearProgress
              sx={{
                position: 'fixed',
                top: 64,
                left: 0,
                right: 0,
                zIndex: 9999,
                height: 3,
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #8b5cf6, #3b82f6)'
                }
              }}
            />
          )}

          <Box component="main" sx={{ flex: 1, pt: '64px' }}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<UploadForm />} />
              <Route path="/results/:datasetId" element={<Results />} />
              <Route path="/visualizations" element={<VizGallery />} />
              <Route path="/visualizations/:datasetId" element={<VizGallery />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Box>

          {/* Enhanced Footer */}
          <Box
            component="footer"
            sx={{
              py: 6,
              px: 3,
              borderTop: '1px solid rgba(255,255,255,0.1)',
              bgcolor: 'rgba(10, 10, 15, 0.95)',
              backdropFilter: 'blur(12px)'
            }}
          >
            <Container maxWidth="lg">
              <Grid container spacing={4}>
                {/* Brand Section */}
                <Grid item xs={12} md={4}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{
                      width: 40,
                      height: 40,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 1.5
                    }}>
                      <BarChartIcon sx={{ color: 'white', fontSize: 24 }} />
                    </Box>
                    <Typography variant="h6" sx={{
                      fontWeight: 800,
                      background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent'
                    }}>
                      DataViz AI
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)', maxWidth: 280, lineHeight: 1.7 }}>
                    Transform your data into beautiful visualizations with AI-powered insights. Upload, ask, and discover.
                  </Typography>
                </Grid>

                {/* Quick Links */}
                <Grid item xs={6} md={2}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'white', mb: 2 }}>
                    Quick Links
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {[{ text: 'Home', path: '/' }, { text: 'Upload', path: '/upload' }, { text: 'Gallery', path: '/visualizations' }].map((link) => (
                      <Box
                        key={link.text}
                        component={Link}
                        to={link.path}
                        sx={{
                          color: 'rgba(255,255,255,0.5)',
                          textDecoration: 'none',
                          fontSize: '0.875rem',
                          transition: 'color 0.2s',
                          '&:hover': { color: '#8b5cf6' }
                        }}
                      >
                        {link.text}
                      </Box>
                    ))}
                  </Box>
                </Grid>

                {/* Resources */}
                <Grid item xs={6} md={2}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'white', mb: 2 }}>
                    Resources
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {['Documentation', 'API Reference', 'Support'].map((text) => (
                      <Typography
                        key={text}
                        variant="body2"
                        sx={{
                          color: 'rgba(255,255,255,0.5)',
                          cursor: 'pointer',
                          transition: 'color 0.2s',
                          '&:hover': { color: '#8b5cf6' }
                        }}
                      >
                        {text}
                      </Typography>
                    ))}
                  </Box>
                </Grid>

                {/* Social Links */}
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'white', mb: 2 }}>
                    Connect With Us
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {[
                      { icon: <GitHubIcon />, label: 'GitHub' },
                      { icon: <TwitterIcon />, label: 'Twitter' },
                      { icon: <LinkedInIcon />, label: 'LinkedIn' }
                    ].map((social) => (
                      <IconButton
                        key={social.label}
                        size="small"
                        sx={{
                          color: 'rgba(255,255,255,0.5)',
                          bgcolor: 'rgba(255,255,255,0.05)',
                          border: '1px solid rgba(255,255,255,0.1)',
                          transition: 'all 0.3s',
                          '&:hover': {
                            color: '#8b5cf6',
                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.3)'
                          }
                        }}
                      >
                        {social.icon}
                      </IconButton>
                    ))}
                  </Box>
                </Grid>
              </Grid>

              {/* Bottom Bar */}
              <Box sx={{
                mt: 4,
                pt: 3,
                borderTop: '1px solid rgba(255,255,255,0.05)',
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 2
              }}>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.4)' }}>
                  © {new Date().getFullYear()} DataViz AI. All rights reserved.
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.4)' }}>
                  Powered by{' '}
                  <Box component="span" sx={{ color: '#8b5cf6', fontWeight: 600 }}>
                    Llama 3.1  3B
                  </Box>
                </Typography>
              </Box>
            </Container>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;