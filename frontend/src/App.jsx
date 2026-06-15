import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  LinearProgress,
  Container,
  Typography
} from '@mui/material';
import Navbar from './components/Navbar';
import Home from './components/Home';
import UploadForm from './components/UploadForm';
import Results from './components/Results';
import VizGallery from './components/VizGallery';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#7c3aed',
    },
    secondary: {
      main: '#0ea5e9',
    },
    background: {
      default: '#0b1020',
      paper: '#111827',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#94a3b8',
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
          backgroundColor: '#0b1020',
          backgroundImage: 'linear-gradient(180deg, #0b1020 0%, #0f172a 52%, #0b1020 100%)',
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
          backgroundColor: '#111827',
          border: '1px solid rgba(148, 163, 184, 0.16)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
          boxShadow: 'none',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderColor: 'rgba(148, 163, 184, 0.16)',
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

          <Box
            component="footer"
            sx={{
              py: 3,
              px: 3,
              borderTop: '1px solid rgba(148, 163, 184, 0.14)',
              bgcolor: '#0b1020'
            }}
          >
            <Container maxWidth="lg">
              <Box sx={{
                display: 'flex',
                flexDirection: { xs: 'column', sm: 'row' },
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 2
              }}>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  © {new Date().getFullYear()} DataViz AI. All rights reserved.
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Powered by{' '}
                  <Box component="span" sx={{ color: 'primary.main', fontWeight: 600 }}>
                    Ollama / Llama 3.2 3B
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
