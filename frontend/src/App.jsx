import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { 
  Box, 
  CssBaseline, 
  Container, 
  ThemeProvider, 
  createTheme,
  LinearProgress
} from '@mui/material';
import Navbar from './components/Navbar';
import Home from './components/Home';
import UploadForm from './components/UploadForm';
import Results from './components/Results';
import VizGallery from './components/VizGallery';
// Import the existing Footer instead of defining it again
// import Footer from './components/Footer';

// Create a responsive theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#9c27b0',
    },
    background: {
      default: '#f9f9fb',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  const [isLoading, setIsLoading] = useState(true);

  // Simulate initial app loading
  useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
    }, 800);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <Router>
        <CssBaseline />
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          minHeight: '100vh',
          bgcolor: 'background.default'
        }}>
          <Navbar />
          
          {isLoading ? (
            <LinearProgress sx={{ position: 'fixed', top: 64, left: 0, right: 0, zIndex: 9999 }} />
          ) : null}
          
          <Container 
            component="main" 
            sx={{ 
              mt: { xs: 2, sm: 4 }, 
              mb: { xs: 2, sm: 4 }, 
              flex: 1,
              maxWidth: {
                xs: '100%',
                lg: '1200px',
              },
              px: { xs: 2, sm: 3 }
            }}
          >
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<UploadForm />} />
              <Route path="/results/:datasetId" element={<Results />} />
              <Route path="/visualizations" element={<VizGallery />} />
              <Route path="/visualizations/:datasetId" element={<VizGallery />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Container>
          
          {/* Use your existing Footer component instead */}
          {/* <Footer /> */}
          
          {/* Or use a simple footer implementation if you don't have a Footer component yet */}
          <Box 
            component="footer" 
            sx={{ 
              py: 3, 
              px: 2, 
              mt: 'auto', 
              backgroundColor: '#f5f5f5',
              borderTop: '1px solid #e0e0e0',
              textAlign: 'center',
              color: 'text.secondary',
              fontSize: 14
            }}
          >
            © {new Date().getFullYear()} DataViz Explorer • All Rights Reserved
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;