import React, { useState } from 'react';
import { Box, Typography, Grid, Paper, Button, Chip, CircularProgress, Alert, Collapse, IconButton } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import CodeIcon from '@mui/icons-material/Code';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import QueryInput from './QueryInput';

const VisualizationDisplay = ({ visualizations, datasetId, columns = [] }) => {
    const [queryHistory, setQueryHistory] = useState([]);
    const [filteredViz, setFilteredViz] = useState(null);
    const [dynamicViz, setDynamicViz] = useState(null);  // For LLM-generated visualizations
    const [loading, setLoading] = useState(false);
    const [interpretation, setInterpretation] = useState('');
    const [explanation, setExplanation] = useState('');
    const [generatedCode, setGeneratedCode] = useState('');
    const [showCode, setShowCode] = useState(false);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('initial');  // 'initial', 'dynamic', 'filter'

    // Helper function to filter visualizations based on requested types
    const filterVisualizations = (vizData, requestedTypes) => {
        if (!vizData || !vizData.data || !vizData.types) return vizData;

        if (requestedTypes === 'all' || (Array.isArray(requestedTypes) && requestedTypes.includes('all'))) {
            return vizData;
        }

        const typesArray = Array.isArray(requestedTypes) ? requestedTypes : [requestedTypes];

        const filteredTypes = vizData.types.filter(type => {
            return typesArray.some(reqType => {
                if (type === reqType) return true;
                if (reqType.endsWith('_') && type.startsWith(reqType)) return true;
                if (type.includes(reqType)) return true;
                return false;
            });
        });

        if (filteredTypes.length === 0) return vizData;

        const filteredData = {};
        filteredTypes.forEach(type => {
            if (vizData.data[type]) {
                filteredData[type] = vizData.data[type];
            }
        });

        return {
            types: filteredTypes,
            data: filteredData
        };
    };

    const handleQuery = async (query) => {
        setLoading(true);
        setError(null);
        setDynamicViz(null);
        setGeneratedCode('');
        setShowCode(false);
        setQueryHistory([...queryHistory, { query, timestamp: new Date() }]);

        try {
            const response = await fetch('http://localhost:5000/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, dataset_id: datasetId, columns })
            });
            const data = await response.json();

            if (data.status === 'success') {
                setInterpretation(data.interpretation);

                // Check if this is a dynamic (LLM-generated) visualization
                if (data.mode === 'dynamic' && data.figures && data.figures.length > 0) {
                    setMode('dynamic');
                    setDynamicViz(data.figures);
                    setExplanation(data.explanation || '');
                    setGeneratedCode(data.code || '');
                    setFilteredViz(null);
                } else {
                    // Fallback to filtering existing visualizations
                    setMode('filter');
                    setDynamicViz(null);
                    const filtered = filterVisualizations(visualizations, data.visualization_types);
                    setFilteredViz(filtered);
                }
            } else {
                setError(data.error || 'Query failed');
            }
        } catch (err) {
            console.error('Query error:', err);
            setFilteredViz(visualizations);
            setInterpretation(`Showing all visualizations for: ${query}`);
            setMode('filter');
        } finally {
            setLoading(false);
        }
    };

    const downloadVisualization = (imgData, name = 'visualization') => {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${imgData}`;
        link.download = `${name}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    if (!visualizations || !visualizations.data || !visualizations.types || visualizations.types.length === 0) {
        return (
            <Box sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: '#ef4444', mb: 1 }}>
                    No visualizations available
                </Typography>
                <Typography sx={{ color: 'rgba(255,255,255,0.5)' }}>
                    Upload a dataset to get started
                </Typography>
            </Box>
        );
    }

    const { data, types } = filteredViz || visualizations;

    return (
        <Box sx={{ p: 3 }}>
            {/* Query Input */}
            <QueryInput onSubmit={handleQuery} loading={loading} disabled={false} />

            {/* Interpretation */}
            {interpretation && (
                <Alert
                    severity="info"
                    icon={<AutoAwesomeIcon />}
                    sx={{
                        mb: 3,
                        bgcolor: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.3)',
                        color: '#c4b5fd',
                        '& .MuiAlert-icon': { color: '#8b5cf6' }
                    }}
                >
                    <Box>
                        <Typography variant="body1">{interpretation}</Typography>
                        {explanation && (
                            <Typography variant="body2" sx={{ mt: 1, opacity: 0.8 }}>
                                {explanation}
                            </Typography>
                        )}
                    </Box>
                </Alert>
            )}

            {/* Show Generated Code Toggle */}
            {generatedCode && (
                <Box sx={{ mb: 3 }}>
                    <Button
                        size="small"
                        startIcon={<CodeIcon />}
                        endIcon={<ExpandMoreIcon sx={{ transform: showCode ? 'rotate(180deg)' : 'none', transition: '0.3s' }} />}
                        onClick={() => setShowCode(!showCode)}
                        sx={{
                            color: '#8b5cf6',
                            border: '1px solid rgba(139, 92, 246, 0.3)',
                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                            mb: 1
                        }}
                    >
                        {showCode ? 'Hide' : 'Show'} Generated Code
                    </Button>
                    <Collapse in={showCode}>
                        <Paper sx={{
                            p: 2,
                            bgcolor: 'rgba(0,0,0,0.5)',
                            borderRadius: 2,
                            overflow: 'auto',
                            maxHeight: 300
                        }}>
                            <pre style={{
                                margin: 0,
                                fontSize: '12px',
                                color: '#a5f3fc',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word'
                            }}>
                                {generatedCode}
                            </pre>
                        </Paper>
                    </Collapse>
                </Box>
            )}

            {/* Query History */}
            {queryHistory.length > 0 && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)', mb: 1 }}>
                        Recent Queries
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {queryHistory.slice(-5).map((item, index) => (
                            <Chip
                                key={index}
                                label={item.query}
                                size="small"
                                onClick={() => handleQuery(item.query)}
                                sx={{
                                    bgcolor: 'rgba(139, 92, 246, 0.1)',
                                    color: '#c4b5fd',
                                    border: '1px solid rgba(139, 92, 246, 0.2)',
                                    cursor: 'pointer',
                                    '&:hover': {
                                        bgcolor: 'rgba(139, 92, 246, 0.2)'
                                    }
                                }}
                            />
                        ))}
                    </Box>
                </Box>
            )}

            {/* Error */}
            {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                </Alert>
            )}

            {/* Loading */}
            {loading && (
                <Box sx={{ textAlign: 'center', py: 6 }}>
                    <CircularProgress sx={{ color: '#8b5cf6', mb: 2 }} />
                    <Typography sx={{ color: 'rgba(255,255,255,0.6)' }}>
                        🧠 Generating visualization with AI...
                    </Typography>
                </Box>
            )}

            {/* Dynamic Visualizations (LLM Generated) */}
            {!loading && mode === 'dynamic' && dynamicViz && dynamicViz.length > 0 && (
                <Grid container spacing={3}>
                    {dynamicViz.map((figure, index) => (
                        <Grid item xs={12} key={index}>
                            <Paper sx={{
                                p: 3,
                                borderRadius: 3,
                                bgcolor: 'rgba(19, 19, 26, 0.8)',
                                border: '1px solid rgba(139, 92, 246, 0.3)',
                                transition: 'all 0.3s',
                            }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <AutoAwesomeIcon sx={{ color: '#8b5cf6' }} />
                                        <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                                            AI Generated Visualization
                                        </Typography>
                                    </Box>
                                    <Button
                                        size="small"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => downloadVisualization(figure, `ai_visualization_${index + 1}`)}
                                        sx={{
                                            color: '#8b5cf6',
                                            border: '1px solid rgba(139, 92, 246, 0.3)',
                                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                                            '&:hover': {
                                                bgcolor: 'rgba(139, 92, 246, 0.2)'
                                            }
                                        }}
                                    >
                                        Download
                                    </Button>
                                </Box>
                                <Box sx={{
                                    borderRadius: 2,
                                    overflow: 'hidden',
                                    bgcolor: 'white',
                                    border: '1px solid rgba(255,255,255,0.1)'
                                }}>
                                    <img
                                        src={`data:image/png;base64,${figure}`}
                                        alt={`Generated visualization ${index + 1}`}
                                        style={{ width: '100%', height: 'auto', display: 'block' }}
                                    />
                                </Box>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>
            )}

            {/* Static Visualizations Grid (Filtered or Initial) */}
            {!loading && (mode === 'filter' || mode === 'initial') && (
                <Grid container spacing={3}>
                    {types.map((type) => (
                        <Grid item xs={12} md={6} key={type}>
                            <Paper sx={{
                                p: 3,
                                borderRadius: 3,
                                bgcolor: 'rgba(19, 19, 26, 0.8)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                transition: 'all 0.3s',
                                '&:hover': {
                                    borderColor: 'rgba(139, 92, 246, 0.3)'
                                }
                            }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                                        {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </Typography>
                                    <Button
                                        size="small"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => downloadVisualization(data[type], type)}
                                        sx={{
                                            color: '#8b5cf6',
                                            border: '1px solid rgba(139, 92, 246, 0.3)',
                                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                                            '&:hover': {
                                                bgcolor: 'rgba(139, 92, 246, 0.2)'
                                            }
                                        }}
                                    >
                                        Download
                                    </Button>
                                </Box>
                                <Box sx={{
                                    borderRadius: 2,
                                    overflow: 'hidden',
                                    bgcolor: 'rgba(255,255,255,0.02)',
                                    border: '1px solid rgba(255,255,255,0.05)'
                                }}>
                                    <img
                                        src={`data:image/png;base64,${data[type]}`}
                                        alt={type}
                                        style={{ width: '100%', height: 'auto', display: 'block' }}
                                    />
                                </Box>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>
            )}

            {/* Ask Another Question */}
            {!loading && queryHistory.length > 0 && (
                <Box sx={{ textAlign: 'center', mt: 6 }}>
                    <Typography sx={{ color: 'rgba(255,255,255,0.5)', mb: 2 }}>
                        Ask another question to generate a new visualization
                    </Typography>
                    <Button
                        variant="outlined"
                        onClick={() => document.querySelector('input')?.focus()}
                        sx={{
                            color: 'white',
                            borderColor: 'rgba(255,255,255,0.3)',
                            '&:hover': {
                                borderColor: 'rgba(255,255,255,0.5)',
                                bgcolor: 'rgba(255,255,255,0.05)'
                            }
                        }}
                    >
                        Ask Another Question
                    </Button>
                </Box>
            )}
        </Box>
    );
};

export default VisualizationDisplay;
