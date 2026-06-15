import React, { useState } from 'react';
import { Box, Typography, Grid, Paper, Button, Chip, CircularProgress, Alert, Collapse, IconButton, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import CodeIcon from '@mui/icons-material/Code';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import QueryInput from './QueryInput';

const VisualizationDisplay = ({ visualizations, datasetId, columns = [] }) => {
    const [queryHistory, setQueryHistory] = useState([]);
    const [filteredViz, setFilteredViz] = useState(null);
    const [dynamicViz, setDynamicViz] = useState(null);  // For LLM-generated visualizations
    const [tableResult, setTableResult] = useState(null);
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
        setTableResult(null);
        setGeneratedCode('');
        setShowCode(false);
        setQueryHistory([...queryHistory, { query, timestamp: new Date() }]);

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, dataset_id: datasetId, columns })
            });
            const data = await response.json();

            if (data.status === 'success') {
                setInterpretation(data.interpretation);

                if (data.mode === 'table' && data.rows) {
                    setMode('table');
                    setTableResult(data);
                    setDynamicViz(null);
                    setFilteredViz(null);
                    setExplanation(data.returned_count < data.row_count
                        ? `Showing ${data.returned_count} of ${data.row_count} matching rows.`
                        : `Showing ${data.row_count} matching rows.`);
                } else if (data.mode === 'dynamic' && data.figures && data.figures.length > 0) {
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
        <Box sx={{ p: { xs: 2, md: 3 }, bgcolor: '#0f172a' }}>
            {/* Query Input */}
            <QueryInput onSubmit={handleQuery} loading={loading} disabled={false} />

            {/* Interpretation */}
            {interpretation && (
                <Alert
                    severity="info"
                    icon={<AutoAwesomeIcon />}
                    sx={{
                        mb: 3,
                        bgcolor: 'rgba(14, 165, 233, 0.08)',
                        border: '1px solid rgba(14, 165, 233, 0.22)',
                        color: '#e0f2fe',
                        borderRadius: 2,
                        '& .MuiAlert-icon': { color: '#38bdf8' }
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
                            color: '#c4b5fd',
                            border: '1px solid rgba(148, 163, 184, 0.18)',
                            bgcolor: '#111827',
                            borderRadius: 1.5,
                            mb: 1
                        }}
                    >
                        {showCode ? 'Hide' : 'Show'} Generated Code
                    </Button>
                    <Collapse in={showCode}>
                        <Paper sx={{
                            p: 2,
                            bgcolor: '#020617',
                            borderRadius: 2,
                            border: '1px solid rgba(148, 163, 184, 0.14)',
                            overflow: 'auto',
                            maxHeight: 300
                        }}>
                            <pre style={{
                                margin: 0,
                                fontSize: '12px',
                                color: '#bae6fd',
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
                    <Typography variant="body2" sx={{ color: '#94a3b8', mb: 1 }}>
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
                                    bgcolor: '#111827',
                                    color: '#cbd5e1',
                                    border: '1px solid rgba(148, 163, 184, 0.18)',
                                    borderRadius: 1.5,
                                    cursor: 'pointer',
                                    '&:hover': {
                                        bgcolor: 'rgba(124, 58, 237, 0.18)'
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
                    <CircularProgress sx={{ color: '#7c3aed', mb: 2 }} />
                    <Typography sx={{ color: '#94a3b8' }}>
                        Generating answer...
                    </Typography>
                </Box>
            )}

            {!loading && mode === 'table' && tableResult && (
                <Paper sx={{
                    p: 3,
                    borderRadius: 2,
                    bgcolor: '#111827',
                    border: '1px solid rgba(148, 163, 184, 0.16)',
                    mb: 3
                }}>
                    <Typography variant="h6" sx={{ color: '#f8fafc', fontWeight: 700, mb: 2 }}>
                        Matching Records
                    </Typography>
                    <TableContainer sx={{ maxHeight: 520 }}>
                        <Table stickyHeader size="small">
                            <TableHead>
                                <TableRow>
                                    {tableResult.columns.map((column) => (
                                        <TableCell key={column} sx={{ fontWeight: 700 }}>
                                            {column}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {tableResult.rows.map((row, rowIndex) => (
                                    <TableRow key={rowIndex} hover>
                                        {tableResult.columns.map((column) => (
                                            <TableCell key={column}>
                                                {row[column] === null || row[column] === undefined ? '' : String(row[column])}
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Paper>
            )}

            {/* Dynamic Visualizations (LLM Generated) */}
            {!loading && mode === 'dynamic' && dynamicViz && dynamicViz.length > 0 && (
                <Grid container spacing={3}>
                    {dynamicViz.map((figure, index) => (
                        <Grid item xs={12} key={index}>
                            <Paper sx={{
                                p: { xs: 2, md: 2.5 },
                                borderRadius: 2,
                                bgcolor: '#111827',
                                border: '1px solid rgba(148, 163, 184, 0.16)',
                            }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <AutoAwesomeIcon sx={{ color: '#a78bfa', fontSize: 20 }} />
                                        <Typography variant="h6" sx={{ color: '#f8fafc', fontWeight: 700, fontSize: '1rem' }}>
                                            Generated Visualization
                                        </Typography>
                                    </Box>
                                    <Button
                                        size="small"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => downloadVisualization(figure, `ai_visualization_${index + 1}`)}
                                        sx={{
                                            color: '#c4b5fd',
                                            border: '1px solid rgba(148, 163, 184, 0.18)',
                                            bgcolor: '#0f172a',
                                            borderRadius: 1.5,
                                            '&:hover': {
                                                bgcolor: 'rgba(124, 58, 237, 0.18)'
                                            }
                                        }}
                                    >
                                        Download
                                    </Button>
                                </Box>
                                <Box sx={{
                                    borderRadius: 1.5,
                                    overflow: 'hidden',
                                    bgcolor: 'white',
                                    border: '1px solid rgba(148, 163, 184, 0.18)'
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
                                p: 2.5,
                                borderRadius: 2,
                                bgcolor: '#111827',
                                border: '1px solid rgba(148, 163, 184, 0.16)',
                                transition: 'all 0.3s',
                                '&:hover': {
                                    borderColor: 'rgba(148, 163, 184, 0.3)'
                                }
                            }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Typography variant="h6" sx={{ color: '#f8fafc', fontWeight: 700, fontSize: '1rem' }}>
                                        {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </Typography>
                                    <Button
                                        size="small"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => downloadVisualization(data[type], type)}
                                        sx={{
                                            color: '#c4b5fd',
                                            border: '1px solid rgba(148, 163, 184, 0.18)',
                                            bgcolor: '#0f172a',
                                            borderRadius: 1.5,
                                            '&:hover': {
                                                bgcolor: 'rgba(124, 58, 237, 0.18)'
                                            }
                                        }}
                                    >
                                        Download
                                    </Button>
                                </Box>
                                <Box sx={{
                                    borderRadius: 1.5,
                                    overflow: 'hidden',
                                    bgcolor: '#020617',
                                    border: '1px solid rgba(148, 163, 184, 0.12)'
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
