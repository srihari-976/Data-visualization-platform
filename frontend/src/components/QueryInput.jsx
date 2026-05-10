import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Chip, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

const QueryInput = ({ onSubmit, loading, disabled }) => {
    const [query, setQuery] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() && !loading) {
            onSubmit(query);
            setQuery('');
        }
    };

    const exampleQueries = [
        "Visualize MEDV using a histogram",
        "Compare average MEDV by CHAS",
        "Show RM vs MEDV as a scatter plot",
        "Visualize CHAS using a pie chart"
    ];

    return (
        <Box sx={{
            p: { xs: 2, md: 2.5 },
            mb: 3,
            borderRadius: 2,
            bgcolor: '#111827',
            border: '1px solid rgba(148, 163, 184, 0.16)'
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mb: 2 }}>
                <Box sx={{
                    width: 32,
                    height: 32,
                    borderRadius: 1.5,
                    bgcolor: 'rgba(124, 58, 237, 0.16)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <AutoAwesomeIcon sx={{ color: '#a78bfa', fontSize: 18 }} />
                </Box>
                <Typography variant="h6" sx={{ color: '#f8fafc', fontWeight: 700, fontSize: '1rem' }}>
                    Ask about your data
                </Typography>
            </Box>

            <form onSubmit={handleSubmit}>
                <Box sx={{ display: 'flex', gap: 1.25, mb: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
                    <TextField
                        fullWidth
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask for one chart, for example: Visualize MEDV using a histogram"
                        disabled={disabled || loading}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                bgcolor: 'rgba(10, 10, 15, 0.8)',
                                borderRadius: 1.5,
                                color: '#f8fafc',
                                '& fieldset': {
                                    borderColor: 'rgba(148, 163, 184, 0.22)'
                                },
                                '&:hover fieldset': {
                                    borderColor: 'rgba(148, 163, 184, 0.4)'
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: '#7c3aed'
                                }
                            },
                            '& input::placeholder': {
                                color: '#64748b'
                            }
                        }}
                    />
                    <Button
                        type="submit"
                        variant="contained"
                        disabled={!query.trim() || loading || disabled}
                        sx={{
                            px: 2.5,
                            borderRadius: 1.5,
                            bgcolor: '#7c3aed',
                            minWidth: { xs: '100%', sm: 56 },
                            '&:hover': { bgcolor: '#6d28d9' },
                            '&:disabled': {
                                background: 'rgba(148, 163, 184, 0.16)'
                            }
                        }}
                    >
                        {loading ? (
                            <CircularProgress size={24} sx={{ color: 'white' }} />
                        ) : (
                            <SendIcon />
                        )}
                    </Button>
                </Box>
            </form>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                    Try:
                </Typography>
                {exampleQueries.map((example, index) => (
                    <Chip
                        key={index}
                        label={example}
                        onClick={() => !loading && !disabled && setQuery(example)}
                        disabled={loading || disabled}
                        size="small"
                        sx={{
                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                            color: '#cbd5e1',
                            border: '1px solid rgba(148, 163, 184, 0.18)',
                            borderRadius: 1.5,
                            '&:hover': {
                                bgcolor: 'rgba(124, 58, 237, 0.18)',
                                color: '#f8fafc'
                            }
                        }}
                    />
                ))}
            </Box>
        </Box>
    );
};

export default QueryInput;
