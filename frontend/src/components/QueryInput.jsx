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
        "Show correlation between columns",
        "Display feature importance",
        "Show distribution plots",
        "Show me all visualizations"
    ];

    return (
        <Box sx={{
            p: 3,
            mb: 4,
            borderRadius: 3,
            bgcolor: 'rgba(26, 26, 36, 0.7)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.1)'
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                <Box sx={{
                    width: 36,
                    height: 36,
                    borderRadius: 2,
                    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <AutoAwesomeIcon sx={{ color: 'white', fontSize: 20 }} />
                </Box>
                <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                    Ask about your data
                </Typography>
            </Box>

            <form onSubmit={handleSubmit}>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                    <TextField
                        fullWidth
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="e.g., Show correlation between age and salary..."
                        disabled={disabled || loading}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                bgcolor: 'rgba(10, 10, 15, 0.8)',
                                borderRadius: 2,
                                color: 'white',
                                '& fieldset': {
                                    borderColor: 'rgba(255,255,255,0.15)'
                                },
                                '&:hover fieldset': {
                                    borderColor: 'rgba(139, 92, 246, 0.5)'
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: '#8b5cf6'
                                }
                            },
                            '& input::placeholder': {
                                color: 'rgba(255,255,255,0.4)'
                            }
                        }}
                    />
                    <Button
                        type="submit"
                        variant="contained"
                        disabled={!query.trim() || loading || disabled}
                        sx={{
                            px: 3,
                            borderRadius: 2,
                            background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                            minWidth: 56,
                            '&:disabled': {
                                background: 'rgba(255,255,255,0.1)'
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
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)' }}>
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
                            color: '#c4b5fd',
                            border: '1px solid rgba(139, 92, 246, 0.3)',
                            '&:hover': {
                                bgcolor: 'rgba(139, 92, 246, 0.2)'
                            }
                        }}
                    />
                ))}
            </Box>
        </Box>
    );
};

export default QueryInput;
