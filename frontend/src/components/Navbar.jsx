import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
    AppBar,
    Toolbar,
    Typography,
    Button,
    Box,
    Container,
    IconButton,
    Drawer,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    useTheme,
    useMediaQuery
} from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BarChartIcon from '@mui/icons-material/BarChart';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';

const Navbar = () => {
    const theme = useTheme();
    const location = useLocation();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navItems = [
        { text: 'Home', icon: <HomeIcon />, path: '/' },
        { text: 'Upload', icon: <CloudUploadIcon />, path: '/upload' },
        { text: 'Gallery', icon: <BarChartIcon />, path: '/visualizations' }
    ];

    const isActive = (path) => {
        if (path === '/') return location.pathname === '/';
        return location.pathname.startsWith(path);
    };

    const drawerContent = (
        <Box sx={{ width: 280, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2, borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <Typography variant="h6" sx={{
                    fontFamily: "'Rye', serif",
                    fontWeight: 400,
                    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                }}>
                    DataViz AI
                </Typography>
                <IconButton onClick={() => setDrawerOpen(false)} sx={{ color: 'white' }}>
                    <CloseIcon />
                </IconButton>
            </Box>
            <List sx={{ pt: 2 }}>
                {navItems.map((item) => (
                    <ListItem
                        button
                        key={item.text}
                        component={Link}
                        to={item.path}
                        onClick={() => setDrawerOpen(false)}
                        sx={{
                            mb: 1,
                            mx: 1,
                            borderRadius: 2,
                            bgcolor: isActive(item.path) ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(139, 92, 246, 0.1)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: isActive(item.path) ? '#8b5cf6' : 'rgba(255,255,255,0.7)', minWidth: 40 }}>
                            {item.icon}
                        </ListItemIcon>
                        <ListItemText
                            primary={item.text}
                            sx={{
                                color: isActive(item.path) ? '#8b5cf6' : 'white',
                                '& .MuiListItemText-primary': { fontFamily: "'Rye', serif" }
                            }}
                        />
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    return (
        <AppBar
            position="fixed"
            elevation={scrolled ? 4 : 0}
            sx={{
                bgcolor: scrolled ? 'rgba(10, 10, 15, 0.95)' : 'rgba(10, 10, 15, 0.8)',
                backdropFilter: 'blur(12px)',
                borderBottom: scrolled ? 'none' : '1px solid rgba(255,255,255,0.1)'
            }}
        >
            <Container maxWidth="lg">
                <Toolbar sx={{ py: 1 }}>
                    <Box component={Link} to="/" sx={{ display: 'flex', alignItems: 'center', textDecoration: 'none', flexGrow: 1 }}>
                        <Box sx={{
                            width: 48,
                            height: 48,
                            borderRadius: 2,
                            background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 2,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                transform: 'scale(1.1)',
                                boxShadow: '0 0 25px rgba(139, 92, 246, 0.5)'
                            }
                        }}>
                            <BarChartIcon sx={{ color: 'white', fontSize: 28 }} />
                        </Box>
                        <Typography variant="h5" sx={{
                            fontFamily: "'Rye', serif",
                            fontWeight: 400,
                            background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            letterSpacing: '0.02em'
                        }}>
                            DataViz AI
                        </Typography>
                    </Box>

                    {isMobile ? (
                        <IconButton onClick={() => setDrawerOpen(true)} sx={{ color: 'white' }}>
                            <MenuIcon />
                        </IconButton>
                    ) : (
                        <Box sx={{ display: 'flex', gap: 1 }}>
                            {navItems.map((item) => (
                                <Button
                                    key={item.text}
                                    component={Link}
                                    to={item.path}
                                    startIcon={item.icon}
                                    sx={{
                                        fontFamily: "'Rye', serif",
                                        px: 2,
                                        py: 1,
                                        borderRadius: 2,
                                        color: isActive(item.path) ? '#8b5cf6' : 'rgba(255,255,255,0.8)',
                                        bgcolor: isActive(item.path) ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
                                        border: isActive(item.path) ? '1px solid rgba(139, 92, 246, 0.3)' : '1px solid transparent',
                                        '&:hover': {
                                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                                            color: 'white'
                                        }
                                    }}
                                >
                                    {item.text}
                                </Button>
                            ))}
                        </Box>
                    )}

                    <Drawer
                        anchor="left"
                        open={drawerOpen}
                        onClose={() => setDrawerOpen(false)}
                        PaperProps={{
                            sx: {
                                bgcolor: 'rgba(10, 10, 15, 0.98)',
                                backdropFilter: 'blur(12px)'
                            }
                        }}
                    >
                        {drawerContent}
                    </Drawer>
                </Toolbar>
            </Container>
        </AppBar>
    );
};

export default Navbar;
