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
        <Box sx={{ width: 280, height: '100%', bgcolor: '#0f172a' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2.25, borderBottom: '1px solid rgba(148,163,184,0.16)' }}>
                <Typography variant="h6" sx={{
                    fontWeight: 800,
                    color: 'white'
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
                            borderRadius: 1.5,
                            bgcolor: isActive(item.path) ? 'rgba(124, 58, 237, 0.16)' : 'transparent',
                            '&:hover': { bgcolor: 'rgba(148, 163, 184, 0.08)' }
                        }}
                    >
                        <ListItemIcon sx={{ color: isActive(item.path) ? '#a78bfa' : '#94a3b8', minWidth: 40 }}>
                            {item.icon}
                        </ListItemIcon>
                        <ListItemText
                            primary={item.text}
                            sx={{
                                color: isActive(item.path) ? '#f8fafc' : '#cbd5e1',
                                '& .MuiListItemText-primary': { fontWeight: 600 }
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
            elevation={0}
            sx={{
                bgcolor: scrolled ? 'rgba(15, 23, 42, 0.96)' : 'rgba(15, 23, 42, 0.88)',
                backdropFilter: 'blur(14px)',
                borderBottom: '1px solid rgba(148,163,184,0.14)'
            }}
        >
            <Container maxWidth="xl">
                <Toolbar sx={{ minHeight: 64, px: { xs: 0, sm: 2 } }}>
                    <Box component={Link} to="/" sx={{ display: 'flex', alignItems: 'center', textDecoration: 'none', flexGrow: 1 }}>
                        <Box sx={{
                            width: 38,
                            height: 38,
                            borderRadius: 1.5,
                            background: 'linear-gradient(135deg, #7c3aed 0%, #0ea5e9 100%)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 1.5
                        }}>
                            <BarChartIcon sx={{ color: 'white', fontSize: 23 }} />
                        </Box>
                        <Typography variant="h5" sx={{
                            fontWeight: 800,
                            color: '#f8fafc',
                            letterSpacing: 0,
                            fontSize: { xs: '1.1rem', sm: '1.25rem' }
                        }}>
                            DataViz AI
                        </Typography>
                    </Box>

                    {isMobile ? (
                        <IconButton onClick={() => setDrawerOpen(true)} sx={{ color: 'white' }}>
                            <MenuIcon />
                        </IconButton>
                    ) : (
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                            {navItems.map((item) => (
                                <Button
                                    key={item.text}
                                    component={Link}
                                    to={item.path}
                                    startIcon={item.icon}
                                    sx={{
                                        px: 1.5,
                                        py: 0.8,
                                        borderRadius: 1.5,
                                        color: isActive(item.path) ? '#f8fafc' : '#94a3b8',
                                        bgcolor: isActive(item.path) ? 'rgba(124, 58, 237, 0.18)' : 'transparent',
                                        border: '1px solid transparent',
                                        '&:hover': {
                                            bgcolor: 'rgba(148, 163, 184, 0.08)',
                                            color: '#f8fafc'
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
                                bgcolor: '#0f172a'
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
