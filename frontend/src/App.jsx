import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Wifi,
    Users,
    TrendingUp,
    Clock,
    LayoutDashboard,
    Activity,
    Timer,
    MessageSquare,
    Hourglass,
    Zap,
    Map as MapIcon,
    BarChart3,
    Filter,
    ChevronRight,
    ShieldCheck,
    AlertCircle,
    Upload,
    X,
    Maximize2,
    Trash2
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    PieChart,
    Pie,
    Cell
} from 'recharts';

import './App.css';

// --- Dashboard V3: The Interactive Control Center ---

const EMOTION_COLORS = {
    happy: '#10b981',
    neutral: '#94a3b8',
    sad: '#3b82f6',
    angry: '#ef4444',
    surprise: '#f59e0b',
    disgust: '#8b5cf6',
    fear: '#ec4899'
};

function App() {
    const [data, setData] = useState([]);
    const [activePersons, setActivePersons] = useState([]);
    const [logs, setLogs] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);

    // Drill-down Filters
    const [filterType, setFilterType] = useState(null); // 'interaction', 'emotion', 'person'
    const [filterValue, setFilterValue] = useState(null);
    const [selectedPerson, setSelectedPerson] = useState(null);

    const [globalStats, setGlobalStats] = useState({
        total_frames: 0,
        unique_persons_count: 0,
        emotions_breakdown: {},
        total_interactions: 0,
        interaction_types: {}
    });

    const canvasRef = useRef(null);
    const socketRef = useRef(null);
    const fileInputRef = useRef(null);

    // 1. Initial Data & Periodic Stats Fetch
    useEffect(() => {
        const fetchData = () => {
            fetch('http://localhost:8000/scene-data')
                .then(res => res.json())
                .then(result => result.data && setData(result.data))
                .catch(err => console.error("Data Fetch Error:", err));

            fetch('http://localhost:8000/stats/summary')
                .then(res => res.json())
                .then(result => setGlobalStats(result))
                .catch(err => console.error("Stats Fetch Error:", err));
        };

        fetchData();
        const interval = setInterval(fetchData, 5000); // Update stats every 5s
        return () => clearInterval(interval);
    }, []);

    // 2. WebSocket Real-time Stream
    useEffect(() => {
        const connectWS = () => {
            socketRef.current = new WebSocket('ws://localhost:8000/ws');
            socketRef.current.onopen = () => setIsConnected(true);
            socketRef.current.onclose = () => {
                setIsConnected(false);
                setTimeout(connectWS, 3000);
            };

            socketRef.current.onmessage = (event) => {
                try {
                    const frameData = JSON.parse(event.data);
                    handleNewFrame(frameData);
                } catch (e) { console.warn("WS Parse Error:", e); }
            };
        };

        connectWS();
        return () => socketRef.current?.close();
    }, []);

    const handleNewFrame = (frame) => {
        setData(prev => [...prev.slice(-200), frame]);
        setActivePersons(frame.persons || []);

        if (frame.interactions?.length > 0) {
            const newLogs = frame.interactions.map(inter => ({
                id: Math.random().toString(36).substr(2, 9),
                time: frame.timestamp?.split(' ')[1]?.split('.')[0] || '00:00:00',
                persons: inter.ids.map(id => `P-${id}`).join(', '),
                type: inter.type,
                raw_ids: inter.ids
            }));
            setLogs(prev => [...newLogs, ...prev].slice(0, 100));
        }

        drawFrame(frame);
    };

    const drawFrame = (frame) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const [imgWidth, imgHeight] = [1280, 720];

        const rect = canvas.parentNode.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        const scaleX = canvas.width / imgWidth;
        const scaleY = canvas.height / imgHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Grid
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.03)';
        ctx.lineWidth = 1;
        for (let i = 0; i < canvas.width; i += 40) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke(); }
        for (let i = 0; i < canvas.height; i += 40) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke(); }

        frame.persons?.forEach(p => {
            const [x1, y1, x2, y2] = p.bbox.map((v, i) => i % 2 === 0 ? v * scaleX : v * scaleY);
            const isSelected = selectedPerson === p.id || (filterType === 'person' && filterValue === p.id);

            // Bounding Box
            ctx.strokeStyle = isSelected ? '#f59e0b' : '#06b6d4';
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Label
            ctx.fillStyle = isSelected ? 'rgba(245, 158, 11, 0.9)' : 'rgba(6, 182, 212, 0.8)';
            ctx.fillRect(x1, y1 - 20, 60, 20);
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 10px Inter';
            ctx.fillText(`ID: ${p.id}`, x1 + 5, y1 - 7);

            // Highlight selection glow
            if (isSelected) {
                ctx.shadowBlur = 15;
                ctx.shadowColor = '#f59e0b';
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.shadowBlur = 0;
            }
        });

        // Interaction lines
        frame.interactions?.forEach(inter => {
            const centers = inter.ids.map(id => {
                const p = frame.persons.find(per => per.id === id);
                if (!p) return null;
                const b = p.bbox;
                return [(b[0] + b[2]) / 2 * scaleX, (b[1] + b[3]) / 2 * scaleY];
            }).filter(c => c !== null);

            if (centers.length >= 2) {
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
                ctx.setLineDash([5, 5]);
                ctx.moveTo(centers[0][0], centers[0][1]);
                ctx.lineTo(centers[1][0], centers[1][1]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });
    };

    // --- Handlers ---

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        setUploadProgress(10);

        const formData = new FormData();
        formData.append('file', file);

        try {
            setUploadProgress(40);
            const res = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            });
            setUploadProgress(80);
            const result = await res.json();
            setUploadProgress(100);
            setTimeout(() => { setIsUploading(false); setUploadProgress(0); }, 1000);
        } catch (err) {
            console.error("Upload Error:", err);
            setIsUploading(false);
        }
    };

    const filteredLogs = useMemo(() => {
        if (!filterType) return logs;
        if (filterType === 'interaction') return logs.filter(l => l.type === filterValue);
        if (filterType === 'person') return logs.filter(l => l.raw_ids.includes(filterValue));
        return logs;
    }, [logs, filterType, filterValue]);

    const emotionData = useMemo(() => {
        return Object.entries(globalStats.emotions_breakdown || {}).map(([name, value]) => ({ name, value }));
    }, [globalStats.emotions_breakdown]);

    const clearFilters = () => {
        setFilterType(null);
        setFilterValue(null);
        setSelectedPerson(null);
    };

    return (
        <div className="v3-dashboard">
            {/* GLOSSY HEADER BAR */}
            <nav className="top-nav">
                <div className="brand" onClick={clearFilters} style={{ cursor: 'pointer' }}>
                    <ShieldCheck className="brand-icon" />
                    <div className="brand-text">
                        <h2>OMNI-V3</h2>
                        <span>INTEGRATED CONTROL ROOM</span>
                    </div>
                </div>

                <div className="nav-center">
                    {isUploading ? (
                        <div className="upload-progress-bar">
                            <div className="progress-fill" style={{ width: `${uploadProgress}%` }}></div>
                            <span>UPLOADING: {uploadProgress}%</span>
                        </div>
                    ) : (
                        <div className="upload-trigger" onClick={() => fileInputRef.current.click()}>
                            <Upload size={16} />
                            <span>UPLOAD NEW VIDEO</span>
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileUpload}
                                style={{ display: 'none' }}
                                accept="video/*"
                            />
                        </div>
                    )}
                </div>

                <div className="nav-stats">
                    <div className={`status-pill ${isConnected ? 'live' : 'offline'}`}>
                        <span className="dot"></span>
                        {isConnected ? 'LIVE ENGINE' : 'ENGINE OFFLINE'}
                    </div>
                </div>
            </nav>

            <div className="control-layout">
                <aside className="v3-sidebar">
                    <div className="sidebar-group">
                        <label>FILTERS</label>
                        {filterType ? (
                            <div className="active-filter-badge">
                                <span>{filterType.toUpperCase()}: {filterValue}</span>
                                <X size={14} onClick={clearFilters} style={{ cursor: 'pointer' }} />
                            </div>
                        ) : <span className="dim-text">No active filters</span>}
                    </div>

                    <div className="sidebar-group">
                        <label>EMOTION DISTRIBUTION</label>
                        <div className="chart-mini">
                            <ResponsiveContainer width="100%" height={150}>
                                <PieChart>
                                    <Pie
                                        data={emotionData}
                                        innerRadius={40}
                                        outerRadius={60}
                                        paddingAngle={5}
                                        dataKey="value"
                                        onClick={(d) => { setFilterType('emotion'); setFilterValue(d.name); }}
                                    >
                                        {emotionData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name] || '#6366f1'} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        contentStyle={{ background: '#0a0d11', border: '1px solid #1a2228', fontSize: '10px' }}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="sidebar-group">
                        <label>ACTIVE PERSONS ({activePersons.length})</label>
                        <div className="person-list">
                            {activePersons.map(p => (
                                <div
                                    key={p.id}
                                    className={`person-card ${selectedPerson === p.id ? 'active' : ''}`}
                                    onClick={() => setSelectedPerson(p.id)}
                                >
                                    <div className="person-avatar">{p.id}</div>
                                    <div className="person-summary">
                                        <span className="p-id">Person #{p.id}</span>
                                        <span className="p-emo">{p.attributes?.emotion || 'Neutral'}</span>
                                    </div>
                                    {selectedPerson === p.id && <div className="p-indicator"></div>}
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>

                <main className="v3-main">
                    {/* INTERACTIVE KPI ROW */}
                    <section className="kpi-grid">
                        <div className="kpi-card-v3 clickable" onClick={() => { setFilterType('person'); setFilterValue('ANY'); }}>
                            <Users className="i-cyan" />
                            <div className="kpi-content">
                                <span className="label">UNIQUE PERSONS</span>
                                <span className="val">{globalStats.unique_persons_count}</span>
                            </div>
                        </div>
                        <div className="kpi-card-v3 clickable" onClick={() => setFilterType('interaction')}>
                            <Zap className="i-amber" />
                            <div className="kpi-content">
                                <span className="label">TOTAL INTERACTIONS</span>
                                <span className="val">{globalStats.total_interactions}</span>
                            </div>
                        </div>
                        <div className="kpi-card-v3">
                            <Hourglass className="i-emerald" />
                            <div className="kpi-content">
                                <span className="label">AVG ENGAGEMENT</span>
                                <span className="val">24s</span>
                            </div>
                        </div>
                        <div className="kpi-card-v3">
                            <Activity className="i-blue" />
                            <div className="kpi-content">
                                <span className="label">FRAME INDEX</span>
                                <span className="val">#{data[data.length - 1]?.frame_idx || 0}</span>
                            </div>
                        </div>
                    </section>

                    <div className="primary-grid">
                        {/* TELEMETRY FEED */}
                        <section className="feed-card-v3">
                            <div className="card-header">
                                <h3><Maximize2 size={14} /> ANALYTICS TELEMETRY</h3>
                                <div className="feed-controls">
                                    <span className="v-tag">1080p</span>
                                    <span className="v-tag">GPU-ACCEL</span>
                                </div>
                            </div>
                            <div className="viewport-v3">
                                <canvas ref={canvasRef}></canvas>
                                <div className="scanner-line"></div>
                            </div>
                        </section>

                        {/* INTERACTIVE AUDIT LOG */}
                        <section className="audit-card-v3">
                            <div className="card-header">
                                <h3><MessageSquare size={14} /> INTERACTION AUDIT</h3>
                                <span className="audit-count">{filteredLogs.length}</span>
                            </div>
                            <div className="audit-list">
                                {filteredLogs.map(log => (
                                    <div
                                        key={log.id}
                                        className="audit-row"
                                        onClick={() => { setFilterType('person'); setFilterValue(log.raw_ids[0]); }}
                                    >
                                        <div className="a-time">{log.time}</div>
                                        <div className="a-content">
                                            <strong>{log.persons}</strong>
                                            <span>{log.type.toUpperCase()}</span>
                                        </div>
                                        <ChevronRight size={14} className="a-icon" />
                                    </div>
                                ))}
                            </div>
                        </section>
                    </div>

                    {/* BOTTOM ANALYTICS FLOOR */}
                    <section className="floor-grid">
                        <div className="floor-card-v3">
                            <div className="card-header">
                                <h3><TrendingUp size={14} /> TEMPORAL DENSITY</h3>
                            </div>
                            <div className="chart-container">
                                <ResponsiveContainer width="100%" height={180}>
                                    <AreaChart data={data.slice(-50)}>
                                        <defs>
                                            <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <Tooltip contentStyle={{ background: '#0a0d11', border: '1px solid #1a2228' }} />
                                        <Area
                                            type="monotone"
                                            dataKey="persons.length"
                                            stroke="#06b6d4"
                                            fillOpacity={1}
                                            fill="url(#colorVal)"
                                            name="Active Persons"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="floor-card-v3">
                            <div className="card-header">
                                <h3><BarChart3 size={14} /> TYPE DISTRIBUTION</h3>
                            </div>
                            <div className="chart-container">
                                <ResponsiveContainer width="100%" height={180}>
                                    <BarChart data={Object.entries(globalStats.interaction_types || {}).map(([name, val]) => ({ name, val }))}>
                                        <XAxis dataKey="name" fontSize={10} axisLine={false} tickLine={false} />
                                        <Tooltip contentStyle={{ background: '#0a0d11', border: '1px solid #1a2228' }} />
                                        <Bar
                                            dataKey="val"
                                            fill="#f59e0b"
                                            radius={[4, 4, 0, 0]}
                                            onClick={(d) => { setFilterType('interaction'); setFilterValue(d.name); }}
                                            style={{ cursor: 'pointer' }}
                                        />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </section>
                </main>
            </div>
        </div>
    );
}

export default App;
