import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
    Users, Zap, Activity, MessageSquare, ShieldCheck, Upload, X,
    Eye, TrendingUp, BarChart3, Navigation, Brain, Loader, ChevronRight, Star
} from 'lucide-react';
import {
    AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
    XAxis, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import './App.css';

const API = 'http://localhost:8000';

const EMO_COLORS = {
    happy: '#10b981', neutral: '#64748b', sad: '#3b82f6',
    angry: '#ef4444', surprise: '#f59e0b', disgust: '#8b5cf6', fear: '#ec4899'
};
const INT_COLORS = {
    'Talking': '#06b6d4', 'Walking Together': '#10b981', 'Group_Bond': '#f59e0b',
    'Approaching': '#3b82f6', 'Physical Contact': '#ef4444', 'Service/Helping': '#8b5cf6'
};

const POSE_CONNECTIONS = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
];

function drawOverlay(canvas, frame, selectedPerson) {
    if (!canvas || !frame) return;
    const ctx = canvas.getContext('2d');
    const container = canvas.parentNode?.getBoundingClientRect();
    if (!container) return;

    if (canvas.width !== Math.round(container.width)) canvas.width = Math.round(container.width);
    if (canvas.height !== Math.round(container.height)) canvas.height = Math.round(container.height);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const srcW = 1920, srcH = 1080;
    const cw = canvas.width, ch = canvas.height;
    const cAspect = cw / ch, sAspect = srcW / srcH;
    let rW, rH, ox, oy;
    if (cAspect > sAspect) { rH = ch; rW = rH * sAspect; ox = (cw - rW) / 2; oy = 0; }
    else { rW = cw; rH = rW / sAspect; ox = 0; oy = (ch - rH) / 2; }
    const tx = v => ox + (v / srcW) * rW;
    const ty = v => oy + (v / srcH) * rH;

    const bgrToRgb = (bgr) => `rgb(${bgr[2]}, ${bgr[1]}, ${bgr[0]})`;

    (frame.persons || []).forEach(p => {
        const [bx1, by1, bx2, by2] = p.bbox;
        const cx1 = tx(bx1), cy1 = ty(by1), cx2 = tx(bx2), cy2 = ty(by2);
        const bw = cx2 - cx1, bh = cy2 - cy1;

        const isSelected = selectedPerson === p.id;
        const colorBGR = p.color || [0, 255, 0];
        const color = bgrToRgb(colorBGR);

        // 1. Human Bounding Box (cv2.rectangle style)
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 4 : 2;
        ctx.strokeRect(cx1, cy1, bw, bh);

        // 2. ID Label (cv2.putText style)
        ctx.fillStyle = color;
        ctx.font = `600 ${isSelected ? 16 : 13}px "Inter", sans-serif`;
        ctx.fillText(`ID:${p.id}`, cx1, cy1 - 5);

        // 3. Skeleton (replicate cv2.line and cv2.circle)
        if (p.pose_keypoints) {
            const kpts = p.pose_keypoints;
            // Bones
            ctx.strokeStyle = '#ffff00'; // Yellow lines
            ctx.lineWidth = 2;
            POSE_CONNECTIONS.forEach(([s, e]) => {
                if (kpts[s] && kpts[e] && kpts[s][2] > 0.4 && kpts[e][2] > 0.4) {
                    ctx.beginPath();
                    ctx.moveTo(tx(kpts[s][0]), ty(kpts[s][1]));
                    ctx.lineTo(tx(kpts[e][0]), ty(kpts[e][1]));
                    ctx.stroke();
                }
            });
            // Joints
            kpts.forEach((kp, i) => {
                if (i >= 5 && kp[2] > 0.4) {
                    const kpx = tx(kp[0]), kpy = ty(kp[1]);
                    ctx.fillStyle = '#ff0000'; // Red fill
                    ctx.beginPath(); ctx.arc(kpx, kpy, 4, 0, Math.PI * 2); ctx.fill();
                    ctx.strokeStyle = '#ffffff'; // White border
                    ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(kpx, kpy, 5, 0, Math.PI * 2); ctx.stroke();
                }
            });
        }

        // 4. Faces (Blue boxes)
        if (p.faces) {
            p.faces.forEach(f => {
                const [fx1, fy1, fx2, fy2] = f.bbox;
                const fcx1 = tx(fx1), fcy1 = ty(fy1), fcx2 = tx(fx2), fcy2 = ty(fy2);
                ctx.strokeStyle = '#0000ff';
                ctx.lineWidth = 2;
                ctx.strokeRect(fcx1, fcy1, fcx2 - fcx1, fcy2 - fcy1);
            });
        }

        // 5. Retail Badges (replicate run_full_pipeline.py draw_badge style)
        let by = cy1 - 25;
        const badges = [];
        const attrs = p.attributes || {};
        if (attrs.role) badges.push({ text: `ROLE: ${attrs.role}`, bg: attrs.role.includes('Staff') ? 'rgb(255, 100, 0)' : 'rgb(100, 100, 100)' });
        if (attrs.intent) badges.push({ text: `INTENT: ${attrs.intent}`, bg: attrs.intent !== 'Normal' ? 'rgb(0, 0, 255)' : 'rgb(50, 50, 50)' });
        if (attrs.emotion) badges.push({ text: `EMO: ${attrs.emotion} ${attrs.mood_trend || ''}`, bg: EMO_COLORS[attrs.emotion.toLowerCase()] || 'rgb(100, 100, 100)' });
        if (attrs.gender) badges.push({ text: `${attrs.gender} | Age: ${attrs.age || 'N/A'}`, bg: 'rgb(50, 50, 50)' });

        ctx.font = '700 11px "Inter", sans-serif';
        badges.forEach(b => {
            const tw = ctx.measureText(b.text).width;
            ctx.fillStyle = b.bg;
            ctx.fillRect(cx1, by - 18, tw + 10, 15);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(b.text, cx1 + 5, by - 7);
            by -= 18;
        });
    });

    // 6. Interaction Lines (Green dashed lines)
    (frame.interactions || []).forEach(inter => {
        const pts = inter.ids.map(id => {
            const p = (frame.persons || []).find(x => x.id === id);
            return p ? [tx((p.bbox[0] + p.bbox[2]) / 2), ty((p.bbox[1] + p.bbox[3]) / 2)] : null;
        }).filter(Boolean);
        if (pts.length < 2) return;

        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        ctx.lineTo(pts[1][0], pts[1][1]);
        ctx.stroke();

        const mx = (pts[0][0] + pts[1][0]) / 2, my = (pts[0][1] + pts[1][1]) / 2;
        ctx.fillStyle = '#00ff00';
        ctx.font = '700 16px "Inter", sans-serif';
        ctx.fillText(`<< ${inter.type} >>`, mx - 50, my);
    });
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
    const [data, setData] = useState([]);
    const [activePersons, setActivePersons] = useState([]);
    const [logs, setLogs] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [pipeline, setPipeline] = useState({ status: 'idle', progress: 0, total_frames: 0 });
    const [selectedPerson, setSelectedPerson] = useState(null);
    const [filterType, setFilterType] = useState(null);
    const [filterValue, setFilterValue] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [uploadedVideoUrl, setUploadedVideoUrl] = useState(null);
    const [outputVideoUrl, setOutputVideoUrl] = useState(null);
    const [stats, setStats] = useState({
        total_frames: 0, unique_persons_count: 0, emotions_breakdown: {},
        total_interactions: 0, interaction_types: {}, roles_breakdown: {},
        intents_breakdown: {}, avg_persons_per_frame: 0,
        avg_dwell_time_seconds: 0, engagement_score: 0, zone_activity: {}
    });

    const processPct = pipeline.total_frames > 0 ? Math.round(pipeline.progress / pipeline.total_frames * 100) : 0;
    const isProcessing = pipeline.status === 'processing';

    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const latestFrame = useRef(null);
    const animRef = useRef(null);

    const fetchStats = useCallback(() => {
        fetch(`${API}/stats/summary`).then(r => r.json()).then(setStats).catch(() => { });
    }, []);

    useEffect(() => {
        fetch(`${API}/reset`, { method: 'POST' }).catch(() => { });
        fetchStats();
        const intervalTime = isProcessing ? 2000 : 5000;
        const iv = setInterval(fetchStats, intervalTime);
        return () => clearInterval(iv);
    }, [fetchStats, isProcessing]);

    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket('ws://localhost:8000/ws');
            const ref = { current: ws };
            ws.onopen = () => setIsConnected(true);
            ws.onclose = () => { setIsConnected(false); setTimeout(connect, 2000); };
            ws.onmessage = e => {
                try {
                    const d = JSON.parse(e.data);
                    if (d.__type === 'status') {
                        setPipeline(d);
                        if (d.status === 'done') {
                            setOutputVideoUrl(prev => {
                                if (!prev) {
                                    fetchStats();
                                    return `${API}/output-video?t=${Date.now()}`;
                                }
                                return prev;
                            });
                        } else if (d.status === 'processing') {
                            // Only restore preview if we don't already have one AND we aren't done yet
                            setUploadedVideoUrl(prev => {
                                if (!prev && d.video_url && !outputVideoUrl) return `${API}${d.video_url}`;
                                return prev;
                            });
                        }
                    } else if (d.frame_idx !== undefined) {
                        latestFrame.current = d;
                        setData(prev => [...prev.slice(-200), d]);
                        setActivePersons(d.persons || []);
                        if (d.interactions?.length) {
                            const newLogs = d.interactions.map(i => ({
                                id: Math.random().toString(36).substr(2, 7),
                                time: d.timestamp?.split(' ')[1]?.split('.')[0] || '--:--:--',
                                persons: i.ids.map(x => `P${x}`).join(' + '),
                                type: i.type, raw_ids: i.ids
                            }));
                            setLogs(prev => [...newLogs, ...prev].slice(0, 100));
                        }
                    }
                } catch { }
            };
            return () => ws.close();
        };
        const cleanup = connect();
        return cleanup;
    }, [fetchStats]);

    // Render loop — canvas updated at 30fps from latest frame data
    useEffect(() => {
        const render = () => {
            if (latestFrame.current && canvasRef.current && !outputVideoUrl) {
                drawOverlay(canvasRef.current, latestFrame.current, selectedPerson);
            }
            animRef.current = requestAnimationFrame(render);
        };
        animRef.current = requestAnimationFrame(render);
        return () => cancelAnimationFrame(animRef.current);
    }, [selectedPerson, outputVideoUrl]);

    const handleFile = useCallback(async file => {
        if (!file?.type.startsWith('video/')) return;
        setIsUploading(true);
        setOutputVideoUrl(null);
        setUploadedVideoUrl(URL.createObjectURL(file));

        // Clear all old dashboard data immediately
        setData([]);
        setLogs([]);
        setActivePersons([]);
        setStats({
            total_frames: 0, unique_persons_count: 0, emotions_breakdown: {},
            total_interactions: 0, interaction_types: {}, roles_breakdown: {},
            intents_breakdown: {}, avg_persons_per_frame: 0,
            avg_dwell_time_seconds: 0, engagement_score: 0, zone_activity: {}
        });

        setUploadProgress(10);
        const fd = new FormData(); fd.append('file', file);
        try {
            setUploadProgress(40);
            const res = await fetch(`${API}/upload`, { method: 'POST', body: fd });
            setUploadProgress(90);
            await res.json();
            setUploadProgress(100);
            setTimeout(() => { setIsUploading(false); setUploadProgress(0); }, 600);
        } catch { setIsUploading(false); alert('Backend offline?'); }
    }, []);

    const clearFilters = () => { setFilterType(null); setFilterValue(null); setSelectedPerson(null); };


    const emotionData = useMemo(() => Object.entries(stats.emotions_breakdown || {}).map(([n, v]) => ({ name: n, value: v })), [stats]);
    const interData = useMemo(() => Object.entries(stats.interaction_types || {}).map(([n, v]) => ({ name: n, value: v })), [stats]);
    const rolesData = useMemo(() => Object.entries(stats.roles_breakdown || {}).map(([n, v]) => ({ name: n.split(' ')[0], value: v })), [stats]);
    const zonesData = useMemo(() => Object.entries(stats.zone_activity || {}).map(([n, v]) => ({ name: n, value: v })), [stats]);
    const densityData = useMemo(() => data.slice(-60).map((d, i) => ({ i, count: d.persons?.length || 0 })), [data]);

    const filteredLogs = useMemo(() => {
        if (!filterType) return logs;
        if (filterType === 'interaction') return logs.filter(l => l.type === filterValue);
        if (filterType === 'person') return logs.filter(l => l.raw_ids.includes(filterValue));
        return logs;
    }, [logs, filterType, filterValue]);

    const TOOLTIP_STYLE = { background: '#060810', border: '1px solid rgba(255,255,255,0.08)', fontSize: 11, borderRadius: 8, color: '#e2e8f0' };

    return (
        <div className="app" onDrop={e => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); }}
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)}>

            {isDragging && <div className="drop-zone"><Upload size={52} /><p>DROP VIDEO TO ANALYZE</p></div>}

            {/* ── HEADER ── */}
            <header className="header">
                <div className="logo" onClick={clearFilters}>
                    <div className="logo-icon"><Activity size={20} /></div>
                    <div>
                        <div className="logo-title">VISIONINTEL PRO</div>
                        <div className="logo-sub">Advanced Video Analysis Platform</div>
                    </div>
                </div>

                <div className="header-center">
                    {isUploading ? (
                        <div className="upload-bar"><div className="upload-fill" style={{ width: `${uploadProgress}%` }} /><span>{uploadProgress}%</span></div>
                    ) : (
                        <button className="upload-btn" onClick={() => fileInputRef.current.click()}>
                            <Upload size={14} /> NEW ANALYSIS
                            <input type="file" ref={fileInputRef} onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} accept="video/*" />
                        </button>
                    )}
                </div>

                <div className="header-right">
                    {isProcessing && (
                        <div className="pill amber"><Loader size={11} className="spin" />{processPct}% PROCESSING</div>
                    )}
                    <div className={`pill ${isConnected ? 'green' : 'gray'}`}>
                        <span className="dot" />{isConnected ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
                    </div>
                </div>
            </header>

            <div className="body">
                {/* ── SIDEBAR ── */}
                <aside className="sidebar">
                    {/* Mini KPIs */}
                    <div className="mini-kpis">
                        {[
                            { icon: <Users size={14} />, label: 'VISITORS', val: stats.unique_persons_count, color: 'cyan' },
                            { icon: <Activity size={14} />, label: 'AVG DWELL', val: `${stats.avg_dwell_time_seconds || 0}s`, color: 'emerald' },
                            { icon: <Zap size={14} />, label: 'ENGAGEMENT', val: `${stats.engagement_score || 0}%`, color: 'amber' },
                            { icon: <MessageSquare size={14} />, label: 'INTERACTIONS', val: stats.total_interactions, color: 'purple' },
                        ].map(k => (
                            <div key={k.label} className={`mini-kpi ${k.color}`}>
                                {k.icon}<div><div className="mini-label">{k.label}</div><div className="mini-val">{k.val}</div></div>
                            </div>
                        ))}
                    </div>

                    <div className="sidebar-section">
                        <div className="section-title"><Eye size={12} /> EMOTION SPLIT</div>
                        <ResponsiveContainer width="100%" height={130}>
                            <PieChart><Pie data={emotionData} innerRadius={35} outerRadius={55} paddingAngle={3} dataKey="value" onClick={d => { setFilterType('emotion'); setFilterValue(d.name); }}>
                                {emotionData.map((e, i) => <Cell key={i} fill={EMO_COLORS[e.name] || '#6366f1'} />)}
                            </Pie><Tooltip contentStyle={TOOLTIP_STYLE} /></PieChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="sidebar-section">
                        <div className="section-title"><Users size={12} /> ROLE SPLIT</div>
                        <ResponsiveContainer width="100%" height={100}>
                            <PieChart><Pie data={rolesData} innerRadius={28} outerRadius={44} paddingAngle={3} dataKey="value">
                                {rolesData.map((e, i) => <Cell key={i} fill={e.name.includes('Staff') ? '#10b981' : '#3b82f6'} />)}
                            </Pie><Tooltip contentStyle={TOOLTIP_STYLE} /><Legend iconSize={8} wrapperStyle={{ fontSize: 9, color: '#64748b' }} /></PieChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="sidebar-section">
                        <div className="section-title"><Users size={12} /> ACTIVE ({activePersons.length})</div>
                        <div className="person-list">
                            {activePersons.map(p => {
                                const isStaff = (p.attributes?.role || '').includes('Staff');
                                return (
                                    <div key={p.id} className={`person-chip ${selectedPerson === p.id ? 'selected' : ''}`}
                                        onClick={() => setSelectedPerson(p.id)}>
                                        <div className={`chip-dot ${isStaff ? 'green' : 'cyan'}`} />
                                        <div className="chip-info">
                                            <span className="chip-id">P-{p.id} {isStaff ? '(Staff)' : ''}</span>
                                            <span className="chip-emo">{p.attributes?.emotion || 'neutral'}</span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </aside>

                {/* ── MAIN CONTENT ── */}
                <main className="main">
                    {/* Video + Canvas */}
                    <div className="video-panel">
                        <div className="panel-header">
                            <span className="panel-title"><Eye size={13} /> TELEMETRY FEED</span>
                            <div className="panel-tags">
                                {pipeline.status === 'done' && <span className="tag green">PROCESSED ✓</span>}
                                {isProcessing && <span className="tag amber">PROCESSING {processPct}%</span>}
                                <span className="tag">HD · GPU</span>
                            </div>
                        </div>
                        <div className="viewport">
                            {/* Final output video */}
                            {outputVideoUrl && (
                                <video key={outputVideoUrl} src={outputVideoUrl} autoPlay muted loop controls
                                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 3, background: '#000' }} />
                            )}
                            {/* Live source video */}
                            {!outputVideoUrl && uploadedVideoUrl && (
                                <video key={uploadedVideoUrl} src={uploadedVideoUrl} autoPlay muted loop
                                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 1, background: '#000' }} />
                            )}
                            {/* Canvas overlay */}
                            {!outputVideoUrl && (
                                <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 2, pointerEvents: 'none' }} />
                            )}
                            {/* Processing overlay */}
                            {isProcessing && (
                                <div className="processing-overlay">
                                    <div className="hud-scanner" />
                                    <div className="hud-corners" />
                                    <div className="hud-data top-left"><Activity size={12} /> AI SENSORS: ONLINE</div>
                                    <div className="hud-data top-right"><Zap size={12} /> TRACKING: {processPct}%</div>
                                    <div className="hud-data bottom-left"><Brain size={12} /> RETAIL INTELLIGENCE v4.0</div>
                                    <div className="hud-data bottom-right"><ShieldCheck size={12} /> GDPR COMPLIANT</div>

                                    <div className="hud-center" style={{ top: 'auto', bottom: '15%', height: 'auto', gap: '0.4rem' }}>
                                        <div className="hud-title" style={{ fontSize: '0.9rem', letterSpacing: '4px' }}>ANALYZING SHOPPER BEHAVIOR...</div>
                                        <div className="prog-track hud-track" style={{ width: '400px' }}><div className="prog-fill" style={{ width: `${processPct}%` }} /></div>
                                        <div className="hud-pct" style={{ fontSize: '0.6rem' }}>COMPUTING RETAIL INTELLIGENCE ({processPct}%)</div>
                                    </div>
                                </div>
                            )}
                            {/* Insights Overlay (Final Summary) */}
                            {pipeline.status === 'done' && (
                                <div className="insights-overlay">
                                    {(() => {
                                        const topZone = Object.entries(stats.zone_activity || {}).sort((a, b) => b[1] - a[1])[0];
                                        const posEmos = (stats.emotions_breakdown?.happy || 0) + (stats.emotions_breakdown?.surprise || 0);
                                        const totalEmos = Object.values(stats.emotions_breakdown || {}).reduce((a, b) => a + b, 0);
                                        const sentiment = totalEmos ? Math.round((posEmos / totalEmos) * 100) : 0;

                                        return (
                                            <>
                                                <div className="insight-card">
                                                    <div className="insight-icon"><Navigation size={18} /></div>
                                                    <div className="insight-content">
                                                        <span className="insight-label">Top Conversion Zone</span>
                                                        <span className="insight-value">{topZone ? topZone[0] : 'Scanning...'}</span>
                                                    </div>
                                                </div>
                                                <div className="insight-card">
                                                    <div className="insight-icon"><TrendingUp size={18} /></div>
                                                    <div className="insight-content">
                                                        <span className="insight-label">Shopper Sentiment</span>
                                                        <span className="insight-value">{sentiment > 50 ? 'Positive' : 'Balanced'} ({sentiment}%)</span>
                                                    </div>
                                                </div>
                                                <div className="insight-card">
                                                    <div className="insight-icon"><Star size={18} /></div>
                                                    <div className="insight-content">
                                                        <span className="insight-label">Engagement Level</span>
                                                        <span className="insight-value">{stats.engagement_score > 70 ? 'PREMIUM' : 'OPTIMAL'}</span>
                                                    </div>
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>
                            )}

                            {/* Idle state */}
                            {!uploadedVideoUrl && !outputVideoUrl && !isProcessing && (
                                <div className="idle-state" onClick={() => fileInputRef.current.click()}>
                                    <Upload size={36} opacity={0.4} />
                                    <p>Upload or drop a video to begin</p>
                                </div>
                            )}
                            {!outputVideoUrl && <div className="scan-line" />}
                        </div>
                    </div>

                    {/* Interaction Audit */}
                    <div className="audit-panel">
                        <div className="panel-header">
                            <span className="panel-title"><MessageSquare size={13} /> INTERACTIONS</span>
                            <span className="audit-count">{filteredLogs.length}</span>
                        </div>
                        <div className="audit-list">
                            {filteredLogs.length === 0 && <div className="empty-state">No interactions yet</div>}
                            {filteredLogs.map(log => (
                                <div key={log.id} className="audit-row"
                                    onClick={() => { setFilterType('person'); setFilterValue(log.raw_ids[0]); }}>
                                    <span className="audit-time">{log.time}</span>
                                    <div className="audit-body">
                                        <strong>{log.persons}</strong>
                                        <span style={{ color: INT_COLORS[log.type] || '#94a3b8', fontSize: 10, fontWeight: 800, letterSpacing: 1 }}>
                                            {log.type.toUpperCase()}
                                        </span>
                                    </div>
                                    <ChevronRight size={12} opacity={0.4} />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Bottom Chart Grid */}
                    <div className="chart-grid">
                        {[
                            {
                                title: 'CUSTOMER FLOW', icon: <Activity size={12} />, chart: (
                                    <AreaChart data={densityData}>
                                        <defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.25} />
                                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                        </linearGradient></defs>
                                        <XAxis dataKey="i" hide />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Area type="monotone" dataKey="count" stroke="#06b6d4" strokeWidth={2} fill="url(#ag)" name="Customers" />
                                    </AreaChart>
                                )
                            },
                            {
                                title: 'ZONE ACTIVITY', icon: <Navigation size={12} />, chart: (
                                    <BarChart data={zonesData}>
                                        <XAxis dataKey="name" fontSize={8} axisLine={false} tickLine={false} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                )
                            },
                            {
                                title: 'BEHAVIORAL ACTIONS', icon: <BarChart3 size={12} />, chart: (
                                    <BarChart data={interData}>
                                        <XAxis dataKey="name" fontSize={8} axisLine={false} tickLine={false} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="value" radius={[4, 4, 0, 0]} onClick={d => { setFilterType('interaction'); setFilterValue(d.name); }}>
                                            {interData.map((e, i) => <Cell key={i} fill={INT_COLORS[e.name] || '#f59e0b'} />)}
                                        </Bar>
                                    </BarChart>
                                )
                            },
                            {
                                title: 'SHOPPER EMOTION', icon: <TrendingUp size={12} />, chart: (
                                    <BarChart data={emotionData}>
                                        <XAxis dataKey="name" fontSize={8} axisLine={false} tickLine={false} />
                                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                            {emotionData.map((e, i) => <Cell key={i} fill={EMO_COLORS[e.name] || '#6366f1'} />)}
                                        </Bar>
                                    </BarChart>
                                )
                            },
                        ].map(({ title, icon, chart }) => (
                            <div key={title} className="chart-card">
                                <div className="panel-header"><span className="panel-title">{icon} {title}</span></div>
                                <div className="chart-body">
                                    <ResponsiveContainer width="100%" height={140}>{chart}</ResponsiveContainer>
                                </div>
                            </div>
                        ))}
                    </div>
                </main>
            </div>
        </div>
    );
}
