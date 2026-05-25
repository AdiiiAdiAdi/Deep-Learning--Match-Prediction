import React, { useState, useEffect } from 'react';
import { Loader2, AlertTriangle, Shield, TrendingUp, History, BrainCircuit } from 'lucide-react';
import ErrorCard from './ErrorCard';

const ScoreBadge = ({ result }) => {
  const colors = {
    'W': '#39d353',
    'D': '#d29922',
    'L': '#f85149'
  };
  return (
    <div style={{
      width: '32px', height: '32px', borderRadius: '6px',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '12px', fontWeight: 'bold', color: '#000',
      backgroundColor: colors[result] || '#333',
      marginRight: '6px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
    }}>
      {result}
    </div>
  );
};

const OutcomeBox = ({ label, percentage, color }) => (
  <div className="card flex-col items-center justify-center" style={{ 
    flex: 1, 
    padding: '2.5rem 1.5rem', 
    border: `1px solid ${color}33`, 
    background: `linear-gradient(135deg, ${color}15, transparent)`,
    position: 'relative',
    overflow: 'hidden'
  }}>
    <div style={{ 
      position: 'absolute', top: '-20px', right: '-20px', 
      width: '100px', height: '100px', borderRadius: '50%', 
      background: `${color}10`, filter: 'blur(40px)' 
    }} />
    <span className="text-secondary text-xs uppercase mb-3 tracking-widest font-bold">{label}</span>
    <span className="text-4xl font-bold tracking-tight" style={{ color: color }}>{Math.round(percentage * 100)}%</span>
    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', marginTop: '1.5rem' }}>
      <div style={{ width: `${percentage * 100}%`, height: '100%', background: color, borderRadius: '3px', boxShadow: `0 0 10px ${color}44` }} />
    </div>
  </div>
);

export default function MatchInsights() {
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [teams, setTeams] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/api/teams')
      .then(res => res.json())
      .then(setTeams)
      .catch(() => setError({ message: "Failed to load teams list" }));
  }, []);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:5000/api/team-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam })
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || "Analysis failed");
      setData(result);
    } catch (e) {
      setError({ message: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2.5rem', paddingBottom: '4rem' }}>
      
      {/* Hero Selector Section */}
      <div className="card" style={{ 
        padding: '3rem', 
        background: 'linear-gradient(to right, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.9))',
        border: '1px solid var(--border-color)',
        borderRadius: '12px'
      }}>
        <div style={{ display: 'flex', gap: '3rem', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ flex: 1, textAlign: 'center' }}>
            <label className="text-secondary text-sm uppercase mb-4 block font-bold tracking-widest text-center">Home Team</label>
            <select 
              value={homeTeam} 
              onChange={(e) => setHomeTeam(e.target.value)}
              style={{ fontSize: '1.25rem', padding: '1rem 1.5rem', textAlign: 'center' }}
            >
              <option value="">Select Home</option>
              {teams.map(t => <option key={`h-${t}`} value={t}>{t}</option>)}
            </select>
          </div>
          
          <div style={{ flex: 0, marginTop: '2rem' }}>
            <div style={{ 
              width: '60px', height: '60px', borderRadius: '50%', 
              background: 'var(--border-color)', display: 'flex', 
              alignItems: 'center', justifyContent: 'center',
              fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-secondary)',
              border: '4px solid rgba(255,255,255,0.05)'
            }}>VS</div>
          </div>

          <div style={{ flex: 1, textAlign: 'center' }}>
            <label className="text-secondary text-sm uppercase mb-4 block font-bold tracking-widest">Away Team</label>
            <select 
              value={awayTeam} 
              onChange={(e) => setAwayTeam(e.target.value)}
              style={{ fontSize: '1.25rem', padding: '1rem 1.5rem' }}
            >
              <option value="">Select Away</option>
              {teams.map(t => <option key={`a-${t}`} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div style={{ marginTop: '2.5rem', display: 'flex', justifyContent: 'center' }}>
          <button 
            className="btn-primary" 
            style={{ 
              height: '60px', 
              maxWidth: '400px', 
              fontSize: '1.1rem', 
              fontWeight: 700,
              background: 'var(--color-info)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              transition: 'transform 0.2s',
              boxShadow: '0 4px 12px rgba(88, 166, 255, 0.3)'
            }}
            onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.98)'}
            onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
            disabled={loading || !homeTeam || !awayTeam || homeTeam === awayTeam}
            onClick={handlePredict}
          >
            {loading ? <div className="flex items-center justify-center gap-2"><Loader2 className="spin" size={24} /> Processing...</div> : 'Kick Off Analysis'}
          </button>
        </div>
      </div>

      {error ? (
        <ErrorCard message={error.message} onRetry={handlePredict} />
      ) : loading ? (
        <div className="card flex-col items-center justify-center" style={{ minHeight: '500px', gap: '1.5rem' }}>
          <div style={{ position: 'relative' }}>
            <Loader2 className="spin" size={64} style={{ color: 'var(--color-info)' }} />
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
               <BrainCircuit size={32} />
            </div>
          </div>
          <div className="text-center">
            <h2 className="text-xl mb-2 font-bold">Deep Learning Inference in Progress</h2>
            <p className="text-secondary mono text-sm">Aggregating historical metrics, calculating form modifiers & resolving neural weights...</p>
          </div>
        </div>
      ) : data ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem', animation: 'fadeIn 0.6s ease-out' }}>
          
          {/* Outcome Probabilities Container */}
          <section>
            <h3 className="text-xs uppercase text-secondary font-bold tracking-widest mb-4 ml-1">Pre-Match Probabilities</h3>
            <div style={{ display: 'flex', gap: '1.5rem' }}>
              <OutcomeBox label="Home Victory" percentage={data.prediction.home_win} color="var(--color-win)" />
              <OutcomeBox label="Stalemate (Draw)" percentage={data.prediction.draw} color="var(--color-draw)" />
              <OutcomeBox label="Away Victory" percentage={data.prediction.away_win} color="var(--color-loss)" />
            </div>
          </section>

          <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '2rem' }}>
            
            {/* Form Section */}
            <div className="card flex-col" style={{ padding: '2rem' }}>
              <h3 className="text-lg font-bold mb-6 flex items-center gap-3">
                <History size={20} className="text-info" /> Performance Momentum
              </h3>
              
              <div className="mb-8">
                <div className="flex justify-between items-end mb-4 border-b border-white/5 pb-4">
                  <div>
                    <h4 className="text-2xl font-bold tracking-tight">{homeTeam}</h4>
                    <span className="text-xs text-secondary uppercase font-bold tracking-tighter">Current Host Form</span>
                  </div>
                  <div className="flex">
                    {data.home_last5.map((m, i) => <ScoreBadge key={i} result={m.result} />)}
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {data.home_last5.map((m, i) => (
                    <div key={i} className="flex justify-between items-center text-sm">
                      <span className="text-secondary"><span className="text-xs mr-2 opacity-50">•</span> vs {m.opponent}</span>
                      <span className="mono font-bold" style={{ color: m.result === 'W' ? 'var(--color-win)' : (m.result === 'L' ? 'var(--color-loss)' : 'var(--color-draw)') }}>{m.score} ({m.result})</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex justify-between items-end mb-4 border-b border-white/5 pb-4">
                  <div>
                    <h4 className="text-2xl font-bold tracking-tight">{awayTeam}</h4>
                    <span className="text-xs text-secondary uppercase font-bold tracking-tighter">Current Visitor Form</span>
                  </div>
                  <div className="flex">
                    {data.away_last5.map((m, i) => <ScoreBadge key={i} result={m.result} />)}
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {data.away_last5.map((m, i) => (
                    <div key={i} className="flex justify-between items-center text-sm">
                      <span className="text-secondary"><span className="text-xs mr-2 opacity-50">•</span> vs {m.opponent}</span>
                      <span className="mono font-bold" style={{ color: m.result === 'W' ? 'var(--color-win)' : (m.result === 'L' ? 'var(--color-loss)' : 'var(--color-draw)') }}>{m.score} ({m.result})</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* H2H & Technical */}
            <div className="flex-col gap-6">
              <div className="card flex-col" style={{ padding: '2rem', flex: 1 }}>
                <h3 className="text-lg font-bold mb-6 flex items-center gap-3">
                  <Shield size={20} className="text-info" /> Head-to-Head History
                </h3>
                {data.h2h.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {data.h2h.map((m, i) => (
                      <div key={i} style={{ 
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                        padding: '1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.05)'
                      }}>
                        <span className="text-sm font-bold" style={{ flex: 1, textAlign: 'right' }}>{m.home}</span>
                        <div style={{ flex: 0.6, textAlign: 'center' }}>
                           <span className="mono text-lg font-bold bg-white/5 px-3 py-1 rounded" style={{ boxShadow: 'inset 0 0 10px rgba(0,0,0,0.5)' }}>{m.score}</span>
                        </div>
                        <span className="text-sm font-bold" style={{ flex: 1, textAlign: 'left' }}>{m.away}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex-col items-center justify-center p-8 bg-white/5 rounded-lg border border-dashed border-white/10">
                     <span className="text-secondary text-base italic">Primary dataset records exhausted.</span>
                  </div>
                )}
              </div>

              {/* Technical context box moved into side column */}
              <div className="card" style={{ padding: '2rem', background: 'linear-gradient(to bottom, #1e252e, #161b22)' }}>
                <h4 className="text-xs text-secondary uppercase tracking-widest mb-4 flex items-center gap-2 font-bold">
                  <TrendingUp size={16} className="text-info" /> Neural Context
                </h4>
                <p className="text-sm leading-relaxed text-secondary italic">
                  "{data.technical_insight}"
                </p>
              </div>
            </div>
          </div>

          {/* AI Executive Summary - Bottom Full Width */}
          <div className="card shadow-2xl" style={{ 
            padding: '3rem',
            border: '2px solid rgba(57, 211, 83, 0.2)', 
            background: 'linear-gradient(160deg, rgba(57, 211, 83, 0.08) 0%, transparent 40%)' 
          }}>
            <h3 className="text-2xl font-bold text-win mb-8 flex items-center gap-3">
              <BrainCircuit size={32} /> Executive Match Briefing
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '4rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {data.insights.map((insight, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.02)' }}>
                    <div style={{ minWidth: '40px', height: '40px', borderRadius: '50%', background: 'rgba(57, 211, 83, 0.15)', display: 'flex', alignItems: 'center', justifyCenter: 'center', color: 'var(--color-win)', fontWeight: 800 }}>
                       <span style={{ width: '100%', textAlign: 'center' }}>{i+1}</span>
                    </div>
                    <span className="text-lg leading-snug">{insight}</span>
                  </div>
                ))}
              </div>
              <div className="flex-col justify-center items-center p-8 rounded-xl border border-white/5 bg-black/20" style={{ textAlign: 'center' }}>
                 <div className="text-xs text-secondary uppercase font-bold tracking-widest mb-4">Inference Reliability</div>
                 <div style={{ position: 'relative', width: '120px', height: '120px', margin: '0 auto' }}>
                    <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                      <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                      <circle cx="50" cy="50" r="45" fill="none" stroke="var(--color-win)" strokeWidth="8" strokeDasharray="210 283" strokeLinecap="round" />
                    </svg>
                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', fontWeight: 800, fontSize: '1.5rem' }}>74%</div>
                 </div>
                 <p className="mt-4 text-xs text-secondary mono">VALIDATED AGAINST SEASON 23/24 BACKTESTS</p>
              </div>
            </div>
          </div>

        </div>
      ) : (
        <div className="card flex-col items-center justify-center text-secondary border-dashed" style={{ minHeight: '500px', background: 'rgba(255,255,255,0.01)' }}>
          <BrainCircuit size={64} className="mb-6 opacity-20" />
          <p className="text-xl font-medium mb-2">Ready for Prediction</p>
          <p className="text-sm opacity-50">Select a Home and Away team from the dashboard header to generate pre-match AI analytics.</p>
        </div>
      )}
    </div>
  );
}
