import React, { useState } from 'react';

const teams = [
  "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
  "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", 
  "Liverpool", "Luton Town", "Manchester City", "Manchester United", 
  "Newcastle United", "Nottingham Forest", "Sheffield United", 
  "Tottenham", "West Ham", "Wolverhampton Wanderers",
  "Leicester", "Leeds United", "Southampton"
];

export default function OngoingMatch() {
  const [data, setData] = useState({
    home_team: "",
    away_team: "",
    minute: 45,
    home_goals: 0,
    away_goals: 0,
    home_possession: 50,
    shots_home: 3,
    shots_away: 3,
    home_form: "average",
    away_form: "average"
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (key, value) => {
    setData(prev => ({ ...prev, [key]: value }));
  };

  const predict = async () => {
    setError(null);
    setResult(null);
    try {
      const res = await fetch("http://localhost:5000/api/live-predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
      });
      
      const json = await res.json();
      if (!res.ok) {
         setError(json.error || "Prediction failed");
      } else {
         setResult(json);
      }
    } catch (err) {
      setError("Failed to connect to server");
    }
  };

  return (
    <div className="ongoing-match-container">
      <h2>⚡ Live Match Simulator</h2>

      <div className="global-controls">
        <label>
          Match Minute
          <input type="range" min="0" max="120" value={data.minute} onChange={e => handleChange("minute", parseFloat(e.target.value))} />
          <span className="minute-display">{data.minute}'</span>
        </label>
        
        <label>
          Home Possession: {data.home_possession}%
          <input type="range" min="0" max="100"
            value={data.home_possession}
            onChange={e => handleChange("home_possession", parseFloat(e.target.value))}
          />
        </label>
      </div>

      <div className="teams-split-panel">
        {/* HOME TEAM PANEL */}
        <div className="team-panel home-panel">
          <h3>Home Team</h3>
          
          <select value={data.home_team} onChange={e => handleChange("home_team", e.target.value)}>
            <option value="">Select Home Team</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>

          <div className="form-group">
            <label>Recent Form</label>
            <select value={data.home_form} onChange={e => handleChange("home_form", e.target.value)}>
              <option value="good">Good</option>
              <option value="average">Average</option>
              <option value="bad">Bad</option>
            </select>
          </div>

          <div className="form-group">
            <label>Goals Scored</label>
            <div className="goal-counter">
              <button onClick={() => handleChange("home_goals", Math.max(0, data.home_goals - 1))}>-</button>
              <span>{Math.floor(data.home_goals)}</span>
              <button onClick={() => handleChange("home_goals", data.home_goals + 1)}>+</button>
            </div>
          </div>

          <div className="form-group">
            <label>Total Shots</label>
            <input type="number" min="0" value={data.shots_home} onChange={e => handleChange("shots_home", parseFloat(e.target.value))} />
          </div>
        </div>

        <div className="vs-divider">VS</div>

        {/* AWAY TEAM PANEL */}
        <div className="team-panel away-panel">
          <h3>Away Team</h3>
          
          <select value={data.away_team} onChange={e => handleChange("away_team", e.target.value)}>
            <option value="">Select Away Team</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>

          <div className="form-group">
            <label>Recent Form</label>
            <select value={data.away_form} onChange={e => handleChange("away_form", e.target.value)}>
              <option value="good">Good</option>
              <option value="average">Average</option>
              <option value="bad">Bad</option>
            </select>
          </div>

          <div className="form-group">
            <label>Goals Scored</label>
            <div className="goal-counter">
              <button onClick={() => handleChange("away_goals", Math.max(0, data.away_goals - 1))}>-</button>
              <span>{Math.floor(data.away_goals)}</span>
              <button onClick={() => handleChange("away_goals", data.away_goals + 1)}>+</button>
            </div>
          </div>

          <div className="form-group">
            <label>Total Shots</label>
            <input type="number" min="0" value={data.shots_away} onChange={e => handleChange("shots_away", parseFloat(e.target.value))} />
          </div>
        </div>
      </div>

      <button className="predict-btn" onClick={predict} disabled={!data.home_team || !data.away_team}>
        Simulate Outcome
      </button>

      {error && <div className="error-message">⚠️ {error}</div>}

      {result && (
        <div className="result-card">
          <h3>Predicted Outcome: {result.prediction}</h3>
          <p className="confidence">Confidence: {(result.confidence*100).toFixed(1)}%</p>
          <div className="probabilities">
             <div className="prob-item">
               <span>Home Win</span>
               <strong>{(result.home_win_prob * 100).toFixed(1)}%</strong>
             </div>
             <div className="prob-item draw-prob">
               <span>Draw</span>
               <strong>{(result.draw_prob * 100).toFixed(1)}%</strong>
             </div>
             <div className="prob-item">
               <span>Away Win</span>
               <strong>{(result.away_win_prob * 100).toFixed(1)}%</strong>
             </div>
          </div>
        </div>
      )}
      
      <style>{`
        .ongoing-match-container {
          background: #111; border: 1px solid #333; padding: 2rem; border-radius: 12px; color: #eee;
          max-width: 800px; margin: 0 auto;
        }
        .ongoing-match-container h2 { margin-top: 0; color: #39d353; border-bottom: 1px solid #333; padding-bottom: 1rem; text-align: center;}
        
        .global-controls {
          display: flex; gap: 2rem; background: #1a1a1a; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; justify-content: space-around;
        }
        .global-controls label { display: flex; flex-direction: column; width: 40%; font-size: 0.9rem; color: #ccc;}
        .minute-display { font-size: 1.5rem; color: #39d353; font-weight: bold; text-align: center; margin-top: 0.5rem;}
        
        .teams-split-panel {
          display: flex; align-items: stretch; justify-content: space-between; gap: 1rem; position: relative;
        }
        .team-panel {
          flex: 1; background: #1a1a1a; padding: 1.5rem; border-radius: 8px; display: flex; flex-direction: column; gap: 1rem;
        }
        .team-panel h3 { margin: 0; text-align: center; border-bottom: 1px solid #333; padding-bottom: 0.5rem; color: #fff;}
        .home-panel { border-top: 4px solid #2563eb; }
        .away-panel { border-top: 4px solid #dc2626; }
        
        .vs-divider {
          display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.5rem; color: #555;
        }

        .team-panel select, .team-panel input {
          width: 100%; background: #222; border: 1px solid #444; color: white; padding: 0.75rem; border-radius: 6px; font-size: 1rem;
        }
        
        .form-group { display: flex; flex-direction: column; gap: 0.3rem; }
        .form-group label { font-size: 0.85rem; color: #aaa; }
        
        .goal-counter { display: flex; align-items: center; justify-content: space-between; background: #222; border-radius: 6px; padding: 0.25rem;}
        .goal-counter button { background: #333; color: white; border: none; width: 40px; height: 40px; border-radius: 4px; font-size: 1.2rem; cursor: pointer;}
        .goal-counter button:hover { background: #444; }
        .goal-counter span { font-size: 1.5rem; font-weight: bold; }

        .predict-btn { 
          width: 100%; margin-top: 2rem; padding: 1rem; background: #39d353; color: black; font-size: 1.2rem; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;
        }
        .predict-btn:hover:not(:disabled) { background: #2ea043; }
        .predict-btn:disabled { background: #333; color: #888; cursor: not-allowed; }

        .error-message { background: #fee2e2; color: #dc2626; padding: 1rem; border-radius: 6px; margin-top: 1.5rem; text-align: center; font-weight: bold;}

        .result-card { margin-top: 2rem; padding: 1.5rem; background: rgba(57, 211, 83, 0.1); border: 1px solid #39d353; border-radius: 8px; text-align: center;}
        .result-card h3 { margin: 0; color: #39d353; font-size: 1.5rem;}
        .confidence { color: #aaa; margin-top: 0.5rem;}
        .probabilities { display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; }
        .prob-item { display: flex; flex-direction: column; align-items: center; background: #222; padding: 1rem; border-radius: 8px; min-width: 100px; }
        .prob-item span { font-size: 0.8rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem;}
        .prob-item strong { font-size: 1.5rem; color: #fff; }
        .draw-prob { opacity: 0.8; }
      `}</style>
    </div>
  );
}
