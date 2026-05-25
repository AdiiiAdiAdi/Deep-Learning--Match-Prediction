export const TEAMS = [
  "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
  "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
  "Liverpool", "Luton Town", "Manchester City", "Manchester United",
  "Newcastle United", "Nottingham Forest", "Sheffield United",
  "Tottenham Hotspur", "West Ham United", "Wolverhampton"
];

export const generateMockPrediction = (homeTeam, awayTeam) => {
  // Deterministic but "random" values based on team names
  const seed = homeTeam.length + awayTeam.length;
  const homeWinProb = Math.floor(Math.abs(Math.sin(seed) * 50)) + 20;
  const drawProb = Math.floor(Math.abs(Math.cos(seed) * 20)) + 10;
  const awayWinProb = 100 - homeWinProb - drawProb;
  
  const hGoals = Math.max(0, Math.floor((homeWinProb / 100) * 4) + (seed % 2));
  const aGoals = Math.max(0, Math.floor((awayWinProb / 100) * 4) - (seed % 2));

  const confidence = 45 + Math.floor(Math.abs(Math.sin(seed * 2) * 50));
  
  return {
    probs: { home: homeWinProb, draw: drawProb, away: awayWinProb },
    score: `${hGoals} - ${aGoals}`,
    confidence,
    explanation: {
      lstm: 68,
      ann: 32
    },
    keyFactors: [
      `${homeTeam} won last ${Math.floor(seed % 4) + 1} of 5 meetings`,
      `${homeTeam} xG home avg: ${(1.2 + (seed % 15) / 10).toFixed(2)} vs ${awayTeam} xG away avg: ${(0.8 + (seed % 10) / 10).toFixed(2)}`,
      `${awayTeam} missing key midfielder (Embedding dist shift: -0.14)`,
      `High temporal momentum for ${homeTeam} (LSTM cell state > 0.8)`,
      `Historical draw rate for this matchup is low (12%)`
    ],
    shap: [
      { feature: 'Home xG (rolling 5)', value: 0.14, raw: '1.85' },
      { feature: 'Away Def. Embed', value: 0.08, raw: '[-0.2, 0.4...]' },
      { feature: 'Home Win Streak', value: 0.06, raw: '3' },
      { feature: 'Rest Days Diff', value: -0.05, raw: '-2' },
      { feature: 'Away xG (rolling 5)', value: -0.04, raw: '1.62' },
      { feature: 'Home Injuries', value: -0.03, raw: '2' },
      { feature: 'Match Importance', value: 0.02, raw: '0.88' },
    ]
  };
};

export const DATASET_STATS = {
  totalMatches: "8,340",
  totalFeatures: "142",
  seasons: "12 (2012-2024)",
  nullCellPct: "0.24%"
};

export const PIPELINE_STEPS = [
  { id: 1, name: "Raw CSV", desc: "Loaded 12 seasons of EPL match and event data. (284,512 rows)", status: "Done" },
  { id: 2, name: "Null Detection", desc: "Fixed 847 null cells, imputed with column median or previous known value.", status: "Done" },
  { id: 3, name: "Type Fixing", desc: "Dates parsed to ISO 8601. Numeric columns cast to float32.", status: "Done" },
  { id: 4, name: "Label Encoding", desc: "Team names, referee names encoded to integer IDs.", status: "Done" },
  { id: 5, name: "Normalization", desc: "Applied StandardScaler to continuous features (xG, Possession, etc.).", status: "Done" },
  { id: 6, name: "Sequence Building", desc: "Created t-5 lookback sliding windows for LSTM input. Target = Match Outcome (0/1/2).", status: "Done" }
];

export const QUALITY_HEATMAP = [
  { category: "Goals", nulls: 0, types: 0, outliers: 1, labels: 0 },
  { category: "Shots", nulls: 0.5, types: 0, outliers: 2, labels: 0 },
  { category: "Possession", nulls: 1.2, types: 0.1, outliers: 4, labels: 0.5 },
  { category: "Discipline", nulls: 0, types: 0, outliers: 8, labels: 0 },
  { category: "Physical", nulls: 15, types: 2, outliers: 1, labels: 0 }, // Red/Yellow warnings mapped visually
  { category: "Embeddings", nulls: 0, types: 0, outliers: 0, labels: 0 },
];

export const MODEL_METRICS = {
  accuracy: { value: "68.4%", baseline: "+18.4% vs Baseline" },
  precision: { value: "65.2%", baseline: "+14.2% vs Baseline" },
  recall: { value: "64.8%", baseline: "+14.0% vs Baseline" },
  f1: { value: "65.0%", baseline: "+14.1% vs Baseline" }
};

export const CONFUSION_MATRIX = [
  // rows: Actual (Win, Draw, Loss), cols: Predicted (Win, Draw, Loss)
  [450, 85, 65], // Actual Home Win
  [120, 180, 110], // Actual Draw
  [70, 95, 380] // Actual Away Win (Home Loss)
];

// Generate realistic loss curves
const generateEpochData = () => {
  let data = [];
  let tLoss = 1.2;
  let vLoss = 1.25;
  let tAcc = 0.33;
  let vAcc = 0.33;
  
  for(let i=1; i<=50; i++) {
    tLoss = tLoss * 0.92 - (Math.random() * 0.02);
    vLoss = i < 35 ? vLoss * 0.93 + (Math.random() * 0.03) : vLoss + (Math.random() * 0.015);
    
    tAcc = Math.min(0.95, tAcc + 0.02 + Math.random() * 0.01);
    vAcc = i < 35 ? Math.min(0.85, vAcc + 0.018 + Math.random() * 0.01) : vAcc - (Math.random() * 0.005);
    
    data.push({
      epoch: i,
      trainLoss: Math.max(0.4, tLoss).toFixed(3),
      valLoss: Math.max(0.45, vLoss).toFixed(3),
      trainAcc: (tAcc * 100).toFixed(1),
      valAcc: (vAcc * 100).toFixed(1)
    });
  }
  return data;
};

export const EPOCH_DATA = generateEpochData();

export const GLOBAL_FEATURES = [
  { name: "Team Form (Last 5 Pt Avg)", score: 0.18, group: "Goals" },
  { name: "Opponent Strength (Elo diff)", score: 0.15, group: "Embeddings" },
  { name: "Expected Goals (xG) Avg", score: 0.12, group: "Goals" },
  { name: "Possession Transition Rate", score: 0.09, group: "Possession/Passing" },
  { name: "Defensive Action Density", score: 0.08, group: "Physical" },
  { name: "Prog. Passes allowed L5", score: 0.07, group: "Possession/Passing" },
  { name: "Fatigue/Rest Days", score: 0.06, group: "Physical" },
  { name: "Manager Tactical Embedding", score: 0.05, group: "Embeddings" },
  { name: "Fouls Suffered L5", score: 0.04, group: "Discipline" },
  { name: "Yellow Card Accumulation", score: 0.03, group: "Discipline" },
];

export const SEQUENCE_DATA = [
  { t: "-4", opp: "Chelsea", ha: "A", res: "L", score: "1 - 2", xG: 0.8, shots: 3, poss: 42, date: "Oct 12" },
  { t: "-3", opp: "Brighton", ha: "H", res: "D", score: "1 - 1", xG: 1.4, shots: 5, poss: 55, date: "Oct 19" },
  { t: "-2", opp: "Bournemouth", ha: "A", res: "W", score: "3 - 0", xG: 2.1, shots: 8, poss: 61, date: "Oct 26" },
  { t: "-1", opp: "Luton Town", ha: "H", res: "W", score: "4 - 1", xG: 3.2, shots: 11, poss: 72, date: "Nov 02" },
  { t: "0",  opp: "West Ham United", ha: "A", res: "W", score: "2 - 1", xG: 1.8, shots: 6, poss: 58, date: "Nov 09" }
];
