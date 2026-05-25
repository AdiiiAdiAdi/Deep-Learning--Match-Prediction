import pandas as pd
import numpy as np

df = pd.read_csv('src/dataset/epl_match_dataset_v2.csv')
# Pre-clean
df.replace(['##', 'NA', 'null', '-', '?', ''], np.nan, inplace=True)

cat_cols = [
    'home_strategy_embedding', 'away_strategy_embedding',
    'home_player_performance', 'away_player_performance',
    'home_formation', 'away_formation'
]
non_feature_cols = {
    'date', 'target', 'season', 'kick_off_time', 'stadium', 'referee', 'weather',
    'match_importance', 'home_team', 'away_team', 'home_manager', 'away_manager',
    'home_goal_timestamps', 'away_goal_timestamps'
}

test_cols = [c for c in df.columns if c not in non_feature_cols and c not in cat_cols and c != 'target']
for col in test_cols:
    df[col] = df[col].astype(str).str.replace(r'x$', '', regex=True).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'target']

static_features = [
    'betting_odds_home_win', 'betting_odds_draw', 'betting_odds_away_win'
] + cat_cols
static_features = [c for c in static_features if c in df.columns]
temporal_features = [c for c in numeric_cols if c not in static_features]

print(f"LEN TEMPORAL: {len(temporal_features)}")
print(f"TEMPORAL: {temporal_features}")
print(f"LEN STATIC: {len(static_features)}")
print(f"STATIC: {static_features}")
