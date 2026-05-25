import os
import json
import pickle
import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.preprocessing import LabelEncoder

def preprocess_pipeline():
    print("Starting Preprocessing Pipeline...")

    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH     = os.path.join(BASE_DIR, 'dataset', 'epl_match_dataset_v2.csv')
    ENCODERS_DIR  = os.path.join(BASE_DIR, 'encoders')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

    os.makedirs(ENCODERS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load
    df = pd.read_csv(DATA_PATH)

    # FIX: Replace ALL known bad-value sentinels with NaN.
    # '##', '-', '?' and 'x'-suffixed values (e.g. '2.51x') are present
    # across 25+ columns and cause pandas to infer them as str dtype.
    df.replace(['##', 'NA', 'null', '-', '?', ''], np.nan, inplace=True)

    # 2. Clean
    df = df.drop_duplicates()
    df = df.dropna(subset=['home_team', 'away_team'])

    # 🔥 Encode teams
    team_encoder = LabelEncoder()

    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_encoder.fit(all_teams)

    df['home_team_id'] = team_encoder.transform(df['home_team'])
    df['away_team_id'] = team_encoder.transform(df['away_team'])

    # Save encoder
    with open(os.path.join(ENCODERS_DIR, 'team_encoder.pkl'), 'wb') as f:
        pickle.dump(team_encoder, f)

    print("✅ Team encoding done:", len(all_teams), "teams")

    # 3. Parse dates
    def parse_date(x):
        try:    return parser.parse(str(x))
        except: return pd.NaT

    df['date'] = df['date'].apply(parse_date)
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 4. TARGET
    result_map = {'H': 2, 'D': 1, 'A': 0}
    df['target'] = df['result'].map(result_map).fillna(1).astype(np.int32)

    with open(os.path.join(ENCODERS_DIR, 'result_map.json'), 'w') as f:
        json.dump(result_map, f)

    # Remove leakage features
    leak_cols = [
        'home_goals_ft', 'away_goals_ft',
        'home_goals_conceded', 'away_goals_conceded',
        'home_points_earned', 'away_points_earned',
        'result'
    ]
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])

    # 5. CATEGORICAL — normalize messy abbreviations before label encoding.
    strategy_map = {
        'attk': 'attack',
        'bal':  'balanced',
        'def':  'defense',
    }
    performance_map = {
        'avg':  'average',
        'poor': 'bad',
    }

    cat_cols = [
        'home_strategy_embedding', 'away_strategy_embedding',
        'home_player_performance', 'away_player_performance',
        'home_formation', 'away_formation'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()
            df[col] = df[col].replace({'nan': 'unknown', 'none': 'unknown', '': 'unknown'})
            if 'strategy' in col:
                df[col] = df[col].replace(strategy_map)
            if 'performance' in col:
                df[col] = df[col].replace(performance_map)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            with open(os.path.join(ENCODERS_DIR, f'{col}.pkl'), 'wb') as f:
                pickle.dump(le, f)

    # 6. DEFINE columns that should NOT be used as features at all
    non_feature_cols = {
        'date', 'target',
        # Text / identifier columns — no predictive signal as-is
        'season', 'kick_off_time', 'stadium', 'referee', 'weather',
        'match_importance', 'home_team', 'away_team',
        'home_manager', 'away_manager',
        # Timestamp strings — not useful as raw features
        'home_goal_timestamps', 'away_goal_timestamps',
    }

    # 7. NUMERIC CONVERSION
    # FIX: Almost every column in this CSV is read as str/object dtype because
    # dirty values like '2.51x', '-', '?' prevent pandas from inferring float.
    # We must explicitly call pd.to_numeric on all intended numeric columns
    # BEFORE using select_dtypes — otherwise only 2 cols pass the filter.
    #
    # Strip trailing 'x' characters (e.g. '2.51x' -> '2.51') then coerce.
    intended_numeric_cols = [
        c for c in df.columns
        if c not in non_feature_cols
        and c not in cat_cols
        and c != 'target'
    ]

    for col in intended_numeric_cols:
        # Strip trailing 'x' that appears in betting odds and some pct cols
        df[col] = df[col].astype(str).str.replace(r'x$', '', regex=True).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # 8. FEATURES — now select_dtypes works correctly
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'target']

    static_features = [
        'betting_odds_home_win', 'betting_odds_draw', 'betting_odds_away_win'
    ] + cat_cols
    static_features   = [c for c in static_features if c in df.columns]
    temporal_features = [c for c in numeric_cols if c not in static_features]

    print(f"   Temporal features ({len(temporal_features)}): {temporal_features[:8]} ...")
    print(f"   Static features   ({len(static_features)}): {static_features}")

    # 9. BUILD SEQUENCES
    X_seq, X_static, y = [], [], []
    home_team_ids = []
    away_team_ids = []

    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    for team in teams:
        team_df = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].sort_values('date')

        if len(team_df) < 6:
            continue

        seq_vals  = team_df[temporal_features].values.astype(np.float64)
        stat_vals = team_df[static_features].values.astype(np.float64)
        targets   = team_df['target'].values.astype(np.int32)
        home_ids  = team_df['home_team_id'].values
        away_ids  = team_df['away_team_id'].values

        for i in range(5, len(team_df)):
            X_seq.append(seq_vals[i-5:i])
            X_static.append(stat_vals[i])
            y.append(targets[i])

            # 🔥 NEW
            home_team_ids.append(home_ids[i])
            away_team_ids.append(away_ids[i])

    X_seq    = np.array(X_seq,    dtype=np.float64)
    X_static = np.array(X_static, dtype=np.float64)
    y        = np.array(y,        dtype=np.int32)
    home_team_ids = np.array(home_team_ids)
    away_team_ids = np.array(away_team_ids)

    print(f"   Total samples: {len(y)}")

    # 10. SHUFFLE
    idx = np.random.permutation(len(X_seq))
    X_seq, X_static, y = X_seq[idx], X_static[idx], y[idx]
    home_team_ids, away_team_ids = home_team_ids[idx], away_team_ids[idx]

    # 11. SPLIT — 70 / 15 / 15
    n         = len(X_seq)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_seq_train  = X_seq[:train_end];    X_seq_val  = X_seq[train_end:val_end];    X_seq_test  = X_seq[val_end:]
    X_stat_train = X_static[:train_end]; X_stat_val = X_static[train_end:val_end]; X_stat_test = X_static[val_end:]
    y_train      = y[:train_end];        y_val      = y[train_end:val_end];        y_test      = y[val_end:]

    # 12. SAVE
    np.save(os.path.join(PROCESSED_DIR, 'X_seq_train.npy'),    X_seq_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_seq_val.npy'),      X_seq_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_seq_test.npy'),     X_seq_test)
    np.save(os.path.join(PROCESSED_DIR, 'X_static_train.npy'), X_stat_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_static_val.npy'),   X_stat_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_static_test.npy'),  X_stat_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'),        y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'),          y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'),         y_test)

    np.save(os.path.join(PROCESSED_DIR, 'home_team_train.npy'), home_team_ids[:train_end])
    np.save(os.path.join(PROCESSED_DIR, 'home_team_val.npy'),   home_team_ids[train_end:val_end])
    np.save(os.path.join(PROCESSED_DIR, 'home_team_test.npy'),  home_team_ids[val_end:])

    np.save(os.path.join(PROCESSED_DIR, 'away_team_train.npy'), away_team_ids[:train_end])
    np.save(os.path.join(PROCESSED_DIR, 'away_team_val.npy'),   away_team_ids[train_end:val_end])
    np.save(os.path.join(PROCESSED_DIR, 'away_team_test.npy'),  away_team_ids[val_end:])

    print("\n✅ Preprocessing complete!")
    print(f"   X_seq    shape : {X_seq_train.shape}   dtype={X_seq_train.dtype}")
    print(f"   X_static shape : {X_stat_train.shape}   dtype={X_stat_train.dtype}")
    print(f"   y        shape : {y_train.shape}        dtype={y_train.dtype}")
    print(f"   Val  samples   : {len(y_val)}")
    print(f"   Test samples   : {len(y_test)}")


if __name__ == "__main__":
    preprocess_pipeline()