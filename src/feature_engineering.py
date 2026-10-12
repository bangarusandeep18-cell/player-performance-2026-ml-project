import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest

def engineer_cricket_features(df):
    """
    Engineers cricket-specific domain features.
    Can be run on raw dataframe prior to one-hot encoding/scaling.
    """
    df = df.copy()
    
    if 'Runs' in df.columns and 'Strike_Rate' in df.columns:
        sr_clipped = df['Strike_Rate'].clip(upper=400.0)
        df['Batting_Impact'] = df['Runs'] * (sr_clipped / 100.0)
        
    if 'Wickets' in df.columns and 'Economy' in df.columns:
        df['Bowling_Impact'] = df['Wickets'] * (10.0 / (df['Economy'] + 1.0))
        
    if 'Fours' in df.columns and 'Sixes' in df.columns and 'Runs' in df.columns:
        boundary_runs = df['Fours'] * 4.0 + df['Sixes'] * 6.0
        df['Boundary_Run_Ratio'] = boundary_runs / (df['Runs'] + 1.0)
        
    if 'Recent_Form' in df.columns and 'Consistency_Score' in df.columns:
        df['Form_Consistency_Index'] = (df['Recent_Form'] / 100.0) * (df['Consistency_Score'] / 100.0)
        
    if 'Fitness_Score' in df.columns and 'Pressure_Score' in df.columns:
        df['Pressure_Resilience'] = df['Fitness_Score'] / (df['Pressure_Score'] + 1.0)
        
    all_round_parts = []
    if 'Runs' in df.columns:
        all_round_parts.append(df['Runs'] / 50.0)
    if 'Wickets' in df.columns:
        all_round_parts.append(df['Wickets'] * 2.0)
    if 'Catches' in df.columns:
        all_round_parts.append(df['Catches'] * 0.5)
    if 'Stumpings' in df.columns:
        all_round_parts.append(df['Stumpings'] * 1.0)
        
    if all_round_parts:
        df['All_Round_Index'] = sum(all_round_parts)
        
    return df

def compute_feature_importances(model, feature_names):
    """Extracts and sorts feature importances from a tree-based estimator."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return fi_df
    return None

def select_top_features_mi(X_train, y_train, k=30):
    """Selects top k features based on Mutual Information with target y."""
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()].tolist()
    scores_df = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': selector.scores_
    }).sort_values('MI_Score', ascending=False).reset_index(drop=True)
    return selected_cols, scores_df
