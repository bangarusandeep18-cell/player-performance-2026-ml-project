import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import DATA_RAW_DIR, set_seed

def load_raw_data(filepath=None):
    if filepath is None:
        filepath = DATA_RAW_DIR / "player_performance_dataset_10000.csv"
    df = pd.read_csv(filepath)
    return df

def create_binary_target(df, threshold=60.0):
    """
    Academic Problem Formulation:
    Defines Elite Performer (High_Performer = 1) if Performance_Score >= threshold.
    Strictly removes Performance_Score from features X to avoid 100% data leakage!
    """
    df = df.copy()
    y = (df['Performance_Score'] >= threshold).astype(int)
    X = df.drop(columns=['Performance_Score'])
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified split to preserve the minority class proportion.
    Executed BEFORE any transformations to ensure zero leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

class SportsDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Academic Preprocessing Pipeline:
    - Drops identifiers (Player_ID, Player_Name, Match_ID)
    - Extracts temporal features from Match_Date
    - Capping extreme Strike_Rate anomalies
    - One-hot encodes nominal categorical columns
    - Standardizes numeric features (fit on train, transform on test)
    """
    def __init__(self, sr_cap=400.0):
        self.sr_cap = sr_cap
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.id_cols = ['Player_ID', 'Player_Name', 'Match_ID']
        self.cat_cols = ['Role', 'Match_Format', 'Home_Away', 'Team', 'Opponent', 'Venue']
        self.num_cols = None
        self.feature_names_out_ = None
        self.fitted = False

    def _extract_date_features(self, df):
        df = df.copy()
        if 'Match_Date' in df.columns:
            date_s = pd.to_datetime(df['Match_Date'])
            df['Match_Year'] = date_s.dt.year
            df['Match_Month'] = date_s.dt.month
            df['Match_DayOfWeek'] = date_s.dt.dayofweek
            df = df.drop(columns=['Match_Date'])
        return df

    def fit(self, X, y=None):
        X_df = X.copy()
        drop_existing = [c for c in self.id_cols if c in X_df.columns]
        X_df = X_df.drop(columns=drop_existing)
        
        X_df = self._extract_date_features(X_df)
        
        self.cat_cols = [c for c in self.cat_cols if c in X_df.columns]
        self.num_cols = [c for c in X_df.columns if c not in self.cat_cols]

        self.encoder.fit(X_df[self.cat_cols])
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        
        X_num = X_df[self.num_cols].copy()
        if 'Strike_Rate' in X_num.columns:
            X_num['Strike_Rate'] = X_num['Strike_Rate'].clip(upper=self.sr_cap)
            
        self.scaler.fit(X_num)
        
        self.feature_names_out_ = self.num_cols + encoded_cat_names
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("SportsDataPreprocessor must be fitted before transform.")
            
        X_df = X.copy()
        drop_existing = [c for c in self.id_cols if c in X_df.columns]
        X_df = X_df.drop(columns=drop_existing)
        
        X_df = self._extract_date_features(X_df)
        
        cat_encoded = self.encoder.transform(X_df[self.cat_cols])
        cat_df = pd.DataFrame(
            cat_encoded,
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=X.index
        )
        
        X_num = X_df[self.num_cols].copy()
        if 'Strike_Rate' in X_num.columns:
            X_num['Strike_Rate'] = X_num['Strike_Rate'].clip(upper=self.sr_cap)
            
        num_scaled = self.scaler.transform(X_num)
        num_df = pd.DataFrame(num_scaled, columns=self.num_cols, index=X.index)
        
        transformed_df = pd.concat([num_df, cat_df], axis=1)
        return transformed_df

    def get_feature_names_out(self):
        return self.feature_names_out_
