import pickle
import numpy as np
import pandas as pd
import json
import sqlite3
from pathlib import Path


class BulkDataLoader:
    """Load bulk data from multiple file formats"""
    
    @staticmethod
    def load_csv(file):
        """Load CSV file"""
        return pd.read_csv(file)
    
    @staticmethod
    def load_xlsx(file):
        """Load Excel file"""
        return pd.read_excel(file)
    
    @staticmethod
    def load_json(file):
        """Load JSON file"""
        data = json.load(file)
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                data = [data]
        return pd.DataFrame(data)
    
    @staticmethod
    def load_sql(file_path):
        """Load SQL database file"""
        try:
            conn = sqlite3.connect(file_path)
            query = "SELECT * FROM water_quality"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except:
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn
            ).iloc[0, 0]
            df = pd.read_sql_query(f"SELECT * FROM {tables}", conn)
            conn.close()
            return df
    
    @staticmethod
    def load_file(file, file_type):
        """Universal loader for different file formats"""
        if file_type == 'csv':
            return BulkDataLoader.load_csv(file)
        elif file_type == 'xlsx':
            return BulkDataLoader.load_xlsx(file)
        elif file_type == 'json':
            return BulkDataLoader.load_json(file)
        elif file_type == 'sql':
            return BulkDataLoader.load_sql(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


class BulkPredictionEngine:
    """Process batch predictions"""
    
    @staticmethod
    def standardize_columns(df):
        """Standardize column names"""
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        return df
    
    @staticmethod
    def validate_data(df, required_cols):
        """Validate if required columns exist"""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    @staticmethod
    def process_batch_data(df, feature_names, scaler):
        """Add engineered features for batch data"""
        df = BulkPredictionEngine.standardize_columns(df.copy())
        
        # Ensure all required features exist in dataframe
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Essential water quality parameters
        core_params = [
            'ph', 'dissolved_oxygen_mg_l', 'turbidity_ntu', 'hardness_mg_l',
            'conductivity_us_cm', 'bod_mg_l', 'ammonia_mg_l', 'nitrate_mg_l',
            'phosphate_mg_l', 'total_coliform_cfu_100ml', 'e_coli_cfu_100ml'
        ]
        
        # Add rolling features
        for param in core_params:
            if param in df.columns:
                df[f'{param}_rolling_7d'] = df[param].rolling(window=7, min_periods=1).mean()
                df[f'{param}_rolling_30d'] = df[param].rolling(window=30, min_periods=1).mean()
                df[f'{param}_rolling_std'] = df[param].rolling(window=7, min_periods=1).std().fillna(1.0)
        
        # Add WQI rolling features if WQI exists
        if 'wqi' in df.columns:
            df['wqi_rolling_7d'] = df['wqi'].rolling(window=7, min_periods=1).mean()
            df['wqi_rolling_30d'] = df['wqi'].rolling(window=30, min_periods=1).mean()
            df['wqi_rolling_std'] = df['wqi'].rolling(window=7, min_periods=1).std().fillna(2.0)
        else:
            df['wqi_rolling_7d'] = 50.0
            df['wqi_rolling_30d'] = 50.0
            df['wqi_rolling_std'] = 2.0
        
        # Location and Season defaults
        if 'location' not in df.columns:
            df['location'] = 0
        if 'season' not in df.columns:
            df['season'] = 1
        
        # Ensure all features are present
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0.0
        
        return df
    
    @staticmethod
    def add_prediction_results(df, predictions):
        """Add prediction results and categories"""
        df['predicted_wqi'] = predictions
        
        # Categorize predictions
        conditions = [
            df['predicted_wqi'] >= 70,
            (df['predicted_wqi'] >= 50) & (df['predicted_wqi'] < 70),
            df['predicted_wqi'] < 50
        ]
        categories = ['Potable', 'Questionable', 'Not Potable']
        df['potability'] = np.select(conditions, categories, default='Unknown')
        
        return df
