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
        """Process batch data - RETRAINED MODEL (NO data leakage)
        
        This uses the fixed model that was retrained WITHOUT WQI rolling features.
        The model now uses only 19 input features and was trained on UNSCALED data.
        """
        df = df.copy()
        
        # Ensure raw features exist
        raw_features = [
            'pH', 'Dissolved_Oxygen_mg_L', 'Turbidity_NTU', 'Conductivity_uS_cm',
            'Temperature_C', 'Hardness_mg_L', 'Chloride_mg_L', 'Ammonia_mg_L',
            'Nitrate_mg_L', 'Phosphate_mg_L', 'Iron_mg_L', 'Manganese_mg_L',
            'Sulfate_mg_L', 'Total_Coliform_CFU_100mL', 'E_Coli_CFU_100mL',
            'BOD_mg_L', 'COD_mg_L'
        ]
        
        for feat in raw_features:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Add categorical features
        if 'Location' not in df.columns:
            df['Location'] = 'Unknown'
        if 'Season' not in df.columns:
            df['Season'] = 'Winter'
        
        # Factorize categorical variables to match training
        df['Location'] = pd.factorize(df['Location'])[0]
        df['Season'] = pd.factorize(df['Season'])[0]
        
        # Select ONLY the features model expects (NO scaling - model trained on unscaled data)
        df_features = df[feature_names].fillna(0).copy()
        
        print(f"\n=== BATCH PROCESSING DEBUG ===")
        print(f"Input shape: {df_features.shape}")
        print(f"Expected features: {len(feature_names)}")
        print(f"First row: pH={df['pH'].iloc[0]:.2f}, DO={df['Dissolved_Oxygen_mg_L'].iloc[0]:.2f}, E_Coli={df['E_Coli_CFU_100mL'].iloc[0]:.0f}")
        print(f"Last row: pH={df['pH'].iloc[-1]:.2f}, DO={df['Dissolved_Oxygen_mg_L'].iloc[-1]:.2f}, E_Coli={df['E_Coli_CFU_100mL'].iloc[-1]:.0f}")
        print(f"==================================\n")
        
        # Return unscaled features (model was trained on unscaled data)
        return df_features
    
    @staticmethod
    def add_prediction_results(df, predictions):
        """Add prediction results and categories"""
        df['predicted_wqi'] = predictions
        
        # UPDATED THRESHOLDS based on actual model training ranges:
        # NOT POTABLE: 41.62 - 54.00  (mean 49.49)
        # QUESTIONABLE: 60.24 - 66.38 (mean 63.36)
        # POTABLE: 85.07 - 87.56      (mean 86.31)
        # Using boundaries: <60 -> Not Potable, 60-75 -> Questionable, >=75 -> Potable
        
        print(f"\n=== PREDICTION DEBUG (Corrected Thresholds) ===")
        print(f"Min WQI Prediction: {predictions.min():.2f}")
        print(f"Max WQI Prediction: {predictions.max():.2f}")
        print(f"Mean WQI Prediction: {predictions.mean():.2f}")
        print(f"Std WQI Prediction: {predictions.std():.2f}")
        
        # Show distribution by ranges (CORRECTED)
        print(f"\nPrediction ranges:")
        print(f"  < 60 (Not Potable): {(predictions < 60).sum()}")
        print(f"  60-75 (Questionable): {((predictions >= 60) & (predictions < 75)).sum()}")
        print(f"  >= 75 (Potable): {(predictions >= 75).sum()}")
        
        print(f"\nLast 5 predictions: {predictions[-5:]}")
        print(f"==============================================\n")
        
        # Categorize predictions using CORRECTED thresholds
        conditions = [
            df['predicted_wqi'] >= 75,  # Potable (model learned 85-87)
            (df['predicted_wqi'] >= 60) & (df['predicted_wqi'] < 75),  # Questionable (60-66)
            df['predicted_wqi'] < 60  # Not Potable (model learned 41-54)
        ]
        categories = ['Potable', 'Questionable', 'Not Potable']
        df['potability'] = np.select(conditions, categories, default='Unknown')
        
        return df
