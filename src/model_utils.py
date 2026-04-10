"""
AquaVision - Model Utilities
Utility functions for model loading, prediction, and serving
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WQIPredictor:
    """
    Water Quality Index Predictor
    Handles model loading, preprocessing, and prediction
    """
    
    def __init__(self, model_path='outputs/best_model_xgboost.pkl', 
                 scaler_path='outputs/scaler.pkl'):
        """
        Initialize the predictor with model and scaler
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        scaler_path : str
            Path to the fitted scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        logger.info("WQIPredictor initialized successfully")
    
    def load_model(self):
        """Load the trained XGBoost model"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_scaler(self):
        """Load the fitted scaler"""
        try:
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {self.scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return None
    
    def predict(self, features):
        """
        Make WQI prediction
        
        Parameters:
        -----------
        features : array-like
            Feature array for prediction
        
        Returns:
        --------
        float : Predicted WQI value
        """
        try:
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            elif isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
            
            prediction = self.model.predict(features)[0]
            # Ensure prediction is within valid range [0, 100]
            prediction = np.clip(prediction, 0, 100)
            
            return prediction
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def predict_batch(self, features_df):
        """
        Make batch predictions
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame with features
        
        Returns:
        --------
        np.ndarray : Array of predictions
        """
        try:
            predictions = self.model.predict(features_df)
            predictions = np.clip(predictions, 0, 100)
            return predictions
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def get_potability(self, wqi):
        """
        Determine potability based on WQI
        
        Parameters:
        -----------
        wqi : float
            Water Quality Index value
        
        Returns:
        --------
        dict : Potability status and recommendation
        """
        if wqi >= 70:
            return {
                'status': 'POTABLE',
                'emoji': '✅',
                'description': 'Safe to drink',
                'color': 'green'
            }
        elif wqi >= 50:
            return {
                'status': 'QUESTIONABLE',
                'emoji': '⚠️',
                'description': 'May require treatment',
                'color': 'orange'
            }
        else:
            return {
                'status': 'NOT POTABLE',
                'emoji': '❌',
                'description': 'Unsafe for consumption',
                'color': 'red'
            }
    
    def get_feature_importance(self, top_n=10):
        """Get top N important features"""
        try:
            importance_df = pd.DataFrame({
                'feature': self.model.get_booster().feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None


def create_prediction_input(ph, dissolved_oxygen, turbidity, conductivity, temperature,
                           hardness, chloride, ammonia, nitrate, phosphate, iron, 
                           manganese, sulfate, coliform, e_coli, bod, cod, 
                           location_code=0, **kwargs):
    """
    Create input features for WQI prediction
    
    Parameters:
    -----------
    All water quality parameters as floats
    location_code : int
        Encoded location (0-4)
    
    Returns:
    --------
    np.ndarray : Feature array ready for prediction
    """
    base_features = [
        ph, dissolved_oxygen, turbidity, conductivity, temperature,
        hardness, chloride, ammonia, nitrate, phosphate, iron,
        manganese, sulfate, coliform, e_coli, bod, cod, location_code
    ]
    
    # Add derived features
    pollutant_index = (ammonia + nitrate + phosphate) / 3
    redox_indicator = dissolved_oxygen - (bod + cod) / 10
    organic_load = (bod + cod) / 2
    microbial_load = np.log1p(coliform + e_coli)
    
    all_features = base_features + [
        pollutant_index, redox_indicator, organic_load, microbial_load
    ]
    
    # Pad to match model's expected number of features
    while len(all_features) < 50:
        all_features.append(0)
    
    return np.array(all_features[:50])


def get_water_quality_recommendations(wqi):
    """Get recommendations based on WQI value"""
    recommendations = {
        'high': (90, 100, [
            '✅ Water quality is excellent',
            '✅ Safe for direct consumption',
            '✅ No treatment required',
            '✅ Suitable for all uses'
        ]),
        'good': (70, 89, [
            '✅ Water quality is good',
            '⚠️ Generally safe for consumption',
            '✓ May benefit from standard treatment',
            '✓ Suitable for most uses'
        ]),
        'fair': (50, 69, [
            '⚠️ Moderate water quality concerns',
            '⚠️ Treatment recommended before consumption',
            '⚠️ Regular monitoring advised',
            '⚠️ Limited use for sensitive applications'
        ]),
        'poor': (25, 49, [
            '❌ Poor water quality',
            '❌ Significant treatment required',
            '❌ Not recommended for drinking',
            '❌ Industrial use only with precautions'
        ]),
        'bad': (0, 24, [
            '❌ Severely contaminated water',
            '❌ Extensive treatment needed',
            '❌ Unsafe for any direct use',
            '❌ Emergency intervention required'
        ])
    }
    
    for category, (min_val, max_val, recs) in recommendations.items():
        if min_val <= wqi <= max_val:
            return {
                'category': category.upper(),
                'wqi_range': f'{min_val}-{max_val}',
                'recommendations': recs
            }
    
    return None


# Model performance metrics
class ModelMetrics:
    """Calculate and store model performance metrics"""
    
    EXCELLENT = 0.95
    GOOD = 0.90
    ACCEPTABLE = 0.85
    POOR = 0.80
    
    @staticmethod
    def evaluate_r2(r2_score):
        """Evaluate R² score quality"""
        if r2_score >= ModelMetrics.EXCELLENT:
            return "Excellent - Model explains >95% variance"
        elif r2_score >= ModelMetrics.GOOD:
            return "Good - Model explains 90-95% variance"
        elif r2_score >= ModelMetrics.ACCEPTABLE:
            return "Acceptable - Model explains 85-90% variance"
        else:
            return "Poor - Model accuracy needs improvement"
    
    @staticmethod
    def evaluate_rmse(rmse, wqi_range=100):
        """Evaluate RMSE relative to target range"""
        percentage = (rmse / wqi_range) * 100
        if percentage < 2:
            return "Excellent - Average error < 2 points"
        elif percentage < 5:
            return "Good - Average error 2-5 points"
        elif percentage < 10:
            return "Acceptable - Average error 5-10 points"
        else:
            return "Poor - Average error > 10 points"


if __name__ == "__main__":
    # Example usage
    print("WQI Predictor Example")
    print("=" * 50)
    
    # Initialize predictor
    predictor = WQIPredictor()
    
    # Create sample input
    sample_input = create_prediction_input(
        ph=7.5,
        dissolved_oxygen=8.0,
        turbidity=3.5,
        conductivity=450,
        temperature=20,
        hardness=180,
        chloride=45,
        ammonia=0.8,
        nitrate=25,
        phosphate=2.5,
        iron=0.3,
        manganese=0.15,
        sulfate=150,
        coliform=50,
        e_coli=10,
        bod=3,
        cod=8,
        location_code=0
    )
    
    # Make prediction
    wqi = predictor.predict(sample_input)
    print(f"Predicted WQI: {wqi:.2f}")
    
    # Get potability
    potability = predictor.get_potability(wqi)
    print(f"Status: {potability['emoji']} {potability['status']} ({potability['description']})")
    
    # Get recommendations
    recommendations = get_water_quality_recommendations(wqi)
    print(f"\nCategory: {recommendations['category']}")
    print("Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"  {rec}")
