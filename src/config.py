"""
AquaVision Configuration File
Central configuration for the project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for dir_path in [DATA_DIR, NOTEBOOKS_DIR, SRC_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'raw_data': DATA_DIR / "water_quality_data.csv",
    'colors': {
        'excellent': '#00A86B',    # Green
        'good': '#87CEEB',         # Light blue
        'fair': '#FFD700',         # Gold
        'poor': '#FF8C00',         # Orange
        'bad': '#FF0000'           # Red
    },
    'wqi_thresholds': {
        'excellent': 90,
        'good': 70,
        'fair': 50,
        'poor': 25,
        'bad': 0
    }
}

# Model configuration
MODEL_CONFIG = {
    'model_path': OUTPUTS_DIR / "best_model_xgboost.pkl",
    'scaler_path': OUTPUTS_DIR / "scaler.pkl",
    'features_path': OUTPUTS_DIR / "02_data_with_features.csv",
    'predictions_path': OUTPUTS_DIR / "model_predictions.csv",
    
    'xgboost_params': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    
    'performance_thresholds': {
        'excellent_r2': 0.95,
        'good_r2': 0.90,
        'acceptable_r2': 0.85,
        'minimum_r2': 0.80
    }
}

# Feature configuration
FEATURE_CONFIG = {
    'numeric_features': [
        'pH', 'Dissolved_Oxygen_mg_L', 'Turbidity_NTU', 'Conductivity_uS_cm',
        'Temperature_C', 'Hardness_mg_L', 'Chloride_mg_L', 'Ammonia_mg_L',
        'Nitrate_mg_L', 'Phosphate_mg_L', 'Iron_mg_L', 'Manganese_mg_L',
        'Sulfate_mg_L', 'Total_Coliform_CFU_100mL', 'E_Coli_CFU_100mL',
        'BOD_mg_L', 'COD_mg_L'
    ],
    
    'categorical_features': ['Location', 'Season'],
    
    'target': 'WQI',
    'target_binary': 'Potability',
    
    'engineered_features': {
        'interactions': [
            'DO_pH_Interaction',
            'Hardness_Conductivity_Ratio',
            'Pollutant_Index',
            'Redox_Indicator',
            'Ion_Balance'
        ],
        'contamination': [
            'Microbial_Load',
            'E_Coli_Ratio',
            'Organic_Load',
            'BOD_COD_Ratio',
            'Metal_Index'
        ],
        'temporal': [
            'Month_sin', 'Month_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos',
            'Season'
        ]
    }
}

# WQI interpretation
WQI_SCALE = {
    'excellent': {'range': (90, 100), 'status': 'Excellent', 'emoji': '😊'},
    'good': {'range': (70, 89), 'status': 'Good', 'emoji': '👍'},
    'fair': {'range': (50, 69), 'status': 'Fair', 'emoji': '⚠️'},
    'poor': {'range': (25, 49), 'status': 'Poor', 'emoji': '😞'},
    'bad': {'range': (0, 24), 'status': 'Bad', 'emoji': '🚫'}
}

# Potability classification
POTABILITY_THRESHOLDS = {
    'potable': 70,           # >= 70: Safe to drink
    'questionable': 50,      # 50-69: May need treatment
    'not_potable': 0         # < 50: Unsafe
}

# Geographic locations
LOCATIONS = [
    'Urban_North',
    'Urban_South',
    'Rural_East',
    'Rural_West',
    'Industrial'
]

# Seasons
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'AquaVision - Water Quality Dashboard',
    'page_icon': '💧',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'AquaVision API',
    'version': '1.0.0',
    'description': 'Water Quality Index Prediction API'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': OUTPUTS_DIR / 'aquavision.log'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

# Environment variables
ENVIRONMENT = os.getenv('ENV', 'development')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

if __name__ == "__main__":
    print("🌊 AquaVision Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Path: {MODEL_CONFIG['model_path']}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("=" * 50)
