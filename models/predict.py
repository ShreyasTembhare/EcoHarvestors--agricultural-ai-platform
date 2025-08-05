"""
Prediction module for EcoHarvestors project.
Loads trained models and makes predictions for agricultural forecasts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from src.data_loader import DataLoader
from src.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    """Handles loading trained models and making predictions."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize Predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.load_models()
    
    def load_models(self) -> None:
        """Load all trained models."""
        try:
            model_files = {
                'crop': 'crop_model.pkl',
                'yield': 'yield_model.pkl',
                'rainfall': 'rainfall_model.pkl',
                'fertilizer': 'fertilizer_model.pkl'
            }
            
            for model_type, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[model_type] = joblib.load(model_path)
                    logger.info(f"Loaded {model_type} model from {model_path}")
                else:
                    logger.warning(f"Model file {model_path} not found")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict_crop(self, N: float, P: float, K: float, temperature: float,
                    humidity: float, ph: float, rainfall: float) -> str:
        """
        Predict crop recommendation.
        
        Args:
            N: Nitrogen content
            P: Phosphorus content
            K: Potassium content
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            ph: pH value
            rainfall: Rainfall in mm
            
        Returns:
            Recommended crop name
        """
        try:
            if 'crop' not in self.models:
                return "Model not available"
            
            # Create input data
            input_data = pd.DataFrame({
                'N': [N],
                'P': [P],
                'K': [K],
                'temperature': [temperature],
                'humidity': [humidity],
                'ph': [ph],
                'rainfall': [rainfall]
            })
            
            # Transform input data
            X_scaled = self.models['crop']['scaler'].transform(input_data)
            
            # Make prediction
            prediction = self.models['crop']['model'].predict(X_scaled)
            
            # Decode prediction
            crop_name = self.models['crop']['encoder'].inverse_transform(prediction)[0]
            
            return crop_name
            
        except Exception as e:
            logger.error(f"Error predicting crop: {e}")
            return "Prediction failed"
    
    def predict_yield(self, state: str, district: str, season: str, crop: str,
                     area: float, production: float) -> float:
        """
        Predict crop yield.
        
        Args:
            state: State name
            district: District name
            season: Season name
            crop: Crop name
            area: Area in hectares
            production: Production in tons
            
        Returns:
            Predicted yield
        """
        try:
            if 'yield' not in self.models:
                return 0.0
            
            # Create input data
            input_data = pd.DataFrame({
                'State': [state],
                'District': [district],
                'Season': [season],
                'Crop': [crop],
                'Area': [area],
                'Production': [production]
            })
            
            # Transform input data
            X_scaled = self.feature_engineer.transform_new_data(input_data, 'yield')
            
            if X_scaled is None:
                return 0.0
            
            # Make prediction
            prediction = self.models['yield']['model'].predict(X_scaled)
            
            return float(prediction[0])
            
        except Exception as e:
            logger.error(f"Error predicting yield: {e}")
            return 0.0
    
    def predict_rainfall(self, month: str, temperature: float, humidity: float) -> float:
        """
        Predict rainfall.
        
        Args:
            month: Month name
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            
        Returns:
            Predicted rainfall in mm
        """
        try:
            if 'rainfall' not in self.models:
                return 0.0
            
            # Create input data
            input_data = pd.DataFrame({
                'Month': [month],
                'Temperature': [temperature],
                'Humidity': [humidity]
            })
            
            # Transform input data
            X_scaled = self.feature_engineer.transform_new_data(input_data, 'rainfall')
            
            if X_scaled is None:
                return 0.0
            
            # Make prediction
            prediction = self.models['rainfall']['model'].predict(X_scaled)
            
            return float(prediction[0])
            
        except Exception as e:
            logger.error(f"Error predicting rainfall: {e}")
            return 0.0
    
    def predict_fertilizer(self, temperature: float, humidity: float, moisture: float,
                          soil_type: str, crop_type: str, nitrogen: float,
                          phosphorous: float, potassium: float, ph: float) -> str:
        """
        Predict fertilizer recommendation.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            moisture: Moisture content
            soil_type: Type of soil
            crop_type: Type of crop
            nitrogen: Nitrogen content
            phosphorous: Phosphorous content
            potassium: Potassium content
            ph: pH value
            
        Returns:
            Recommended fertilizer name
        """
        try:
            if 'fertilizer' not in self.models:
                return "Model not available"
            
            # Create input data
            input_data = pd.DataFrame({
                'Temparature': [temperature],
                'Humidity': [humidity],
                'Moisture': [moisture],
                'Soil Type': [soil_type],
                'Crop Type': [crop_type],
                'Nitrogen': [nitrogen],
                'Phosphorous': [phosphorous],
                'Potassium': [potassium],
                'pH': [ph]
            })
            
            # Transform input data
            X_scaled = self.feature_engineer.transform_new_data(input_data, 'fertilizer')
            
            if X_scaled is None:
                return "Prediction failed"
            
            # Make prediction
            prediction = self.models['fertilizer']['model'].predict(X_scaled)
            
            # Decode prediction
            fertilizer_name = self.models['fertilizer']['encoders']['fertilizer_target'].inverse_transform(prediction)[0]
            
            return fertilizer_name
            
        except Exception as e:
            logger.error(f"Error predicting fertilizer: {e}")
            return "Prediction failed"
    
    def get_available_crops(self) -> list:
        """Get list of available crops for prediction."""
        try:
            if 'crop' in self.models:
                return list(self.models['crop']['encoder'].classes_)
            return []
        except Exception as e:
            logger.error(f"Error getting available crops: {e}")
            return []
    
    def get_available_states(self) -> list:
        """Get list of available states for yield prediction."""
        return ['Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Maharashtra']
    
    def get_available_districts(self) -> list:
        """Get list of available districts for yield prediction."""
        return ['Anantapur', 'Kurnool', 'Chittoor', 'Kadapa']
    
    def get_available_seasons(self) -> list:
        """Get list of available seasons."""
        return ['Kharif', 'Rabi', 'Whole Year', 'Autumn', 'Summer', 'Winter']
    
    def get_available_months(self) -> list:
        """Get list of available months for rainfall prediction."""
        return ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
    
    def get_available_soil_types(self) -> list:
        """Get list of available soil types."""
        return ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    
    def get_available_crop_types(self) -> list:
        """Get list of available crop types for fertilizer prediction."""
        return ['Maize', 'Sugarcane', 'Cotton', 'Jute', 'Coffee']
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        info = {}
        for model_type, model_data in self.models.items():
            info[model_type] = {
                'model_name': model_data.get('model_name', 'Unknown'),
                'model_type': type(model_data['model']).__name__,
                'available': True
            }
        
        # Add missing models
        expected_models = ['crop', 'yield', 'rainfall', 'fertilizer']
        for model_type in expected_models:
            if model_type not in info:
                info[model_type] = {
                    'model_name': 'Not loaded',
                    'model_type': 'Not available',
                    'available': False
                }
        
        return info

# Example usage
if __name__ == "__main__":
    predictor = Predictor()
    
    # Example predictions
    print("Crop Prediction:", predictor.predict_crop(50, 50, 50, 25, 70, 6.5, 100))
    print("Yield Prediction:", predictor.predict_yield("Andhra Pradesh", "Anantapur", "Kharif", "Rice", 1000, 5000))
    print("Rainfall Prediction:", predictor.predict_rainfall("July", 30, 80))
    print("Fertilizer Prediction:", predictor.predict_fertilizer(25, 70, 20, "Loamy", "Maize", 50, 50, 50, 6.5)) 