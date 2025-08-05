"""
Training module for EcoHarvestors project.
Trains and evaluates machine learning models for agricultural predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import logging

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.metrics import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training and evaluation of agricultural prediction models."""
    
    def __init__(self):
        """Initialize ModelTrainer."""
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.best_models = {}
    
    def train_crop_model(self) -> dict:
        """
        Train crop recommendation model.
        
        Returns:
            Dictionary containing model performance metrics
        """
        try:
            logger.info("Loading crop data...")
            data = self.data_loader.load_crop_data()
            
            if data.empty:
                logger.error("No crop data available")
                return {}
            
            logger.info("Preprocessing crop data...")
            X_train, X_test, y_train, y_test, scaler, encoder = self.feature_engineer.preprocess_crop_data(data)
            
            if X_train is None:
                logger.error("Failed to preprocess crop data")
                return {}
            
            # Define models to try
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'DecisionTree': DecisionTreeClassifier(random_state=42)
            }
            
            best_score = 0
            best_model = None
            best_model_name = None
            
            logger.info("Training crop models...")
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                metrics = self.evaluator.evaluate_classification_model(y_test, y_pred, f"crop_{name}")
                
                # Store model
                self.models[f"crop_{name}"] = model
                
                # Check if this is the best model
                if metrics['accuracy'] > best_score:
                    best_score = metrics['accuracy']
                    best_model = model
                    best_model_name = name
            
            # Store best model and preprocessing objects
            self.best_models['crop'] = {
                'model': best_model,
                'scaler': scaler,
                'encoder': encoder,
                'model_name': best_model_name
            }
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')
            joblib.dump(self.best_models['crop'], model_path)
            logger.info(f"Best crop model saved to {model_path}")
            
            return self.evaluator.metrics
            
        except Exception as e:
            logger.error(f"Error training crop model: {e}")
            return {}
    
    def train_yield_model(self) -> dict:
        """
        Train yield prediction model.
        
        Returns:
            Dictionary containing model performance metrics
        """
        try:
            logger.info("Loading yield data...")
            data = self.data_loader.load_yield_data()
            
            if data.empty:
                logger.error("No yield data available")
                return {}
            
            logger.info("Preprocessing yield data...")
            X_train, X_test, y_train, y_test, scaler, encoders = self.feature_engineer.preprocess_yield_data(data)
            
            if X_train is None:
                logger.error("Failed to preprocess yield data")
                return {}
            
            # Define models to try
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'DecisionTree': DecisionTreeRegressor(random_state=42)
            }
            
            best_score = float('inf')
            best_model = None
            best_model_name = None
            
            logger.info("Training yield models...")
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                metrics = self.evaluator.evaluate_regression_model(y_test, y_pred, f"yield_{name}")
                
                # Store model
                self.models[f"yield_{name}"] = model
                
                # Check if this is the best model (lower RMSE is better)
                if metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    best_model = model
                    best_model_name = name
            
            # Store best model and preprocessing objects
            self.best_models['yield'] = {
                'model': best_model,
                'scaler': scaler,
                'encoders': encoders,
                'model_name': best_model_name
            }
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), 'yield_model.pkl')
            joblib.dump(self.best_models['yield'], model_path)
            logger.info(f"Best yield model saved to {model_path}")
            
            return self.evaluator.metrics
            
        except Exception as e:
            logger.error(f"Error training yield model: {e}")
            return {}
    
    def train_rainfall_model(self) -> dict:
        """
        Train rainfall prediction model.
        
        Returns:
            Dictionary containing model performance metrics
        """
        try:
            logger.info("Loading rainfall data...")
            data = self.data_loader.load_rainfall_data()
            
            if data.empty:
                logger.error("No rainfall data available")
                return {}
            
            logger.info("Preprocessing rainfall data...")
            X_train, X_test, y_train, y_test, scaler, encoder = self.feature_engineer.preprocess_rainfall_data(data)
            
            if X_train is None:
                logger.error("Failed to preprocess rainfall data")
                return {}
            
            # Define models to try
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'DecisionTree': DecisionTreeRegressor(random_state=42)
            }
            
            best_score = float('inf')
            best_model = None
            best_model_name = None
            
            logger.info("Training rainfall models...")
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                metrics = self.evaluator.evaluate_regression_model(y_test, y_pred, f"rainfall_{name}")
                
                # Store model
                self.models[f"rainfall_{name}"] = model
                
                # Check if this is the best model (lower RMSE is better)
                if metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    best_model = model
                    best_model_name = name
            
            # Store best model and preprocessing objects
            self.best_models['rainfall'] = {
                'model': best_model,
                'scaler': scaler,
                'encoder': encoder,
                'model_name': best_model_name
            }
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), 'rainfall_model.pkl')
            joblib.dump(self.best_models['rainfall'], model_path)
            logger.info(f"Best rainfall model saved to {model_path}")
            
            return self.evaluator.metrics
            
        except Exception as e:
            logger.error(f"Error training rainfall model: {e}")
            return {}
    
    def train_fertilizer_model(self) -> dict:
        """
        Train fertilizer recommendation model.
        
        Returns:
            Dictionary containing model performance metrics
        """
        try:
            logger.info("Loading fertilizer data...")
            data = self.data_loader.load_fertilizer_data()
            
            if data.empty:
                logger.error("No fertilizer data available")
                return {}
            
            logger.info("Preprocessing fertilizer data...")
            X_train, X_test, y_train, y_test, scaler, encoders = self.feature_engineer.preprocess_fertilizer_data(data)
            
            if X_train is None:
                logger.error("Failed to preprocess fertilizer data")
                return {}
            
            # Define models to try
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'DecisionTree': DecisionTreeClassifier(random_state=42)
            }
            
            best_score = 0
            best_model = None
            best_model_name = None
            
            logger.info("Training fertilizer models...")
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                metrics = self.evaluator.evaluate_classification_model(y_test, y_pred, f"fertilizer_{name}")
                
                # Store model
                self.models[f"fertilizer_{name}"] = model
                
                # Check if this is the best model
                if metrics['accuracy'] > best_score:
                    best_score = metrics['accuracy']
                    best_model = model
                    best_model_name = name
            
            # Store best model and preprocessing objects
            self.best_models['fertilizer'] = {
                'model': best_model,
                'scaler': scaler,
                'encoders': encoders,
                'model_name': best_model_name
            }
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), 'fertilizer_model.pkl')
            joblib.dump(self.best_models['fertilizer'], model_path)
            logger.info(f"Best fertilizer model saved to {model_path}")
            
            return self.evaluator.metrics
            
        except Exception as e:
            logger.error(f"Error training fertilizer model: {e}")
            return {}
    
    def train_all_models(self) -> dict:
        """
        Train all models for the EcoHarvestors project.
        
        Returns:
            Dictionary containing all model performance metrics
        """
        logger.info("Starting training of all models...")
        
        all_metrics = {}
        
        # Train crop model
        logger.info("Training crop recommendation model...")
        crop_metrics = self.train_crop_model()
        all_metrics.update(crop_metrics)
        
        # Train yield model
        logger.info("Training yield prediction model...")
        yield_metrics = self.train_yield_model()
        all_metrics.update(yield_metrics)
        
        # Train rainfall model
        logger.info("Training rainfall prediction model...")
        rainfall_metrics = self.train_rainfall_model()
        all_metrics.update(rainfall_metrics)
        
        # Train fertilizer model
        logger.info("Training fertilizer recommendation model...")
        fertilizer_metrics = self.train_fertilizer_model()
        all_metrics.update(fertilizer_metrics)
        
        # Save metrics report
        self.evaluator.save_metrics_report("training_metrics_report.txt")
        
        logger.info("All models trained successfully!")
        return all_metrics

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.train_all_models()
    print("Training completed!")
    print("Metrics:", metrics) 