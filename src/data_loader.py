"""
Data loader module for EcoHarvestors project.
Handles loading and preprocessing of agricultural datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of agricultural datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing the datasets
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_crop_data(self) -> pd.DataFrame:
        """
        Load crop recommendation dataset.
        
        Returns:
            DataFrame with crop recommendation data
        """
        try:
            # This would typically load from a CSV file
            # For now, creating sample data structure
            data = {
                'N': np.random.randint(0, 140, 1000),
                'P': np.random.randint(5, 145, 1000),
                'K': np.random.randint(5, 205, 1000),
                'temperature': np.random.uniform(8.0, 44.0, 1000),
                'humidity': np.random.uniform(14.0, 100.0, 1000),
                'ph': np.random.uniform(3.5, 10.0, 1000),
                'rainfall': np.random.uniform(20.0, 300.0, 1000),
                'label': np.random.choice(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                                         'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                                         'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
                                         'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute',
                                         'coffee'], 1000)
            }
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading crop data: {e}")
            return pd.DataFrame()
    
    def load_yield_data(self) -> pd.DataFrame:
        """
        Load crop yield prediction dataset.
        
        Returns:
            DataFrame with yield prediction data
        """
        try:
            # Sample yield data structure
            data = {
                'State': np.random.choice(['Andhra Pradesh', 'Karnataka', 'Tamil Nadu', 'Maharashtra'], 1000),
                'District': np.random.choice(['Anantapur', 'Kurnool', 'Chittoor', 'Kadapa'], 1000),
                'Season': np.random.choice(['Kharif', 'Rabi', 'Whole Year', 'Autumn', 'Summer', 'Winter'], 1000),
                'Crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton'], 1000),
                'Area': np.random.uniform(100, 10000, 1000),
                'Production': np.random.uniform(100, 50000, 1000),
                'Yield': np.random.uniform(0.5, 5.0, 1000)
            }
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading yield data: {e}")
            return pd.DataFrame()
    
    def load_rainfall_data(self) -> pd.DataFrame:
        """
        Load rainfall prediction dataset.
        
        Returns:
            DataFrame with rainfall prediction data
        """
        try:
            # Sample rainfall data structure
            data = {
                'Month': np.random.choice(['January', 'February', 'March', 'April', 'May', 'June',
                                         'July', 'August', 'September', 'October', 'November', 'December'], 1000),
                'Year': np.random.randint(2010, 2024, 1000),
                'Rainfall': np.random.uniform(0, 500, 1000),
                'Temperature': np.random.uniform(15, 40, 1000),
                'Humidity': np.random.uniform(30, 90, 1000)
            }
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading rainfall data: {e}")
            return pd.DataFrame()
    
    def load_fertilizer_data(self) -> pd.DataFrame:
        """
        Load fertilizer recommendation dataset.
        
        Returns:
            DataFrame with fertilizer recommendation data
        """
        try:
            # Sample fertilizer data structure
            data = {
                'Temparature': np.random.uniform(8.0, 44.0, 1000),
                'Humidity': np.random.uniform(14.0, 100.0, 1000),
                'Moisture': np.random.uniform(5.0, 50.0, 1000),
                'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], 1000),
                'Crop Type': np.random.choice(['Maize', 'Sugarcane', 'Cotton', 'Jute', 'Coffee'], 1000),
                'Nitrogen': np.random.randint(0, 140, 1000),
                'Phosphorous': np.random.randint(5, 145, 1000),
                'Potassium': np.random.randint(5, 205, 1000),
                'pH': np.random.uniform(3.5, 10.0, 1000),
                'Fertilizer Name': np.random.choice(['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea'], 1000)
            }
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading fertilizer data: {e}")
            return pd.DataFrame()
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to the processed directory.
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save
        """
        try:
            filepath = self.processed_dir / filename
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from the processed directory.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            filepath = self.processed_dir / filename
            if filepath.exists():
                return pd.read_csv(filepath)
            else:
                logger.warning(f"File {filepath} not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return pd.DataFrame() 