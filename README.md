# EcoHarvestors - Agricultural AI Platform

EcoHarvestors is an advanced agricultural intelligence platform that leverages machine learning to provide farmers with data-driven insights and predictions. The platform offers crop recommendations, yield predictions, rainfall forecasting, and fertilizer optimization to help farmers make informed decisions and increase productivity.

## Features

### Crop Recommendation
- Get personalized crop suggestions based on soil conditions and climate data
- Analyze multiple parameters including N-P-K levels, temperature, humidity, pH, and rainfall
- Support for 22+ different crop types

### Yield Prediction
- Forecast crop yields with high accuracy using historical data
- Consider factors like state, district, season, crop type, area, and production
- Help in planning harvest schedules and resource allocation

### Rainfall Prediction
- Predict rainfall patterns and seasonal variations
- Plan irrigation schedules and protect crops from adverse weather
- Based on temperature, humidity, and seasonal data

### Fertilizer Recommendation
- Optimize fertilizer usage with AI-powered recommendations
- Reduce costs while maximizing crop health and productivity
- Consider soil type, crop type, and nutrient requirements

### Analytics Dashboard
- Interactive visualizations of agricultural trends
- Crop distribution analysis
- Yield trends over time
- Weather pattern analysis

## Project Structure

```
Crop-management-project-Final-Year-/
├── data/
│   ├── raw/            # Place your CSV datasets here
│   └── processed/      # Cleaned & feature-engineered files
├── models/
│   ├── train.py        # Trains & evaluates your model
│   ├── predict.py      # Loads model.pkl and makes forecasts
│   ├── *.pkl          # Your trained model files
│   └── __init__.py     # Package initialization
├── src/
│   ├── data_loader.py  # Functions to read + prep data
│   ├── features.py     # Feature-engineering routines
│   ├── metrics.py      # Evaluation helpers
│   └── __init__.py     # Package initialization
├── app/
│   ├── streamlit_app.py    # Main Streamlit script
│   └── pages/              # Multipage files
├── requirements.txt    # Pin versions
├── run_app.py         # Easy run script
└── README.md          # Project overview & how to run
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Crop-management-project-Final-Year-
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training Models with Your Data

### Step 1: Prepare Your Data

Place your agricultural datasets in the `data/raw/` directory. You'll need the following CSV files:

#### 1. Crop Recommendation Dataset (`crop_data.csv`)
```csv
N,P,K,temperature,humidity,ph,rainfall,label
50,50,50,25,70,6.5,100,rice
45,60,40,30,80,7.0,150,maize
...
```

**Required Columns:**
- `N`: Nitrogen content (kg/ha)
- `P`: Phosphorus content (kg/ha)
- `K`: Potassium content (kg/ha)
- `temperature`: Temperature in Celsius
- `humidity`: Humidity percentage
- `ph`: pH value
- `rainfall`: Rainfall in mm
- `label`: Crop name (target variable)

#### 2. Yield Prediction Dataset (`yield_data.csv`)
```csv
State,District,Season,Crop,Area,Production,Yield
Andhra Pradesh,Anantapur,Kharif,Rice,1000,5000,2.5
Karnataka,Bangalore,Rabi,Wheat,800,3200,2.0
...
```

**Required Columns:**
- `State`: State name
- `District`: District name
- `Season`: Season (Kharif/Rabi/etc.)
- `Crop`: Crop name
- `Area`: Area in hectares
- `Production`: Production in tons
- `Yield`: Yield per hectare (target variable)

#### 3. Rainfall Prediction Dataset (`rainfall_data.csv`)
```csv
Month,Year,Temperature,Humidity,Rainfall
January,2020,22,65,25
February,2020,24,70,30
...
```

**Required Columns:**
- `Month`: Month name
- `Year`: Year
- `Temperature`: Temperature in Celsius
- `Humidity`: Humidity percentage
- `Rainfall`: Rainfall in mm (target variable)

#### 4. Fertilizer Recommendation Dataset (`fertilizer_data.csv`)
```csv
Temparature,Humidity,Moisture,Soil Type,Crop Type,Nitrogen,Phosphorous,Potassium,pH,Fertilizer Name
25,70,20,Loamy,Maize,50,50,50,6.5,Urea
30,80,25,Sandy,Rice,60,40,45,7.0,DAP
...
```

**Required Columns:**
- `Temparature`: Temperature in Celsius
- `Humidity`: Humidity percentage
- `Moisture`: Moisture content
- `Soil Type`: Type of soil
- `Crop Type`: Type of crop
- `Nitrogen`: Nitrogen content
- `Phosphorous`: Phosphorous content
- `Potassium`: Potassium content
- `pH`: pH value
- `Fertilizer Name`: Recommended fertilizer (target variable)

### Step 2: Update Data Loading

Edit `src/data_loader.py` to load your actual data:

```python
def load_crop_data(self) -> pd.DataFrame:
    """Load crop recommendation dataset."""
    try:
        # Load your actual CSV file
        filepath = self.raw_dir / "crop_data.csv"
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error loading crop data: {e}")
        return pd.DataFrame()

def load_yield_data(self) -> pd.DataFrame:
    """Load yield prediction dataset."""
    try:
        # Load your actual CSV file
        filepath = self.raw_dir / "yield_data.csv"
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error loading yield data: {e}")
        return pd.DataFrame()

# Similar updates for rainfall and fertilizer data
```

### Step 3: Train Models

Run the training script to train models with your data:

```bash
python models/train.py
```

This will:
- Load your datasets from `data/raw/`
- Preprocess and feature engineer the data
- Train Random Forest models for each prediction task
- Save trained models as `.pkl` files in `models/`
- Display training metrics (accuracy, RMSE, etc.)

### Step 4: Run the Application

Start the Streamlit application:

```bash
# Option 1: Using the run script
python run_app.py

# Option 2: Direct streamlit command
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Data Requirements for Indian Agriculture

### Recommended Data Sources

1. **Government Sources:**
   - [Ministry of Agriculture & Farmers Welfare](https://agriculture.gov.in/)
   - [Indian Council of Agricultural Research (ICAR)](https://icar.org.in/)
   - [Department of Agriculture Cooperation & Farmers Welfare](https://agricoop.gov.in/)

2. **State Agricultural Departments:**
   - State-wise agricultural statistics
   - District-level crop data
   - Weather and rainfall data

3. **Research Institutions:**
   - ICAR research centers
   - Agricultural universities
   - Krishi Vigyan Kendras (KVKs)

### Data Collection Guidelines

#### For Crop Recommendation:
- Collect soil test reports from your region
- Include weather data (temperature, humidity, rainfall)
- Sample size: At least 500-1000 records per crop
- Geographic coverage: District/state level

#### For Yield Prediction:
- Historical crop production data
- Area under cultivation
- Weather patterns and seasonal data
- Sample size: 5-10 years of data
- Geographic coverage: State/district level

#### For Rainfall Prediction:
- Historical rainfall data from IMD
- Temperature and humidity records
- Seasonal patterns
- Sample size: 10-20 years of monthly data

#### For Fertilizer Recommendation:
- Soil test results
- Crop-specific fertilizer trials
- Local agricultural practices
- Sample size: 500-1000 records

### Data Quality Checklist

- **Completeness**: No missing values in critical columns
- **Accuracy**: Data verified against official sources
- **Consistency**: Uniform units and formats
- **Relevance**: Data from your target region
- **Timeliness**: Recent data (within 5 years)

## Customization

### Adding New Crops

1. **Update crop data** with new crop types
2. **Retrain the crop model**:
   ```bash
   python models/train.py
   ```

### Adding New Regions

1. **Add new states/districts** to your yield data
2. **Include regional weather patterns** in rainfall data
3. **Retrain models** to include new geographic data

### Model Tuning

Edit `models/train.py` to adjust model parameters:

```python
# For better accuracy, increase n_estimators
model = RandomForestClassifier(n_estimators=200, random_state=42)

# For faster training, decrease n_estimators
model = RandomForestClassifier(n_estimators=50, random_state=42)
```

## Model Performance

### Expected Performance with Good Data:

- **Crop Recommendation**: 85-95% accuracy
- **Yield Prediction**: RMSE < 0.5 tons/hectare
- **Rainfall Prediction**: RMSE < 50mm
- **Fertilizer Recommendation**: 80-90% accuracy

### Improving Model Performance:

1. **More Data**: Increase dataset size
2. **Better Features**: Add relevant variables
3. **Data Quality**: Clean and validate data
4. **Feature Engineering**: Create derived features
5. **Model Tuning**: Adjust hyperparameters

## Key Benefits

- **Increase Crop Productivity**: Data-driven crop recommendations
- **Reduce Resource Wastage**: Optimized fertilizer usage
- **Make Informed Decisions**: Accurate yield and rainfall predictions
- **Adapt to Climate Changes**: Weather pattern analysis
- **User-Friendly Interface**: Modern web application with intuitive design

## Troubleshooting

### Common Issues:

1. **"Model not available" error:**
   - Train models first using `python models/train.py`
   - Check if `.pkl` files exist in `models/` directory

2. **"No data available" error:**
   - Place your CSV files in `data/raw/`
   - Update data loading functions in `src/data_loader.py`

3. **Poor prediction accuracy:**
   - Improve data quality
   - Increase dataset size
   - Add more relevant features

4. **Installation issues:**
   - Use virtual environment
   - Install requirements: `pip install -r requirements.txt`

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit for the web interface
- Powered by scikit-learn for machine learning
- Agricultural data insights and best practices
- Indian agricultural research institutions

## Support

For questions, issues, or feature requests:

- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for common solutions

---

**EcoHarvestors** - Empowering Indian farmers with AI-driven agricultural intelligence

*Train once, predict anywhere - Your agricultural AI companion for better farming decisions.* 