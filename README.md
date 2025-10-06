# EcoHarvestors - Agricultural AI Platform

EcoHarvestors is an advanced agricultural intelligence platform that leverages machine learning to provide farmers with data-driven insights and predictions. The platform offers crop recommendations, yield predictions, rainfall forecasting, and fertilizer optimization to help farmers make informed decisions and increase productivity.

## Features

- **Crop Recommendation**: Personalized crop suggestions based on soil conditions and climate data
- **Yield Prediction**: Forecast crop yields with high accuracy using historical data
- **Rainfall Prediction**: Predict rainfall patterns and seasonal variations
- **Fertilizer Recommendation**: Optimize fertilizer usage with AI-powered recommendations
- **Analytics Dashboard**: Interactive visualizations of agricultural trends

## Project Structure

```
├── data/
│   ├── raw/            # CSV datasets
│   └── processed/      # Cleaned & feature-engineered files
├── models/
│   ├── train.py        # Model training script
│   ├── predict.py      # Prediction functions
│   └── *.pkl          # Trained model files
├── src/
│   ├── data_loader.py  # Data loading functions
│   └── metrics.py      # Evaluation helpers
├── app/
│   ├── streamlit_app.py    # Main Streamlit application
│   └── pages/              # Multipage files
├── requirements.txt    # Dependencies
└── run_app.py         # Application launcher
```

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EcoHarvestors--agricultural-ai-platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

Place your agricultural datasets in the `data/raw/` directory:

- **Crop Recommendation**: `crop_data.csv` (N, P, K, temperature, humidity, pH, rainfall, label)
- **Yield Prediction**: `yield_data.csv` (State, District, Season, Crop, Area, Production, Yield)
- **Rainfall Prediction**: `rainfall_data.csv` (Month, Year, Temperature, Humidity, Rainfall)
- **Fertilizer Recommendation**: `fertilizer_data.csv` (Temperature, Humidity, Moisture, Soil Type, Crop Type, N, P, K, pH, Fertilizer Name)

### Training & Running

1. **Train models**
   ```bash
   python models/train.py
   ```

2. **Run application**
   ```bash
   python run_app.py
   # or: streamlit run app/streamlit_app.py
   ```

The application will be available at `http://localhost:8501`

## Data Sources

- [Ministry of Agriculture & Farmers Welfare](https://agriculture.gov.in/)
- [Indian Council of Agricultural Research (ICAR)](https://icar.org.in/)
- State Agricultural Departments
- Research Institutions and KVKs

## Model Performance

- **Crop Recommendation**: 85-95% accuracy
- **Yield Prediction**: RMSE < 0.5 tons/hectare
- **Rainfall Prediction**: RMSE < 50mm
- **Fertilizer Recommendation**: 80-90% accuracy

## Troubleshooting

- **Model not available**: Train models first using `python models/train.py`
- **No data available**: Place CSV files in `data/raw/` directory
- **Poor accuracy**: Improve data quality and increase dataset size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**EcoHarvestors** - Empowering farmers with AI-driven agricultural intelligence 