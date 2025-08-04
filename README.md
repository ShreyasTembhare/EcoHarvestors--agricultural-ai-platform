# Ecoharvestors 🌱

**AI-powered crop yield prediction system for sustainable agriculture**

Ecoharvestors is a comprehensive agricultural intelligence platform that leverages machine learning to predict crop yields, optimize harvest timing, and minimize post-harvest waste. Built with modern technologies and designed for scalability, it serves farmers, agronomists, and cooperative managers.

## 🚀 Features

### Core Capabilities
- **Multi-Model ML Ensemble**: Random Forest, XGBoost, and Bidirectional LSTM for robust predictions
- **Real-time Processing**: Sub-3-second predictions deployed on cloud infrastructure
- **Smart Recommendations**: Harvest timing, quantity adjustments, and distribution optimization
- **Spoilage Risk Assessment**: Real-time monitoring and alerts for post-harvest losses
- **Demand Forecasting**: Market-aware production planning

### User Management
- **Role-based Access**: Farmers, Agronomists, Admins, Cooperative Managers
- **Guided Onboarding**: Step-by-step farm setup and data collection
- **Multi-farm Support**: Cooperative management and aggregate forecasting

### Data Integration
- **Weather APIs**: Real-time and historical weather data
- **Satellite Imagery**: NDVI and vegetation health monitoring
- **IoT Sensors**: Soil moisture, temperature, and environmental tracking
- **Market Data**: Price trends and demand signals

### Advanced Analytics
- **What-if Simulations**: Test different scenarios and strategies
- **Risk-adjusted Recommendations**: Personalized based on farmer preferences
- **Performance Tracking**: Historical accuracy and improvement metrics
- **Drift Detection**: Automated model monitoring and retraining

## 🏗️ Architecture

```
Ecoharvestors/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # REST API endpoints
│   │   ├── core/           # Configuration and security
│   │   ├── models/         # ML models and data models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities and helpers
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API integration
│   │   └── utils/          # Frontend utilities
├── ml_models/              # ML model training and deployment
├── data/                   # Sample data and datasets
└── docs/                   # Documentation
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and session management
- **Celery**: Background task processing

### Frontend
- **React**: Modern UI framework
- **TypeScript**: Type safety
- **Material-UI**: Component library
- **Chart.js**: Data visualization
- **React Query**: Server state management

### Machine Learning
- **TensorFlow**: Deep learning models
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting
- **Pandas/NumPy**: Data processing
- **SMOTE**: Synthetic data generation

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Local development
- **Nginx**: Reverse proxy
- **JWT**: Authentication

## 📊 ML Models

### Yield Prediction
- **Random Forest**: Baseline predictions
- **XGBoost**: Gradient boosting for complex patterns
- **Bidirectional LSTM**: Sequential data modeling
- **Ensemble Methods**: Weighted combination for robustness

### Feature Engineering
- **50,000+ Data Points**: Comprehensive agricultural dataset
- **SMOTE Optimization**: 5,000+ synthetic samples
- **Feature Reduction**: 100+ → 25 optimized features
- **Real-time Processing**: Millions of records

### Model Performance
- **Accuracy**: 85%+ prediction accuracy
- **Latency**: <3 seconds response time
- **Scalability**: Handles millions of records
- **Adaptability**: Continuous learning from feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker and Docker Compose
- PostgreSQL 13+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/eco-harvest-predictor.git
cd eco-harvest-predictor
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Database Setup**
```bash
# Using Docker Compose
docker-compose up -d postgres redis
```

5. **Run the Application**
```bash
# Backend
cd backend
uvicorn app.main:app --reload

# Frontend
cd frontend
npm start
```

## 📈 Key Metrics

- **Prediction Accuracy**: 85%+
- **Response Time**: <3 seconds
- **Data Points Analyzed**: 50,000+
- **Synthetic Samples**: 5,000+
- **Feature Optimization**: 100+ → 25 features
- **Real-time Processing**: Millions of records

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Agricultural research institutions for data insights
- Open-source community for tools and libraries
- Farmers and agronomists for domain expertise

## 📞 Support

For support, email support@ecoharvestors.com or join our Slack channel.

---

**Ecoharvestors** - Empowering sustainable agriculture through AI-driven insights.
