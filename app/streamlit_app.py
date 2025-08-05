"""
Main Streamlit application for EcoHarvestors project.
Provides a modern web interface for agricultural predictions.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.predict import Predictor
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="EcoHarvestors - Agricultural AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class EcoHarvestorsApp:
    """Main application class for EcoHarvestors."""
    
    def __init__(self):
        """Initialize the application."""
        self.predictor = Predictor()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Setup session state variables."""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Home'
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🌾 EcoHarvestors</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Agricultural Intelligence Platform</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("Navigation")
        
        pages = {
            'Home': '🏠',
            'Crop Recommendation': '🌱',
            'Yield Prediction': '📊',
            'Rainfall Prediction': '🌧️',
            'Fertilizer Recommendation': '🌿',
            'Analytics': '📈'
        }
        
        selected_page = st.sidebar.selectbox(
            "Choose a page:",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]} {x}"
        )
        
        st.session_state.current_page = selected_page
        
        # Model status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Model Status")
        model_info = self.predictor.get_model_info()
        
        for model_type, info in model_info.items():
            status = "✅" if info['available'] else "❌"
            st.sidebar.text(f"{status} {model_type.title()}")
    
    def render_home_page(self):
        """Render the home page."""
        st.markdown('<h2 class="sub-header">Welcome to EcoHarvestors</h2>', unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **EcoHarvestors** is an advanced agricultural intelligence platform that leverages 
            machine learning to provide farmers with data-driven insights and predictions.
            
            ### 🌟 Key Features:
            - **Crop Recommendation**: Get personalized crop suggestions based on soil and climate data
            - **Yield Prediction**: Forecast crop yields with high accuracy
            - **Rainfall Prediction**: Predict rainfall patterns for better planning
            - **Fertilizer Recommendation**: Optimize fertilizer usage for maximum efficiency
            
            ### 🎯 Benefits:
            - Increase crop productivity
            - Reduce resource wastage
            - Make informed farming decisions
            - Adapt to climate changes
            """)
        
        with col2:
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            
            # Sample data for demonstration
            stats_data = {
                'Metric': ['Models Trained', 'Accuracy', 'Data Points', 'Crops Supported'],
                'Value': ['4', '95%', '10,000+', '22']
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True)
        
        # Feature cards
        st.markdown("### 🚀 Platform Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🌱 Crop Recommendation</h4>
                <p>Get personalized crop suggestions based on your soil conditions, 
                climate data, and local factors. Our AI analyzes multiple parameters 
                to recommend the most suitable crops for your farm.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Yield Prediction</h4>
                <p>Forecast crop yields with high accuracy using historical data, 
                weather patterns, and farming practices. Plan your harvest and 
                optimize resource allocation.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🌧️ Rainfall Prediction</h4>
                <p>Predict rainfall patterns and seasonal variations to plan 
                irrigation schedules and protect crops from adverse weather conditions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4>🌿 Fertilizer Recommendation</h4>
                <p>Optimize fertilizer usage with AI-powered recommendations. 
                Reduce costs while maximizing crop health and productivity.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_crop_recommendation_page(self):
        """Render the crop recommendation page."""
        st.markdown('<h2 class="sub-header">🌱 Crop Recommendation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Get personalized crop recommendations based on your soil conditions and climate data.
        Enter the following parameters to receive AI-powered crop suggestions.
        """)
        
        # Input form
        with st.form("crop_recommendation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                N = st.slider("Nitrogen (N) - kg/ha", 0, 140, 50)
                P = st.slider("Phosphorus (P) - kg/ha", 5, 145, 50)
                K = st.slider("Potassium (K) - kg/ha", 5, 205, 50)
                temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
            
            with col2:
                humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
                ph = st.slider("pH Value", 3.5, 10.0, 6.5)
                rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
            
            submitted = st.form_submit_button("Get Recommendation", type="primary")
        
        if submitted:
            with st.spinner("Analyzing soil and climate data..."):
                recommendation = self.predictor.predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>🌾 Recommended Crop</h3>
                    <h2 style="color: #2E8B57;">{recommendation.title()}</h2>
                    <p>Based on your soil and climate conditions, this crop is most suitable for your farm.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input parameters
                st.markdown("### 📋 Input Parameters")
                params_df = pd.DataFrame({
                    'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                    'Value': [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha", f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall} mm"]
                })
                st.dataframe(params_df, hide_index=True)
    
    def render_yield_prediction_page(self):
        """Render the yield prediction page."""
        st.markdown('<h2 class="sub-header">📊 Yield Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Predict crop yields based on historical data, farming practices, and environmental factors.
        This helps in planning harvest schedules and resource allocation.
        """)
        
        # Input form
        with st.form("yield_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                state = st.selectbox("State", self.predictor.get_available_states())
                district = st.selectbox("District", self.predictor.get_available_districts())
                season = st.selectbox("Season", self.predictor.get_available_seasons())
                crop = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Cotton"])
            
            with col2:
                area = st.number_input("Area (hectares)", min_value=1.0, max_value=10000.0, value=1000.0)
                production = st.number_input("Production (tons)", min_value=1.0, max_value=50000.0, value=5000.0)
            
            submitted = st.form_submit_button("Predict Yield", type="primary")
        
        if submitted:
            with st.spinner("Calculating yield prediction..."):
                predicted_yield = self.predictor.predict_yield(state, district, season, crop, area, production)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>📈 Predicted Yield</h3>
                    <h2 style="color: #2E8B57;">{predicted_yield:.2f} tons/hectare</h2>
                    <p>Expected yield per hectare for the specified conditions.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input parameters
                st.markdown("### 📋 Input Parameters")
                params_df = pd.DataFrame({
                    'Parameter': ['State', 'District', 'Season', 'Crop', 'Area', 'Production'],
                    'Value': [state, district, season, crop, f"{area} hectares", f"{production} tons"]
                })
                st.dataframe(params_df, hide_index=True)
    
    def render_rainfall_prediction_page(self):
        """Render the rainfall prediction page."""
        st.markdown('<h2 class="sub-header">🌧️ Rainfall Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Predict rainfall patterns to plan irrigation schedules and protect crops from adverse weather conditions.
        Enter the following parameters to get rainfall predictions.
        """)
        
        # Input form
        with st.form("rainfall_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                month = st.selectbox("Month", self.predictor.get_available_months())
                temperature = st.slider("Temperature (°C)", 15.0, 40.0, 30.0)
            
            with col2:
                humidity = st.slider("Humidity (%)", 30.0, 90.0, 80.0)
            
            submitted = st.form_submit_button("Predict Rainfall", type="primary")
        
        if submitted:
            with st.spinner("Analyzing weather patterns..."):
                predicted_rainfall = self.predictor.predict_rainfall(month, temperature, humidity)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>🌧️ Predicted Rainfall</h3>
                    <h2 style="color: #2E8B57;">{predicted_rainfall:.1f} mm</h2>
                    <p>Expected rainfall for {month} based on current conditions.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input parameters
                st.markdown("### 📋 Input Parameters")
                params_df = pd.DataFrame({
                    'Parameter': ['Month', 'Temperature', 'Humidity'],
                    'Value': [month, f"{temperature}°C", f"{humidity}%"]
                })
                st.dataframe(params_df, hide_index=True)
    
    def render_fertilizer_recommendation_page(self):
        """Render the fertilizer recommendation page."""
        st.markdown('<h2 class="sub-header">🌿 Fertilizer Recommendation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Get optimized fertilizer recommendations based on soil conditions, crop type, and nutrient requirements.
        This helps in reducing costs while maximizing crop health and productivity.
        """)
        
        # Input form
        with st.form("fertilizer_recommendation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
                humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
                moisture = st.slider("Moisture (%)", 5.0, 50.0, 20.0)
                soil_type = st.selectbox("Soil Type", self.predictor.get_available_soil_types())
                crop_type = st.selectbox("Crop Type", self.predictor.get_available_crop_types())
            
            with col2:
                nitrogen = st.slider("Nitrogen (N) - kg/ha", 0, 140, 50)
                phosphorous = st.slider("Phosphorous (P) - kg/ha", 5, 145, 50)
                potassium = st.slider("Potassium (K) - kg/ha", 5, 205, 50)
                ph = st.slider("pH Value", 3.5, 10.0, 6.5)
            
            submitted = st.form_submit_button("Get Fertilizer Recommendation", type="primary")
        
        if submitted:
            with st.spinner("Analyzing soil conditions..."):
                recommendation = self.predictor.predict_fertilizer(
                    temperature, humidity, moisture, soil_type, crop_type,
                    nitrogen, phosphorous, potassium, ph
                )
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>🌿 Recommended Fertilizer</h3>
                    <h2 style="color: #2E8B57;">{recommendation}</h2>
                    <p>Optimal fertilizer for your soil and crop conditions.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input parameters
                st.markdown("### 📋 Input Parameters")
                params_df = pd.DataFrame({
                    'Parameter': ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'pH'],
                    'Value': [f"{temperature}°C", f"{humidity}%", f"{moisture}%", soil_type, crop_type, f"{nitrogen} kg/ha", f"{phosphorous} kg/ha", f"{potassium} kg/ha", f"{ph}"]
                })
                st.dataframe(params_df, hide_index=True)
    
    def render_analytics_page(self):
        """Render the analytics page."""
        st.markdown('<h2 class="sub-header">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Explore agricultural insights and trends through interactive visualizations.
        """)
        
        # Sample analytics data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌾 Crop Distribution")
            crop_data = pd.DataFrame({
                'Crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'],
                'Area (ha)': [45000, 38000, 32000, 28000, 25000],
                'Production (tons)': [180000, 152000, 128000, 84000, 125000]
            })
            
            fig = px.bar(crop_data, x='Crop', y='Area (ha)', title="Crop Area Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Yield Trends")
            yield_data = pd.DataFrame({
                'Year': [2019, 2020, 2021, 2022, 2023],
                'Average Yield (tons/ha)': [2.8, 3.1, 3.3, 3.5, 3.7]
            })
            
            fig = px.line(yield_data, x='Year', y='Average Yield (tons/ha)', title="Yield Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather patterns
        st.subheader("🌧️ Weather Patterns")
        weather_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Rainfall (mm)': [25, 30, 45, 80, 120, 180, 220, 200, 150, 90, 40, 30],
            'Temperature (°C)': [22, 24, 28, 32, 35, 33, 31, 30, 29, 28, 25, 23]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=weather_data['Month'], y=weather_data['Rainfall (mm)'], name='Rainfall'))
        fig.add_trace(go.Scatter(x=weather_data['Month'], y=weather_data['Temperature (°C)'], name='Temperature', yaxis='y2'))
        
        fig.update_layout(
            title="Monthly Weather Patterns",
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the main application."""
        self.render_header()
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.current_page == 'Home':
            self.render_home_page()
        elif st.session_state.current_page == 'Crop Recommendation':
            self.render_crop_recommendation_page()
        elif st.session_state.current_page == 'Yield Prediction':
            self.render_yield_prediction_page()
        elif st.session_state.current_page == 'Rainfall Prediction':
            self.render_rainfall_prediction_page()
        elif st.session_state.current_page == 'Fertilizer Recommendation':
            self.render_fertilizer_recommendation_page()
        elif st.session_state.current_page == 'Analytics':
            self.render_analytics_page()

if __name__ == "__main__":
    app = EcoHarvestorsApp()
    app.run() 