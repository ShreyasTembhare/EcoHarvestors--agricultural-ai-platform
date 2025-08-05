"""
Metrics module for EcoHarvestors project.
Provides evaluation helpers for model performance assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance metrics."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.metrics = {}
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str = "model") -> dict:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for storing metrics
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Store metrics
            self.metrics[model_name] = metrics
            
            logger.info(f"Classification metrics for {model_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {}
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "model") -> dict:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for storing metrics
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
            
            # Store metrics
            self.metrics[model_name] = metrics
            
            logger.info(f"Regression metrics for {model_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            return {}
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: list = None, title: str = "Confusion Matrix") -> go.Figure:
        """
        Create confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(cm))]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")
            return go.Figure()
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Regression Results") -> go.Figure:
        """
        Create regression results plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Actual vs Predicted', 'Residuals'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Actual vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Residuals
            residuals = y_true - y_pred
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='green', size=8)
                ),
                row=1, col=2
            )
            
            # Zero line for residuals
            fig.add_trace(
                go.Scatter(
                    x=[y_pred.min(), y_pred.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=title,
                width=1000,
                height=500
            )
            
            fig.update_xaxes(title_text="Actual", row=1, col=1)
            fig.update_yaxes(title_text="Predicted", row=1, col=1)
            fig.update_xaxes(title_text="Predicted", row=1, col=2)
            fig.update_yaxes(title_text="Residuals", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regression results plot: {e}")
            return go.Figure()
    
    def plot_feature_importance(self, feature_names: list, importance_scores: np.ndarray,
                               title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance plot.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            # Sort features by importance
            sorted_indices = np.argsort(importance_scores)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_scores = importance_scores[sorted_indices]
            
            fig = go.Figure(data=go.Bar(
                x=sorted_scores,
                y=sorted_features,
                orientation='h',
                marker=dict(color='lightblue')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return go.Figure()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    target_names: list = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Formatted classification report
        """
        try:
            report = classification_report(y_true, y_pred, target_names=target_names)
            return report
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return ""
    
    def plot_metrics_comparison(self, metrics_dict: dict, title: str = "Model Comparison") -> go.Figure:
        """
        Create comparison plot for multiple models.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics as values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            models = list(metrics_dict.keys())
            metrics = list(metrics_dict[models[0]].keys())
            
            fig = make_subplots(
                rows=1, cols=len(metrics),
                subplot_titles=metrics,
                specs=[[{"type": "bar"} for _ in range(len(metrics))]]
            )
            
            for i, metric in enumerate(metrics):
                values = [metrics_dict[model][metric] for model in models]
                
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=values,
                        name=metric,
                        marker=dict(color=f'rgb({50 + i*50}, {100 + i*30}, {150 + i*20})')
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title=title,
                width=300 * len(metrics),
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics comparison plot: {e}")
            return go.Figure()
    
    def save_metrics_report(self, filename: str = "metrics_report.txt") -> None:
        """
        Save all stored metrics to a file.
        
        Args:
            filename: Name of the file to save
        """
        try:
            with open(filename, 'w') as f:
                f.write("EcoHarvestors Model Metrics Report\n")
                f.write("=" * 40 + "\n\n")
                
                for model_name, metrics in self.metrics.items():
                    f.write(f"Model: {model_name}\n")
                    f.write("-" * 20 + "\n")
                    for metric_name, value in metrics.items():
                        f.write(f"{metric_name}: {value:.4f}\n")
                    f.write("\n")
            
            logger.info(f"Metrics report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving metrics report: {e}") 