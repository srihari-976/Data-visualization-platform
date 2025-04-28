# feature_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import logging
from itertools import combinations
from utils.data_cleaning import DataCleaner

def plot_to_base64():
    """Convert current matplotlib plot to base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def analyze_features(file_path, target_col=None):
    """
    Perform comprehensive feature analysis on the uploaded CSV file
    
    Args:
        file_path (str): Path to the CSV file
        target_col (str, optional): Name of the target column for analysis
    """
    try:
        # Read the data
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded with shape: {df.shape}")
        
        # Initialize data cleaner
        cleaner = DataCleaner()
        
        # Clean and preprocess data
        df_cleaned = cleaner.clean_data(df)
        
        # Always include null_values and data_types
        null_values = df.isnull().sum().to_dict()
        data_types = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                data_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                data_types[col] = "datetime"
            else:
                data_types[col] = "categorical"

        cleaning_summary = {
            'original_shape': [int(x) for x in df.shape],  # Convert to Python int
            'cleaned_shape': [int(x) for x in df_cleaned.shape],  # Convert to Python int
            'column_types': {
                'numeric_columns': [str(col) for col in df.select_dtypes(include=['int64', 'float64']).columns],
                'categorical_columns': [str(col) for col in df.select_dtypes(include=['object']).columns]
            },
            'null_values': null_values,
            'data_types': data_types
        }
        logging.info("Data cleaning completed")
        
        # Generate visualizations
        visualizations = {}
        
        # 1. Correlation Matrix for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix Heatmap')
            visualizations['correlation_heatmap'] = plot_to_base64()
            plt.close()
        
        # 2. Distribution Plots for numeric columns (limit to first 5)
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                visualizations[f'dist_{col}'] = plot_to_base64()
                plt.close()
        
        # 3. Box Plots for numeric columns (limit to first 5)
        if len(numeric_cols) > 0:
            plt.figure(figsize=(15, 5))
            df[numeric_cols[:5]].boxplot()  # Limit to first 5 columns
            plt.title('Box Plots of Numeric Features')
            plt.xticks(rotation=45)
            visualizations['box_plots'] = plot_to_base64()
            plt.close()
        
        # 4. Scatter Matrix for numeric columns (limit to first 5)
        if len(numeric_cols) > 0:
            cols_for_scatter = numeric_cols[:5]  # Limit to first 5 columns
            if len(cols_for_scatter) >= 2:
                plt.figure(figsize=(15, 15))
                pd.plotting.scatter_matrix(df[cols_for_scatter], diagonal='kde')
                plt.tight_layout()
                visualizations['scatter_matrix'] = plot_to_base64()
                plt.close()
        
        # 5. Bar plots for categorical columns (limit to first 5)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            value_counts = df[col].value_counts()
            # Only plot if number of unique values is between 2 and 15
            if 2 <= len(value_counts) <= 15:
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar', color=plt.get_cmap('tab10').colors[:len(value_counts)])
                plt.title(f'Value Counts for {col}')
                plt.xticks(rotation=45)
                visualizations[f'bar_{col}'] = plot_to_base64()
                plt.close()
        
        # 6. Pair plots for numeric columns (limit to first 5)
        if len(numeric_cols) >= 2:
            cols_for_pair = numeric_cols[:5]
            plt.figure(figsize=(15, 15))
            sns.pairplot(df[cols_for_pair])
            plt.tight_layout()
            visualizations['pair_plot'] = plot_to_base64()
            plt.close()
        
        # Prepare final results with compressed visualization data
        results = {
            'status': 'success',
            'data': {
                'visualizations': {
                    'count': len(visualizations),
                    'types': list(visualizations.keys()),
                    'data': visualizations  # Store actual visualization data
                },
                'cleaning_summary': cleaning_summary,
                'dataset_info': {
                    'total_rows': int(df.shape[0]),
                    'total_columns': int(df.shape[1]),
                    'numeric_columns': [str(col) for col in numeric_cols],
                    'categorical_columns': [str(col) for col in categorical_cols]
                }
            }
        }
        
        # Ensure all data is JSON serializable
        results = convert_numpy_types(results)
        
        return results
        
    except Exception as e:
        logging.error(f"Error analyzing features: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage
    file_path = "your_data.csv"  # Replace with actual file path
    results = analyze_features(file_path)
    print("Analysis completed successfully") 