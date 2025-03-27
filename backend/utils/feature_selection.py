import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def select_features(df, n_components=0.95):
    """
    Select the most important features using PCA and correlation analysis.
    Returns a list of selected feature names.
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        return list(df.columns)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(scaled_features)
    
    # Get feature importance scores
    feature_importance = pd.DataFrame({
        'feature': numeric_df.columns,
        'importance': np.abs(pca.components_).sum(axis=0)
    })
    
    # Sort features by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = feature_importance['feature'].tolist()
    
    # Add categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    selected_features.extend(categorical_columns)
    
    return selected_features 