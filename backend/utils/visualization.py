import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def generate_visualizations(df, selected_features):
    """
    Generate various visualizations for the dataset using Plotly.
    Returns a list of visualization objects.
    """
    visualizations = []
    
    # Get numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # 1. Correlation Heatmap
    if len(numeric_columns) > 1:
        corr_matrix = df[numeric_columns].corr()
        heatmap = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            color_continuous_scale='RdBu'
        )
        visualizations.append({
            'type': 'heatmap',
            'data': heatmap.to_json()
        })
    
    # 2. Distribution Plots
    for col in numeric_columns:
        if col in selected_features:
            hist = px.histogram(
                df,
                x=col,
                title=f'Distribution of {col}',
                nbins=30
            )
            visualizations.append({
                'type': 'histogram',
                'data': hist.to_json()
            })
    
    # 3. Scatter Plots for numeric columns
    if len(numeric_columns) >= 2:
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                if numeric_columns[i] in selected_features and numeric_columns[j] in selected_features:
                    scatter = px.scatter(
                        df,
                        x=numeric_columns[i],
                        y=numeric_columns[j],
                        title=f'{numeric_columns[i]} vs {numeric_columns[j]}'
                    )
                    visualizations.append({
                        'type': 'scatter',
                        'data': scatter.to_json()
                    })
    
    # 4. Bar Plots for categorical columns
    for col in categorical_columns:
        if col in selected_features:
            value_counts = df[col].value_counts()
            bar = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {col}'
            )
            visualizations.append({
                'type': 'bar',
                'data': bar.to_json()
            })
    
    # 5. Box Plots for numeric columns by categorical columns
    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
        for num_col in numeric_columns:
            for cat_col in categorical_columns:
                if num_col in selected_features and cat_col in selected_features:
                    box = px.box(
                        df,
                        x=cat_col,
                        y=num_col,
                        title=f'{num_col} by {cat_col}'
                    )
                    visualizations.append({
                        'type': 'box',
                        'data': box.to_json()
                    })
    
    return visualizations 