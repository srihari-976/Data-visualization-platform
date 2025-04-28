# visualization.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
from itertools import combinations
import json

logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert numpy and other non-serializable objects to serializable format."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (go.Figure, go.FigureWidget)):
        return json.loads(obj.to_json())
    return obj

def create_line_plots(df, selected_features, limit=10):
    """
    Create line plots for each feature against index.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit: Maximum number of plots to generate
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    for i, feature in enumerate(numeric_features[:limit]):
        fig = px.line(
            df, 
            y=feature, 
            title=f'Line Plot of {feature}',
            labels={'index': 'Index', 'y': feature}
        )
        
        fig.update_layout(
            height=500,
            width=800,
            xaxis_title='Index',
            yaxis_title=feature
        )
        
        plots.append({
            'type': 'line_plot',
            'title': f'Line Plot of {feature}',
            'feature': feature,
            'data': convert_to_serializable(fig)
        })
    
    return plots

def create_scatter_plots(df, selected_features, limit_pairs=15):
    """
    Create scatter plots for all combinations of selected features.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit_pairs: Maximum number of feature pairs to plot
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    # Generate all combinations of 2 features
    feature_pairs = list(combinations(numeric_features, 2))[:limit_pairs]
    
    for x_feature, y_feature in feature_pairs:
        fig = px.scatter(
            df, 
            x=x_feature, 
            y=y_feature, 
            title=f'Scatter Plot: {x_feature} vs {y_feature}',
            trendline="ols"
        )
        
        fig.update_layout(
            height=600,
            width=800,
            xaxis_title=x_feature,
            yaxis_title=y_feature
        )
        
        plots.append({
            'type': 'scatter_plot',
            'title': f'Scatter Plot: {x_feature} vs {y_feature}',
            'features': [x_feature, y_feature],
            'data': convert_to_serializable(fig)
        })
    
    return plots

def create_histograms(df, selected_features, limit=10):
    """
    Create histograms for each selected feature.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit: Maximum number of histograms to generate
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    for feature in numeric_features[:limit]:
        fig = px.histogram(
            df, 
            x=feature,
            title=f'Histogram of {feature}',
            marginal="rug",
            nbins=30
        )
        
        fig.update_layout(
            height=500,
            width=800,
            xaxis_title=feature,
            yaxis_title='Count'
        )
        
        plots.append({
            'type': 'histogram',
            'title': f'Histogram of {feature}',
            'feature': feature,
            'data': convert_to_serializable(fig)
        })
        
        # Also create distribution plot with KDE
        try:
            hist_data = [df[feature].dropna()]
            group_labels = [feature]
            
            fig_dist = ff.create_distplot(
                hist_data, 
                group_labels, 
                bin_size=0.2, 
                show_rug=True
            )
            
            fig_dist.update_layout(
                title=f'Distribution Plot with KDE for {feature}',
                height=500,
                width=800,
                xaxis_title=feature,
                yaxis_title='Density'
            )
            
            plots.append({
                'type': 'kde_plot',
                'title': f'Distribution Plot with KDE for {feature}',
                'feature': feature,
                'data': convert_to_serializable(fig_dist)
            })
        except Exception as e:
            logger.error(f"Error creating KDE plot for {feature}: {str(e)}")
    
    return plots

def create_box_plots(df, selected_features, limit=15):
    """
    Create box plots for each selected feature.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit: Maximum number of box plots to generate
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) > 0:
        # Create individual box plots
        for feature in numeric_features[:limit]:
            fig = px.box(
                df, 
                y=feature,
                title=f'Box Plot of {feature}',
                points="all"
            )
            
            fig.update_layout(
                height=500,
                width=700,
                yaxis_title=feature
            )
            
            plots.append({
                'type': 'box_plot',
                'title': f'Box Plot of {feature}',
                'feature': feature,
                'data': convert_to_serializable(fig)
            })
        
        # Create combined box plot
        fig = go.Figure()
        for feature in numeric_features[:10]:  # Limit to 10 features for combined plot
            fig.add_trace(go.Box(
                y=df[feature].values,
                name=feature,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title='Combined Box Plots of Numeric Features',
            height=600,
            width=max(800, len(numeric_features[:10]) * 80),
            yaxis_title='Value'
        )
        
        plots.append({
            'type': 'combined_box_plot',
            'title': 'Combined Box Plots of Numeric Features',
            'features': numeric_features[:10],
            'data': convert_to_serializable(fig)
        })
    
    return plots

def create_violin_plots(df, selected_features, limit=15):
    """
    Create violin plots for each selected feature.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit: Maximum number of violin plots to generate
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) > 0:
        # Create individual violin plots
        for feature in numeric_features[:limit]:
            fig = px.violin(
                df, 
                y=feature,
                title=f'Violin Plot of {feature}',
                box=True,
                points="all"
            )
            
            fig.update_layout(
                height=500,
                width=700,
                yaxis_title=feature
            )
            
            plots.append({
                'type': 'violin_plot',
                'title': f'Violin Plot of {feature}',
                'feature': feature,
                'data': convert_to_serializable(fig)
            })
        
        # Create combined violin plot
        fig = go.Figure()
        for feature in numeric_features[:10]:  # Limit to 10 features for combined plot
            fig.add_trace(go.Violin(
                y=df[feature].values,
                name=feature,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title='Combined Violin Plots of Numeric Features',
            height=600,
            width=max(800, len(numeric_features[:10]) * 80),
            yaxis_title='Value'
        )
        
        plots.append({
            'type': 'combined_violin_plot',
            'title': 'Combined Violin Plots of Numeric Features',
            'features': numeric_features[:10],
            'data': convert_to_serializable(fig)
        })
    
    return plots

def create_bar_plots(df, selected_features, limit=10):
    """
    Create bar plots for categorical features.
    """
    plots = []
    # Filter categorical features
    categorical_features = df[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
    for feature in categorical_features[:limit]:
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, 'count']
        n_unique = len(value_counts)
        # Only plot if number of unique values is between 2 and 15
        if 2 <= n_unique <= 15:
            # Limit to top 8 categories for clarity
            if n_unique > 8:
                value_counts = value_counts.head(8)
                title = f'Bar Plot of Top 8 Categories in {feature}'
            else:
                title = f'Bar Plot of {feature}'
            # Shorten x labels if too long
            max_label_len = 12
            full_labels = [str(x) for x in value_counts[feature]]
            short_labels = [x[:max_label_len] + ('...' if len(x) > max_label_len else '') for x in full_labels]
            fig = px.bar(
                value_counts,
                x=short_labels,
                y='count',
                title=title,
                color=feature,  # Add color for more distinction
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                height=500,
                width=max(800, len(value_counts) * 80),
                xaxis_title=feature,
                yaxis_title='Count',
                xaxis={
                    'categoryorder': 'total descending',
                    'tickvals': short_labels,
                    'ticktext': short_labels
                }
            )
            fig.update_xaxes(tickangle=45)
            # Add hover template to show full label
            fig.update_traces(hovertemplate=[f'{feature}: {full}<br>Count: %{{y}}<extra></extra>' for full in full_labels])
            plots.append({
                'type': 'bar_plot',
                'title': title,
                'feature': feature,
                'data': convert_to_serializable(fig)
            })
    return plots

def create_correlation_heatmap(df, selected_features=None):
    """
    Create correlation heatmap for numeric features.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
    
    Returns:
        Plotly figure object
    """
    # Filter numeric features
    if selected_features:
        numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 2:
        logger.warning("Not enough numeric features for correlation heatmap")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title='Correlation Heatmap',
        color_continuous_scale='Viridis',  # More colorful
        aspect="auto",
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=max(600, len(numeric_features) * 30),
        width=max(600, len(numeric_features) * 30)
    )
    
    return {
        'type': 'correlation_heatmap',
        'title': 'Correlation Heatmap',
        'features': numeric_features,
        'data': convert_to_serializable(fig)
    }

def create_pair_plots(df, selected_features, limit=6):
    """
    Create pair plots for combinations of selected features.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit: Maximum number of features to include (to avoid too many plots)
    
    Returns:
        Plotly figure object
    """
    # Filter numeric features and limit
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    features_subset = numeric_features[:min(limit, len(numeric_features))]
    
    if len(features_subset) < 2:
        logger.warning("Not enough numeric features for pair plot")
        return None
    
    # Create pair plot
    fig = px.scatter_matrix(
        df[features_subset],
        title=f'Pair Plot Matrix ({len(features_subset)} features)',
        dimensions=features_subset,
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        height=200 * len(features_subset),
        width=200 * len(features_subset)
    )
    
    # Update traces (diagonal)
    for i in range(len(features_subset)):
        fig.update_traces(histnorm='probability density', selector=dict(type='histogram'))
    
    return {
        'type': 'pair_plot',
        'title': f'Pair Plot Matrix ({len(features_subset)} features)',
        'features': features_subset,
        'data': convert_to_serializable(fig)
    }

def create_3d_plots(df, selected_features, limit_combinations=5):
    """
    Create 3D scatter plots for combinations of selected features.
    
    Args:
        df: DataFrame with features
        selected_features: List of selected feature names
        limit_combinations: Maximum number of feature combinations to plot
    
    Returns:
        List of plotly figure objects
    """
    plots = []
    
    # Filter numeric features
    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 3:
        logger.warning("Not enough numeric features for 3D plot")
        return plots
    
    # Generate combinations of 3 features
    feature_combos = list(combinations(numeric_features, 3))[:limit_combinations]
    
    for x_feature, y_feature, z_feature in feature_combos:
        fig = px.scatter_3d(
            df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            title=f'3D Scatter Plot: {x_feature}, {y_feature}, {z_feature}'
        )
        
        fig.update_layout(
            height=800,
            width=1000,
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature
            )
        )
        
        plots.append({
            'type': 'scatter_3d',
            'title': f'3D Scatter Plot: {x_feature}, {y_feature}, {z_feature}',
            'features': [x_feature, y_feature, z_feature],
            'data': convert_to_serializable(fig)
        })
    
    # Also create PCA-based 3D plot if we have enough features
    if len(numeric_features) > 3:
        try:
            # Scale the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_features])
            
            # Apply PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(
                data=pca_result,
                columns=['PC1', 'PC2', 'PC3']
            )
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                pca_df,
                x='PC1',
                y='PC2',
                z='PC3',
                title='PCA 3D Projection'
            )
            
            fig.update_layout(
                height=800,
                width=1000,
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
                )
            )
            
            plots.append({
                'type': 'pca_3d',
                'title': 'PCA 3D Projection',
                'features': numeric_features,
                'data': convert_to_serializable(fig)
            })
        except Exception as e:
            logger.error(f"Error creating PCA 3D plot: {str(e)}")
    
    return plots

def create_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Create ROC curve visualization."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600,
            width=800
        )
        
        return {
            'type': 'roc_curve',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating ROC curve: {str(e)}")
        return None

def create_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Create precision-recall curve visualization."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR curve (AUC = {pr_auc:.2f})'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=600,
            width=800
        )
        
        return {
            'type': 'precision_recall_curve',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating precision-recall curve: {str(e)}")
        return None

def create_learning_curve(estimator, X, y, title="Learning Curve"):
    """Create learning curve visualization."""
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines',
            name='Training score',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines',
            name='Cross-validation score',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Training score (std)'
        ))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='CV score (std)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Training examples',
            yaxis_title='Score',
            height=600,
            width=800
        )
        
        return {
            'type': 'learning_curve',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating learning curve: {str(e)}")
        return None

def create_feature_importance_plot(importance_scores, title="Feature Importance"):
    """Create feature importance plot visualization."""
    try:
        if not importance_scores or 'feature' not in importance_scores[0]:
            logger.warning("Invalid importance scores format")
            return None
            
        # Get feature names and importance scores
        features = [item['feature'] for item in importance_scores]
        scores = [item['importance'] if 'importance' in item 
                 else item['importance_score'] if 'importance_score' in item
                 else item['score'] if 'score' in item
                 else 0 for item in importance_scores]
        
        # Sort by importance
        sorted_indices = np.argsort(scores)
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        fig = px.bar(
            x=scores,
            y=features,
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(500, len(features) * 20),
            width=800
        )
        
        return {
            'type': 'feature_importance',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return None

def create_residual_plot(y_true, y_pred, title="Residual Plot"):
    """Create residual plot visualization."""
    try:
        residuals = y_true - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Residuals'
        ))
        fig.add_trace(go.Scatter(
            x=[min(y_pred), max(y_pred)],
            y=[0, 0],
            mode='lines',
            name='Zero line',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted values',
            yaxis_title='Residuals',
            height=600,
            width=800
        )
        
        # Add histogram of residuals
        fig_hist = px.histogram(
            x=residuals,
            title='Residuals Distribution',
            nbins=30
        )
        
        fig_hist.update_layout(
            xaxis_title='Residual',
            yaxis_title='Count',
            height=400,
            width=600
        )
        
        return {
            'type': 'residual_plot',
            'title': title,
            'plots': [
                convert_to_serializable(fig),
                convert_to_serializable(fig_hist)
            ]
        }
    except Exception as e:
        logger.error(f"Error creating residual plot: {str(e)}")
        return None

def create_decision_boundary_plot(X, y, model, title="Decision Boundary"):
    """Create decision boundary visualization."""
    try:
        if X.shape[1] != 2:
            logger.warning("Decision boundary plot requires exactly 2 features")
            return None
            
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale='Viridis',
            showscale=False
        ))
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(color=y, colorscale='Viridis'),
            name='Data points'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            height=600,
            width=800
        )
        
        return {
            'type': 'decision_boundary',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating decision boundary plot: {str(e)}")
        return None

def create_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix visualization."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Get labels for x and y axis based on matrix size
        num_classes = cm.shape[0]
        labels = [f'Class {i}' for i in range(num_classes)]
        
        if num_classes == 2:
            x_labels = ['Predicted Negative', 'Predicted Positive']
            y_labels = ['Actual Negative', 'Actual Positive']
        else:
            x_labels = labels
            y_labels = labels
            
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            annotation_text=cm
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=max(500, num_classes * 70),
            width=max(500, num_classes * 70)
        )
        
        # Fix for axis labels
        fig.update_xaxes(side="bottom")
        
        return {
            'type': 'confusion_matrix',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating confusion matrix plot: {str(e)}")
        return None

def create_elbow_plot(df, selected_features, max_clusters=10, title="Elbow Plot for K-means Clustering"):
    """Create elbow plot for K-means clustering."""
    try:
        # Filter numeric features
        numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            logger.warning("No numeric features for elbow plot")
            return None
            
        # Scale the data
        X = df[numeric_features].values
        X = StandardScaler().fit_transform(X)
        
        # Calculate inertia for different k values
        inertias = []
        k_values = range(1, min(max_clusters + 1, len(df) - 1))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Create elbow plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_values),
            y=inertias,
            mode='lines+markers',
            name='Inertia'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Number of Clusters',
            yaxis_title='Inertia',
            height=600,
            width=800
        )
        
        # Calculate silhouette scores for additional information
        silhouette_scores = []
        for k in k_values[1:]:  # Silhouette score isn't defined for k=1
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            try:
                silhouette_scores.append(silhouette_score(X, labels))
            except:
                silhouette_scores.append(0)
        
        # Create silhouette score plot
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(
            x=list(k_values[1:]),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score'
        ))
        
        fig_silhouette.update_layout(
            title='Silhouette Score for K-means Clustering',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score',
            height=600,
            width=800
        )
        
        return {
            'type': 'elbow_plot',
            'title': title,
            'plots': [
                convert_to_serializable(fig),
                convert_to_serializable(fig_silhouette)
            ],
            'optimal_k': np.argmin(np.diff(inertias)) + 1 if len(inertias) > 1 else 1
        }
    except Exception as e:
        logger.error(f"Error creating elbow plot: {str(e)}")
        return None

def create_cluster_plot(df, selected_features, n_clusters=3, title="Cluster Plot"):
    """Create cluster plot visualization using PCA for dimensionality reduction."""
    try:
        # Filter numeric features
        numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            logger.warning("No numeric features for cluster plot")
            return None
            
        # Scale the data
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        pca = PCA(n_components=min(3, len(numeric_features)))
        pca_result = pca.fit_transform(X_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        
        if pca_result.shape[1] >= 3:
            # 3D plot
            fig = px.scatter_3d(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                z=pca_result[:, 2],
                color=clusters,
                title=f"{title} (3D PCA Projection)",
                labels={
                    'x': f'PC1 ({explained_variance[0]:.2%})',
                    'y': f'PC2 ({explained_variance[1]:.2%})',
                    'z': f'PC3 ({explained_variance[2]:.2%})'
                }
            )
            
            fig.update_layout(
                height=800,
                width=1000
            )
            
            plots = [convert_to_serializable(fig)]
        elif pca_result.shape[1] == 2:
            # 2D plot
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                color=clusters,
                title=f"{title} (2D PCA Projection)",
                labels={
                    'x': f'PC1 ({explained_variance[0]:.2%})',
                    'y': f'PC2 ({explained_variance[1]:.2%})'
                }
            )
            
            fig.update_layout(
                height=700,
                width=900
            )
            
            plots = [convert_to_serializable(fig)]
        else:
            # 1D plot
            fig = px.scatter(
                x=pca_result[:, 0],
                y=np.zeros_like(pca_result[:, 0]),
                color=clusters,
                title=f"{title} (1D PCA Projection)",
                labels={
                    'x': f'PC1 ({explained_variance[0]:.2%})',
                    'y': ''
                }
            )
            
            fig.update_layout(
                height=500,
                width=800,
                yaxis_visible=False
            )
            
            plots = [convert_to_serializable(fig)]
        
        # Also create a parallel coordinates plot to visualize clusters across all features
        fig_parallel = px.parallel_coordinates(
            pd.DataFrame(X_scaled, columns=numeric_features).assign(cluster=clusters),
            color="cluster",
            title=f"Parallel Coordinates Plot for {n_clusters} Clusters"
        )
        
        fig_parallel.update_layout(
            height=600,
            width=max(800, len(numeric_features) * 100)
        )
        
        plots.append(convert_to_serializable(fig_parallel))
        
        return {
            'type': 'cluster_plot',
            'title': title,
            'plots': plots,
            'n_clusters': n_clusters,
            'cluster_centers': convert_to_serializable(kmeans.cluster_centers_),
            'features': numeric_features
        }
    except Exception as e:
        logger.error(f"Error creating cluster plot: {str(e)}")
        return None

def create_dendogram_plot(df, selected_features, title="Hierarchical Clustering Dendrogram"):
    """Create dendrogram for hierarchical clustering."""
    try:
        # Filter numeric features
        numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            logger.warning("No numeric features for dendrogram plot")
            return None
            
        # Scale the data
        X = df[numeric_features].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Limit data size for computational efficiency
        max_samples = 1000
        if len(X_scaled) > max_samples:
            logger.info(f"Limiting dendrogram to {max_samples} samples to avoid excessive processing")
            indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            X_scaled = X_scaled[indices]
        
        # Calculate linkage
        Z = linkage(X_scaled, method='ward')
        
        # Create dendrogram plot
        fig = go.Figure()
        
        dendro = sch.dendrogram(Z)
        
        # Extract dendrogram data
        x = []
        y = []
        for i, d in enumerate(dendro['dcoord']):
            x.extend([dendro['icoord'][i][j] for j in range(4)])
            y.extend(d)
            x.append(None)
            y.append(None)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='black')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Samples',
            yaxis_title='Distance',
            height=600,
            width=max(800, min(len(X_scaled) * 10, 2000)),
            xaxis_showticklabels=False
        )
        
        return {
            'type': 'dendrogram',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating dendrogram plot: {str(e)}")
        return None

def create_time_series_plot(df, time_column, value_columns, title="Time Series Plot"):
    """Create time series visualization."""
    try:
        if time_column not in df.columns:
            logger.warning(f"Time column {time_column} not found in dataframe")
            return None
        
        # Make sure value_columns are in the dataframe
        valid_value_columns = [col for col in value_columns if col in df.columns]
        
        if not valid_value_columns:
            logger.warning("No valid value columns for time series plot")
            return None
        
        # Convert time column to datetime if it's not already
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column])
        except:
            logger.warning(f"Could not convert {time_column} to datetime for time series plot")
            return None
        
        # Sort by time
        df_sorted = df.sort_values(by=time_column)
        
        # Create time series plot
        fig = go.Figure()
        
        for column in valid_value_columns:
            fig.add_trace(go.Scatter(
                x=df_sorted[time_column],
                y=df_sorted[column],
                mode='lines',
                name=column
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=time_column,
            yaxis_title='Value',
            height=600,
            width=1000
        )
        
        return {
            'type': 'time_series',
            'title': title,
            'time_column': time_column,
            'value_columns': valid_value_columns,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")
        return None

def create_bubble_chart(df, x_column, y_column, size_column, color_column=None, title="Bubble Chart"):
    """Create bubble chart visualization."""
    try:
        # Check if required columns exist
        required_columns = [x_column, y_column, size_column]
        if color_column:
            required_columns.append(color_column)
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for bubble chart: {missing_columns}")
            return None
            
        # Create bubble chart
        if color_column:
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                size=size_column,
                color=color_column,
                title=title,
                hover_name=df.index if df.index.name else None
            )
        else:
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                size=size_column,
                title=title,
                hover_name=df.index if df.index.name else None
            )
        
        fig.update_layout(
            height=700,
            width=900,
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        
        return {
            'type': 'bubble_chart',
            'title': title,
            'x_column': x_column,
            'y_column': y_column,
            'size_column': size_column,
            'color_column': color_column,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating bubble chart: {str(e)}")
        return None

def create_radar_chart(df, categories, values, group_column=None, title="Radar Chart"):
    """Create radar/spider chart visualization."""
    try:
        # Check if required columns exist
        all_columns = categories.copy()
        if group_column:
            all_columns.append(group_column)
            
        missing_columns = [col for col in all_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for radar chart: {missing_columns}")
            return None
            
        fig = go.Figure()
        
        if group_column:
            # Create radar chart grouped by group_column
            groups = df[group_column].unique()
            
            for group in groups:
                group_df = df[df[group_column] == group]
                
                fig.add_trace(go.Scatterpolar(
                    r=group_df[categories].mean().values.tolist(),
                    theta=categories,
                    fill='toself',
                    name=str(group)
                ))
        else:
            # Create single radar chart using means
            fig.add_trace(go.Scatterpolar(
                r=df[categories].mean().values.tolist(),
                theta=categories,
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df[categories].max().max() * 1.1]
                )
            ),
            title=title,
            height=700,
            width=700
        )
        
        return {
            'type': 'radar_chart',
            'title': title,
            'categories': categories,
            'group_column': group_column,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating radar chart: {str(e)}")
        return None

def create_sankey_diagram(df, source_col, target_col, value_col=None, title="Sankey Diagram"):
    """Create Sankey diagram visualization."""
    try:
        # Check if required columns exist
        required_cols = [source_col, target_col]
        if value_col:
            required_cols.append(value_col)
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for Sankey diagram: {missing_cols}")
            return None
            
        # Create node labels and indices
        sources = df[source_col].unique().tolist()
        targets = df[target_col].unique().tolist()
        
        all_nodes = list(set(sources + targets))
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        links = []
        for _, row in df.iterrows():
            link = {
                'source': node_indices[row[source_col]],
                'target': node_indices[row[target_col]]
            }
            
            if value_col:
                link['value'] = row[value_col]
            else:
                link['value'] = 1
                
            links.append(link)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links]
            )
        )])
        
        fig.update_layout(
            title=title,
            height=800,
            width=1000
        )
        
        return {
            'type': 'sankey_diagram',
            'title': title,
            'source_col': source_col,
            'target_col': target_col,
            'value_col': value_col,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating Sankey diagram: {str(e)}")
        return None

def create_sunburst_chart(df, path_columns, values_column=None, title="Sunburst Chart"):
    """Create sunburst chart visualization."""
    try:
        # Check if required columns exist
        required_cols = path_columns.copy()
        if values_column:
            required_cols.append(values_column)
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for sunburst chart: {missing_cols}")
            return None
            
        # Create sunburst chart
        fig = px.sunburst(
            df,
            path=path_columns,
            values=values_column,
            title=title
        )
        
        fig.update_layout(
            height=800,
            width=800
        )
        
        return {
            'type': 'sunburst_chart',
            'title': title,
            'path_columns': path_columns,
            'values_column': values_column,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating sunburst chart: {str(e)}")
        return None

def create_treemap(df, path_columns, values_column=None, title="Treemap"):
    """Create treemap visualization."""
    try:
        # Check if required columns exist
        required_cols = path_columns.copy()
        if values_column:
            required_cols.append(values_column)
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for treemap: {missing_cols}")
            return None
            
        # Create treemap
        fig = px.treemap(
            df,
            path=path_columns,
            values=values_column,
            title=title
        )
        
        fig.update_layout(
            height=700,
            width=900
        )
        
        return {
            'type': 'treemap',
            'title': title,
            'path_columns': path_columns,
            'values_column': values_column,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating treemap: {str(e)}")
        return None

def create_waterfall_chart(names, values, title="Waterfall Chart"):
    """Create waterfall chart visualization."""
    try:
        # Calculate cumulative sums
        cumulative = np.cumsum(values)
        totals = np.append(values, sum(values))
        names.append("Total")
        
        # Determine colors (green for positive, red for negative, blue for total)
        colors = ['green' if v > 0 else 'red' for v in values]
        colors.append('blue')
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Waterfall",
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=names,
            textposition="outside",
            y=totals,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            marker={"color": colors}
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=900,
            showlegend=False
        )
        
        return {
            'type': 'waterfall_chart',
            'title': title,
            'data': convert_to_serializable(fig)
        }
    except Exception as e:
        logger.error(f"Error creating waterfall chart: {str(e)}")
        return None

def create_visualization_summary(plots):
    """Create a summary of all visualizations."""
    summary = {
        'visualization_count': len(plots),
        'visualization_types': {},
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Count visualization types
    for plot in plots:
        plot_type = plot.get('type', 'unknown')
        if plot_type in summary['visualization_types']:
            summary['visualization_types'][plot_type] += 1
        else:
            summary['visualization_types'][plot_type] = 1
    
    return summary

def batch_create_visualizations(df, selected_features=None, y_column=None, time_column=None):
    """Create a batch of standard visualizations for exploratory data analysis."""
    if selected_features is None:
        selected_features = df.columns.tolist()
    
    plots = []
    
    # Basic plots
    plots.extend(create_line_plots(df, selected_features))
    plots.extend(create_scatter_plots(df, selected_features))
    plots.extend(create_histograms(df, selected_features))
    plots.extend(create_box_plots(df, selected_features))
    plots.extend(create_violin_plots(df, selected_features))
    plots.extend(create_bar_plots(df, selected_features))
    
    # Correlation plots
    correlation_heatmap = create_correlation_heatmap(df, selected_features)
    if correlation_heatmap:
        plots.append(correlation_heatmap)
    
    pair_plot = create_pair_plots(df, selected_features)
    if pair_plot:
        plots.append(pair_plot)
    
    # 3D plots
    plots.extend(create_3d_plots(df, selected_features))
    
    # Clustering plots
    elbow_plot = create_elbow_plot(df, selected_features)
    if elbow_plot:
        plots.append(elbow_plot)
        optimal_k = elbow_plot.get('optimal_k', 3)
        
        cluster_plot = create_cluster_plot(df, selected_features, n_clusters=optimal_k)
        if cluster_plot:
            plots.append(cluster_plot)
    
    dendogram_plot = create_dendogram_plot(df, selected_features)
    if dendogram_plot:
        plots.append(dendogram_plot)
    
    # Time series plot if time column is provided
    if time_column and time_column in df.columns:
        numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        
        # Select a subset of numeric features for time series plot
        value_columns = numeric_features[:5]  # Limit to first 5 numeric features
        
        time_series_plot = create_time_series_plot(df, time_column, value_columns)
        if time_series_plot:
            plots.append(time_series_plot)
    
    # Create summary
    summary = create_visualization_summary(plots)
    
    return plots, summary