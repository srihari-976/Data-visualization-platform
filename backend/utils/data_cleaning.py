import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Clean the input dataframe by handling missing values, outliers, and scaling features.
    """
    # Create a copy of the dataframe
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    
    # Impute numeric values with median
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        cleaned_df[numeric_columns] = numeric_imputer.fit_transform(cleaned_df[numeric_columns])
    
    # Impute categorical values with mode
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        cleaned_df[categorical_columns] = categorical_imputer.fit_transform(cleaned_df[categorical_columns])
    
    # Handle outliers using IQR method for numeric columns
    for column in numeric_columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
    
    # Scale numeric features
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        cleaned_df[numeric_columns] = scaler.fit_transform(cleaned_df[numeric_columns])
    
    return cleaned_df 