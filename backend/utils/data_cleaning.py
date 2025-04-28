import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
import logging

class DataCleaner:
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.scaler = None
        self.encoder = None
        self.tfidf_vectorizers = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.text_columns = []

    def detect_column_types(self, df):
        self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.text_columns = []

        for col in self.categorical_columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                self.text_columns.append(col)

        self.categorical_columns = list(set(self.categorical_columns) - set(self.text_columns))

    def handle_missing_values(self, df, numeric_strategy='knn', categorical_strategy='most_frequent'):
        df_cleaned = df.copy()

        if self.numeric_columns:
            if numeric_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=numeric_strategy)
            df_cleaned[self.numeric_columns] = imputer.fit_transform(df_cleaned[self.numeric_columns])

        if self.categorical_columns:
            for col in self.categorical_columns:
                if categorical_strategy == 'most_frequent':
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                else:
                    df_cleaned[col] = df_cleaned[col].fillna('Unknown')

        if self.text_columns:
            for col in self.text_columns:
                df_cleaned[col] = df_cleaned[col].fillna('')

        return df_cleaned

    def detect_and_handle_outliers(self, df):
        df_cleaned = df.copy()
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df_cleaned[col]))
            outlier_indices = np.where(z_scores > 3)[0]
            if len(outlier_indices) > 0:
                median = df_cleaned[col].median()
                df_cleaned.loc[outlier_indices, col] = median
        return df_cleaned

    def encode_categorical(self, df):
        if not self.categorical_columns:
            return df
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        encoded = self.encoder.fit_transform(df[self.categorical_columns])
        encoded_df = pd.DataFrame(encoded.toarray(), columns=self.encoder.get_feature_names_out(self.categorical_columns), index=df.index)
        df = df.drop(columns=self.categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def vectorize_text(self, df):
        for col in self.text_columns:
            tfidf = TfidfVectorizer(max_features=100)
            vectors = tfidf.fit_transform(df[col])
            tfidf_df = pd.DataFrame(vectors.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(vectors.shape[1])], index=df.index)
            df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)
            self.tfidf_vectorizers[col] = tfidf
        return df

    def scale_features(self, df):
        if not self.numeric_columns:
            return df
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            return df

        df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        return df

    def clean_data(self, df):
        try:
            logging.info("Starting data cleaning...")
            self.detect_column_types(df)
            df = self.handle_missing_values(df)
            df = self.detect_and_handle_outliers(df)
            df = self.encode_categorical(df)
            df = self.vectorize_text(df)
            df = self.scale_features(df)
            logging.info("Data cleaning completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error during data cleaning: {str(e)}")
            raise
