# backend/utils/feature_analysis.py

import pandas as pd
import logging
from utils.data_cleaning import DataCleaner
from utils.visualization import batch_create_visualizations, convert_to_serializable

logger = logging.getLogger(__name__)

class FeatureSelector:
    @staticmethod
    def analyze_features(file_path, target_col=None):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded with shape: {df.shape}")

            cleaner = DataCleaner()
            selector = FeatureSelector()

            df_cleaned = cleaner.clean_data(df)
            cleaning_metadata = None

            logger.info("Data cleaning completed")

            feature_results = None
            if target_col and target_col in df_cleaned.columns:
                feature_results = selector.select_features(df_cleaned, target_col)
                logger.info("Feature selection completed")

            # Visualizations using Plotly (batch)
            visualizations, visualization_summary = batch_create_visualizations(df_cleaned)

            results = {
                'status': 'success',
                'data': {
                    'visualizations': visualizations,
                    'visualization_summary': visualization_summary,
                    'cleaning_summary': cleaning_metadata,
                    'feature_selection_summary': feature_results if feature_results else "Target column not provided or invalid.",
                    'dataset_info': {
                        'total_rows': int(df_cleaned.shape[0]),
                        'total_columns': int(df_cleaned.shape[1]),
                    }
                }
            }

            return convert_to_serializable(results)

        except Exception as e:
            logger.error(f"Error analyzing features: {str(e)}")
            return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    file_path = "your_data.csv"
    results = analyze_features(file_path, target_col="target")
    print("Analysis completed successfully")
