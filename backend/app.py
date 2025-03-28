from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
import pandas as pd
from utils.data_cleaning import clean_data
from utils.feature_selection import select_features
from utils.visualization import generate_visualizations

# Initialize Flask App
app = Flask(__name__)

# Setup CORS for your frontend only (Remove "*" in production)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "https://data-visualization-platform.vercel.app/"]}})

# Configure Logging
logging.basicConfig(level=logging.INFO)

# File Upload Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure Upload Folder Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read the data
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Clean the data
            cleaned_df = clean_data(df)
            
            # Select features
            selected_features = select_features(cleaned_df)
            
            # Generate visualizations
            visualizations = generate_visualizations(cleaned_df, selected_features)
            
            return jsonify({
                'message': 'File processed successfully',
                'visualizations': visualizations,
                'features': selected_features
            })
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run()
