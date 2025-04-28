from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from utils.feature_analysis import analyze_features

# Logging config
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome! Server is running. Use /api/upload to upload files."})

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
        
    logging.info(f"Received file upload request")
    
    try:
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'status': 'error', 'error': 'No file part in request'}), 400
        
        file = request.files['file']
        target_col = request.form.get('target_col')  # Optional: User can submit target column

        if not file or file.filename == '':
            logging.error("No selected file")
            return jsonify({'status': 'error', 'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.csv'):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'status': 'error', 'error': 'Invalid file type. Please upload a CSV'}), 400

        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.info(f"File saved successfully: {file_path}")

        try:
            # Analyze the dataset
            result = analyze_features(file_path)
            logging.info("Analysis completed successfully")
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            return jsonify({'status': 'error', 'error': f'Analysis failed: {str(e)}'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed uploaded file: {file_path}")

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    logging.info("Starting Flask application")
    print("✅ Server running at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000, use_reloader=False)
