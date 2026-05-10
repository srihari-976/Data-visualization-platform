from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import logging
import pandas as pd
import uuid
from utils.feature_analysis import analyze_features
from utils.llm_service import get_llm_service
from utils.code_executor import execute_visualization_code
from utils.rag_service import get_rag_service
from utils.query_engine import answer_table_query
from utils.dataset_preprocessing import clean_for_analysis

# Logging config
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

MAX_ROWS_FOR_ANALYSIS = 10000

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dataviz-ai-secret-key-2024')

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000", "https://data-visualization-platform.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# In-memory storage for datasets (for demo purposes)
datasets = {}


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome! DataViz AI Server is running.", "status": "healthy"})


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Upload dataset, analyze with LLM, and store for queries."""
    if request.method == 'OPTIONS':
        return '', 200
        
    logging.info("Received file upload request")
    
    try:
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'status': 'error', 'error': 'No file part in request'}), 400
        
        file = request.files['file']

        if not file or file.filename == '':
            logging.error("No selected file")
            return jsonify({'status': 'error', 'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.csv'):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'status': 'error', 'error': 'Invalid file type. Please upload a CSV'}), 400

        # Use provided dataset_id or generate new unique one
        dataset_id = request.form.get('datasetId')
        if not dataset_id:
            dataset_id = str(uuid.uuid4())[:8]
        
        # Save file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}.csv")
        file.save(file_path)
        logging.info(f"File saved: {file_path}")

        try:
            # Load dataframe
            df = pd.read_csv(file_path)
            cleaned_df, preprocessing_summary = clean_for_analysis(df)
            
            # Store cleaned dataframe in memory for later RAG/query/visualization work.
            datasets[dataset_id] = {
                'df': cleaned_df,
                'raw_df': df,
                'filename': file.filename,
                'columns': list(cleaned_df.columns),
                'preprocessing_summary': preprocessing_summary
            }
            
            # Use sampled df for analysis if too large
            analysis_df = cleaned_df
            if len(cleaned_df) > MAX_ROWS_FOR_ANALYSIS:
                analysis_df = cleaned_df.sample(n=MAX_ROWS_FOR_ANALYSIS, random_state=42)
                logging.info(f"Sampling dataset to {MAX_ROWS_FOR_ANALYSIS} rows for analysis")
            
            # Get LLM to analyze dataset
            llm = get_llm_service()
            dataset_analysis = llm.analyze_dataset(analysis_df)
            
            # Initialize RAG Service and add dataset schema
            rag = get_rag_service()
            rag.add_dataset_schema(dataset_id, analysis_df)

            # Use the cleaned version for traditional analysis and initial charts too.
            cleaned_df.to_csv(file_path, index=False)
            
            # Also run traditional feature analysis for initial visualizations
            result = analyze_features(file_path)
            
            # Add LLM analysis and dataset info to result
            result['dataset_id'] = dataset_id
            result['llm_analysis'] = dataset_analysis.get('llm_summary', 'Dataset loaded successfully')
            result['columns'] = list(cleaned_df.columns)
            result['preprocessing_summary'] = preprocessing_summary
            if result.get('data', {}).get('cleaning_summary'):
                result['data']['cleaning_summary']['preprocessing_summary'] = preprocessing_summary
            
            logging.info(f"Analysis completed for dataset {dataset_id}")
            
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            return jsonify({'status': 'error', 'error': f'Analysis failed: {str(e)}'}), 500
        finally:
            # Keep the file for now (needed for queries)
            pass

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/query', methods=['POST', 'OPTIONS'])
def query_visualizations():
    """Process user query using LLM to generate and execute visualization code."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        dataset_id = data.get('dataset_id', '')
        columns = data.get('columns', [])
        
        if not query:
            return jsonify({'status': 'error', 'error': 'No query provided'}), 400
        
        logging.info(f"Processing query: {query} for dataset: {dataset_id}")
        
        # Get dataset from memory
        dataset_info = datasets.get(dataset_id)
        
        if dataset_info:
            df = dataset_info['df']

            table_result = answer_table_query(query, df)
            if table_result:
                return jsonify(table_result)
            
            # Sample for execution if needed
            exec_df = df
            if len(df) > MAX_ROWS_FOR_ANALYSIS:
                exec_df = df.sample(n=MAX_ROWS_FOR_ANALYSIS, random_state=42)
            
            # Retrieve relevant schema chunks using RAG
            rag = get_rag_service()
            schema_chunks = rag.query_schema(dataset_id, query, n_results=3)
            logging.info(f"Retrieved {len(schema_chunks)} schema chunks for query")

            # Use LLM to generate visualization code
            llm = get_llm_service()
            code_result = llm.generate_visualization_code(query, exec_df, schema_chunks)
            
            logging.info(f"Generated code for: {code_result.get('visualization_type', 'unknown')}")
            
            # Execute the generated code
            execution_result = execute_visualization_code(
                code_result.get('code', ''),
                exec_df
            )
            
            if execution_result['success'] and execution_result['figures']:
                return jsonify({
                    'status': 'success',
                    'mode': 'dynamic',
                    'figures': execution_result['figures'],
                    'explanation': code_result.get('explanation', ''),
                    'visualization_type': code_result.get('visualization_type', ''),
                    'code': code_result.get('code', ''),  # Include code for transparency
                    'interpretation': f"Generated {code_result.get('visualization_type', 'visualization')} based on your query"
                })
            else:
                # Code execution failed; report it instead of showing unrelated old plots.
                logging.warning(f"Code execution failed: {execution_result.get('error')}")
                return jsonify({
                    'status': 'error',
                    'mode': 'dynamic_error',
                    'error': execution_result.get('error', 'Code execution failed'),
                    'code': code_result.get('code', ''),
                    'interpretation': 'The requested visualization could not be generated.'
                }), 500
        else:
            # No dataset in memory, use legacy query understanding
            llm = get_llm_service()
            result = llm.understand_query(query, columns)
            
            columns_str = ', '.join(result['columns'][:5]) if result['columns'] else 'all columns'
            viz_str = result['visualization_types'] if isinstance(result['visualization_types'], str) else ', '.join(result['visualization_types'])
            
            return jsonify({
                'status': 'success',
                'mode': 'filter',
                'columns': result['columns'],
                'visualization_types': result['visualization_types'],
                'interpretation': f"Showing {viz_str} visualizations for: {columns_str}"
            })
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'interpretation': 'An error occurred while processing your query'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "llm": "available", "datasets_loaded": len(datasets)}), 200


@app.route('/api/llm-status', methods=['GET'])
def llm_status():
    """Check LLM model status"""
    try:
        llm = get_llm_service()
        return jsonify({
            'status': 'success',
            'model_loaded': llm.model_loaded,
            'device': llm.device,
            'fallback_mode': not llm.model_loaded,
            'api_configured': False,
            'load_error': getattr(llm, 'load_error', None)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False,
            'fallback_mode': True
        })


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all loaded datasets"""
    dataset_list = []
    for did, info in datasets.items():
        dataset_list.append({
            'id': did,
            'filename': info.get('filename', 'unknown'),
            'columns': info.get('columns', []),
            'rows': len(info['df']) if 'df' in info else 0
        })
    return jsonify({'status': 'success', 'datasets': dataset_list})


if __name__ == '__main__':
    logging.info("Starting DataViz AI Flask application")
    print("=" * 60)
    print("  DataViz AI Server - LLM Powered")
    print("=" * 60)
    print("✅ Server running at http://127.0.0.1:5000/")
    print("📊 Upload endpoint: http://127.0.0.1:5000/api/upload")
    print("🤖 Query endpoint: http://127.0.0.1:5000/api/query")
    print("=" * 60)
    
    llm = get_llm_service()
    if llm.model_loaded:
        print("🧠 LLM: Local Llama model loaded")
    else:
        print("⚠️  LLM: Using template fallback")
    print("=" * 60)
    
    app.run(debug=True, port=5000, use_reloader=False)
