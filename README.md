
# AI-Powered Data Visualization Platform

A modern web application that automatically generates insightful visualizations from uploaded datasets using AI and machine learning techniques.

## Live Demo

[![Live Application](https://img.shields.io/badge/Live%20Application-Click%20Here-brightgreen)](https://data-visualization-platform.vercel.app/)

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/your-username/Data-visualization-platform.git
cd Data-visualization-platform

# Pull pre-built images from Docker Hub
docker pull sri235/dataviz-backend:latest
docker pull sri235/dataviz-frontend:latest
docker pull ollama/ollama:latest

# Start all services
docker-compose up
```

App available at **http://localhost**

No build required — all images are pulled from Docker Hub:
- `sri235/dataviz-backend:latest`
- `sri235/dataviz-frontend:latest`
- `ollama/ollama:latest` (official image)

First run downloads the Llama model (~2GB) into a Docker volume. Subsequent runs use the cached volume.

### Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Backend runs at `http://localhost:5000`, Frontend at `http://localhost:5173`

## Features

- Automatic data cleaning and preprocessing
- AI-powered visualization code generation via Ollama LLM
- Interactive visualizations using Plotly and Matplotlib
- Support for CSV and Excel files
- Modern, responsive UI with Material-UI and TailwindCSS
- Real-time data processing
- Feature importance rankings via PCA analysis

## Tech Stack

### Backend

- Python / Flask
- Ollama (primary LLM) — HuggingFace Transformers (fallback)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- ChromaDB + SentenceTransformers (RAG)

### Frontend

- React
- Material-UI + TailwindCSS
- Plotly.js
- Axios

### Infrastructure

- Docker + Docker Compose
- Nginx (reverse proxy)
- Ollama (LLM serving)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model to use |
| `FLASK_ENV` | `production` | Flask environment |

## Setup Instructions

### Docker Deployment

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Data-visualization-platform.git
   cd Data-visualization-platform
   ```

2. Start all services:

   ```bash
   docker-compose up
   ```

   First run downloads the Llama model (~2GB). Subsequent runs use the cached volume.

3. Access the app at **http://localhost**

4. Stop services:

   ```bash
   docker-compose down
   ```

### Rebuilding Images (if you made changes)

```bash
docker-compose up --build
```

Or rebuild and push individual services:

```bash
# Backend
docker build -f backend/Dockerfile -t sri235/dataviz-backend:latest .
docker push sri235/dataviz-backend:latest

# Frontend
docker build -f frontend/Dockerfile -t sri235/dataviz-frontend:latest .
docker push sri235/dataviz-frontend:latest
```

### Local Development (Without Docker)

1. Install and start [Ollama](https://ollama.ai):

   ```bash
   ollama pull llama3.2:3b
   ```

2. Backend setup:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. Frontend setup:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. Open `http://localhost:5173`

## Usage

1. Open the app in your browser
2. Click "Get Started" or navigate to the Upload page
3. Upload your CSV or Excel file
4. Wait for the AI to process your data
5. View the generated visualizations on the Results page
6. Ask natural language questions to generate custom visualizations

## Project Structure

```
Data-visualization-platform/
├── docker-compose.yml          # Docker orchestration
├── .dockerignore               # Docker build exclusions
├── backend/                    # Flask Backend
│   ├── Dockerfile              # Backend container build
│   ├── main.py                 # Flask API entry point
│   ├── requirements.txt        # Python dependencies
│   ├── utils/
│   │   ├── llm_service.py      # Ollama / HuggingFace LLM
│   │   ├── rag_service.py      # ChromaDB + embeddings
│   │   ├── code_executor.py    # Sandboxed code execution
│   │   ├── feature_analysis.py # Feature analysis + PCA
│   │   └── dataset_preprocessing.py
│   └── models/                 # HuggingFace model cache (local dev only)
├── frontend/                   # React Frontend
│   ├── Dockerfile              # Frontend container build
│   ├── nginx.conf              # Nginx reverse proxy config
│   ├── src/
│   │   ├── components/         # UI Components
│   │   ├── pages/              # App Pages
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # Entry point
│   ├── package.json
│   └── vite.config.js          # Vite dev server + proxy
└── README.md
```

## Architecture

```
User Browser → Nginx (port 80) → Flask API (port 5000) → Ollama (port 11434)
                  │                     │                       │
              Serves SPA          REST endpoints         LLM inference
              Proxies /api        Code generation        llama3.2:3b
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and analyze a dataset |
| `POST` | `/api/query` | Generate visualization from natural language |
| `GET` | `/api/datasets` | List all loaded datasets |
| `GET` | `/api/visualizations/<id>` | Regenerate visualizations for a dataset |

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
