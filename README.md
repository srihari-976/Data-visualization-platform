# AI-Powered Data Visualization Platform

A modern web application that automatically generates insightful visualizations from uploaded datasets using AI and machine learning techniques.

## 🚀 Live Application
Check out the deployed version of Data Visulaization Platform:

🔗 [![LUMINA AI](https://img.shields.io/badge/Live%20Application-Click%20Here-brightgreen)](https://data-visualization-platform.vercel.app/)

---

## Features

- Automatic data cleaning and preprocessing
- AI-powered feature selection
- Interactive visualizations using Plotly
- Support for CSV and Excel files
- Modern, responsive UI with TailwindCSS
- Real-time data processing

## Tech Stack

### Backend
- Python/Flask
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Plotly

### Frontend
- React
- TailwindCSS
- Plotly.js
- Axios

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python main.py
```

The backend server will start at `http://localhost:5000`

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend application will start at `http://localhost:5173`

## Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Click "Get Started" or navigate to the Upload page
3. Upload your CSV or Excel file
4. Wait for the AI to process your data
5. View the generated visualizations on the Results page

## Project Structure

```
data-viz-platform/
│── backend/                   # Flask Backend
│   ├── static/                # Stores generated plots
│   ├── uploads/               # Stores uploaded datasets
│   ├── models/                # Deep learning models
│   ├── main.py                # Flask API entry point
│   ├── requirements.txt       # Python dependencies
│   ├── utils/                 # Helper functions
│       ├── data_cleaning.py   # Data cleaning functions
│       ├── feature_selection.py # Feature selection
│       └── visualization.py   # Plot generation
│
│── frontend/                  # React Frontend
│   ├── public/                # Static files
│   ├── src/                   # React source files
│   │   ├── components/        # UI Components
│   │   ├── pages/            # App Pages
│   │   ├── App.jsx           # Main app component
│   │   └── main.jsx          # Entry point
│   ├── package.json          # Dependencies
│   └── tailwind.config.js    # TailwindCSS config
│
│── README.md                  # Documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
