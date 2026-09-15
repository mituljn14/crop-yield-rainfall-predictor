# 🌾 Crop Yield & Rainfall-Based Crop Recommendation System

A machine learning web app that predicts agricultural crop yield and recommends the most suitable crop to grow, based on historical rainfall patterns, state, season, and farming inputs across India.

## 🎯 What It Does

Given a set of agricultural inputs — crop type, year, season, state, cultivated area, fertilizer and pesticide usage, and annual rainfall — the app:

1. **Predicts crop yield** using a regression model trained on historical Indian agricultural data
2. **Recommends the optimal crop** to grow under those conditions using a trained classification model

Built on real datasets: India rainfall records (1901–2015), state-wise crop yield data, and a crop recommendation dataset (soil N-P-K, temperature, humidity, pH, rainfall).

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **ML/Data:** scikit-learn, pandas, NumPy
- **Frontend:** HTML/CSS (Jinja2 templates)
- **Model persistence:** Pickle

## 📁 Project Structure
crop-yield-rainfall-predictor/
├── app.py # Flask app — routes for home & prediction
├── preprocess.ipynb # Data cleaning & preprocessing
├── training.ipynb # Model training (yield regression + crop classifier)
├── data/
│ ├── crop_yield.csv
│ ├── rainfall in india 1901-2015.csv
│ └── Crop_recommendation.csv
├── models/
│ ├── crop_model.pkl # Yield prediction model
│ ├── crop_classifier.pkl # Crop recommendation model
│ ├── scaler.pkl
│ ├── crop_mapping.json
│ ├── season_mapping.json
│ ├── state_mapping.json
│ └── columns.json
├── templates/
│ └── index.html
└── requirements.txt


## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/mituljn14/crop-yield-rainfall-predictor.git
cd crop-yield-rainfall-predictor
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 📊 How It Works

1. **Data preprocessing** (`preprocess.ipynb`) — merges and cleans rainfall, crop yield, and crop recommendation datasets; encodes categorical fields (crop, season, state) into numerical mappings.
2. **Model training** (`training.ipynb`) — trains a regression model to predict yield and a classifier to recommend the best crop, both evaluated on held-out data.
3. **Web interface** (`app.py` + `templates/index.html`) — a Flask app takes user input through a form, loads the trained models, and returns both a predicted yield and a recommended crop in real time.

## 🔮 Future Improvements

- Add model evaluation metrics (R², accuracy, confusion matrix) to the README
- Deploy live demo (Render/Railway/HuggingFace Spaces)
- Add input validation and better error handling on the form
- Expand recommendation model with soil and weather API integration

## 👤 Author

**Mitul Jain**
[GitHub](https://github.com/mituljn14) · [LinkedIn](https://linkedin.com/in/mitul-jain) · mita140605@gmail.com
