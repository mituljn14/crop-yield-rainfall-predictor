from flask import Flask, render_template, request
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# =========================
# 📂 PATH SETUP (SAFE WAY)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
crop_map_path = os.path.join(BASE_DIR, "models", "crop_mapping.json")
season_map_path = os.path.join(BASE_DIR, "models", "season_mapping.json")
state_map_path = os.path.join(BASE_DIR, "models", "state_mapping.json")

# =========================
# 🤖 LOAD MODEL
# =========================
model = pickle.load(open(model_path, "rb"))

# =========================
# 📊 LOAD MAPPINGS
# =========================
with open(crop_map_path) as f:
    crop_mapping = json.load(f)

with open(season_map_path) as f:
    season_mapping = json.load(f)

with open(state_map_path) as f:
    state_mapping = json.load(f)

# =========================
# 🏠 HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        crops=crop_mapping,
        seasons=season_mapping,
        states=state_mapping
    )

# =========================
# 🔮 PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        crop = int(request.form["Crop"])
        year = float(request.form["Crop_Year"])
        season = int(request.form["Season"])
        state = int(request.form["State"])
        area = float(request.form["Area"])
        production = float(request.form["Production"])
        rainfall = float(request.form["Annual_Rainfall"])
        fertilizer = float(request.form["Fertilizer"])
        pesticide = float(request.form["Pesticide"])

        features = np.array([[crop, year, season, state, area, production, rainfall, fertilizer, pesticide]])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"🌾 Predicted Yield: {round(prediction,2)}",
            crops=crop_mapping,
            seasons=season_mapping,
            states=state_mapping
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            crops=crop_mapping,
            seasons=season_mapping,
            states=state_mapping
        )

# =========================
# ▶️ RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)