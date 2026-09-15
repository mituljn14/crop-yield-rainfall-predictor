from flask import Flask, render_template, request
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# =========================
# 📂 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
classifier_path = os.path.join(BASE_DIR, "models", "crop_classifier.pkl")

crop_map_path = os.path.join(BASE_DIR, "models", "crop_mapping.json")
season_map_path = os.path.join(BASE_DIR, "models", "season_mapping.json")
state_map_path = os.path.join(BASE_DIR, "models", "state_mapping.json")

# =========================
# 🤖 LOAD MODELS
# =========================
model = pickle.load(open(model_path, "rb"))
classifier = pickle.load(open(classifier_path, "rb"))

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
# 🏠 HOME
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",
        crops=crop_mapping,
        seasons=season_mapping,
        states=state_mapping,
        recommended_crop=None,
        prediction_text=None
    )

# =========================
# 🔮 PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ========= INPUT =========
        crop = int(request.form["Crop"])
        year = float(request.form["Crop_Year"])
        season = int(request.form["Season"])
        state = int(request.form["State"])
        area = float(request.form["Area"])
        production = float(request.form["Production"])
        rainfall = float(request.form["Annual_Rainfall"])
        fertilizer = float(request.form["Fertilizer"])
        pesticide = float(request.form["Pesticide"])

        # ========= YIELD =========
        features = np.array([[crop, year, season, state,
                              area, production, rainfall,
                              fertilizer, pesticide]])

        predicted_yield = model.predict(features)[0]

        # ========= RECOMMENDATION =========
        clf_input = np.array([[year, season, state,
                               area, production, rainfall,
                               fertilizer, pesticide]])

        pred_crop_encoded = classifier.predict(clf_input)[0]

        # 🔥 FIXED MAPPING
        recommended_crop = None
        for name, val in crop_mapping.items():
            if int(val) == int(pred_crop_encoded):
                recommended_crop = name
                break

        if recommended_crop is None:
            recommended_crop = f"Unknown ({pred_crop_encoded})"

        # ========= DEBUG PRINT =========
        print("\n========== DEBUG ==========", flush=True)
        print("Classifier Output:", pred_crop_encoded, flush=True)
        print("Recommended Crop:", recommended_crop, flush=True)
        print("Predicted Yield:", predicted_yield, flush=True)
        print("================================\n", flush=True)

        # ========= RETURN =========
        return render_template(
            "index.html",
            prediction_text=f"🌾 Predicted Yield: {round(predicted_yield,2)}",
            recommended_crop=recommended_crop,
            crops=crop_mapping,
            seasons=season_mapping,
            states=state_mapping
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            recommended_crop="Error",
            crops=crop_mapping,
            seasons=season_mapping,
            states=state_mapping
        )

# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)