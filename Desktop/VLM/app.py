import os
import re
import traceback

import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from src.weather_service import WeatherService
from src.history_manager import ImageHistoryManager
from src.translations import get_translation, get_supported_languages
from src.perenual_service import PerenualService


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
CLASS_NAMES_PATH = os.path.join(MODELS_FOLDER, "class_names.json")
MODEL_PATH = os.path.join(MODELS_FOLDER, "best_model.pth")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# External plant disease detection API configuration
EXTERNAL_API_URL = os.getenv("PLANT_API_URL")
EXTERNAL_API_KEY = os.getenv("PLANT_API_KEY")


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)


# Global services
weather_service = WeatherService()
history_manager = ImageHistoryManager(history_file=os.path.join(BASE_DIR, "data", "image_history.json"))
perenual_service = PerenualService(api_key="sk-7LiM695f79b64eb8814239")


# Load model (with graceful fallback to demo mode)
model = None
class_names = None
DEMO_MODE = False

# Only import PyTorch utilities if we are NOT using an external API
if not EXTERNAL_API_URL or not EXTERNAL_API_KEY:
    try:
        from src.utils import (
            load_model,
            load_class_names,
            preprocess_image,
            predict_disease,
            format_prediction_results,
        )
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
                print(f"[DEBUG] Loading model from {MODEL_PATH}...")
                class_names = load_class_names(CLASS_NAMES_PATH)
                model = load_model(MODEL_PATH, num_classes=len(class_names))
                print(f"[DEBUG] Model loaded successfully with {len(class_names)} classes.")
            else:
                print(f"[DEBUG] Model or class names missing. Paths: {MODEL_PATH}, {CLASS_NAMES_PATH}")
                DEMO_MODE = True
        except Exception as e:
            print(f"[DEBUG] Error loading model: {e}")
            traceback.print_exc()
            DEMO_MODE = True
    except Exception as e:
        print(f"[DEBUG] Import error: {e}")
        traceback.print_exc()
        DEMO_MODE = True
else:
    print("[DEBUG] Using external API.")
    DEMO_MODE = False


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def call_external_disease_api(image_path: str):
    """Call external Plant.id v3 health_assessment API if configured.

    - Reads the image from ``image_path`` and sends it as base64 inside JSON,
      as required by Plant.id v3.
    - Uses ``PLANT_API_URL`` (should point to the v3 base URL or
      ``.../health_assessment``) and ``PLANT_API_KEY``.
    - Normalizes the response into our internal analysis_result dict with keys:
      prediction, confidence, severity, description, treatment (list),
      symptoms, top3.
    """
    if not EXTERNAL_API_URL or not EXTERNAL_API_KEY:
        print("[DEBUG] External API not configured; skipping Plant.id call")
        return None

    try:
        import base64

        # Build the correct health_assessment URL
        target_url = EXTERNAL_API_URL.rstrip("/") if EXTERNAL_API_URL else ""
        if target_url and "health_assessment" not in target_url:
            target_url = f"{target_url}/health_assessment"

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "images": [f"data:image/jpeg;base64,{img_b64}"],
            "language": "en",
            # Ask for rich plant health details; adjust as desired
            "modifiers": ["similar_images"],
            "plant_details": [
                "disease",
                "description",
                "treatment",
                "common_names",
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Api-Key": EXTERNAL_API_KEY,
        }

        print(f"[DEBUG] Calling Plant.id at {target_url}")
        resp = requests.post(target_url, json=payload, headers=headers, timeout=30)
        print(f"[DEBUG] Plant.id status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"[DEBUG] Plant.id error body: {resp.text[:500]}")
            return None

        data = resp.json() or {}

        # Plant.id health_assessment usually returns a structure with suggestions/result
        suggestions = (
            data.get("result")
            or data.get("health_assessment")
            or data.get("suggestions")
        )

        if isinstance(suggestions, dict):
            # Sometimes wrapped in a dict
            suggestions = suggestions.get("suggestions") or suggestions.get("results")

        if not isinstance(suggestions, list) or not suggestions:
            print("[DEBUG] Plant.id response has no suggestions/result")
            return None

        best = suggestions[0] or {}

        # Try to extract disease name and probability in a robust way
        prediction = (
            best.get("name")
            or best.get("plant_name")
            or best.get("disease_name")
            or "Unknown disease"
        )

        prob = best.get("probability") or best.get("score") or best.get("confidence")
        if isinstance(prob, (int, float)):
            confidence_pct = float(prob) * (100.0 if prob <= 1 else 1)
        else:
            confidence_pct = 0.0

        details = best.get("details") or {}
        description = details.get("description") or best.get("description") or ""
        severity = details.get("severity") or best.get("severity") or "unknown"
        symptoms = details.get("local_name") or ""

        raw_treatment = (
            details.get("treatment")
            or best.get("treatment")
            or details.get("treatment_description")
            or ""
        )

        treatment_steps: list[str] = []
        if isinstance(raw_treatment, list):
            treatment_steps = [str(x).strip() for x in raw_treatment if str(x).strip()]
        elif isinstance(raw_treatment, str) and raw_treatment.strip():
            parts = re.split(r"[.;]\s*", raw_treatment)
            treatment_steps = [p.strip() for p in parts if p.strip()]

        # Build top-3 alternative suggestions if present
        top3 = []
        for alt in suggestions[1:4]:
            if not isinstance(alt, dict):
                continue
            name_alt = (
                alt.get("name")
                or alt.get("plant_name")
                or alt.get("disease_name")
            )
            prob_alt = alt.get("probability") or alt.get("score") or alt.get("confidence")
            if isinstance(prob_alt, (int, float)):
                prob_alt_pct = float(prob_alt) * (100.0 if prob_alt <= 1 else 1)
            else:
                prob_alt_pct = 0.0
            if name_alt:
                top3.append({
                    "label": name_alt,
                    "confidence": round(prob_alt_pct, 2),
                })

        return {
            "prediction": prediction,
            "confidence": round(confidence_pct, 2),
            "severity": severity,
            "description": description,
            "symptoms": symptoms,
            "treatment": treatment_steps,
            "top3": top3,
        }
    except Exception:
        traceback.print_exc()
        return None


def call_perenual_search(query: str):
    """Search Perenual species-list by name and normalize to our analysis_result format."""
    if not EXTERNAL_API_URL or not EXTERNAL_API_KEY:
        return None

    try:
        params = {"key": EXTERNAL_API_KEY, "q": query}
        resp = requests.get(EXTERNAL_API_URL, params=params, timeout=30)

        if resp.status_code != 200:
            print(f"[DEBUG] Perenual search error: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json() or {}
        items = data.get("data", [])
        if not items:
            return None

        # Use first result
        best = items[0] or {}

        # Map Perenual fields to our format
        prediction = best.get("common_name") or best.get("scientific_name") or "Unknown plant"
        description = best.get("description") or ""
        # Perenual doesn’t provide disease-specific fields; improvise
        severity = "unknown"
        symptoms = ""
        treatment_steps = []

        # Try to pull care guide as treatment placeholder
        care = best.get("care_guide") or {}
        if isinstance(care, dict):
            for section, txt in care.items():
                if isinstance(txt, str):
                    treatment_steps.append(f"{section.title()}: {txt}")

        top3 = []
        for alt in items[1:4]:
            if isinstance(alt, dict):
                alt_name = alt.get("common_name") or alt.get("scientific_name")
                if alt_name:
                    top3.append({"label": alt_name, "confidence": 0.0})

        return {
            "prediction": prediction,
            "confidence": 0.0,  # Perenual text search doesn’t give probability
            "severity": severity,
            "description": description,
            "symptoms": symptoms,
            "treatment": treatment_steps,
            "top3": top3,
        }
    except Exception:
        traceback.print_exc()
        return None


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search Perenual by plant/disease name."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    result = call_perenual_search(query)
    if not result:
        return jsonify({"error": "No results from Perenual"}), 404

    return jsonify(result)


@app.route("/debug-env")
def debug_env():
    """Debug endpoint to show environment variables."""
    return jsonify({
        "PLANT_API_URL": os.getenv("PLANT_API_URL"),
        "PLANT_API_KEY": os.getenv("PLANT_API_KEY"),
        "EXTERNAL_API_URL": EXTERNAL_API_URL,
        "EXTERNAL_API_KEY": EXTERNAL_API_KEY,
    })


@app.route("/")
def index():
    """Main landing page."""
    # Preferred Premium template
    try:
        return render_template("index_premium.html")
    except Exception:
        try:
            return render_template("index_new.html")
        except Exception:
            return render_template("index.html")


@app.route("/api/languages", methods=["GET"])
def api_languages():
    """Return list of supported languages for the frontend selector."""
    try:
        langs = get_supported_languages()
        return jsonify({"languages": langs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/translations", methods=["GET"])
def api_translations():
    """Return all translation strings for a given language (new API)."""
    lang = request.args.get("lang", "en")
    try:
        translations = get_translation(lang, "all")
        return jsonify({"lang": lang, "translations": translations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/translate", methods=["GET"])
def api_translate_legacy():
    """Legacy translation endpoint for existing frontend JS.

    Returns shape: {"translation": {...}} so index_new.html can use window.translations.
    """
    lang = request.args.get("lang", "en")
    try:
        translations = get_translation(lang, "all")
        return jsonify({"lang": lang, "translation": translations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/weather", methods=["GET"])
def api_weather():
    """Return current weather-based disease risk analysis."""
    try:
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        if lat is not None and lon is not None:
            data = weather_service.get_weather_data(lat=lat, lon=lon)
        else:
            data = weather_service.get_weather_data()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return recent analysis history."""
    try:
        limit = request.args.get("limit", default=10, type=int)
        history = history_manager.get_analysis_history(limit=limit)
        stats = history_manager.get_disease_statistics()
        trends = history_manager.get_recent_trends(days=30)
        return jsonify({
            "history": history,
            "statistics": stats,
            "trends": trends,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<analysis_id>", methods=["GET"])
def api_history_item(analysis_id):
    """Return a single analysis record by ID."""
    try:
        entry = history_manager.get_analysis_by_id(analysis_id)
        if not entry:
            return jsonify({"error": "Analysis not found"}), 404
        return jsonify(entry)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Handle image upload and plant disease prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        analysis_result = None
        used_external = False
        local_confidence = 0.0

        # Try local model first to save bandwidth/cost
        if not DEMO_MODE and model is not None and class_names is not None:
            image_tensor = preprocess_image(save_path)
            predicted_class, confidence, top3 = predict_disease(model, image_tensor, class_names)
            analysis_result = format_prediction_results(predicted_class, confidence, top3)
            local_confidence = confidence
            print(f"[DEBUG] Local confidence: {local_confidence}%")

        # Prefer external API if configured
        ext_result = call_external_disease_api(save_path)
        if ext_result is not None:
            analysis_result = ext_result
            used_external = True
        elif not DEMO_MODE and model is not None and class_names is not None:
            # Try local model if external API is not configured
            image_tensor = preprocess_image(save_path)
            predicted_class, confidence, top3 = predict_disease(model, image_tensor, class_names)
            analysis_result = format_prediction_results(predicted_class, confidence, top3)
            local_confidence = confidence
            print(f"[DEBUG] Local confidence: {local_confidence}%")

        # Final fallback: demo mode, or local result is too uncertain (< 40%)
        # If we have a local result but it's very low confidence and no external API was used
        if analysis_result is not None and not used_external and local_confidence < 40.0:
             print(f"[GUARDRAIL] Confidence {local_confidence}% is too low. Returning inconclusive.")
             analysis_result = {
                "prediction": "Result Inconclusive / Unknown",
                "confidence": local_confidence,
                "severity": "Unknown",
                "description": "The AI is uncertain about this image. It may be a plant type outside our database or the image quality is insufficient.",
                "treatment": [
                    "Try taking a clearer photo in better lighting.",
                    "Ensure the leaf is the main focus of the image.",
                    "Use the 'Deep Scan' feature if available for global analysis."
                ],
                "top3": analysis_result.get("top3", [])
            }

        if analysis_result is None:
            analysis_result = {
                "prediction": "Demo Mode - Model Not Available",
                "confidence": 0.0,
                "severity": "unknown",
                "description": "The trained model files were not found and no external API is configured.",
                "treatment": "In demo mode, no specific treatment steps are available.",
                "top3": [],
            }

        # Weather context
        weather_data = weather_service.get_weather_data()

        # Combine for history
        combined = {
            **analysis_result,
            "weather": weather_data,
            "used_external": used_external,
            "local_confidence": local_confidence
        }

        analysis_id = history_manager.add_analysis(save_path, combined)

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "analysis": analysis_result,
            "weather": weather_data,
            "demo_mode": not used_external and (DEMO_MODE or model is None or class_names is None),
            "deep_scan_available": EXTERNAL_API_URL is not None and EXTERNAL_API_KEY is not None,
            "used_external": used_external
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_legacy():
    """Legacy prediction endpoint used by older templates (index.html, index_new.html).

    Accepts form field 'file' (or 'image') and returns a flat JSON structure:
    {
        prediction, confidence, severity, description, treatment,
        top3, image_path, weather, analysis_id, demo_mode
    }
    """
    # Accept both 'file' and 'image' form field names
    file = request.files.get("file") or request.files.get("image")

    if file is None:
        return jsonify({"error": "No image file provided"}), 400
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        analysis_result = None
        used_external = False
        local_confidence = 0.0

        # Try local model first
        if not DEMO_MODE and model is not None and class_names is not None:
            image_tensor = preprocess_image(save_path)
            predicted_class, confidence, top3 = predict_disease(model, image_tensor, class_names)
            analysis_result = format_prediction_results(predicted_class, confidence, top3)
            local_confidence = confidence

        # Confidence-based fallback
        force_deep = request.form.get("deep_scan") == "true"
        threshold = 50.0

        if (local_confidence < threshold or force_deep) and EXTERNAL_API_URL and EXTERNAL_API_KEY:
            ext_result = call_external_disease_api(save_path)
            if ext_result is not None:
                analysis_result = ext_result
                used_external = True

        # Enrich with Perenual Data (if we have a valid prediction)
        if analysis_result and analysis_result.get("prediction") and "Inconclusive" not in analysis_result["prediction"]:
             try:
                 # Fetch rich data for the predicted class
                 p_data = perenual_service.get_disease_details(analysis_result["prediction"])
                 if p_data:
                     print(f"[DEBUG] Enriched with Perenual data for {analysis_result['prediction']}")
                     # Override description and treatment with professional data
                     if p_data.get('description'):
                        analysis_result['description'] = p_data['description']
                     if p_data.get('treatment'):
                        analysis_result['treatment'] = p_data['treatment']
             except Exception as e:
                 print(f"[WARN] Failed to enrich data: {e}")

        if analysis_result is None:
            analysis_result = {
                "prediction": "Demo Mode - Model Not Available",
                "confidence": 0.0,
                "severity": "unknown",
                "description": "The trained model files were not found and no external API is configured.",
                "treatment": "In demo mode, no specific treatment steps are available.",
                "top3": [],
            }

        weather_data = weather_service.get_weather_data()

        combined = {
            **analysis_result,
            "weather": weather_data,
        }

        analysis_id = history_manager.add_analysis(save_path, combined)

        # Normalize treatment to always be a list for the frontend timeline
        raw_treatment = analysis_result.get("treatment", "")
        treatment_steps = []
        if isinstance(raw_treatment, list):
            treatment_steps = raw_treatment
        elif isinstance(raw_treatment, str) and raw_treatment.strip():
            # Split into sentences/steps heuristically
            parts = re.split(r"[.;]\s*", raw_treatment)
            treatment_steps = [p.strip() for p in parts if p.strip()]

        # Build weather_analysis structure expected by index_new.html
        overall_risk = weather_data.get("overall_disease_risk", 0)
        try:
            risk_percent = round(float(overall_risk) * 100, 1)
        except Exception:
            risk_percent = 0.0

        weather_analysis = {
            "temperature": weather_data.get("current_temp"),
            "humidity": weather_data.get("humidity"),
            "wind_speed": weather_data.get("wind_speed"),
            "conditions": weather_data.get("conditions"),
            "disease_risk": risk_percent,
            "recommendations": weather_data.get("recommendations", []),
        }

        # Legacy response shape expected by frontend JS
        return jsonify({
            "prediction": analysis_result.get("prediction"),
            "confidence": analysis_result.get("confidence", 0.0),
            "severity": analysis_result.get("severity", "unknown"),
            "description": analysis_result.get("description", ""),
            "symptoms": analysis_result.get("symptoms", ""),
            "treatment": treatment_steps,
            "top3": analysis_result.get("top3", []),
            "image_path": f"/static/uploads/{filename}",
            "weather": weather_data,
            "weather_analysis": weather_analysis,
            "analysis_id": analysis_id,
            "demo_mode": not used_external and (DEMO_MODE or model is None or class_names is None),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
