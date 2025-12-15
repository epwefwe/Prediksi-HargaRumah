from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import util as utils
import joblib
import os
import json
from datetime import datetime
from collections import deque
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit Cloud

# -----------------------------------------------------------------------------
# PREDICTION LOGGING SYSTEM
# -----------------------------------------------------------------------------
MAX_LOG_SIZE = 100
prediction_logs = deque(maxlen=MAX_LOG_SIZE)

def log_prediction(input_data, prediction, status="success", error_msg=None):
    """Log each prediction request"""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "prediction": prediction,
        "status": status,
        "error": error_msg
    }
    prediction_logs.append(log_entry)
    return log_entry

# -----------------------------------------------------------------------------
# DATA DRIFT DETECTION
# -----------------------------------------------------------------------------
# Reference statistics from training data (will be loaded on startup)
reference_stats = None

def load_reference_stats():
    """Load training data statistics for drift detection"""
    global reference_stats
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to load from processed data
        x_train_path = os.path.join(base_dir, "..", "data", "processed", "x_train.pkl")
        if os.path.exists(x_train_path):
            x_train = pd.read_pickle(x_train_path)
            reference_stats = {
                "mean": x_train.mean().to_dict(),
                "std": x_train.std().to_dict(),
                "min": x_train.min().to_dict(),
                "max": x_train.max().to_dict(),
                "count": len(x_train)
            }
        else:
            # Fallback default stats based on config ranges
            reference_stats = {
                "mean": {"LB": 150, "LT": 180, "KT": 4, "KM": 3, "GRS": 2},
                "std": {"LB": 80, "LT": 100, "KT": 2, "KM": 1.5, "GRS": 1},
                "min": {"LB": 30, "LT": 20, "KT": 1, "KM": 1, "GRS": 0},
                "max": {"LB": 2000, "LT": 2000, "KT": 15, "KM": 15, "GRS": 15},
                "count": 0
            }
    except Exception as e:
        print(f"Error loading reference stats: {e}")
        reference_stats = None

def calculate_drift(recent_predictions):
    """Calculate data drift by comparing recent inputs to reference stats"""
    if not reference_stats or len(recent_predictions) < 5:
        return None
    
    # Extract input features from recent predictions
    recent_inputs = []
    for log in recent_predictions:
        if log["status"] == "success":
            input_data = log["input"]
            # Handle both list format [value] and scalar format
            clean_input = {}
            for key, val in input_data.items():
                if isinstance(val, list):
                    clean_input[key] = val[0] if len(val) > 0 else 0
                else:
                    clean_input[key] = val
            recent_inputs.append(clean_input)
    
    if len(recent_inputs) < 5:
        return None
    
    df_recent = pd.DataFrame(recent_inputs)
    
    drift_report = {}
    features = ["LB", "LT", "KT", "KM", "GRS"]
    
    for feature in features:
        if feature in df_recent.columns:
            recent_mean = df_recent[feature].mean()
            recent_std = df_recent[feature].std()
            ref_mean = reference_stats["mean"].get(feature, 0)
            ref_std = reference_stats["std"].get(feature, 1)
            
            # Calculate drift score using normalized difference
            if ref_std > 0:
                drift_score = abs(recent_mean - ref_mean) / ref_std
            else:
                drift_score = 0
            
            # Drift severity: low < 0.5, medium 0.5-1.5, high > 1.5
            if drift_score < 0.5:
                severity = "low"
            elif drift_score < 1.5:
                severity = "medium"
            else:
                severity = "high"
            
            drift_report[feature] = {
                "reference_mean": round(ref_mean, 2),
                "recent_mean": round(recent_mean, 2),
                "drift_score": round(drift_score, 3),
                "severity": severity
            }
    
    # Overall drift status
    severities = [d["severity"] for d in drift_report.values()]
    if "high" in severities:
        overall_status = "high"
    elif "medium" in severities:
        overall_status = "medium"
    else:
        overall_status = "low"
    
    return {
        "overall_status": overall_status,
        "features": drift_report,
        "sample_size": len(recent_inputs),
        "reference_size": reference_stats.get("count", "N/A")
    }

# Load config and model
config_path = utils.get_config_path()
config = utils.load_params(config_path)
model_path = utils.get_model_path(config)
model = utils.pickle_load(model_path)

# Load reference stats for drift detection
load_reference_stats()

import data_preparation
import preprocessing
# Helper needed for pickle loading if it uses classes from these modules

@app.route('/')
def home():
    return "House Price Prediction API is Up!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_json = request.get_json()
        
        # Expecting input keys matching the predictors
        predictors = config['prediktor'] # LB, LT, KT, KM, GRS
        
        # Ensure all predictors are present
        input_data = {}
        missing_fields = []
        for p in predictors:
            if p not in data_json:
                missing_fields.append(p)
            else:
                input_data[p] = [data_json[p]]
        
        if missing_fields:
             return jsonify({"error": f"Missing features: {missing_fields}"}), 400

        # Create DataFrame
        df = pd.DataFrame(input_data)
        
        # Ensure correct data types (int64)
        for p in predictors:
            df[p] = df[p].astype('int64')

        # Validate data
        try:
           data_preparation.cek_data(df, config, True)
        except AssertionError as ae:
            return jsonify({"status": "error", "message": f"Validation Error: {str(ae)}"}), 400

        # Predict
        prediction = model.predict(df)
        
        # Result
        result = prediction[0]
        
        # Log the successful prediction with clean data (not list format)
        log_input = {p: data_json[p] for p in predictors}
        log_prediction(log_input, float(result), "success")
        
        return jsonify({
            "status": "success",
            "prediction": float(result)
        })
        
    except Exception as e:
        # Log the failed prediction
        log_prediction(data_json if 'data_json' in dir() else {}, None, "error", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        # Resolve metrics.json relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_path = os.path.join(base_dir, "models", "metrics.json")
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return jsonify({"status": "success", "data": metrics})
        else:
            return jsonify({"status": "error", "message": "Metrics not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """Get prediction logs with optional filtering"""
    try:
        limit = request.args.get('limit', 50, type=int)
        status_filter = request.args.get('status', None)
        
        logs_list = list(prediction_logs)
        
        # Filter by status if specified
        if status_filter:
            logs_list = [log for log in logs_list if log["status"] == status_filter]
        
        # Return most recent first, limited
        logs_list = logs_list[-limit:][::-1]
        
        # Calculate summary stats
        total_logs = len(prediction_logs)
        success_count = sum(1 for log in prediction_logs if log["status"] == "success")
        error_count = total_logs - success_count
        
        return jsonify({
            "status": "success",
            "data": {
                "logs": logs_list,
                "summary": {
                    "total_requests": total_logs,
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": round(success_count / total_logs * 100, 2) if total_logs > 0 else 0
                }
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/drift', methods=['GET'])
def get_drift():
    """Get data drift analysis based on recent predictions"""
    try:
        recent_logs = list(prediction_logs)
        drift_analysis = calculate_drift(recent_logs)
        
        if drift_analysis:
            return jsonify({
                "status": "success",
                "data": drift_analysis
            })
        else:
            return jsonify({
                "status": "success",
                "data": {
                    "overall_status": "insufficient_data",
                    "message": "Minimal 5 prediksi berhasil diperlukan untuk analisis drift",
                    "current_samples": len([l for l in recent_logs if l.get("status") == "success"])
                }
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
