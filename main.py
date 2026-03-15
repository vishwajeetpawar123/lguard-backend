import os
import pickle
import numpy as np
import pandas as pd
import shap
import base64
import io
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Initialize FastAPI ---
app = FastAPI(
    title="HepaGuard AI API V2",
    description="Disease staging and SHAP XAI for Liver Disease Detection",
    version="2.0.0"
)

# Allow CORS for SvelteKit Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models & Imputers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'models', 'ensemble_model.pkl'))
IMPUTER_PATH = os.environ.get('IMPUTER_PATH', os.path.join(BASE_DIR, 'models', 'imputers.pkl'))

ensemble = None
imputers = None

@app.on_event("startup")
def load_pipeline():
    global ensemble, imputers
    try:
        with open(MODEL_PATH, 'rb') as f:
            ensemble = pickle.load(f)
        with open(IMPUTER_PATH, 'rb') as f:
            imputers = pickle.load(f)
        print("Pipeline successfully loaded into memory.")
    except Exception as e:
        print(f"Error loading model files: {e}")

# --- Input Schema ---
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)
    gender: str = Field(..., description="'Male' or 'Female'")
    tot_bilirubin: float = Field(..., ge=0.0)
    direct_bilirubin: float = Field(..., ge=0.0)
    alkphos: float = Field(..., ge=0.0)
    sgpt: float = Field(..., ge=0.0)
    sgot: float = Field(..., ge=0.0)
    tot_proteins: float = Field(..., ge=0.0)
    albumin: float = Field(..., ge=0.0)
    ag_ratio: float = Field(..., ge=0.0)

# --- Staging Logic ---
def get_stage_confidence(score, prediction):
    """
    Generates a simulated confidence distribution across stages 0-4
    based on the clinical severity score and the AI's binary prediction.
    """
    import math
    def gaussian(x, mu, sig):
        return math.exp(-pow(x - mu, 2.) / (2 * pow(sig, 2.)))

    # Midpoints of stage scoring ranges
    mids = [0.5, 3.0, 7.5, 12.5, 17.5]
    
    if prediction == 0:
        # Healthy prediction: Heavy weight on Stage 0
        raw = [0.95, 0.03, 0.01, 0.005, 0.005]
    else:
        # Disease prediction: Use Gaussian centered at the patient's score
        sigma = 3.5 # Spread to ensure neighboring stages get some weight
        raw = [gaussian(m, score, sigma) for m in mids]
    
    total = sum(raw)
    return [round(r/total, 3) for r in raw]

def get_robust_liver_score(row):
    score = 0.0
    bilirubin = row['tot_bilirubin']
    if bilirubin > 1.2: score += min((bilirubin - 1.2) * 1.5, 10.0)
        
    ast, alt = row['sgot'], row['sgpt']
    ratio = ast / alt if alt > 0 else 1.0
    if ratio > 1.5: score += 3.0
    if ratio > 2.0: score += 5.0
        
    albumin = row['albumin']
    if albumin < 3.5: score += ((3.5 - albumin) * 4.0)
        
    alkphos = row['alkphos']
    if alkphos > 300: score += 4.0
        
    age = row['age']
    if age > 60: score *= 1.1
        
    return score

def assign_liver_stage(row, prediction):
    score = get_robust_liver_score(row)
    
    # If the model predicts no disease OR the biochemical score is completely normal,
    # classify as Healthy Liver (Stage 0).
    if prediction == 0 or score < 1.0:
        return 0, "Healthy Liver"
        
    if score >= 15: return 4, "Stage 4 (End-Stage/Critical)"
    if score >= 10: return 3, "Stage 3 (Cirrhosis/Severe)"
    if score >= 5:  return 2, "Stage 2 (Fibrosis/Significant)"
    return 1, "Stage 1 (Early/Mild)"

# --- Main Endpoint ---
@app.post("/predict")
async def predict_liver_disease(patient: PatientData):
    if ensemble is None or imputers is None:
        raise HTTPException(status_code=500, detail="Model pipeline is currently offline.")
    
    try:
        # Convert Pydantic object to Pandas DataFrame
        data = patient.dict()
        df_input = pd.DataFrame([data])
        
        # Apply strict Imputation & Preprocessing
        num_cols = df_input.select_dtypes(include=np.number).columns.drop('gender', errors='ignore')
        df_input.loc[:, num_cols] = imputers['num_imputer'].transform(df_input[num_cols])
        df_input[['gender']] = imputers['cat_imputer'].transform(df_input[['gender']])
        df_input['gender'] = df_input['gender'].map({'Male': 1, 'Female': 0}).astype(int)

        # 1. Base Prediction (Ensemble)
        base_prediction = int(ensemble.predict(df_input)[0])
        disease_probability = float(ensemble.predict_proba(df_input)[0][1]) # Class 1 Prob
        
        # 2. Stage Disease
        stage_code, stage_text = assign_liver_stage(df_input.iloc[0], base_prediction)
        
        # Determine Final Confidence Score
        final_prob = disease_probability
        if stage_code == 0:
            final_prob = 1.0 - disease_probability
            # If the Clinical Heuristic overrides an ML Disease prediction
            if base_prediction == 1:
                final_prob = 0.95

        # Determine Response Metadata
        stage_map = {
            0: {"label": "Healthy Liver", "color": "#00cc66", "prob": final_prob, "desc": "No significant clinical markers detected. Liver tests are completely within normal ranges."},
            1: {"label": "Stage 1: Early/Mild", "color": "#f1c40f", "prob": final_prob, "desc": "Inflammation detected. Early warning signs."},
            2: {"label": "Stage 2: Significant", "color": "#e67e22", "prob": final_prob, "desc": "Significant risk. Early scarring and enzyme elevations."},
            3: {"label": "Stage 3: Severe", "color": "#e74c3c", "prob": final_prob, "desc": "Severe risk. Sharp protein drops/critical damage."},
            4: {"label": "Stage 4: Critical", "color": "#8b0000", "prob": final_prob, "desc": "Critical risk. Filter and synthesis markers compromised."}
        }
        meta = stage_map[stage_code]
        
        # Calculate Confidence Distribution
        score = get_robust_liver_score(df_input.iloc[0])
        conf_dist = get_stage_confidence(score, base_prediction)

        # 3. Generate SHAP Waterfall Plot Image via explicit Catboost extractor
        cat_model = ensemble.named_estimators_['cat']
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(df_input)
        
        display_names = ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphatase', 'SGPT', 'SGOT', 'Total Proteins', 'Albumin', 'A/G Ratio']
        
        explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=df_input.iloc[0], 
            feature_names=display_names
        )

        fig = plt.figure(figsize=(10, 4))
        shap.waterfall_plot(explanation, show=False)
        
        # Style for Frontend
        fig.patch.set_facecolor('#ffffff') # Send clean to Svelte
        plt.tight_layout()
        
        # Convert plot to Base64 String to send over API
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            "success": True,
            "prediction_code": stage_code,
            "prediction_text": stage_text,
            "confidence_score": meta['prob'],
            "color": meta['color'],
            "clinical_description": meta['desc'],
            "stage_confidence": conf_dist,
            "shap_plot_base64": img_base64
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "operational", "model_loaded": ensemble is not None}
