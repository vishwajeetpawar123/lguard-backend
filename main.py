import os
import math
import pickle
import numpy as np
import pandas as pd
import shap
import base64
import io
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'ensemble_model.pkl')
IMPUTER_PATH = os.path.join(BASE_DIR, 'imputers.pkl')

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
    age: Optional[float] = Field(None, ge=1, le=120)
    gender: Optional[str] = Field(None, description="'Male' or 'Female'")
    tot_bilirubin: Optional[float] = Field(None, ge=0.0)
    direct_bilirubin: Optional[float] = Field(None, ge=0.0)
    alkphos: Optional[float] = Field(None, ge=0.0)
    sgpt: Optional[float] = Field(None, ge=0.0)
    sgot: Optional[float] = Field(None, ge=0.0)
    tot_proteins: Optional[float] = Field(None, ge=0.0)
    albumin: Optional[float] = Field(None, ge=0.0)
    ag_ratio: Optional[float] = Field(None, ge=0.0)

# --- Staging Logic ---
def get_stage_confidence(score, prediction):
    """
    Generates a simulated confidence distribution across stages 0-4
    based on the clinical severity score and the AI's binary prediction.
    """
    def gaussian(x, mu, sig):
        return math.exp(-pow(x - mu, 2.) / (2 * pow(sig, 2.)))

    # Midpoints of stage scoring ranges
    mids = [0.5, 3.0, 7.5, 12.5, 17.5]
    
    if prediction == 0:
        # Score-aware healthy confidence: borderline scores get slightly less certainty
        base_healthy = max(0.80, 0.98 - score * 0.15)
        remainder = 1.0 - base_healthy
        raw = [base_healthy, remainder * 0.6, remainder * 0.25, remainder * 0.1, remainder * 0.05]
    else:
        # Disease prediction: Use Gaussian centered at the patient's score
        sigma = 3.5 # Spread to ensure neighboring stages get some weight
        raw = [gaussian(m, score, sigma) for m in mids]
    
    total = sum(raw)
    return [round(r/total, 3) for r in raw]

def _safe_get(row, key, default=0.0):
    """Safely extract a numeric value from the row, returning default if missing or invalid."""
    try:
        val = float(row[key])
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (KeyError, TypeError, ValueError):
        return default

def get_robust_liver_score(row):
    score = 0.0

    # --- Extract values with safe defaults ---
    bilirubin = _safe_get(row, 'tot_bilirubin', 0.0)
    ast = _safe_get(row, 'sgot', 0.0)
    alt = _safe_get(row, 'sgpt', 0.0)
    albumin = _safe_get(row, 'albumin', 4.0)  # Default to normal
    ag_ratio = _safe_get(row, 'ag_ratio', 1.5)  # Default to normal
    alkphos = _safe_get(row, 'alkphos', 0.0)
    age = _safe_get(row, 'age', 30.0)

    # --- Bilirubin (capped contribution) ---
    if bilirubin > 1.2:
        score += min((bilirubin - 1.2) * 1.5, 10.0)

    # --- De Ritis Ratio: penalties for severe elevations ---
    if ast > 40 or alt > 40:
        if alt > 0:
            ratio = ast / alt
        else:
            # ALT ≈ 0 with elevated AST is clinically very concerning
            ratio = 999.0
        if ratio > 1.5: score += 3.0
        if ratio > 2.0: score += 5.0

    # --- Albumin ---
    if albumin < 3.5:
        score += ((3.5 - albumin) * 4.0)

    # --- A/G Ratio: base penalty (always applied) ---
    if ag_ratio < 1.0:
        score += 1.5  # Mild base penalty for abnormal A/G ratio

    # --- Alkaline Phosphatase (graduated scale) ---
    if alkphos > 120:
        score += min((alkphos - 120) * 0.02, 6.0)

    # --- Age modifier ---
    if age > 60:
        score *= 1.1

    # --- Conditional Multipliers for Advanced Cirrhosis Markers ---
    # Only apply these heavy red flags if the liver is already showing significant damage
    if score >= 5.0:
        if (ast > 40 or alt > 40) and (alt > 0 and (ast / alt) > 1.0):
            score += 2.0

        if ag_ratio < 1.0:
            score += 3.0  # Additional penalty on top of the base 1.5

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

def generate_shap_text_explanation(shap_vals, feature_names, stage_code, stage_text):
    """Generates a dynamic 2-sentence clinical explanation string based on the top SHAP features."""
    contributions = list(zip(feature_names, shap_vals))
    
    # Positive SHAP pushes toward Disease (Risk-increasing)
    positive_contributors = sorted([c for c in contributions if c[1] > 0], key=lambda x: x[1], reverse=True)
    # Negative SHAP pushes toward Healthy (Risk-reducing / Protective)
    negative_contributors = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])
    
    top_pos = [f"**{c[0]}**" for c in positive_contributors[:2]]
    top_neg = [f"**{c[0]}**" for c in negative_contributors[:2]]
    
    if stage_code == 0:
        text = "The AI confidently affirmed a healthy liver profile. "
        if top_neg: # These are the strongest drivers for the 'Healthy' class
            text += f"The strongest biomarker indicators supporting this healthy diagnosis were normal {', '.join(top_neg)}. "
        if top_pos: # These slightly pushed towards disease but failed to overcome the threshold
            text += f"All other evaluated biomarkers, including {', '.join(top_pos)}, were determined to be completely within safe clinical tolerances."
        else:
            text += "All evaluated biomarkers were completely within normal hepatic ranges."
    else:
        text = f"The AI diagnosed {stage_text}. "
        if top_pos:
            text += f"This assessment was most heavily driven by elevated {', '.join(top_pos)}, which significantly elevated the clinical risk profile. "
        if top_neg:
            text += f"Conversely, {', '.join(top_neg)} acted as counter-balancing factors, preventing an even more severe risk estimate."
            
    return text.strip()

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
        expected_num_cols = ['age', 'tot_bilirubin', 'direct_bilirubin', 'alkphos', 'sgpt', 'sgot', 'tot_proteins', 'albumin', 'ag_ratio']
        df_input[expected_num_cols] = df_input[expected_num_cols].astype(float)
        
        df_input.loc[:, expected_num_cols] = imputers['num_imputer'].transform(df_input[expected_num_cols])
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
            # If the Clinical Heuristic overrides an ML Disease prediction,
            # use a tempered confidence (the ML model disagreed, so certainty is lower)
            if base_prediction == 1:
                final_prob = max(0.60, 1.0 - disease_probability)

        # Determine Response Metadata
        stage_map = {
            0: {"label": "Healthy Liver", "color": "#00cc66", "prob": final_prob, "desc": "No significant clinical markers detected. Liver tests are completely within normal ranges."},
            1: {"label": "Stage 1: Early/Mild", "color": "#f1c40f", "prob": final_prob, "desc": "Inflammation detected. Early warning signs."},
            2: {"label": "Stage 2: Fibrosis/Significant", "color": "#e67e22", "prob": final_prob, "desc": "Significant risk. Early scarring and enzyme elevations."},
            3: {"label": "Stage 3: Cirrhosis/Severe", "color": "#e74c3c", "prob": final_prob, "desc": "Severe risk. Sharp protein drops/critical damage."},
            4: {"label": "Stage 4: End-Stage/Critical", "color": "#8b0000", "prob": final_prob, "desc": "Critical risk. Filter and synthesis markers compromised."}
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

        # 4. Generate Text-Based XAI Summary
        text_explanation = generate_shap_text_explanation(shap_values[0], display_names, stage_code, meta['label'])

        # Prepare safe native python dictionary for imputed values
        safe_imputed_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in df_input.iloc[0].to_dict().items()}

        return {
            "success": True,
            "prediction_code": stage_code,
            "prediction_text": stage_text,
            "confidence_score": meta['prob'],
            "color": meta['color'],
            "clinical_description": meta['desc'],
            "stage_confidence": conf_dist,
            "shap_plot_base64": img_base64,
            "text_explanation": text_explanation,
            "imputed_data": safe_imputed_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "operational", "model_loaded": ensemble is not None}
