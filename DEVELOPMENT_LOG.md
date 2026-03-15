# Dataset Preprocessing for v2

- [x] Analyze the raw LPD Dataset
- [x] Write `preprocess.py` to clean the dataset (handling nulls, duplicates, names)
- [x] Install CatBoost and ExplainableBoostingMachine (InterpretML)
- [x] Train a CatBoostClassifier
- [x] Train an ExplainableBoostingMachine
- [x] Create an Ensemble of the two models

## Web Application Deployments

### Prototyping Phase
- [x] Build `v2_streamlit/app.py` (Python UI)
- [x] Connect SHAP generated plots to Streamlit UI
- [x] Test Streamlit prototype with live inference

### Production Phase
- [x] Build FastAPI Backend (`v2_production/backend/main.py`)
- [x] Build SvelteKit Frontend (`v2_production/frontend/`)
- [x] Connect Svelte fetch API to FastAPI `/predict`
- [x] Render SHAP data and Staging on the Svelte Interface

### Tweaks & Documentation
- [x] Add explicit "Healthy Liver" (Stage 0) override for patients with perfectly normal biomarker scores.
- [x] Create Master `README.md` explaining ML vs Heuristic Staging.
- [x] Consolidate Dual-Architecture Documentation in Walkthrough.
- [x] Create `.bat` launchers for one-click execution.
- [x] Implement "Diagnosis Confidence Distribution" chart in both Streamlit and Svelte UIs.
- [x] Document Chart Comparison Logic (Gaussian Distribution) in Docs.
- [x] Add Mathematical Formula explanation for Confidence logic.
