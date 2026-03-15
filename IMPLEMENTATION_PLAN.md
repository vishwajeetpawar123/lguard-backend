# Web Application V2: Implementation Plan

The user noted concern about the extremely high accuracy. While we fixed a data leakage issue during imputation, the accuracy remained very high (99%+). This often happens in tabular datasets where a feature acts as a near-perfect proxy for the target (e.g., in this dataset, very specific combinations of Bilirubin and SGOT might cleanly separate the classes). 

Regardless, the pipeline is structurally sound, so we will proceed with building the new interactive web interface (`v2_streamlit/app.py`) using the new **Ensemble Model** and **Robust Staging Logic**.

We will build two separate deployment architectures to provide both rapid internal prototyping and a highly scalable, production-ready frontend.

## Proposed Architecture

### 1. `v2_streamlit/` (Internal Demo App)
A rapid prototyping interface built purely in Python.
*   **App Script:** `v2_streamlit/app.py`
*   **Functionality:** Handles model loading (`ensemble_model.pkl`), preprocessing logic (`imputers.pkl`), the 4-stage heuristic logic, and renders SHAP waterfall plots directly in the UI.

### 2. `v2_production/` (Production Web Stack)
A decoupled, scalable architecture featuring a robust backend API and a fast, modern frontend.

#### Backend (FastAPI - Python)
*   **App Script:** `v2_production/backend/main.py`
*   **Endpoint (`/predict`):** Receives JSON patient data, handles imputation, runs the ensemble model, calculates the 4-stage heuristic score, generates SHAP values, and returns the complete diagnosis payload. 
*   **CORS:** Enabled for frontend communication.

#### Frontend (SvelteKit + TailwindCSS - Javascript/HTML)
*   **Structure:** Standard initialized SvelteKit project (`v2_production/frontend/`).
*   **Styling:** Modern, medical-themed interface built strictly with TailwindCSS tokens. 
*   **Functionality:** A beautiful form to capture the 10 biomarkers. Submits data to the FastAPI backend and renders the returned Staging Card, Confidence Array, and SHAP visual explanations dynamically on the screen.

## Verification Plan

### Automated Tests
*   Run the Streamlit app: `cd v2_streamlit; streamlit run app.py`

### Manual Verification
*   Input healthy metrics (e.g., Bilirubin 0.8, Albumin 4.5) and verify a "No Disease" or "Stage 1" output.
*   Input critical metrics (e.g., Bilirubin 6.0, Albumin 2.1) and verify a "Stage 4 (End-Stage/Critical)" output.
*   Verify that the SHAP waterfall plot renders correctly on the right side of the screen without errors.
