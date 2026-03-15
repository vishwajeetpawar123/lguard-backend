# Liver Disease Detection V2: Comprehensive Architecture Overview

This document details the complete end-to-end logic, models, and features implemented in the `v2` directory of the Liver Disease Detection application.

## 1. Dataset Preprocessing Pipeline (`preprocess.py` & `train_models.py`)

The data pipeline has been robustly engineered to prevent data leakage and ensure standard machine learning practices are followed.

### Structural Cleaning (`preprocess.py`)
*   **Column Standardization:** Removed invisible characters, trailing spaces, and typos from raw column names (e.g., converting `Alkphos Alkaline Phosphotase` to `alkphos`). This ensures programmatic predictability.
*   **Categorical Encoding:** Converted `Gender` to binary values (`1` for Male, `0` for Female).
*   **Target Normalization:** Mapped the target variable from `[1, 2]` to a standard binary classification map: `1` (Disease) and `0` (No Disease).
*   **Deduplication:** Dropped 11,323 exact duplicate rows from the source LPD dataset to prevent data leakage and artificial weight bias during training.

### Statistical Imputation (`train_models.py`)
*   **Data Leakage Prevention:** Missing values (NaNs) are purposefully *not* filled in `preprocess.py`. 
*   **Post-Split Imputation:** The dataset is first split 80/20 into Training and Testing sets.
*   **Scikit-Learn SimpleImputer:** 
    *   Numerical missing values are imputed using the **Median** derived *only* from `X_train`.
    *   Categorical missing values (`gender`) are imputed using the **Mode** derived *only* from `X_train`.
    *   These imputers are saved to `imputers.pkl` for production use in the web application.

## 2. Model Architecture: Soft Voting Ensemble

To maximize both predictive power and interpretability, the V2 architecture utilizes a heterogeneous ensemble approach.

*   **Model 1: Explainable Boosting Machine (EBM)**
    *   *Purpose:* A "glass-box" model built by InterpretML that provides exact, generalized additive explanations for its predictions. It handles non-linear relationships natively while remaining completely transparent.
*   **Model 2: CatBoostClassifier**
    *   *Purpose:* A highly robust Gradient Boosting framework optimized for datasets with varying feature types. It is configured for 500 iterations with a depth of 6.
*   **Ensemble Strategy (Soft Voting):**
    *   A `VotingClassifier` combines the output probabilities of both EBM and CatBoost. By averaging the probabilities (soft voting) rather than just the final binary class (hard voting), the ensemble is more nuanced and confident in its predictions.
    *   The final model is serialized and saved as `ensemble_model.pkl`.

### Ensemble Performance Metrics (Test Set)
| Metric | Score | 
|---|---|
| **Accuracy** | 99.79% |
| **Precision** | 99.75% |
| **Recall** | 99.96% |
| **F1-Score** | 99.86% |

## 3. Clinical Liver Disease Staging Module (Heuristic)

A custom, robust scoring algorithm (`get_robust_liver_score`) was implemented to categorize patients diagnosed with Liver Disease into one of four clinical severity stages. 

The algorithm assigns weighted points based on critical biomarkers:

1.  **Bilirubin (The Filter Marker):** Adds up to 10 points for levels exceeding the normal baseline (1.2 mg/dL).
2.  **De Ritis Ratio (AST/ALT - The Damage Marker):** Adds 3 points if the ratio exceeds 1.5, and 5 points if it exceeds 2.0 (critical liver damage).
3.  **Albumin (The Protein Synthesis Marker):** Heavily penalizes drops below 3.5 g/dL, as protein loss is a severe indicator.
4.  **Alkaline Phosphatase (The Duct Marker):** Adds 4 points if ALP exceeds three times the normal limit (> 300).
5.  **Age Scaling:** Applies a 10% severity multiplier to the final score for patients older than 60, reflecting reduced liver reserve capacity.

### Staging Output
Based on the total score and model interactions, the algorithm assigns:
*   **Stage 4 (End-Stage/Critical):** Score >= 15
*   **Stage 3 (Cirrhosis/Severe):** Score >= 10
*   **Stage 2 (Fibrosis/Significant):** Score >= 5
*   **Stage 1 (Early/Mild):** Score < 5 but Model Prediction is Disease
*   **Stage 0 (Healthy Liver):** Score < 1.0 (Healthy override) *or* Model Prediction is No Disease

### 3.1. Diagnostic Confidence Distribution
To provide transparency into "borderline" cases, the UI features a **Probability Distribution Chart**. This logic calculates the proximity of the patient's biochemical state to every other clinical stage using a **Gaussian (Normal) Distribution** model.

#### The Mathematical Engine
The backend calculates the "weight" for each stage $i$ using the following formula:

$$W_i = e^{ -\frac{(X_{score} - \mu_i)^2}{2\sigma^2} }$$

*   **$X_{score}$:** The patient's actual Clinical Severity Score.
*   **$\mu_i$ (Midpoint Anchors):** Fixed center points for each stage: 
    *   Stage 0: **0.5** | Stage 1: **3.0** | Stage 2: **7.5** | Stage 3: **12.5** | Stage 4: **17.5**
*   **$\sigma$ (Spread):** Set to **3.5** to allow for clinical overlap.

#### Normalization & Final Confidence
To convert these raw weights into percentages, a **Softmax-style normalization** is applied:
$$Confidence_i = \frac{W_i}{\sum_{j=0}^{4} W_j} \times 100\%$$

This ensures that the total across all bars always equals 100%, providing the clinician with a "Soft Classification" view that highlights borderline risks.

**Healthy Guardrail:** If the ML model predicts "No Disease," the logic bypasses the curve and forces a **95%** confidence spike in Stage 0 to maintain clinical baseline accuracy.

> [!NOTE]
> ### Architecture Note: Machine Learning vs. Clinical Staging
> It is important to clarify that the Machine Learning Model (CatBoost/EBM Ensemble) **was only trained on a binary classification task** (0 = No Disease, 1 = Disease). The LPD dataset did *not* contain severity stages, so the model **cannot dynamically predict Stage 1-4 directly**.
> 
> To achieve 4-Stage resolution, this application employs a hybrid **AI + Heuristic Paradigm**:
> 1. **The AI Gate:** The ML model analyzes the data and outputs a strict Binary result (Disease vs No Disease).
> 2. **The Clinical Matrix:** A custom, rule-based Python function (`get_robust_liver_score`) independently analyzes the patient's exact Liver Function Tests (LFTs) against established medical guidelines (like the De Ritis ratio) to assign severity points.
> 3. **The Merger:** If the AI gates the patient as "Disease Detected," the application maps them into stages 1, 2, 3, or 4 strictly based on their biochemical severity score. If the AI gates them as "Healthy," but the Biochemical score is terrible, they are overridden. If the AI gates them as "Disease," but their scores are immaculate, they are safely overridden to "Healthy Liver."
> 
> **Why this matters for retraining:** This decoupling ensures that you can infinitely retrain, swap, or upgrade the underlying Machine Learning Engine (`train_models.py`) without breaking or retraining the Stage 1-4 classification system, as the Stager operates as an independent module layered on top of the AI's base prediction.

*   **Global Explainability (`shap_summary_plot.png`):** Shows the driving features across the entire dataset. For instance, high AST/ALT and Bilirubin levels push the model's output heavily toward predicting disease.
*   **Local Explainability (`shap_waterfall_plot_sample1.png`):** Breaks down an individual patient's prediction, showing exactly how much each feature (e.g., `albumin = 2.4`) contributed to their specific diagnosis.

## 5. Dual Web Application Architectures

To ensure the Liver Disease Detection model can be utilized in various environments, two separate interfaces have been successfully deployed.

### A. Internal Demo Prototype (Streamlit)
*   **Location:** `v2/v2_streamlit/app.py`
*   **Purpose:** A rapid-prototyping, Python-native interface for data scientists to validate model behavior in real-time.
*   **Features:** Loads the `ensemble_model` and `imputers`, handles the Clinical Staging logic natively, and generates real-time matplotlib SHAP waterfall plots directly on the dashboard.

![Streamlit V2 Interface](./media/streamlit_v2_validation_1773518358420.webp)

### B. Production Web Stack (FastAPI + SvelteKit)
*   **Backend Location:** `v2/v2_production/backend/main.py`
*   **Frontend Location:** `v2/v2_production/frontend/`
*   **Purpose:** A decoupled, scalable architecture designed for production deployment and high concurrent traffic.

#### Architecture Details
*   **FastAPI Pipeline:** Operates as a REST API on Port 8000. It manages the inference engine, Pydantic data validation, categorical imputations, and calculates the 4-Stage Heuristic logic. It encodes SHAP explanation plots into **Base64 strings** to send over the network efficiently.
*   **SvelteKit Interface:** A lightning-fast, reactive frontend utilizing **Svelte 5 Runes** and **Tailwind CSS v4** styling. It features a modern, dark-mode medical UI that safely fetches JSON data from the backend to render dynamic classification cards, confidence scores, and SHAP diagrams.

![SvelteKit Production Interface](./media/hepaguard_landing_1773520152880.png)
![SvelteKit Interaction](./media/hepaguard_partial_fill_1773520256100.png)
![SvelteKit Analysis Proof](./media/svelte_v2_final_successful_syntax_check_1773520600526.webp)
