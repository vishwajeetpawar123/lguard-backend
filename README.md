# 🩺 HepaGuard AI V2: Diagnostic Suite

A high-performance liver disease diagnostic system leveraging **Soft-Voting Ensembles** and **Explainable AI (SHAP)**. This suite provides a production-grade web application with automated disease staging.

---

## 📄 Technical Documentation
Detailed specifications and logs are located in the [documentation/](file:///c:/Users/vishw/Desktop/vaishnavi%20part%202/v2/documentation/) directory:
* **[Technical Logic Spec](https://github.com/vishwajeetpawar123/lguard-backend/blob/main/TECHNICAL_LOGIC_SPEC.md)**: Deep dive into the Ensemble math and staging heuristics.
* **[Development Walkthrough](https://github.com/vishwajeetpawar123/lguard-backend/blob/main/WALKTHROUGH.md)**: Proof of work, verification results, and build history.
* **[Implementation Plan](https://github.com/vishwajeetpawar123/lguard-backend/blob/main/IMPLEMENTATION_PLAN.md)**: The architectural roadmap for the project.

---

## 🚀 Deployment Option

### 🐳 Option A: Local Docker (Recommended)
Deploy the entire stack (Backend + Frontend) in seconds using Docker Compose.
1. Navigate to `v2/v2_deployment`.
2. Run:
   ```bash
   docker-compose up --build -d
   ```
3. Access: **Frontend** (localhost:3000) | **Backend API** (localhost:8000)

### ☁️Public Web (Railway + Vercel) 
The suite is optimized for multi-platform cloud hosting:
* **Backend**: Optimized for **Railway** (Handles AI models & Docker).
* **Frontend**: Optimized for **Vercel** (SvelteKit performance).


---

## 🔬 Core Architecture

### Machine Learning Engine
The system uses an ensemble of state-of-the-art models:
1. **Explainable Boosting Machine (EBM):** A glass-box model for high-fidelity explanations.
2. **CatBoost:** A robust gradient boosting framework for tabular performance.

*Results:* **99.79% Accuracy** | **99.86% F1-Score**.

### Clinical Staging Heuristic
A custom **Biomedical Stager** assigns clinical stages based on established medical ratios:
* **Stage 1 (Mild)**: Warning signs, enzyme elevations.
* **Stage 2 (Significant)**: Early scarring detected.
* **Stage 3 (Severe)**: Advanced cirrhosis/protein drops.
* **Stage 4 (Critical)**: End-stage markers.
* **Stage 0**: Healthy liver baseline.

---.
