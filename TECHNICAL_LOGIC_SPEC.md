> [!NOTE]
> This document contains mathematical formulas written in LaTeX format (using `$` and `$$` symbols) and standard Markdown (using `*` and `**`). These are best viewed in a Markdown renderer like VS Code, GitHub, or a dedicated Markdown previewer.

# HepaGuard AI V2: Mathematical & Logic Specification

This document provides a deep-dive into the mathematical foundations and logical heuristics used in the HepaGuard AI production environment.

---

## 1. Data Cleaning: Imputation Mathematics

Missing values are handled using statistical central tendency. For a feature vector $X$ with missing value $X_i$:

### Numerical Imputation (Median)
The missing value is replaced by the median of the training set $X_{train}$:
$$ \tilde{x} = \text{median}(X_{train}) $$
*Rationale:* Median is robust to outliers in biomarkers (e.g., extreme Bilirubin spikes) compared to the mean.

### Categorical Imputation (Mode)
The missing value is replaced by the most frequent category $c$:
$$ \tilde{c} = \text{argmax}_{j} \left( \text{count}(c_j) \in X_{train} \right) $$

---

## 2. Model Ensemble: Soft Voting Logic

The system uses a **Soft Voting Classifier** combining an Explainable Boosting Machine (EBM) and CatBoost.

### Probability Aggregation
For each class $k$ (0 = Healthy, 1 = Disease), the ensemble calculates the weighted average probability:
$$ P(y=k | X) = \frac{1}{N} \sum_{i=1}^{N} P_i(y=k | X) $$
Where $P_i$ is the probability assigned by model $i$. 

- If $P(y=1 | X) \geq 0.5$, the initial classification is **Diseased (1)**.
- If $P(y=1 | X) < 0.5$, the initial classification is **Healthy (0)**.

---

## 3. Clinical Staging: The "Robust Liver Score"

This heuristic overrides or refines the ML prediction using clinically validated thresholds.

### Score Calculation ($S$)
The total severity score $S$ is the sum of five biological markers:

1.  **Bilirubin Penalty ($B$):**
    If $\text{Bilirubin} > 1.2$:
    $$ B = \min((\text{Bilirubin} - 1.2) \times 1.5, 10.0) $$
2.  **De Ritis Ratio ($R$):**
    Defined as $AST / ALT$. 
    - If $R > 1.5$, $S_{ratio} = 3.0$
    - If $R > 2.0$, $S_{ratio} = 8.0$ (cumulative)
3.  **Albumin Synthesis Deficit ($A$):**
    If $\text{Albumin} < 3.5$:
    $$ A = (3.5 - \text{Albumin}) \times 4.0 $$
4.  **Cholestasis Marker (AlkPhos):**
    If $\text{AlkPhos} > 300$, add **4.0** to score.
5.  **Age Multiplier ($M$):**
    If $\text{Age} > 60$:
    $$ S_{final} = S_{raw} \times 1.1 $$

### Stage Mapping Functions
The stage $G$ is assigned based on $S_{final}$ and the ML prediction $y$:
- **Healthy ($G=0$):** If $y=0$ OR $S_{final} < 1.0$
- **Stage 1:** If $1.0 \leq S_{final} < 5.0$
- **Stage 2:** If $5.0 \leq S_{final} < 10.0$
- **Stage 3:** If $10.0 \leq S_{final} < 15.0$
- **Stage 4:** If $S_{final} \geq 15.0$

---

## 4. Confidence Distribution: Gaussian Spread

To visualize diagnostic uncertainty, the backend generates a probability curve across all stages using a Gaussian distribution.

### Normalization Formula
For each stage midpoint $m \in \{0.5, 3.0, 7.5, 12.5, 17.5\}$, we calculate a raw weight $w$:
$$ w_m = e^{-\frac{(m - S_{final})^2}{2\sigma^2}} $$
Where $\sigma$ (spread) is globally set to **3.5** to allow for "borderline" overlaps between stages.

The final confidence for each stage $i$ is normalized to 1.0:
$$ C_i = \frac{w_i}{\sum_{j=0}^{4} w_j} $$

---

## 5. Explainable AI: SHAP (Shapley Values)

The system calculates individual feature contributions using the cooperative game theory approach:
$$ \phi_i(v) = \sum_{S \subseteq \{1, \dots, p\} \setminus \{i\}} \frac{|S|! (p - |S| - 1)!}{p!} [v(S \cup \{i\}) - v(S)] $$
Where:
- $\phi_i$ is the contribution of feature $i$ (e.g., Bilirubin).
- Blue bars in the UI represent negative $\phi_i$ (pushing prediction toward Healthy).
- Red bars in the UI represent positive $\phi_i$ (pushing prediction toward Disease).
