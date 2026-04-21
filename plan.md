# Machine Learning Interpretability Project — Development Plan

## Overview
This plan is structured to prioritize blocking tasks, enable parallel execution, and favor vertical slices (end-to-end deliverables early). The goal is to quickly reach a functional pipeline and progressively deepen analysis, especially interpretability across KNN, Naïve Bayes, and Decision Tree models.

---

## 0) Initial Definition (Global Blocking Step)

### Objectives
- Select dataset
- Define classification target
- Define evaluation metric
- Define tech stack

### Tasks
- Choose dataset (see options below)
- Define target variable (categorical)
- Select primary metric (e.g., F1-score for imbalanced data)
- Define stack:
  - Python
  - pandas
  - scikit-learn
  - shap
  - lime

### Deliverable
- Initial README with problem description and dataset justification

---

## Dataset Options (Choose One)

### Option 1 — Bank Marketing (UCI)
- Classification: subscription to financial product
- Mixed feature types
- Strong interpretability potential

### Option 2 — Student Performance (UCI)
- Classification: pass/fail or grade category
- Many categorical variables
- Good for socio-contextual analysis

### Option 3 — Heart Failure Clinical Records (UCI/Kaggle)
- Classification: survival prediction
- Real-world medical relevance
- Strong feature importance signals

All options satisfy:
- ≥100 instances
- Mixed feature types
- Not part of banned “toy datasets”

---

## 1) Minimal Functional Pipeline (First Vertical Slice)

**Dependency:** Dataset selected

### Objective
Produce a complete working pipeline (data → model → evaluation)

### Tasks
- Load dataset
- Basic cleaning (drop or simple imputation)
- Split into X / y
- Train/test split
- Train **Decision Tree**
- Evaluate (accuracy + F1)

### Deliverable
- Single script/notebook running end-to-end
- Baseline metrics

---

## 2) Robust Preprocessing Pipeline

**Dependency:** Step 1 complete

### Tasks
- Build preprocessing pipeline:
  - Numerical imputation
  - Categorical imputation
  - Encoding (OneHotEncoder)
  - Scaling (StandardScaler for KNN)
- Use `ColumnTransformer`
- Wrap everything in sklearn `Pipeline`

### Deliverable
- Reusable preprocessing + training pipeline

### Parallelization
- Can run alongside early model experimentation

---

## 3) Train All Models (Controlled Horizontal Expansion)

**Dependency:** Pipeline ready

### Tasks
- Train:
  - KNN (vary k)
  - Naïve Bayes (Gaussian or Multinomial)
  - Decision Tree (tune depth)
- Evaluate all models using same metrics

### Deliverable
- Comparative metrics table

### Parallel Execution
- Agent 1: KNN
- Agent 2: Naïve Bayes
- Agent 3: Decision Tree

---

## 4) Performance Validation (Quality Gate)

**Dependency:** Models trained

### Tasks
- Hyperparameter tuning (GridSearch or simple sweeps)
- Check overfitting
- Ensure results are meaningful

### Deliverable
- Finalized models suitable for interpretation

---

## 5) Model Interpretability (Vertical per Model)

Each model can be handled independently.

---

### 5.1 Decision Tree (Start Here)

### Tasks
- Visualize tree
- Extract `feature_importances_`
- Analyze decision paths

### Deliverable
- Clear rule-based explanations

---

### 5.2 Naïve Bayes

### Tasks
- Analyze conditional probabilities
- Study feature influence per class
- Compare distributions

### Deliverable
- Probabilistic interpretation

---

### 5.3 KNN (Most Challenging)

### Tasks
- Explain instance-based reasoning
- Apply:
  - SHAP or
  - LIME
- Interpret specific predictions

### Deliverable
- Local explanations

---

## 6) Global Interpretability (Cross-Model)

**Can run in parallel with Step 5**

### Tasks
- Permutation importance
- SHAP (global view)
- Compare feature importance across models

### Deliverable
- Consolidated feature ranking

---

## 7) Comparative Analysis

**Dependency:** Interpretability completed

### Tasks
- Compare:
  - Agreement between models
  - Divergences in feature importance
- Discuss:
  - Interpretability vs performance
  - Model limitations

### Deliverable
- Analytical discussion (README or notebook)

---

## 8) Presentation

**Can be started early, finalized last**

### Tasks
- Structure presentation:
  1. Problem + dataset
  2. Pipeline
  3. Models
  4. Interpretability
  5. Comparison
- Record video (10–15 minutes)

---

## Suggested Agent-Based Structure

### Agent 1 — Data & Pipeline
- Dataset handling
- Cleaning
- Preprocessing pipeline

### Agent 2 — Modeling
- Train and evaluate models
- Basic tuning

### Agent 3 — Interpretability
- SHAP / LIME
- Model-specific analysis

### Agent 4 — Synthesis
- Comparative analysis
- Documentation + presentation

---

## Execution Order (Dependency-Oriented)

1. Dataset selection (global blocker)
2. Minimal pipeline (vertical slice)
3. Robust preprocessing
4. Parallel model training
5. Model validation
6. Interpretability (parallel per model)
7. Comparative analysis
8. Presentation

---

## Practical Notes

- Start with Decision Tree to reduce friction
- KNN requires scaling → ensure pipeline correctness
- SHAP can be computationally expensive → use sampling
- LIME is lighter and useful for demonstrations
