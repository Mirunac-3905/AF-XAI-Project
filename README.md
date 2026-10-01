# AF-XAI — Adaptive Federated Explainable AI for Smart Grid Fault Detection

IEEE research prototype for federated, explainable fault detection in smart grids.

## Project objective

Detect and explain smart-grid faults while keeping substation data local. Each simulated substation trains a local anomaly model; a central server aggregates those models with quality-aware weighting, then supports fault prediction and post-hoc explanations through an API and a planned dashboard.

## Simulated substations

The prototype uses **three simulated substations**. Each site holds its own operational data and trains independently. Raw measurements stay at the substation; only model updates are shared for aggregation.

## Local One-Class SVM

Each substation trains a **One-Class SVM** on local (mostly normal) operating data. The model is intended for unsupervised anomaly detection of unusual grid conditions without requiring labelled fault samples at every site.

## Adaptive quality-aware aggregation

Local updates are combined into a **global model** with **adaptive, quality-aware aggregation**. Contribution of each substation is weighted by estimated update quality (for example data sufficiency or local performance), rather than uniform averaging.

## Fault prediction

The global (and optionally local) models are used for **fault prediction / anomaly scoring** on incoming grid features, so operators can flag likely fault conditions before or as they appear.

## SHAP / LIME explainability

Predictions are accompanied by **SHAP** and **LIME** explanations so that feature contributions (voltage, current, frequency, and related signals) can be inspected. This is the XAI layer of AF-XAI.

## Flask API

A **Flask** backend (`backend/app.py`) will expose training, aggregation, prediction, and explanation endpoints for the prototype.

## React dashboard (planned)

A **React** frontend under `frontend/` is planned for visualising substation status, metrics, predictions, and explanations. UI implementation is not part of this initial structure.

## Repository layout

```
FINAL-YEAR-PROJECT/
├── dataset/          # raw and processed data (not committed)
├── ml/               # preprocessing, local training, aggregation, evaluation, explainability
├── models/           # local and global artefacts (generated, not committed)
├── results/          # metrics, figures, reports (generated, not committed)
├── backend/          # Flask API
├── frontend/         # React dashboard (planned)
├── config/           # configuration
├── requirements.txt
├── README.md
└── .gitignore
```
