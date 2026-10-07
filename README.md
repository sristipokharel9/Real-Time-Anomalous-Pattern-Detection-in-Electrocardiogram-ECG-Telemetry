# Real-Time-Anomalous-Pattern-Detection-in-Electrocardiogram-ECG-Telemetry

A Python-based analytical pipeline for **real-time anomalous pattern detection in electrocardiogram (ECG) telemetry**. The project covers the full data science workflow: data loading, exploratory analysis, feature engineering, multi-class heartbeat classification, unsupervised anomaly detection, and a simulated real-time monitoring stream.

> ⚠️ **Disclaimer:** This is an educational / academic project. The "real-time" component is a playback simulation of recorded ECG data. It is **not** a medical device and must not be used for diagnosis or clinical decisions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Methodology Notes](#methodology-notes)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Acknowledgements](#acknowledgements)

---

## Overview

The project has two complementary goals:

1. **Supervised classification:** classify individual heartbeats from the MIT-BIH Arrhythmia dataset into five AAMI classes.
2. **Unsupervised anomaly detection:** train an Isolation Forest on *normal-only* PTB Diagnostic ECG beats, then flag abnormal beats as anomalies, which mirrors a real monitoring scenario where abnormal examples are rare or unknown.

The trained anomaly detector is then used in a simulated telemetry monitor that scores heartbeats one at a time.

## Dataset

[ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (Kaggle, Shayan Fazeli). Each row is a single heartbeat: **187 signal samples** followed by a class label in the final column. Files have no header row.

| File | Rows | Use |
|---|---|---|
| `mitbih_train.csv` | 87,553 | Classification training + validation |
| `mitbih_test.csv` | 21,891 | Classification final test |
| `ptbdb_normal.csv` | 4,045 | Normal baseline for anomaly detection |
| `ptbdb_abnormal.csv` | 10,505 | Abnormal samples for anomaly evaluation |

**MIT-BIH class labels (AAMI EC57):**

| Label | Class |
|---|---|
| 0 | N: Normal |
| 1 | S: Supraventricular |
| 2 | V: Ventricular |
| 3 | F: Fusion |
| 4 | Q: Unknown / Paced |

> The dataset is not included in this repository. Download the four CSV files from Kaggle and place them in the project root.

## Pipeline

The script is organised into six rounds, each a function:

| Round | Function | Description |
|---|---|---|
| 1 | `round1_load_data` | Verifies file integrity (line counts), loads CSVs, assigns column names, prints shapes and class distribution |
| 2 | `round2_eda` | Class distribution bar chart and mean waveform per class |
| 3 | `round3_preprocess` | Stratified train/validation split, 8 statistical features per beat, `StandardScaler`, class weights, PTBDB normal/abnormal preparation |
| 4 | `round4_classification` | Trains Logistic Regression, Random Forest and SVM; selects best by macro-F1; evaluates on the test set |
| 5 | `round5_anomaly_detection` | Isolation Forest trained on normal beats only; reports accuracy, precision, recall, F1, ROC-AUC and confusion matrix |
| 6 | `round6_realtime_simulation` | Streams beats sequentially through the same preprocessing and scores each one live |

### Engineered features

For every heartbeat, 8 statistical features are appended to the 187 raw samples: mean, standard deviation, min, max, range, median, RMS, and energy.

### Models

- **Logistic Regression:** multinomial, class-weighted
- **Random Forest:** 200 trees, max depth 20, class-weighted
- **SVM:** RBF kernel, class-weighted, trained on a stratified 15k subsample for tractability
- **Isolation Forest:** 200 estimators, contamination = 0.05

## Project Structure

```
.
├── ecg_project_full.py          # Full consolidated pipeline (Rounds 1-6)
├── mitbih_train.csv             # (download separately)
├── mitbih_test.csv              # (download separately)
├── ptbdb_normal.csv             # (download separately)
├── ptbdb_abnormal.csv           # (download separately)
└── README.md
```

## Installation

Requires **Python 3.8+**.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Optionally, use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

> **Note:** the script uses `LogisticRegression(multi_class="multinomial")`, which is deprecated in newer scikit-learn versions (1.5+) and may emit a warning or fail in future releases. If you hit an error, remove the `multi_class` argument; the default behaviour is already multinomial with the `lbfgs` solver.

## Usage

1. Download the four CSV files from Kaggle and place them in the same folder as the script.
2. Run the full pipeline:

```bash
python ecg_project_full.py
```

The script was originally developed in **Google Colab**. To run it there, upload the CSVs to the Colab session and execute the file (or paste it into a cell).

### Using individual rounds

Each round is a standalone function, so you can import and run parts of the pipeline:

```python
from ecg_project_full import round1_load_data, round3_preprocess

mitbih_train, mitbih_test, ptb_n, ptb_a, ptb_full, cols = round1_load_data()
data = round3_preprocess(mitbih_train, mitbih_test, ptb_n, ptb_a, cols[:-1], "target")
```

### Loading saved models

```python
import joblib

clf = joblib.load("best_classification_model.joblib")
iso = joblib.load("isolation_forest_model.joblib")
```

## Outputs

Running the pipeline produces:

| File | Description |
|---|---|
| `fig_class_distribution.png` | Class distribution of MIT-BIH training data |
| `fig_class_comparison.png` | Mean ECG waveform per class |
| `best_classification_model.joblib` | Best classifier (selected by validation macro-F1) |
| `isolation_forest_model.joblib` | Trained anomaly detector |

Console output includes the model comparison table, final test metrics, anomaly detection metrics with confusion matrix, and a 20-reading simulated telemetry log:

```
========================================
ECG TELEMETRY MONITOR — SIMULATION START
========================================
Reading: 1
Anomaly Score: 0.0xxx
Status: NORMAL
----------------------------------------
```

## Methodology Notes

- **No data leakage:** statistical features are computed row-wise, and scalers are fit on training data only.
- **Official splits respected:** MIT-BIH's provided train/test split is kept; a stratified 15% validation set is carved from the training file.
- **Class imbalance handled:** balanced class weights are used, and the best model is chosen by **macro-F1** rather than accuracy alone.
- **Reproducibility:** `random_state=42` is used throughout.
- **Anomaly detection setup:** the Isolation Forest sees only normal beats during training; held-out normal beats plus all abnormal beats form the evaluation set.

## Limitations

- The simulation replays recorded beats; it does not connect to a live ECG device.
- Features are simple statistical summaries of pre-segmented beats. No R-peak detection, filtering, or deep learning is used.
- The anomaly detector is trained and evaluated on the PTB Diagnostic dataset only; generalisation to other recording setups is untested.
- SVM is trained on a subsample, which may understate its potential performance.

## Tech Stack

Python · NumPy · pandas · Matplotlib · seaborn · scikit-learn · joblib

## Acknowledgements

- Dataset: [Shayan Fazeli, ECG Heartbeat Categorization Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Original sources: MIT-BIH Arrhythmia Database and PTB Diagnostic ECG Database (PhysioNet)

## License

Add your preferred license here (e.g., MIT).
