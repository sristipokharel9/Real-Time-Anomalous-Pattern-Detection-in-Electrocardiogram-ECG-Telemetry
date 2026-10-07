"""
================================================================================
A Python-Based Analytical Pipeline for Real-Time Anomalous Pattern Detection
in Electrocardiogram (ECG) Telemetry
================================================================================
College Data Science Project — Full Pipeline (Rounds 1-6 consolidated)

Dataset: ECG Heartbeat Categorization Dataset (Kaggle, Shayan Fazeli)
Files required in the same folder as this script:
    mitbih_train.csv, mitbih_test.csv, ptbdb_normal.csv, ptbdb_abnormal.csv

NOTE: This script was developed and executed in Google Colab. It is presented
here as a single consolidated file for code review / demonstration purposes.
To actually RUN it locally, you need Python with pandas, numpy, matplotlib,
seaborn, and scikit-learn installed, and the 4 CSV files in this folder.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import joblib

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.max_columns", 20)


# ==============================================================================
# ROUND 1 — DATA COLLECTION, LOADING AND UNDERSTANDING
# ==============================================================================

def round1_load_data():
    """Load and verify the four ECG CSV files (no header row; col 187 = label)."""
    required_files = ["mitbih_train.csv", "mitbih_test.csv",
                       "ptbdb_normal.csv", "ptbdb_abnormal.csv"]
    expected_lines = {"mitbih_train.csv": 87554, "mitbih_test.csv": 21892,
                       "ptbdb_normal.csv": 4046, "ptbdb_abnormal.csv": 10506}

    print("STEP 1: Checking file integrity")
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")
        with open(f) as fh:
            lc = sum(1 for _ in fh)
        status = "OK" if lc == expected_lines[f] else "MISMATCH"
        print(f"  {f:22s} {lc:6d} lines -> {status}")

    print("\nSTEP 2: Loading CSVs")
    mitbih_train = pd.read_csv("mitbih_train.csv", header=None)
    mitbih_test = pd.read_csv("mitbih_test.csv", header=None)
    ptbdb_normal = pd.read_csv("ptbdb_normal.csv", header=None)
    ptbdb_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)

    n_features = mitbih_train.shape[1] - 1  # 187 signal points
    col_names = [f"c{i}" for i in range(n_features)] + ["target"]
    for df in (mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal):
        df.columns = col_names

    ptbdb_full = pd.concat([ptbdb_normal, ptbdb_abnormal], axis=0, ignore_index=True)

    print("\nSTEP 3: Shapes")
    print("  mitbih_train  :", mitbih_train.shape)
    print("  mitbih_test   :", mitbih_test.shape)
    print("  ptbdb_normal  :", ptbdb_normal.shape)
    print("  ptbdb_abnormal:", ptbdb_abnormal.shape)

    print("\nSTEP 4: Class distribution (mitbih_train)")
    print(mitbih_train["target"].value_counts().sort_index())

    return mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal, ptbdb_full, col_names


# Class label reference (AAMI EC57 mapping used by MIT-BIH)
MITBIH_CLASS_NAMES = {
    0: "N (Normal)",
    1: "S (Supraventricular)",
    2: "V (Ventricular)",
    3: "F (Fusion)",
    4: "Q (Unknown/Paced)"
}


# ==============================================================================
# ROUND 2 — EXPLORATORY DATA ANALYSIS
# ==============================================================================

def round2_eda(mitbih_train, FEATURE_COLS, TARGET_COL):
    """Generate the core EDA visualizations: class distribution, waveforms,
    amplitude distribution, boxplots, correlation heatmap."""

    # Class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = mitbih_train[TARGET_COL].value_counts().sort_index()
    sns.barplot(x=[MITBIH_CLASS_NAMES[int(c)] for c in counts.index], y=counts.values, ax=ax)
    ax.set_title("MIT-BIH: Class Distribution")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig("fig_class_distribution.png", dpi=150)
    plt.close()

    # Mean waveform per class
    plt.figure(figsize=(12, 6))
    for cls in range(5):
        mean_wave = mitbih_train[mitbih_train[TARGET_COL] == cls][FEATURE_COLS].mean(axis=0)
        plt.plot(mean_wave.values, label=MITBIH_CLASS_NAMES[cls])
    plt.title("Mean ECG Waveform by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_class_comparison.png", dpi=150)
    plt.close()

    print("EDA figures saved: fig_class_distribution.png, fig_class_comparison.png")


# ==============================================================================
# ROUND 3 — PREPROCESSING AND FEATURE ENGINEERING
# ==============================================================================

def extract_statistical_features(signal_df):
    """8 statistical features per heartbeat, computed row-wise (no leakage)."""
    v = signal_df.values
    return pd.DataFrame({
        "feat_mean":   v.mean(axis=1),
        "feat_std":    v.std(axis=1),
        "feat_min":    v.min(axis=1),
        "feat_max":    v.max(axis=1),
        "feat_range":  v.max(axis=1) - v.min(axis=1),
        "feat_median": np.median(v, axis=1),
        "feat_rms":    np.sqrt(np.mean(np.square(v), axis=1)),
        "feat_energy": np.sum(np.square(v), axis=1),
    }, index=signal_df.index)


def round3_preprocess(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal, FEATURE_COLS, TARGET_COL):
    """Stratified split, feature engineering, scaling (no leakage), class weights,
    and PTBDB preparation for anomaly detection."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    # --- MIT-BIH: respect official train/test split, carve stratified validation ---
    X_train_full_raw = mitbih_train[FEATURE_COLS].copy()
    y_train_full = mitbih_train[TARGET_COL].astype(int).copy()
    X_test_raw = mitbih_test[FEATURE_COLS].copy()
    y_test = mitbih_test[TARGET_COL].astype(int).copy()

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full_raw, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)

    # --- Feature engineering (row-wise, no leakage) ---
    X_train_combined = pd.concat([X_train_raw.reset_index(drop=True),
        extract_statistical_features(X_train_raw).reset_index(drop=True)], axis=1)
    X_val_combined = pd.concat([X_val_raw.reset_index(drop=True),
        extract_statistical_features(X_val_raw).reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test_raw.reset_index(drop=True),
        extract_statistical_features(X_test_raw).reset_index(drop=True)], axis=1)

    # --- Scale: fit ONLY on training data ---
    scaler = StandardScaler().fit(X_train_combined)
    X_train_final = scaler.transform(X_train_combined)
    X_val_final = scaler.transform(X_val_combined)
    X_test_final = scaler.transform(X_test_combined)

    # --- Class weights (handle imbalance) ---
    classes = np.unique(y_train)
    class_weight_dict = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

    # --- PTBDB: normal-only training baseline for anomaly detection ---
    ptbdb_normal_train, ptbdb_normal_test = train_test_split(ptbdb_normal, test_size=0.3, random_state=42)

    ptb_train_combined = pd.concat([ptbdb_normal_train[FEATURE_COLS].reset_index(drop=True),
        extract_statistical_features(ptbdb_normal_train[FEATURE_COLS]).reset_index(drop=True)], axis=1)
    ptb_test_normal_combined = pd.concat([ptbdb_normal_test[FEATURE_COLS].reset_index(drop=True),
        extract_statistical_features(ptbdb_normal_test[FEATURE_COLS]).reset_index(drop=True)], axis=1)
    ptb_test_abnormal_combined = pd.concat([ptbdb_abnormal[FEATURE_COLS].reset_index(drop=True),
        extract_statistical_features(ptbdb_abnormal[FEATURE_COLS]).reset_index(drop=True)], axis=1)

    ptbdb_scaler = StandardScaler().fit(ptb_train_combined)
    X_ptb_train_final = ptbdb_scaler.transform(ptb_train_combined)
    X_ptb_test_normal_final = ptbdb_scaler.transform(ptb_test_normal_combined)
    X_ptb_test_abnormal_final = ptbdb_scaler.transform(ptb_test_abnormal_combined)

    print("Preprocessing complete.")
    print("  X_train_final:", X_train_final.shape, "| classes:", sorted(y_train.unique()))
    print("  X_val_final  :", X_val_final.shape)
    print("  X_test_final :", X_test_final.shape)
    print("  PTBDB train baseline:", X_ptb_train_final.shape)

    return {
        "X_train_final": X_train_final, "y_train": y_train,
        "X_val_final": X_val_final, "y_val": y_val,
        "X_test_final": X_test_final, "y_test": y_test,
        "class_weight_dict": class_weight_dict,
        "scaler": scaler, "ptbdb_scaler": ptbdb_scaler,
        "X_ptb_train_final": X_ptb_train_final,
        "X_ptb_test_normal_final": X_ptb_test_normal_final,
        "X_ptb_test_abnormal_final": X_ptb_test_abnormal_final,
        "ptbdb_normal_test": ptbdb_normal_test,
    }


# ==============================================================================
# ROUND 4 — MACHINE LEARNING CLASSIFICATION
# ==============================================================================

def round4_classification(data):
    """Train Logistic Regression, Random Forest, and SVM; compare via macro-F1;
    evaluate the best model on the held-out test set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train_final, y_train = data["X_train_final"], data["y_train"]
    X_val_final, y_val = data["X_val_final"], data["y_val"]
    X_test_final, y_test = data["X_test_final"], data["y_test"]
    class_weight_dict = data["class_weight_dict"]

    def metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight_dict,
                                  multi_class="multinomial", solver="lbfgs", random_state=42)
    log_reg.fit(X_train_final, y_train)
    m_lr = metrics(y_val, log_reg.predict(X_val_final))

    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                       class_weight=class_weight_dict, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_final, y_train)
    m_rf = metrics(y_val, rf_model.predict(X_val_final))

    print("Training SVM (stratified 15k subsample for tractability)...")
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train_final, y_train, train_size=15000, stratify=y_train, random_state=42)
    svm_model = SVC(kernel="rbf", class_weight=class_weight_dict, random_state=42)
    svm_model.fit(X_train_svm, y_train_svm)
    m_svm = metrics(y_val, svm_model.predict(X_val_final))

    comparison_df = pd.DataFrame({
        "Logistic Regression": m_lr, "Random Forest": m_rf, "SVM": m_svm
    }).T.round(4)
    print("\n=== MODEL COMPARISON (Validation) ===")
    print(comparison_df)

    # Select best model by macro-F1 (not accuracy alone — imbalanced classes)
    best_name = comparison_df["F1-score (macro)"].idxmax()
    best_model = {"Logistic Regression": log_reg, "Random Forest": rf_model, "SVM": svm_model}[best_name]
    print(f"\nBest model: {best_name}")

    y_test_pred = best_model.predict(X_test_final)
    test_metrics = metrics(y_test, y_test_pred)
    print(f"\n=== {best_name}: FINAL TEST SET PERFORMANCE ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return best_model, best_name, comparison_df, test_metrics


# ==============================================================================
# ROUND 5 — ECG ANOMALY DETECTION (Isolation Forest, core of the project)
# ==============================================================================

def round5_anomaly_detection(data):
    """Train Isolation Forest on normal ECG baseline only; evaluate against
    held-out normal + all abnormal PTBDB samples."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix)

    X_ptb_train_final = data["X_ptb_train_final"]
    X_ptb_test_normal_final = data["X_ptb_test_normal_final"]
    X_ptb_test_abnormal_final = data["X_ptb_test_abnormal_final"]

    print("Training Isolation Forest on normal-only baseline...")
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_ptb_train_final)

    X_anomaly_test = np.vstack([X_ptb_test_normal_final, X_ptb_test_abnormal_final])
    y_anomaly_true = np.concatenate([
        np.zeros(X_ptb_test_normal_final.shape[0]),
        np.ones(X_ptb_test_abnormal_final.shape[0])
    ])

    raw_pred = iso_forest.predict(X_anomaly_test)
    y_anomaly_pred = np.where(raw_pred == -1, 1, 0)
    anomaly_scores = iso_forest.decision_function(X_anomaly_test)

    roc_auc = roc_auc_score(y_anomaly_true, -anomaly_scores)

    results = {
        "Accuracy": accuracy_score(y_anomaly_true, y_anomaly_pred),
        "Precision": precision_score(y_anomaly_true, y_anomaly_pred, zero_division=0),
        "Recall": recall_score(y_anomaly_true, y_anomaly_pred, zero_division=0),
        "F1-score": f1_score(y_anomaly_true, y_anomaly_pred, zero_division=0),
        "ROC-AUC": roc_auc,
    }

    print("\n=== ANOMALY DETECTION PERFORMANCE ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(y_anomaly_true, y_anomaly_pred)
    print("\nConfusion Matrix:\n", cm)

    return iso_forest, results, X_anomaly_test, y_anomaly_true, y_anomaly_pred, anomaly_scores


# ==============================================================================
# ROUND 6 — REAL-TIME ECG TELEMETRY SIMULATION
# ==============================================================================

def round6_realtime_simulation(iso_forest, ptbdb_scaler, ptbdb_normal_test, ptbdb_abnormal,
                                FEATURE_COLS, n_readings=20, delay_seconds=0.0):
    """Simulate sequential ECG telemetry: each reading is processed through
    the same preprocessing pipeline and scored live by the trained detector.
    NOTE: this is playback of recorded ECG data, not a live medical device."""

    stream_source = pd.concat([
        ptbdb_normal_test[FEATURE_COLS].assign(true_label="Normal"),
        ptbdb_abnormal[FEATURE_COLS].sample(n=min(200, len(ptbdb_abnormal)), random_state=42).assign(true_label="Abnormal")
    ], axis=0).sample(frac=1, random_state=7).reset_index(drop=True)

    def process_reading(raw_row):
        signal_df = pd.DataFrame([raw_row.values], columns=FEATURE_COLS)
        stats = extract_statistical_features(signal_df)
        combined = pd.concat([signal_df.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
        scaled = ptbdb_scaler.transform(combined)
        score = iso_forest.decision_function(scaled)[0]
        pred = iso_forest.predict(scaled)[0]
        status = "ANOMALY" if pred == -1 else "NORMAL"
        return status, score

    print("=" * 40)
    print("ECG TELEMETRY MONITOR — SIMULATION START")
    print("=" * 40)

    log = []
    for i in range(min(n_readings, len(stream_source))):
        row = stream_source.iloc[i]
        status, score = process_reading(row[FEATURE_COLS])
        log.append({"reading": i + 1, "score": score, "status": status, "true_label": row["true_label"]})
        print(f"Reading: {i+1}")
        print(f"Anomaly Score: {score:.4f}")
        print(f"Status: {'⚠ ANOMALY DETECTED' if status == 'ANOMALY' else 'NORMAL'}")
        print("-" * 40)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    log_df = pd.DataFrame(log)
    print("\nSimulation complete.")
    print(log_df["status"].value_counts())
    return log_df


# ==============================================================================
# MAIN — run the full pipeline end to end
# ==============================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("# ECG ANOMALY DETECTION PROJECT — FULL PIPELINE RUN")
    print("#" * 70)

    mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal, ptbdb_full, col_names = round1_load_data()
    FEATURE_COLS = col_names[:-1]
    TARGET_COL = "target"

    print("\n" + "#" * 70)
    print("# ROUND 2 — EDA")
    print("#" * 70)
    round2_eda(mitbih_train, FEATURE_COLS, TARGET_COL)

    print("\n" + "#" * 70)
    print("# ROUND 3 — PREPROCESSING")
    print("#" * 70)
    data = round3_preprocess(mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal, FEATURE_COLS, TARGET_COL)

    print("\n" + "#" * 70)
    print("# ROUND 4 — CLASSIFICATION")
    print("#" * 70)
    best_model, best_name, comparison_df, test_metrics = round4_classification(data)
    joblib.dump(best_model, "best_classification_model.joblib")

    print("\n" + "#" * 70)
    print("# ROUND 5 — ANOMALY DETECTION")
    print("#" * 70)
    iso_forest, anomaly_results, X_anomaly_test, y_anomaly_true, y_anomaly_pred, anomaly_scores = \
        round5_anomaly_detection(data)
    joblib.dump(iso_forest, "isolation_forest_model.joblib")

    print("\n" + "#" * 70)
    print("# ROUND 6 — REAL-TIME SIMULATION")
    print("#" * 70)
    log_df = round6_realtime_simulation(
        iso_forest, data["ptbdb_scaler"], data["ptbdb_normal_test"], ptbdb_abnormal,
        FEATURE_COLS, n_readings=20, delay_seconds=0.0
    )

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70)
