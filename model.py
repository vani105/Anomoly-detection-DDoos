import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Optional: For even faster linear SVM, uncomment the next line and use LinearSVC below
# from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from sklearn.utils import resample  # For stratified subsampling

# Define paths
DATA_PATH = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Make sure this file exists
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)  # Create model directory if it doesn't exist

# Feature columns (must match the ones used in app.py - adjust to your CSV's 51 features if needed)
# Note: Your output shows 51 features, so ensure this list has exactly those (or update based on your CSV inspection)
FEATURES = ['Source Port', 'Destination Port', 'Protocol', 'Flow Duration',
            'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow IAT Mean',
            'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
            'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
            'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
            'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
            'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
            'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']

TARGET = 'Label'  # No leading space

# Configuration for speed
SUBSAMPLE_FRACTION = 0.2  # 20% of data for SVM (adjust to 0.1 if still slow)

def train_and_save_models():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Raw dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please place your CSV file there.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Filter out rows with NaN values or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Cleaned dataset shape (after removing NaNs/infs): {df.shape}")

    # Ensure all FEATURES are present
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in dataset: {missing_features}")
        print("Available columns in CSV:")
        for col in df.columns:
            print(f"  - '{col}'")
        print("Please ensure your dataset has all the required columns.")
        return

    # Map labels to numerical values (0 for BENIGN, 1 for DDoS)
    df[TARGET] = df[TARGET].apply(lambda x: 0 if x == 'BENIGN' else 1)

    X = df[FEATURES]
    y = df[TARGET]

    print(f"Final dataset shape: {df.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Split data (full split for RF, will subsample later for SVM)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Full train/test split: Train {X_train_full.shape}, Test {X_test_full.shape}")

    # Scale features (fit on full train data)
    print("Training StandardScaler on full train data...")
    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_test_scaled_full = scaler.transform(X_test_full)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Scaler saved to model/scaler.pkl")

    # Train Random Forest (on full scaled data - fast)
    print("Training Random Forest model on full data...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled_full, y_train_full)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    print("Random Forest model saved to model/random_forest_model.pkl")

    # Evaluate Random Forest on full test data
    rf_pred_full = rf_model.predict(X_test_scaled_full)
    print("\nRandom Forest Metrics (on full test data):")
    print(f"Accuracy: {accuracy_score(y_test_full, rf_pred_full):.4f}")
    print(f"Precision: {precision_score(y_test_full, rf_pred_full, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_full, rf_pred_full, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test_full, rf_pred_full, zero_division=0):.4f}")

    # Subsample for SVM (stratified to maintain class balance)
    print(f"\nSubsampling {SUBSAMPLE_FRACTION*100}% of data for SVM training (to prevent hanging on large dataset)...")
    n_train_samples = int(len(X_train_full) * SUBSAMPLE_FRACTION)
    n_test_samples = int(len(X_test_full) * SUBSAMPLE_FRACTION)
    X_train_svm, y_train_svm = resample(X_train_full, y_train_full, stratify=y_train_full,
                                        n_samples=n_train_samples, random_state=42)
    X_test_svm, y_test_svm = resample(X_test_full, y_test_full, stratify=y_test_full,
                                      n_samples=n_test_samples, random_state=42)
    print(f"SVM subsample: Train {X_train_svm.shape}, Test {X_test_svm.shape}")

    # Scale subsampled data (using the same scaler fitted on full data)
    X_train_scaled_svm = scaler.transform(X_train_svm)
    X_test_scaled_svm = scaler.transform(X_test_svm)

    # Train SVM (optimized for speed: linear kernel, lower C - FIXED: removed n_jobs)
    print("Training SVM model on subsampled data (linear kernel for speed)...")
    svm_model = SVC(kernel='linear', random_state=42, C=0.1)  # FIXED: No n_jobs for SVC
    # Alternative: Even faster with LinearSVC (uncomment below if you want ultra-speed)
    # svm_model = LinearSVC(random_state=42, C=0.1, max_iter=1000)
    print("  - Fitting SVM... (this may take 1-3 minutes)")
    svm_model.fit(X_train_scaled_svm, y_train_svm)
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    print("SVM model saved to model/svm_model.pkl")

    # Evaluate SVM on subsampled test data
    svm_pred = svm_model.predict(X_test_scaled_svm)
    print("\nSVM Metrics (on subsampled test data):")
    print(f"Accuracy: {accuracy_score(y_test_svm, svm_pred):.4f}")
    print(f"Precision: {precision_score(y_test_svm, svm_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_svm, svm_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test_svm, svm_pred, zero_division=0):.4f}")

    # Combined Model (Majority Voting: on subsampled data for consistency)
    # Get RF predictions on the same subsampled test data
    rf_pred_svm = rf_model.predict(X_test_scaled_svm)
    combined_pred = (rf_pred_svm + svm_pred) >= 1  # 1 if either predicts DDoS
    print("\nCombined Model (Majority Voting) Metrics (on subsampled test data):")
    print(f"Accuracy: {accuracy_score(y_test_svm, combined_pred):.4f}")
    print(f"Precision: {precision_score(y_test_svm, combined_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_svm, combined_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test_svm, combined_pred, zero_division=0):.4f}")

    print("\nAll models and scaler trained and saved successfully!")
    print("You can now run app.py to use the models for predictions.")

if __name__ == '__main__':
    train_and_save_models()