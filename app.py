from flask import Flask, request, render_template, flash, redirect, url_for
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_this_in_production'  # Required for flash messages
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for large CSVs

# Ensure static folder exists for charts
os.makedirs('static', exist_ok=True)

# Load pre-trained models and scaler (run model.py first!)
try:
    rf_model = joblib.load('model/random_forest_model.pkl')
    try:
        svm_model = joblib.load('model/svm_model.pkl')
    except FileNotFoundError:
        svm_model = None  # Handle if SVM was skipped
    scaler = joblib.load('model/scaler.pkl')
    print("Models loaded successfully!")
    print(f"SVM model loaded: {svm_model is not None}")
    if svm_model is not None:
        print("Note: SVM was trained on a 20% subsample for efficiency.")
except FileNotFoundError:
    print("Error: Run model.py first to train and save models!")
    rf_model = svm_model = scaler = None  # Set to None to avoid crashes
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    rf_model = svm_model = scaler = None

# Feature columns (must match training - NO leading spaces, corrected typo)
# Note: If your CSV has only 51 features, trim this list to match exactly (from model.py inspection)
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


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('index'))

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Check if models are loaded
            if rf_model is None or scaler is None:
                raise ValueError("Models not loaded. Please run model.py first to train and save them.")

            # Load and prepare data
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()  # Clean column names

            # Check for required features
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                raise ValueError(
                    f"Uploaded CSV is missing required feature columns: {', '.join(missing_features[:10])}... (and more). Expected {len(FEATURES)} features. Found {len(df.columns) - 1 if 'Label' in df.columns else len(df.columns)} columns.")

            X = df[FEATURES].fillna(0).values  # Fill NaNs with 0 for safety

            has_labels = 'Label' in df.columns
            if has_labels:
                # Map labels to numerical values (0 for BENIGN, 1 for DDoS)
                df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
                y_true = df['Label'].values.astype(int)
                summary = f"Dataset shape: {df.shape}\nFeatures: {len(FEATURES)}\nClasses: {np.bincount(y_true)}"
                print("Labeled data detected - real metrics will be computed.")
            else:
                y_true = None
                summary = f"Dataset shape: {df.shape}\nFeatures: {len(FEATURES)}\n(No 'Label' column for metrics - predictions only)"
                print("No labels - metrics will be N/A; charts based on predictions.")

            # Scale data
            X_scaled = scaler.transform(X)

            # Predictions
            rf_pred = rf_model.predict(X_scaled)
            if svm_model is not None:
                svm_pred = svm_model.predict(X_scaled)
                combined_pred = (rf_pred + svm_pred) >= 1  # Majority voting
            else:
                svm_pred = rf_pred  # Fallback to RF
                combined_pred = rf_pred  # No combined, just RF

            # Prediction summary (ENHANCED: Per-model DDoS counts)
            total_samples = len(combined_pred)
            rf_ddos = np.sum(rf_pred)
            svm_ddos = np.sum(svm_pred)
            num_ddos = np.sum(combined_pred)  # Combined
            ddos_pct = (num_ddos / total_samples * 100) if total_samples > 0 else 0
            benign_count = total_samples - num_ddos  # BENIGN count from combined predictions
            print(
                f"Predictions computed: RF DDoS={rf_ddos}, SVM DDoS={svm_ddos}, Combined DDoS={num_ddos}, BENIGN={benign_count}")

            # Metrics if labels present
            if has_labels:
                rf_acc = accuracy_score(y_true, rf_pred)
                rf_prec = precision_score(y_true, rf_pred, zero_division=0)
                rf_rec = recall_score(y_true, rf_pred, zero_division=0)
                rf_f1 = f1_score(y_true, rf_pred, zero_division=0)

                if svm_model is not None:
                    svm_acc = accuracy_score(y_true, svm_pred)
                    svm_prec = precision_score(y_true, svm_pred, zero_division=0)
                    svm_rec = recall_score(y_true, svm_pred, zero_division=0)
                    svm_f1 = f1_score(y_true, svm_pred, zero_division=0)

                    comb_acc = accuracy_score(y_true, combined_pred)
                    comb_prec = precision_score(y_true, combined_pred, zero_division=0)
                    comb_rec = recall_score(y_true, combined_pred, zero_division=0)
                    comb_f1 = f1_score(y_true, combined_pred, zero_division=0)
                else:
                    # Fallback: Use RF metrics for SVM and combined
                    svm_acc = rf_acc
                    svm_prec = rf_prec
                    svm_rec = rf_rec
                    svm_f1 = rf_f1
                    comb_acc = rf_acc
                    comb_prec = rf_prec
                    comb_rec = rf_rec
                    comb_f1 = rf_f1
            else:
                # Dummy metrics for display
                rf_acc = rf_prec = rf_rec = rf_f1 = "N/A (No labels)"
                svm_acc = svm_prec = svm_rec = svm_f1 = "N/A (No labels)"
                comb_acc = comb_prec = comb_rec = comb_f1 = "N/A (No labels)"

            # Generate Pie Chart: Prediction Distribution (Combined Model - BENIGN vs DDoS)
            plt.figure(figsize=(8, 6))
            pie_labels = ['BENIGN (0)', 'DDoS (1)']
            pie_sizes = [benign_count, num_ddos]
            pie_colors = ['#ff9999', '#66b3ff']
            plt.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            plt.title('Prediction Distribution (Combined Model)')
            plt.savefig('static/pie_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Pie chart generated successfully (based on combined predictions).")

            # Generate Bar Chart: Prediction Counts (BENIGN vs DDoS from Combined Model)
            # Uses same parameters as pie chart: x-axis = predictions (BENIGN, DDoS), y-axis = count
            plt.figure(figsize=(8, 6))
            bar_labels = ['BENIGN', 'DDoS']
            bar_counts = [benign_count, num_ddos]
            sns.barplot(x=bar_labels, y=bar_counts, palette=['#ff9999', '#66b3ff'])
            plt.title('Prediction Counts')
            plt.ylabel('Count')
            plt.xlabel('Prediction')
            # Dynamic y-limit for better visualization
            max_count = max(bar_counts) if bar_counts else 1
            plt.ylim(0, max_count * 1.1)  # 10% margin above max
            # Add count labels on top of bars
            for i, count in enumerate(bar_counts):
                plt.text(i, count + (max_count * 0.02), str(count), ha='center', va='bottom', fontweight='bold')
            plt.savefig('static/bar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Bar chart generated successfully (BENIGN vs DDoS prediction counts).")

            # Cleanup uploaded file (success case)
            os.remove(filepath)

            # Render results
            return render_template('result.html',
                                   summary=summary,
                                   rf_acc=f"{rf_acc:.4f}" if isinstance(rf_acc, (int, float)) else rf_acc,
                                   rf_prec=f"{rf_prec:.4f}" if isinstance(rf_prec, (int, float)) else rf_prec,
                                   rf_rec=f"{rf_rec:.4f}" if isinstance(rf_rec, (int, float)) else rf_rec,
                                   rf_f1=f"{rf_f1:.4f}" if isinstance(rf_f1, (int, float)) else rf_f1,
                                   svm_acc=f"{svm_acc:.4f}" if isinstance(svm_acc, (int, float)) else svm_acc,
                                   svm_prec=f"{svm_prec:.4f}" if isinstance(svm_prec, (int, float)) else svm_prec,
                                   svm_rec=f"{svm_rec:.4f}" if isinstance(svm_rec, (int, float)) else svm_rec,
                                   svm_f1=f"{svm_f1:.4f}" if isinstance(svm_f1, (int, float)) else svm_f1,
                                   comb_acc=f"{comb_acc:.4f}" if isinstance(comb_acc, (int, float)) else comb_acc,
                                   comb_prec=f"{comb_prec:.4f}" if isinstance(comb_prec, (int, float)) else comb_prec,
                                   comb_rec=f"{comb_rec:.4f}" if isinstance(comb_rec, (int, float)) else comb_rec,
                                   comb_f1=f"{comb_f1:.4f}" if isinstance(comb_f1, (int, float)) else comb_f1,
                                   rf_ddos=rf_ddos,
                                   svm_ddos=svm_ddos,
                                   num_ddos=num_ddos,
                                   total_samples=total_samples,
                                   ddos_pct=f"{ddos_pct:.2f}",
                                   svm_available=(svm_model is not None),
                                   has_labels=has_labels,
                                   benign_count=benign_count)  # Pass for template if needed

        except Exception as e:
            # Error handling: Cleanup and return message
            if os.path.exists(filepath):
                os.remove(filepath)
            error_msg = f"Error processing file: {str(e)}. Ensure CSV has the exact features and optional 'Label' column."
            flash(error_msg, 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)