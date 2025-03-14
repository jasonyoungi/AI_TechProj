import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
sensor_data = pd.read_csv("C:/Users/Jason/Desktop/sensor_maintenance_data.csv")

# Preprocess data
def preprocess_data(df, target_column, drop_columns=[]):
    df = df.drop(columns=drop_columns, errors='ignore')
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

X_train_sensor, X_test_sensor, y_train_sensor, y_test_sensor, feature_names_sensor = preprocess_data(
    sensor_data, "Predictive Maintenance Trigger", ["Sensor_ID", "Timestamp", "Equipment_ID", "Last Maintenance Date"]
)

# Train models with time tracking
def train_model(model, X_train, y_train, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time for {model_name}: {training_time:.4f} seconds")
    return model, training_time

xgb_sensor, xgb_time = train_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42), X_train_sensor, y_train_sensor, "XGBoost")
rf_sensor, rf_time = train_model(RandomForestClassifier(n_estimators=100, random_state=42), X_train_sensor, y_train_sensor, "Random Forest")
svm_sensor, svm_time = train_model(SVC(kernel="rbf", probability=True, random_state=42), X_train_sensor, y_train_sensor, "SVM")

# Evaluate models
def evaluate_model(model, X_test, y_test, dataset_name, model_name, training_time):
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc_score = auc(fpr, tpr)
    
    print(f"\n=== {dataset_name} Model Evaluation ({model_name}) ===")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {dataset_name} ({model_name})")
    plt.show()
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name} ({model_name})")
    plt.legend()
    plt.show()

evaluate_model(xgb_sensor, X_test_sensor, y_test_sensor, "Sensor Maintenance", "XGBoost", xgb_time)
evaluate_model(rf_sensor, X_test_sensor, y_test_sensor, "Sensor Maintenance", "Random Forest", rf_time)
evaluate_model(svm_sensor, X_test_sensor, y_test_sensor, "Sensor Maintenance", "SVM", svm_time)
