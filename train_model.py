import joblib
from preprocess import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

def evaluate_model(y_true, y_pred, y_probs, model_name):
    print(f"\n================ {model_name} Evaluation ================")
    print(classification_report(y_true, y_pred))
    
    # Calculate Precision-Recall AUC (The gold standard for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"   Predicted Legit   Predicted Fraud")
    print(f"Actual Legit:  {cm[0][0]}             {cm[0][1]}")
    print(f"Actual Fraud:  {cm[1][0]}             {cm[1][1]}")

def train_and_save_models():
    # Load processed data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # --- Model 1: Logistic Regression Baseline ---
    print("\n🤖 Training Logistic Regression Baseline...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, lr_preds, lr_probs, "Logistic Regression")
    
    # --- Model 2: XGBoost Heavy Lifter ---
    print("\n⚡ Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, xgb_preds, xgb_probs, "XGBoost Classifier")
    
    # Save the best model and scaler for deployment
    print("\n💾 Saving models to disk...")
    joblib.dump(xgb_model, 'models/fraud_xgb_model.pkl')
    joblib.dump(scaler, 'models/data_scaler.pkl')
    print("✅ Models saved successfully in the 'models/' directory!")

if __name__ == "__main__":
    train_and_save_models()