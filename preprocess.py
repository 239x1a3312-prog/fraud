import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath="dataset/creditcard.csv", use_smote=True):
    print("⏳ Loading dataset...")
    df = pd.read_csv(filepath)
    
    # 1. Scale 'Amount' and 'Time' to match the scale of V1-V28
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original unscaled columns
    df = df.drop(['Time', 'Amount'], axis=1)
    
    # 2. Separate Features (X) and Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 3. Split into Train and Test sets (Stratify ensures fraud ratio is kept equal in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Initial split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Original Train Class Distribution: Fraud={sum(y_train==1)}, Legitimate={sum(y_train==0)}")
    
    # 4. Handle Class Imbalance using SMOTE (ONLY on training data!)
    if use_smote:
        print("⚖️ Applying SMOTE to balancing training data...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Balanced Train Class Distribution: Fraud={sum(y_train_res==1)}, Legitimate={sum(y_train_res==0)}")
        return X_train_res, X_test, y_train_res, y_test, scaler
        
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test execution
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()