import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    """
    Trains and compares Baseline, Logistic Regression, Decision Tree, and Random Forest
    on an expanded multi-stock dataset using a strict Global Chronological Split.
    Saves metrics, confusion matrices, ROC curves, feature importances, and metadata.
    """
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load processed features
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Strict Global Chronological Date Cutoff (Option A)
    # Ensures zero lookahead leakage across all stocks simultaneously
    unique_dates = np.sort(df['Date'].unique())
    split_date_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_date_idx]
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    features_to_drop = ['Date', 'Ticker', 'Target']
    feature_columns = [col for col in df.columns if col not in features_to_drop]
    
    X_train = train_df[feature_columns]
    y_train = train_df['Target']
    
    X_test = test_df[feature_columns]
    y_test = test_df['Target']
    
    train_start = train_df['Date'].min().strftime('%Y-%m-%d')
    train_end = train_df['Date'].max().strftime('%Y-%m-%d')
    test_start = test_df['Date'].min().strftime('%Y-%m-%d')
    test_end = test_df['Date'].max().strftime('%Y-%m-%d')
    tickers_list = sorted(df['Ticker'].unique().tolist())
    
    print("==================================================")
    print("CHRONOLOGICAL TRAIN/TEST SPLIT SUMMARY")
    print("==================================================")
    print(f"Total Unique Tickers: {len(tickers_list)}")
    print(f"Global Cutoff Date: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")
    print(f"Training Period: {train_start} to {train_end} ({len(X_train):,} rows)")
    print(f"Testing Period: {test_start} to {test_end} ({len(X_test):,} rows)")
    print(f"Target Class Balance (Train): UP = {y_train.mean():.1%}, DOWN = {1 - y_train.mean():.1%}")
    print(f"Target Class Balance (Test):  UP = {y_test.mean():.1%}, DOWN = {1 - y_test.mean():.1%}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print("==================================================\n")
    
    # 3. Model Definitions
    models = {
        'Baseline (Majority Class)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=-1))
        ])
    }
    
    results = {}
    roc_curves_data = {}
    
    plt.figure(figsize=(8, 6))
    
    for name, pipeline in models.items():
        print(f"Training {name}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        print("Done. Evaluating...", flush=True)
        
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = (y_pred == 1).astype(float)
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': roc_auc,
            'Confusion_Matrix': cm
        }
        
        # Save individual Confusion Matrix Plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['DOWN (0)', 'UP (1)'],
                    yticklabels=['DOWN (0)', 'UP (1)'])
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(f'plots/cm_{clean_name}.png')
        plt.close()
        
        # Compute ROC curve for multi-model plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
    # Finalize Multi-Model ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png')
    plt.close()
    
    # 4. Feature Importance Extraction
    feature_importance_dict = {}
    rf_classifier = models['Random Forest'].named_steps['classifier']
    dt_classifier = models['Decision Tree'].named_steps['classifier']
    lr_classifier = models['Logistic Regression'].named_steps['classifier']
    
    feature_importance_dict['Random_Forest'] = dict(sorted(zip(feature_columns, [float(v) for v in rf_classifier.feature_importances_]), key=lambda x: x[1], reverse=True))
    feature_importance_dict['Decision_Tree'] = dict(sorted(zip(feature_columns, [float(v) for v in dt_classifier.feature_importances_]), key=lambda x: x[1], reverse=True))
    feature_importance_dict['Logistic_Regression_Coefficients'] = dict(sorted(zip(feature_columns, [float(v) for v in lr_classifier.coef_[0]]), key=lambda x: abs(x[1]), reverse=True))
    
    with open('results/feature_importance.json', 'w') as f:
        json.dump(feature_importance_dict, f, indent=4)
        
    # Plot top features for Random Forest
    rf_top = pd.Series(feature_importance_dict['Random_Forest']).head(12)
    plt.figure(figsize=(10, 5))
    rf_top.plot(kind='barh', color='#2563EB')
    plt.gca().invert_yaxis()
    plt.title('Top Feature Importances (Random Forest)')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 5. Model Selection Criterion
    # Primary selection based on Test ROC-AUC (measuring true discrimination ability above 0.500 baseline)
    candidate_models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    best_model_name = max(candidate_models, key=lambda m: results[m]['ROC-AUC'])
    best_pipeline = models[best_model_name]
    
    print("\n==================================================")
    print("FINAL EVALUATION METRICS (TEST SET)")
    print("==================================================")
    results_df = pd.DataFrame(results).T
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']])
    print(f"\nSelected Model: {best_model_name} (Highest ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f})")
    print("==================================================\n")
    
    # 6. Save Results and Metadata
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    results_df.to_csv('results/metrics.csv')
    
    # Save best model and features
    joblib.dump(best_pipeline, 'models/best_model.pkl')
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f, indent=4)
        
    metadata = {
        "project_name": "NEXTICK",
        "description": "Machine Learning-Based Next-Day Stock Price Movement Prediction",
        "stock_universe": tickers_list,
        "num_stocks": len(tickers_list),
        "total_processed_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_date_range": [train_start, train_end],
        "test_date_range": [test_start, test_end],
        "target_definition": "1 if Next_Close > Close else 0",
        "feature_columns": feature_columns,
        "selected_model": best_model_name,
        "metrics": results,
        "feature_importance": feature_importance_dict
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    with open('results/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training and evaluation completed successfully. Artifacts saved to models/, results/, and plots/.")
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
