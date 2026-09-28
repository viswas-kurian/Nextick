import os
import sys

def run_pipeline():
    print("==================================================")
    print("NEXTICK: Multi-Stock ML Prediction Pipeline")
    print("==================================================")
    
    print("\n[Step 1/4] Downloading / Verifying Stock Dataset (36 Tickers, 10 Years)...")
    res_code = os.system(f"{sys.executable} src/data_loader.py")
    if res_code != 0:
        raise RuntimeError("Data loading failed.")
    
    print("\n[Step 2/4] Feature Engineering & Target Creation...")
    res_code = os.system(f"{sys.executable} src/features.py")
    if res_code != 0:
        raise RuntimeError("Feature engineering failed.")
    
    print("\n[Step 3/4] Model Training, Chronological Evaluation & Artifact Generation...")
    res_code = os.system(f"{sys.executable} src/train.py")
    if res_code != 0:
        raise RuntimeError("Model training failed.")
    
    print("\n[Step 4/4] Running Sample Inferences Across Supported Sectors...")
    res_code = os.system(f"{sys.executable} src/predict.py")
    if res_code != 0:
        raise RuntimeError("Inference test failed.")
    
    print("\n==================================================")
    print("Pipeline Execution Completed Successfully.")
    print("Artifacts updated in 'models/', 'results/', and 'plots/'.")
    print("==================================================")

if __name__ == '__main__':
    run_pipeline()
