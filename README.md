# NEXTICK: Machine Learning-Based Next-Day Stock Price Movement Prediction
**MCA Academic Machine Learning Mini-Project**

Nextick is an academic machine learning system designed to analyze historical financial market data, engineer time-series features, and evaluate binary classification models predicting whether a stock's next trading day closing price will move **UP** or **DOWN**.

---

## 1. Project Overview & Problem Formulation
- **Problem Formulation:** Binary classification on daily stock market time-series.
- **Prediction Target:**
  $$\text{Target}_t = \begin{cases} 1 & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0 & \text{if } \text{Close}_{t+1} \le \text{Close}_t \end{cases}$$
- **Prediction Horizon:** Strictly the **Next Trading Day** ($t+1$).
- **Academic Goal:** Demonstrate clean, leakage-free data processing, multi-stock handling, scale-invariant feature engineering, strict chronological evaluation, model comparison, and interactive deployment without artificial performance inflation.

---

## 2. Multi-Stock Dataset & Universe
- **Data Source:** Yahoo Finance API (via `yfinance`).
- **Time Period:** 10+ Years (`2015-01-02` to `2026-09-25`).
- **Universe (36 Large-Cap US Stocks across 5 Sectors):**
  - **Technology (11):** `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `NVDA`, `AVGO`, `ORCL`, `ADBE`, `CSCO`, `INTC`
  - **Financials (6):** `JPM`, `V`, `MA`, `BAC`, `WFC`, `GS`
  - **Healthcare (6):** `JNJ`, `UNH`, `PFE`, `ABBV`, `MRK`, `LLY`
  - **Consumer Goods & Services (7):** `WMT`, `COST`, `KO`, `PEP`, `MCD`, `PG`, `NKE`
  - **Industrial & Energy (6):** `CAT`, `GE`, `XOM`, `CVX`, `BA`, `HON`
- **Total Raw Records:** `106,200`
- **Total Valid Processed Records:** `105,444` (after 20-day rolling window initialization and NaN removal)

---

## 3. Leakage-Free Feature Engineering
Features are strictly derived using information available at prediction time ($t \le T$). Rolling and lag calculations are executed independently per ticker group to prevent cross-ticker contamination.

### Feature Set (26 Features):
1. **Raw Price & Volume:** `Open`, `High`, `Low`, `Close`, `Volume`
2. **Returns & Momentum:** `Daily_Return`, `Lag1_Return`, `Lag2_Return`, `Return_5d`
3. **Trend Indicators:** `SMA_5`, `SMA_10`, `SMA_20`, `EMA_10`, `EMA_20`
4. **Volatility:** `Rolling_Vol_10`, `Rolling_Vol_20` (10-day and 20-day return standard deviation)
5. **Volume Dynamics:** `Volume_Change`, `Volume_MA_5`, `Volume_to_MA5`
6. **Normalized Scale-Invariant Features (Crucial for Multi-Stock Comparison):**
   - `Close_to_SMA5`: `(Close - SMA_5) / SMA_5`
   - `Close_to_SMA10`: `(Close - SMA_10) / SMA_10`
   - `Close_to_SMA20`: `(Close - SMA_20) / SMA_20`
   - `SMA5_to_SMA20`: `(SMA_5 - SMA_20) / SMA_20`
   - `EMA10_to_EMA20`: `(EMA_10 - EMA_20) / EMA_20`
   - `High_Low_Range`: `(High - Low) / Close`
   - `Open_Close_Range`: `(Close - Open) / Open`

---

## 4. Chronological Train/Test Split
To prevent temporal lookahead leakage, a **Global Chronological Date Cutoff** is applied across all 36 stocks simultaneously at the 80% quantile of unique trading dates:
- **Training Set:** `2015-02-02` to `2024-05-22` &rarr; **84,348 observations** (80.0%)
- **Test Set (Out-of-Sample):** `2024-05-23` to `2026-09-24` &rarr; **21,096 observations** (20.0%)
- **Target Distribution:** Train: 52.5% UP / 47.5% DOWN | Test: 52.3% UP / 47.7% DOWN.

---

## 5. Model Evaluation Results (Test Set)

| Model | Accuracy | Precision (UP) | Recall (UP) | F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority Class)** | 0.5226 | 0.5226 | 1.0000 | 0.6864 | 0.5000 |
| **Logistic Regression** | 0.5197 | 0.5228 | 0.9253 | 0.6682 | 0.5007 |
| **Decision Tree** | 0.5193 | 0.5223 | 0.9362 | 0.6706 | 0.4992 |
| **Random Forest (Selected)** | **0.5213** | **0.5238** | **0.9242** | **0.6686** | **0.5012** |

### Academic Discussion on Market Predictability:
On the large 21,096 out-of-sample test set, the models achieve an ROC-AUC of **~0.501**, compared to the **0.500** baseline. This aligns with standard empirical financial literature: daily price direction in highly efficient large-cap equity markets behaves closely to a random walk when analyzed solely through basic historical technical indicators.

---

## 6. Project Architecture
```text
StocksPredictionSystem/
├── app.py                   # Streamlit web application
├── main.py                  # End-to-end ML pipeline runner
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── data/
│   ├── raw/                 # Downloaded raw CSVs (106,200 rows)
│   └── processed/           # Processed features dataset (105,444 rows)
├── models/
│   ├── best_model.pkl       # Serialized Random Forest pipeline
│   ├── feature_columns.json # Feature definitions
│   └── metadata.json        # Machine-readable experiment metadata
├── plots/
│   ├── roc_curves.png       # Multi-model ROC curves
│   ├── feature_importance.png # Random Forest feature rankings
│   └── cm_*.png             # Confusion matrix heatmaps
├── results/
│   ├── metrics.csv          # Evaluation metrics table
│   ├── metrics.json         # Evaluation metrics JSON
│   ├── feature_importance.json # Feature weights
│   └── metadata.json        # Pipeline metadata
└── src/
    ├── __init__.py
    ├── data_loader.py       # Multi-stock historical data ingestion
    ├── features.py          # Scale-invariant feature engineering
    ├── train.py             # Chronological training & evaluation
    └── predict.py           # Live inference module
```

---

## 7. How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Training Pipeline
```bash
python main.py
```
This executes data loading, feature engineering, model training, evaluation, and saves all artifacts.

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Academic Disclaimers & Limitations
1. **Academic Demonstration:** Nextick is an MCA mini-project for academic study and research.
2. **No Investment Advice:** It does not provide trading signals, buy/sell recommendations, or financial advice.
3. **Market Noise:** Macroeconomic shifts, earnings announcements, interest rate decisions, and geopolitical events cannot be captured by simple technical price time-series alone.
4. **Probabilistic Outputs:** Model probabilities reflect classifier leaf distributions from training data, not real-world guarantees.
