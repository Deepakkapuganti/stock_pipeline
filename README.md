# Stock Market Pipeline — End-to-End Data Engineering & ML Project

## Project Overview
Understanding stock market behavior and predicting price movements is a major challenge 
in financial data engineering. This project builds a complete end-to-end data pipeline 
for stock market data, combining data ingestion, feature engineering, machine learning 
predictions, and interactive visualizations.

The dataset contains historical and live stock price data including OHLCV (Open, High, 
Low, Close, Volume) metrics. The project aims to identify trading signals, analyze 
market trends, and predict buy/sell signals using machine learning.

## Objectives
* Build an automated data pipeline to ingest live and historical stock market data.
* Engineer meaningful financial features such as Moving Averages, RSI, and Volatility.
* Develop a machine learning model to predict buy/sell signals based on technical indicators.
* Visualize market trends, model performance, and trading signals interactively.
* Store and manage processed data efficiently using AWS and Apache Airflow.

## Dataset Description
* Domain: Finance / Stock Market
* Type: Time Series / Multivariate
* Associated Tasks: Classification, Feature Engineering, Visualization
* Data Sources: Live quotes & historical OHLCV data
* Features: 
   * Raw: Open, High, Low, Close, Volume, Symbol, Date
   * Engineered: MA_7, MA_30, Daily Return, Volatility_7, RSI, MA Signal

## Feature Engineering
| Feature | Description |
|---|---|
| MA_7 | 7-day Moving Average |
| MA_30 | 30-day Moving Average |
| Daily_Return | Day-over-day percentage return |
| Volatility_7 | 7-day rolling standard deviation |
| RSI | Relative Strength Index (Overbought/Oversold) |
| MA_Signal | Buy/Sell signal based on MA crossover |
| Predicted_Signal | ML model predicted Buy/Sell signal |

## Machine Learning Workflow
1. **Data Ingestion**
   * Live stock quotes via API
   * Historical OHLCV data collection
   
2. **Data Preprocessing (PySpark)**
   * Handling missing values
   * Normalization and scaling
   * Feature engineering at scale

3. **Exploratory Data Analysis (EDA)**
   * Price trend analysis
   * Volume distribution
   * Moving average crossover signals
   * RSI overbought/oversold zones

4. **Model Training**
   * Random Forest Classifier
   * Feature importance analysis
   * SHAP value interpretation

5. **Pipeline Orchestration (Airflow)**
   * Automated daily DAG execution
   * End-to-end pipeline scheduling

## Results & Observations
* Achieved strong classification performance for Buy/Sell signal prediction.
* Identified key technical indicators driving price movement signals.
* Clear patterns observed in:
   * MA crossover signals across different stocks
   * RSI overbought/oversold zones
   * Volatility clustering during market events
* Results suggest memantine Moving Average crossover combined with RSI 
  provides reliable trading signals.

## Tableau Dashboards
| Dashboard | Description | Link |
|---|---|---|
| Market Overview | KPI cards, price trends, volume analysis | [View →](https://public.tableau.com/views/stocks_17728711691440/MarketOverview?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) |
| Technical Analysis | RSI, Moving Averages, Signal Timeline | [View →](https://public.tableau.com/views/stocks_17728711691440/TechnicalAnalysis?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) |
| ML Predictions | Model accuracy, predicted signals, heatmap | [View →](https://public.tableau.com/views/stocks_17728711691440/MLpredicitions?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) |

## Biological Financial Significance
* Provides insight into stock price momentum and trend detection.
* Highlights the power of technical indicators in signal generation.
* Demonstrates scalable data engineering using PySpark and Airflow.
* Supports data-driven evaluation of trading strategies.

## Project Structure
```
stock_market_pipeline/
├── dags/
│   └── stock_pipeline_DAG.py       # Airflow DAG
├── dashboards/
│   └── stockmarket_dashboard.twbx  # Tableau Workbook
├── data/
│   ├── viz/
│   │   ├── stock_features.csv      # Processed features
│   │   ├── monthly_summary.csv     # Monthly aggregations
│   │   └── predictions.csv         # ML predictions
│   └── reports/
│       ├── confusion_matrix.png    # Model evaluation
│       ├── feature_importance.png  # Feature analysis
│       └── shap_summary.png        # SHAP explainability
└── src/
    ├── ingestion.py                # Data ingestion
    ├── pyspark_preprocessing.py    # Feature engineering
    ├── ml_model.py                 # ML model training
    └── aws_storage.py              # AWS S3 storage
```

## Technologies Used
* **Orchestration:** Apache Airflow
* **Processing:** PySpark
* **Storage:** AWS S3
* **Visualization:** Tableau
* **Programming Language:** Python
* **Libraries:**
   * PySpark
   * Pandas
   * Scikit-learn
   * Matplotlib
   * Boto3 (AWS SDK
