from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, stddev, lag, when,
    round as spark_round,
    to_date, year, month,
    max as spark_max,
    min as spark_min,
    count
)
from pyspark.sql.window import Window
import os

# ── Absolute paths — works everywhere ─────────────
BASE  = '/home/deepak/stock_pipeline'
RAW   = f'{BASE}/data/raw'
PROC  = f'{BASE}/data/processed'
VIZ   = f'{BASE}/data/viz'


def create_spark_session():
    spark = SparkSession.builder \
        .appName("StockMarketPipeline") \
        .master("local[*]") \
        .config(
            "spark.sql.shuffle.partitions", "4"
        ) \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark Session started!")
    return spark


def load_data(spark):
    print("\n📥 Loading historical data...")

    df = spark.read.csv(
        f'{RAW}/historical_data.csv',
        header=True,
        inferSchema=True
    )

    # Safety: keep only 7 required columns
    df = df.select(
        'Date', 'Open', 'High',
        'Low', 'Close', 'Volume', 'Symbol'
    )

    df = df.withColumn(
        "Date", to_date(col("Date"))
    )
    for c in ["Open", "High", "Low",
              "Close", "Volume"]:
        df = df.withColumn(
            c, col(c).cast("double")
        )
    df = df.dropna()

    print(f"   Records:  {df.count():,}")
    print(f"   Stocks:   "
          f"{df.select('Symbol').distinct().count()}")
    print(f"   Columns:  {df.columns}")
    df.show(3)
    return df


def add_features(df):
    print("\n Adding features...")

    w   = Window.partitionBy("Symbol") \
                .orderBy("Date")
    w7  = w.rowsBetween(-6, 0)
    w14 = w.rowsBetween(-13, 0)
    w30 = w.rowsBetween(-29, 0)

    # Moving averages
    df = df.withColumn(
        "MA_7",
        spark_round(avg("Close").over(w7), 2)
    )
    df = df.withColumn(
        "MA_14",
        spark_round(avg("Close").over(w14), 2)
    )
    df = df.withColumn(
        "MA_30",
        spark_round(avg("Close").over(w30), 2)
    )
    print("   Moving averages (MA7, MA14, MA30)")

    # Prev close for calculations
    df = df.withColumn(
        "Prev_Close", lag("Close", 1).over(w)
    )

    # Daily return
    df = df.withColumn(
        "Daily_Return",
        spark_round(
            ((col("Close") - col("Prev_Close"))
             / col("Prev_Close")) * 100, 2
        )
    )
    print("  Daily returns")

    # Volatility
    df = df.withColumn(
        "Volatility_7",
        spark_round(stddev("Close").over(w7), 2)
    )
    print("   Volatility")

    # Price range
    df = df.withColumn(
        "Price_Range",
        spark_round(col("High") - col("Low"), 2)
    )
    print("   Price range")

    # RSI
    df = df.withColumn(
        "Price_Change",
        col("Close") - col("Prev_Close")
    )
    df = df.withColumn(
        "Gain",
        when(col("Price_Change") > 0,
             col("Price_Change")).otherwise(0)
    )
    df = df.withColumn(
        "Loss",
        when(col("Price_Change") < 0,
             -col("Price_Change")).otherwise(0)
    )
    df = df.withColumn(
        "Avg_Gain", avg("Gain").over(w14)
    )
    df = df.withColumn(
        "Avg_Loss", avg("Loss").over(w14)
    )
    df = df.withColumn(
        "RSI",
        spark_round(
            100 - (
                100 / (
                    1 + (
                        col("Avg_Gain") /
                        (col("Avg_Loss") + 0.0001)
                    )
                )
            ), 2
        )
    )
    print("   RSI (14-day)")

    # MA Signal — 3 classes!
    df = df.withColumn(
        "MA_Signal",
        when(col("MA_7") > col("MA_30"),
             "Bullish")
        .when(col("MA_7") < col("MA_30"),
              "Bearish")
        .otherwise("Neutral")
    )
    print("  MA Signal (Bullish/Bearish/Neutral)")

    # Year + Month
    df = df.withColumn("Year",  year("Date"))
    df = df.withColumn("Month", month("Date"))

    # Drop helper columns
    df = df.drop(
        "Prev_Close", "Price_Change",
        "Gain", "Loss", "Avg_Gain", "Avg_Loss"
    )
    df = df.dropna()

    print(f"\n  ✅ Final columns: {df.columns}")
    print(f"  ✅ Final records: {df.count():,}")
    return df


def monthly_aggregation(df):
    print("\n Monthly aggregations...")

    monthly = df.groupBy(
        "Symbol", "Year", "Month"
    ).agg(
        spark_round(avg("Close"), 2)
            .alias("Avg_Close"),
        spark_round(spark_max("High"), 2)
            .alias("Month_High"),
        spark_round(spark_min("Low"), 2)
            .alias("Month_Low"),
        spark_round(avg("Volume"), 0)
            .alias("Avg_Volume"),
        spark_round(avg("Daily_Return"), 2)
            .alias("Avg_Return"),
        spark_round(avg("Volatility_7"), 2)
            .alias("Avg_Volatility"),
        spark_round(avg("RSI"), 2)
            .alias("Avg_RSI"),
        count("*").alias("Trading_Days")
    ).orderBy("Symbol", "Year", "Month")

    print(f" Monthly records: {monthly.count():,}")
    monthly.show(5)
    return monthly


def save_data(df, monthly):
    print("\nSaving processed data...")

    os.makedirs(PROC, exist_ok=True)
    os.makedirs(VIZ,  exist_ok=True)

    # Parquet — Big Data format
    df.write.mode("overwrite") \
      .partitionBy("Symbol") \
      .parquet(f'{PROC}/stock_features/')
    print("  Parquet saved!")

    # CSVs for Tableau
    monthly.toPandas().to_csv(
        f'{VIZ}/monthly_summary.csv',
        index=False
    )
    print(" Monthly summary saved!")

    df.toPandas().to_csv(
        f'{VIZ}/stock_features.csv',
        index=False
    )
    print(" Full features saved!")


def print_summary(df):
    print("\n Summary by Stock:")
    print("=" * 60)
    df.groupBy("Symbol").agg(
        spark_round(avg("Close"), 2)
            .alias("Avg_Price"),
        spark_round(avg("Daily_Return"), 2)
            .alias("Avg_Return%"),
        spark_round(avg("RSI"), 2)
            .alias("Avg_RSI"),
        spark_round(avg("Volatility_7"), 2)
            .alias("Avg_Volatility"),
        count("*").alias("Total_Days")
    ).orderBy("Symbol").show(truncate=False)


# ── Main ───────────────────────────────────────────
if __name__ == "__main__":
    print(" PySpark Processing Started!")
    print("=" * 50)

    spark   = create_spark_session()
    df      = load_data(spark)
    df      = add_features(df)
    monthly = monthly_aggregation(df)
    save_data(df, monthly)
    print_summary(df)
    spark.stop()

    print("\n" + "=" * 50)
  
    print(f"  {PROC}/stock_features/ ← Parquet")
    print(f"  {VIZ}/stock_features.csv ← Tableau")
    print(f"  {VIZ}/monthly_summary.csv ← Tableau")