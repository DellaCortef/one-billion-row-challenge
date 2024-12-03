# tools_benchmark.py

import os
import time
import duckdb
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pyspark.sql import SparkSession


# CSV file to save benchmark results
RESULTS_FILE = "benchmark_results.csv"

def benchmark_tool(tool_name, dataset_size, processing_function, dataset_path):
    """
    Benchmark a tool's processing time and save the result.

    Parameters:
        tool_name (str): The name of the tool/framework (e.g., Pandas, DuckDB, Python, Spark).
        dataset_size (str): The dataset size (e.g., '1M', '10M', '100M').
        processing_function (callable): The function that processes the dataset.
        dataset_path (Path): Path to the dataset file.
    """
    # Start the timer
    start_time = time.time()

    # Run the processing function
    processing_function(dataset_path)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Append the results to the CSV file
    results = {
        "Tool": tool_name,
        "Dataset Size": dataset_size,
        "Processing Time (s)": round(elapsed_time, 2),
    }
    results_df = pd.DataFrame([results])

    if Path(RESULTS_FILE).exists():
        results_df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        results_df.to_csv(RESULTS_FILE, index=False)

    print(f"{tool_name} completed on {dataset_size} dataset in {elapsed_time:.2f} seconds.")


def sort_results_by_dataset_size(results_file):
    """
    Reads the benchmark results CSV, sorts by dataset size, and saves it back.
    """
    # Define the order for sorting dataset sizes
    size_order = ["10K", "100K", "1M", "2M", "3M"]
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(results_file)
    
    # Convert the 'Dataset Size' column to a categorical type for sorting
    df["Dataset Size"] = pd.Categorical(df["Dataset Size"], categories=size_order, ordered=True)
    
    # Sort the DataFrame by 'Dataset Size'
    df_sorted = df.sort_values(by="Dataset Size")
    
    # Save the sorted results back to the CSV file
    df_sorted.to_csv(results_file, index=False)
    
    print(f"Benchmark results sorted by dataset size and saved to {results_file}.")


def process_with_pandas(file_path):
    """
    Example processing function using Pandas.
    Reads the dataset and calculates statistics.
    """
    df = pd.read_csv(file_path, sep=";", header=None, names=["station_city_name", "temperature"])
    print(f"Columns in DataFrame: {df.columns}")
    stats = df.groupby("station_city_name")["temperature"].agg(["min", "mean", "max"])
    return stats


def process_with_duckdb(file_path):
    """
    Example processing function using DuckDB.
    Reads the dataset and calculates statistics.
    """
    conn = duckdb.connect()
    query = f"""
        SELECT 
            stations AS station_city_name,
            MIN(measure) AS min_temp,
            AVG(measure) AS mean_temp,
            MAX(measure) AS max_temp
        FROM read_csv_auto('{file_path}', delim=';', header=False, columns={{'stations': 'VARCHAR', 'measure': 'DECIMAL(5,2)'}})
        GROUP BY stations
        ORDER BY stations;
    """
    result = conn.execute(query).fetchdf()
    return result


def process_with_python(file_path):
    """
    Example processing function using plain Python.
    Reads the dataset and calculates statistics.
    """
    temperature_per_station = defaultdict(list)

    # Read the file line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            station, temperature = line.strip().split(";")
            temperature_per_station[station].append(float(temperature))

    # Calculate statistics
    stats = {station: {
        "min": min(temps),
        "mean": sum(temps) / len(temps),
        "max": max(temps)
    } for station, temps in temperature_per_station.items()}
    return stats


def process_with_spark(file_path):
    """
    Example processing function using Apache Spark.
    Reads the dataset and calculates statistics.
    """
    # Initialize Spark session
    spark = SparkSession.builder.appName("Benchmarking").getOrCreate()

    # Read the CSV file into a Spark DataFrame
    df = spark.read.csv(file_path, sep=";", header=False, inferSchema=True)
    df = df.withColumnRenamed("_c0", "station_city_name").withColumnRenamed("_c1", "temperature")

    # Group by and calculate statistics
    stats = df.groupBy("station_city_name").agg(
        {"temperature": "min", "temperature": "avg", "temperature": "max"}
    )
    stats.show()
    spark.stop()
    return stats


if __name__ == "__main__":
    # Paths to datasets
    datasets = {
        "10K":  Path("data/measurements10K.txt"),
        "100K": Path("data/measurements100K.txt"),
        "1M":   Path("data/measurements1M.txt"),
        "2M":   Path("data/measurements2M.txt"),
        "3M":   Path("data/measurements3M.txt"),
    }

    # Benchmark each tool with each dataset
    for dataset_size, dataset_path in datasets.items():
        print(f"Benchmarking Pandas on {dataset_size} dataset...")
        benchmark_tool("Pandas", dataset_size, process_with_pandas, dataset_path)

        print(f"Benchmarking DuckDB on {dataset_size} dataset...")
        benchmark_tool("DuckDB", dataset_size, process_with_duckdb, dataset_path)

        print(f"Benchmarking Python on {dataset_size} dataset...")
        benchmark_tool("Python", dataset_size, process_with_python, dataset_path)

        print(f"Benchmarking Spark on {dataset_size} dataset...")
        benchmark_tool("Spark", dataset_size, process_with_spark, dataset_path)

    print(f"Benchmark results saved to {RESULTS_FILE}")

    # Sort the benchmark results by dataset size
    sort_results_by_dataset_size(RESULTS_FILE)
