# tools_benchmark.py

import os
import time
import duckdb
import pandas as pd
from pathlib import Path

# CSV file to save benchmark results
RESULTS_FILE = "benchmark_results.csv"

def benchmark_tool(tool_name, dataset_size, processing_function, dataset_path):
    """
    Benchmark a tool's processing time and save the result.

    Parameters:
        tool_name (str): The name of the tool/framework (e.g., Pandas, DuckDB).
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


# Example processing functions for tools

def process_with_pandas(file_path):
    """
    Example processing function using Pandas.
    Reads the dataset and calculates statistics.
    """

    df = pd.read_csv(file_path, sep=";")
    # Example operation: group by a column and calculate statistics
    stats = df.groupby("station_city_name")["temperature"].agg(["min", "mean", "max"])
    return stats


def process_with_duckdb(file_path):
    """
    Example processing function using DuckDB.
    Reads the dataset and calculates statistics.
    """

    # Read the dataset and execute a query
    conn = duckdb.connect()
    query = f"""
        SELECT station_city_name, MIN(temperature) AS min_temp, AVG(temperature) AS mean_temp, MAX(temperature) AS max_temp
        FROM read_csv_auto('{file_path}', delim=';')
        GROUP BY station_city_name
    """
    result = conn.execute(query).fetchdf()
    return result


if __name__ == "__main__":
    # Paths to datasets
    datasets = {
        "1M": Path("data/measurements1M.txt"),
        "10M": Path("data/measurements10M.txt"),
        "100M": Path("data/measurements100M.txt"),
    }

    # Benchmark each tool with each dataset
    for dataset_size, dataset_path in datasets.items():
        print(f"Benchmarking Pandas on {dataset_size} dataset...")
        benchmark_tool("Pandas", dataset_size, process_with_pandas, dataset_path)

        print(f"Benchmarking DuckDB on {dataset_size} dataset...")
        benchmark_tool("DuckDB", dataset_size, process_with_duckdb, dataset_path)

    print(f"Benchmark results saved to {RESULTS_FILE}")
