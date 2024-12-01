# Importing libs
import os
import time
from csv import reader
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

# Load .env file
load_dotenv()

# Creating processing_temperature() funtion
def processing_temperature(txt_path: Path):
    ''''
    Function that will process the csv file to load temperatures
    '''
    # Access environment variables
    txt_path_100M = os.getenv("TXT_PATH_100M")
    print("Starting processing!")

    # Starting stopwatch
    start_time = time.time() # start time

    temperature_per_station = defaultdict(list)

    with open(txt_path_100M, "r", encoding='utf-8') as file:
        _reader = reader(file, delimiter=';')
        for row in _reader:
            station_city_name, temperature = str(row[0]), float(row[1])

            temperature_per_station[station_city_name].append(temperature)

    print("Data loaded. Calculating statistics...")

    # Dictionary to store calculated results
    results = {}

    # KPI calculation
    for station, temperature in temperature_per_station.items():
        min_temp = min(temperature)
        mean_temp = sum(temperature) / len(temperature)
        max_temp = max(temperature)
        results[station] = (min_temp, mean_temp, max_temp)
    
    print("Calculated statistics. Ordering....")

    # Ordering results by state name
    sorted_results = dict(sorted(results.items()))

    # Format station temperatures into "min_temp/mean_temp/max_temp" (1 decimal place) for each station in sorted_results
    formated_results = {station: f"{min_temp:.1f}/{mean_temp:.1f}/{max_temp:.1f}" for station, (min_temp, mean_temp, max_temp) in sorted_results.items()}

    # Ending stopwatch
    end_time = time.time()
    print(f"Processamento concluído em {end_time - start_time:.2f} segundos!")

    return formated_results

# Replace "data/measurements10M.txt" with the correct path to your file
if __name__ == "__main__":
    txt_path: Path = Path("data/measurements100M.txt")
    
    # 100M > 5 minutes
    results = processing_temperature(txt_path)