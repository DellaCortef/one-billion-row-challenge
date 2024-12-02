# Importing libs
import time
import duckdb

def create_duckdb():
    duckdb.sql("""SELECT
    stations,
    min(measure) as min_measure,
    avg(measure) as mean_measure,
    max(measure) as max_measure
FROM read_csv(
    'data/measurements1M.txt',
    AUTO_DETECT=FALSE,
    sep=';',
    columns={'stations': 'VARCHAR', 'measure': 'DECIMAL(3,1)'}
)
GROUP BY stations
ORDER BY stations
""").show()

if __name__ == "__main__":
    start_time = time.time()
    create_duckdb()
    took = time.time() - start_time
    print(f"Duckdb Took: {took:.2f} seconds!")