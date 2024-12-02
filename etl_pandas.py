# Importing libs
import pandas as pd

df = pd.read_csv("data/measurements10K.txt",
                 sep=";",
                 header=None,
                 names=['stations_name', 'measure']
                 )

df_agg = df.groupby('stations_name')
df_kpi = df_agg['measure'].agg(
    minimum = 'min',
    maximum = 'max',
    mean    = 'mean'
)

print(df_kpi)