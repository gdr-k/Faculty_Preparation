import csv

with open("results.csv", "r", newline="\n") as file:
    reader = csv.DictReader(file)

    for row in reader:
        print(row)

import pandas as pd

df = pd.read_csv("results.csv")

print(df.head())