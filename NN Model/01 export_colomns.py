# esporta nella cartella parsed le colonne selezionate dal file deciso
# applica inoltre delle modifiche al dataset su alcuni campi


import pandas
import time
import datetime
import numpy as np

filename = "1359418.csv"

#colonne selezionate dal csv
fields = ["hourlyprsentweathertype"]

for i in range(len(fields)):
	fields[i] = fields[i].strip().lower().replace('/', '').replace(' ', '_')

df = pandas.read_csv(filename)
df.columns = df.columns.str.strip().str.lower().str.replace('/', '')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print(df.columns)

print(df[fields].head(20))
df[fields].to_csv("1-"+filename)

print("Exported: " + filename)