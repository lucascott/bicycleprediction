


import pandas as pd
import io
import requests
import datetime as dt

df = pd.DataFrame()
d = dt.date(2017,5,29)
delta = dt.timedelta(days=1)
while d <= dt.date(2017, 11, 16):
	url = "http://www.airnowapi.org/aq/observation/latLong/historical/?format=text/csv&latitude=40.7657&longitude=-73.9614&date=%sT00-0000&distance=25&API_KEY=8A3E850A-51C6-433A-A1C9-C11FACC13529" % (d.strftime("%Y-%m-%d"))

	s=requests.get(url).content
	c=pd.read_csv(io.StringIO(s.decode('utf-8')))

	df = pd.concat([df,c])
	print (df.tail())
	print (d.strftime("%Y-%m-%d"))
	d += delta
	df.to_csv("out3.csv", encoding='utf-8')