import pandas as pd
import sys

f = open(sys.argv[1], 'r')
mega = []
for line in f:
	line =  line.rstrip()
	pcs = line.split(",")
	mega.append(pcs)
print(mega)