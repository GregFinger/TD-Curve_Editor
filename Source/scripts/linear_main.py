import numpy as np

posData = op('positions').cols()
resolution = 0.01


x = [float(x.val) for x in posData[0]]
y = [float(y.val) for y in posData[1]]
base = np.linspace(0,1,(1/resolution))

x = np.asarray(x)
y = np.asarray(y)

samples = np.interp(base,x,y)
table = op('equispaced_x')
table.clear(keepFirstRow = True)
table.appendRows(samples)
