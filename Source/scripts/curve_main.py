# from: https://stackoverflow.com/questions/36644259/cubic-hermit-spline-interpolation-python/48026686#48026686

import numpy as np
import interpolate
import curve_samples as curve

# Input.
posData = op('positions').rows()
tanData = op('tangents').rows()
resolution = op('res')[0]

points = [[float(x[0].val),float(x[1].val)] for x in posData]
tangents = [[float(x[0].val),float(x[1].val)] for x in tanData]
points = np.asarray(points)
tangents = np.asarray(tangents)

# Interpolate with different tangent lengths, but equal direction.
tangents = np.dot(tangents, np.eye(2))
samples = curve.sampleCubicSplinesWithDerivative(points, tangents, resolution)

# equispaced x
x = samples[:,0]
y = samples[:,1]
f = interpolate.interp1d(x, y)
x_new = np.linspace(np.min(x), np.max(x), x.shape[0])
y_new = f(x_new)

table = op('equispaced_x')
table.clear(keepFirstRow = True)
table.appendRows(y_new)

# original
"""
table = op('original')
table.clear(keepFirstRow = True)
table.appendRows(samples)
"""

# equispaced x and y
""" 
pointCount = 20
x,y = samples.T
xd = np.diff(x)
yd = np.diff(y)
dist = np.sqrt(xd**2+yd**2)
u = np.cumsum(dist)
u = np.hstack([[0],u])
t = np.linspace(0,u.max(),pointCount)
xn = np.interp(t, u, x)
yn = np.interp(t, u, y)

table = op('equispaced_xy')
table.clear(keepFirstRow = True)
for i in range(pointCount):
	table.appendRow([xn[i],yn[i]]) 
"""