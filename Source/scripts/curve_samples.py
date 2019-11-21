# from: https://stackoverflow.com/questions/36644259/cubic-hermit-spline-interpolation-python/48026686#48026686

import numpy as np
import interpolate

def sampleCubicSplinesWithDerivative(points, tangents, resolution):

	'''
	Compute and sample the cubic splines for a set of input points with
	optional information about the tangent (direction AND magnitude). The 
	splines are parametrized along the traverse line (piecewise linear), with
	the resolution being the step size of the parametrization parameter.
	The resulting samples have NOT an equidistant spacing.

	Arguments:      points: a list of n-dimensional points
					tangents: a list of tangents
					resolution: parametrization step size
	Returns:        samples

	Notes: Lists points and tangents must have equal length. In case a tangent
			is not specified for a point, just pass None. For example:
					points = [[0,0], [1,1], [2,0]]
					tangents = [[1,1], None, [1,-1]]
	'''

	resolution = float(resolution)
	points = np.asarray(points)
	nPoints, dim = points.shape

	# Parametrization parameter s.
	dp = np.diff(points, axis=0)                 # difference between points
	dp = np.linalg.norm(dp, axis=1)              # distance between points
	d = np.cumsum(dp)                            # cumsum along the segments
	d = np.hstack([[0],d])                       # add distance from first point
	l = d[-1]                                    # length of point sequence
	nSamples = int(l/resolution)                 # number of samples
	s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step

	# Bring points and (optional) tangent information into correct format.
	assert(len(points) == len(tangents))
	data = np.empty([nPoints, dim], dtype=object)
	for i,p in enumerate(points):
		t = tangents[i]
		# Either tangent is None or has the same
		# number of dimensions as the point p.
		assert(t is None or len(t)==dim)
		fuse = list(zip(p,t) if t is not None else zip(p,))
		data[i,:] = fuse

	# Compute splines per dimension separately.
	samples = np.zeros([nSamples, dim])
	for i in range(dim):
		poly = interpolate.BPoly.from_derivatives(d, data[:,i])
		samples[:,i] = poly(s)
	return samples