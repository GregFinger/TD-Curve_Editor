# from scipy's "special" sub-package

import numpy as np

def binom(n, k):

    if n < 0:
        nx = math.floor(n)
        if n == nx:
            # undefined
            return nan

    kx = math.floor(k)
    if k == kx and (math.fabs(n) > 1e-8 or n == 0):
        # Integer case: use multiplication formula for less rounding error
        # for cases where the result is an integer.
        #
        # This cannot be used for small nonzero n due to loss of
        # precision.

        nx = math.floor(n)
        if nx == n and kx > nx/2 and nx > 0:
            # Reduce kx by symmetry
            kx = nx - kx

        if kx >= 0 and kx < 20:
            num = 1.0
            den = 1.0
            for i in range(1, 1 + int(kx)):
                num *= i + n - kx
                den *= i
                if math.fabs(num) > 1e50:
                    num /= den
                    den = 1.0
            return num/den

    # general case:
    if n >= 1e10*k and k > 0:
        # avoid under/overflows in intermediate results
        return exp(-lbeta(1 + n - k, 1 + k) - log(n + 1))
    elif k > 1e8*math.fabs(n):
        # avoid loss of precision
        num = Gamma(1 + n) / math.fabs(k) + Gamma(1 + n) * n / (2*k**2) # + ...
        num /= pi * math.fabs(k)**n
        if k > 0:
            kx = floor(k)
            if int(kx) == kx:
                dk = k - kx
                sgn = 1 if (int(kx)) % 2 == 0 else -1
            else:
                dk = k
                sgn = 1
            return num * sin((dk-n)*pi) * sgn
        else:
            kx = floor(k)
            if int(kx) == kx:
                return 0
            else:
                return num * sin(k*pi)
    else:
        return 1/beta(1 + n - k, 1 + k)/(n + 1)


def comb(N, k, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, ndarray
        The total number of combinations.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120L
    >>> comb(10, 3, exact=True, repetition=True)
    220L

    """
    if repetition:
        return comb(N + k - 1, k, exact)
    if exact:
        N = int(N)
        k = int(k)
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for j in xrange(min(k, N-k)):
            val = (val*(N-j))//(j+1)
        return val
    else:
        k, N = np.asarray(k), np.asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = binom(N, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


def is_nonpos_int(x):
	return x <= 0 and x == ceil(x) and math.fabs(x) < 1e13

def poch(a, m):
    r = 1.0;

    # /*
    #  * 1. Reduce magnitude of `m` to |m| < 1 by using recurrence relations.
    #  *
    #  * This may end up in over/underflow, but then the function itself either
    #  * diverges or goes to zero. In case the remainder goes to the opposite
    #  * direction, we end up returning 0*INF = NAN, which is OK.
    #  */

    # /* Recurse down */
    while (m >= 1.0):
        if (a + m == 1):
            break
        m -= 1.0
        r *= (a + m)
        if (not np.isfinite(r) or r == 0):
            break;

    # /* Recurse up */
    while (m <= -1.0):
        if (a + m == 0):
            break
        r /= (a + m)
        m += 1.0
        if (not np.isfinite(r) or r == 0):
            break

    # /*
    #  * 2. Evaluate function with reduced `m`
    #  *
    #  * Now either `m` is not big, or the `r` product has over/underflown.
    #  * If so, the function itself does similarly.
    #  */

    if (m == 0):
        # /* Easy case */
        return r

    elif (a > 1e4 and math.fabs(m) <= 1):
        # /* Avoid loss of precision */
        return r * pow(a, m) * (1 + m*(m-1)/(2*a) + m*(m-1)*(m-2)*(3*m-1)/(24*a*a) + m*m*(m-1)*(m-1)*(m-2)*(m-3)/(48*a*a*a))

    # /* Check for infinity */
    if (is_nonpos_int(a + m) and not is_nonpos_int(a) and a + m != m):
        return NPY_INFINITY

    # /* Check for zero */
    if (not is_nonpos_int(a + m) and is_nonpos_int(a)):
        return 0

    return r * exp(lgam(a + m) - lgam(a)) * gammasgn(a + m) * gammasgn(a)