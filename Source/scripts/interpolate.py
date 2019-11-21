# from scipy's "interpolate" sub-package

import numpy as np
import special

def evaluate_bpoly1(s,c,ci,cj):
    """
    Evaluate polynomial in the Bernstein basis in a single interval.

    A Bernstein polynomial is defined as

        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}

    with ``0 <= x <= 1``.

    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use

    """

    k = c.shape[0] - 1  # polynomial order
    s1 = 1. - s

    # special-case lowest orders
    if k == 0:
        res = c[0, ci, cj]
    elif k == 1: 
        res = c[0, ci, cj] * s1 + c[1, ci, cj] * s
    elif k == 2:
        res = c[0, ci, cj] * s1*s1 + c[1, ci, cj] * 2.*s1*s + c[2, ci, cj] * s*s
    elif k == 3:
        res = (c[0, ci, cj] * s1*s1*s1 + c[1, ci, cj] * 3.*s1*s1*s +
               c[2, ci, cj] * 3.*s1*s*s + c[3, ci, cj] * s*s*s)
    else:
        # XX: replace with de Casteljau's algorithm if needs be
        res, comb = 0., 1.
        for j in range(k+1):
            res += comb * s**j * s1**(k-j) * c[j, ci, cj]
            comb *= 1. * (k-j) / (j+1.)

    return res

def evaluate_bpoly1_deriv(s,c,ci,cj,nu,wrk):
    """
    Evaluate the derivative of a polynomial in the Bernstein basis 
    in a single interval.

    A Bernstein polynomial is defined as

        .. math:: b_{j, k} = comb(k, j) x^{j} (1-x)^{k-j}

    with ``0 <= x <= 1``.

    The algorithm is detailed in BPoly._construct_from_derivatives.

    Parameters
    ----------
    s : double
        Polynomial x-value
    c : double[:,:,:]
        Polynomial coefficients. c[:,ci,cj] will be used
    ci, cj : int
        Which of the coefs to use
    nu : int
        Order of the derivative to evaluate. Assumed strictly positive
        (no checks are made).
    wrk : double[:,:,::1]
        A work array, shape (c.shape[0]-nu, 1, 1).

    """

    k = c.shape[0] - 1  # polynomial order

    if nu == 0:
        res = evaluate_bpoly1(s, c, ci, cj)
    else:
        poch = 1.
        for a in range(nu):
            poch *= k - a

        term = 0.
        for a in range(k - nu + 1):
            term, comb = 0., 1.
            for j in range(nu+1):
                term += c[j+a, ci, cj] * (-1)**(j+nu) * comb
                comb *= 1. * (nu-j) / (j+1)
            wrk[a, 0, 0] = term * poch
        res = evaluate_bpoly1(s, wrk, 0, 0)
    return res

def find_interval(x,xval,prev_interval=0,extrapolate=1):
    """
    Find an interval such that x[interval] <= xval < x[interval+1]
    or interval == 0 and xval < x[0]
    or interval == n-2 and xval > x[n-1]

    Parameters
    ----------
    x : array of double, shape (m,)
        Piecewise polynomial breakpoints
    xval : double
        Point to find
    prev_interval : int, optional
        Interval where a previous point was found
    extrapolate : int, optional
        Whether to return the last of the first interval if the
        point is ouf-of-bounds. 

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.

    """

    a = x[0]
    b = x[x.shape[0]-1]

    interval = prev_interval
    if interval < 0 or interval >= x.shape[0]:
        interval = 0

    if not (a <= xval <= b):
        # Out-of-bounds (or nan)
        if xval < a and extrapolate:
            # below
            interval = 0
        elif xval > b and extrapolate:
            # above
            interval = x.shape[0] - 2
        else:
            # nan or no extrapolation
            interval = -1
    elif xval == b:
        # Make the interval closed from the right
        interval = x.shape[0] - 2
    else:
        # Find the interval the coordinate is in
        # (binary search with locality)
        if xval >= x[interval]:
            low = interval
            high = x.shape[0]-2
        else:
            low = 0
            high = interval

        if xval < x[low+1]:
            high = low

        while low < high:
            mid = (high + low)//2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 1
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid
                break

        interval = low

    return interval

def evaluate_bernstein(c,x,xp,nu,extrapolate,out):
    """
    Evaluate a piecewise polynomial in the Bernstein basis.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    nu : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.

    """

    # check derivative order
    if nu < 0:
        raise NotImplementedError("Cannot do antiderivatives in the B-basis yet.")

    # shape checks
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[2]:
        raise ValueError("out and c have incompatible shapes")
    if c.shape[1] != x.shape[0] - 1:
        raise ValueError("x and c have incompatible shapes")

    if nu > 0:
        if double_or_complex is double_complex:
            wrk = np.empty((c.shape[0]-nu, 1, 1), dtype=np.complex_)
        else:
            wrk = np.empty((c.shape[0]-nu, 1, 1), dtype=np.float_)
        
    # evaluate
    interval = 0

    for ip in range(len(xp)):
        xval = xp[ip]

        # Find correct interval
        i = find_interval(x, xval, interval, extrapolate)
        if i < 0:
            # xval was nan etc
            for jp in range(c.shape[2]):
                out[ip, jp] = nan
            continue
        else:
            interval = i

        # Evaluate the local polynomial(s)
        ds = x[interval+1] - x[interval]
        ds_nu = ds**nu
        for jp in range(c.shape[2]):
            s = (xval - x[interval]) / ds
            if nu == 0:
                out[ip, jp] = evaluate_bpoly1(s, c, interval, jp)
            else:
                out[ip, jp] = evaluate_bpoly1_deriv(s, c, interval, jp,
                        nu, wrk) / ds_nu


class _PPolyBase(object):
    """
    Base class for piecewise polynomials.
    """
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = np.asarray(c)
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("%s must be between 0 and %s" % (axis, c.ndim-1))

        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = np.rollaxis(self.c, axis+1)
            self.c = np.rollaxis(self.c, axis+1)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        if np.any(self.x[1:] - self.x[:-1] < 0):
            raise ValueError("x-coordinates are not in increasing order")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        `c` and `x` must be arrays of the correct shape and type.  The
        `c` array can only be of dtypes float and complex, and `x`
        array must have dtype float.

        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x, right=True):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals
            ``self.x[-1] <= x < x_right[0]``, ``x_right[0] <= x < x_right[1]``,
            ..., ``x_right[m-2] <= x < x_right[m-1]``
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted and either to
            the right or to the left of the current breakpoints.
        right : bool, optional
            Whether the new intervals are to the right or to the left
            of the current intervals.

        """
        c = np.asarray(c)
        x = np.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("x and c have incompatible sizes")
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("c and self.c have incompatible shapes")
        if right:
            if x[0] < self.x[-1]:
                raise ValueError("new x are not to the right of current ones")
        else:
            if x[-1] > self.x[0]:
                raise ValueError("new x are not to the left of current ones")

        if c.size == 0:
            return

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                      dtype=dtype)

        if right:
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]
        else:
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)
        out = np.empty((len(x), np.prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out


class BPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints

    The polynomial in the ``i``-th interval ``x[i] <= xp < x[i+1]``
    is written in the Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1))

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = comb(k, a) * t**k * (1 - t)**(k - a)

    with ``t = (x - x[i]) / (x[i+1] - x[i])``.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. These must be sorted in
        increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-dimensional array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    extend
    derivative
    antiderivative
    integrate
    construct_fast
    from_power_basis
    from_derivatives

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature.
    Here's a non-exhaustive list:

    .. [1] http://en.wikipedia.org/wiki/Bernstein_polynomial

    .. [2] Kenneth I. Joy, Bernstein polynomials,
      http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf

    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
         vol 2011, article ID 829546, doi:10.1155/2011/829543

    Examples
    --------
    >>> from scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) + 3 \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2

    """

    def _evaluate(self, x, nu, extrapolate, out):
        evaluate_bernstein(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. (Default: 1)
            If negative, the antiderivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k2 = k - nu representing the derivative
            of this polynomial.

        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            # For a polynomial
            #    B(x) = \sum_{a=0}^{k} c_a b_{a, k}(x),
            # we use the fact that
            #   b'_{a, k} = k ( b_{a-1, k-1} - b_{a, k-1} ),
            # which leads to
            #   B'(x) = \sum_{a=0}^{k-1} (c_{a+1} - c_a) b_{a, k-1}
            #
            # finally, for an interval [y, y + dy] with dy != 1,
            # we need to correct for an extra power of dy

            rest = (None,)*(self.c.ndim-2)

            k = self.c.shape[0] - 1
            dx = np.diff(self.x)[(None, slice(None))+rest]
            c2 = k * np.diff(self.c, axis=0) / dx

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. (Default: 1)
            If negative, the derivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k2 = k + nu representing the
            antiderivative of this polynomial.

        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp

        # Construct the indefinite integrals on individual intervals
        c, x = self.c, self.x
        k = c.shape[0]
        c2 = np.zeros((k+1,) + c.shape[1:], dtype=c.dtype)

        c2[1:, ...] = np.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,)*(c.ndim-2)]

        # Now fix continuity: on the very first interval, take the integration
        # constant to be zero; on an interval [x_j, x_{j+1}) with j>0,
        # the integration constant is then equal to the jump of the `bp` at x_j.
        # The latter is given by the coefficient of B_{n+1, n+1}
        # *on the previous interval* (other B. polynomials are zero at the breakpoint)
        # Finally, use the fact that BPs form a partition of unity.
        c2[:,1:] += np.cumsum(c2[k,:], axis=0)[:-1]

        return self.construct_fast(c2, x, self.extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.
            Defaults to ``self.extrapolate``.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]

        """
        # XXX: can probably use instead the fact that
        # \int_0^{1} B_{j, n}(x) \dx = 1/(n+1)
        ib = self.antiderivative()
        if extrapolate is not None:
            ib.extrapolate = extrapolate
        return ib(b) - ib(a)

    def extend(self, c, x, right=True):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x, right)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs. Default: True.

        """
        dx = np.diff(pp.x)
        k = pp.c.shape[0] - 1   # polynomial order

        rest = (None,)*(pp.c.ndim-2)

        c = np.zeros_like(pp.c)
        for a in range(k+1):
            factor = pp.c[a] / comb(k, k-a) * dx[(slice(None),)+rest]**(k-a)
            for j in range(k-a, k+1):
                c[j] += factor * comb(j, k-a)

        if extrapolate is None:
            extrapolate = pp.extrapolate

        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        """Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j``-th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs. Default: True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------

        >>> from scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
        Notice that no restriction is imposed on the derivatives at
        `x = 1` and `x = 2`.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2

        """
        xi = np.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError("xi and yi need to have the same length")
        if np.any(xi[1:] - xi[:1] <= 0):
            raise ValueError("x coordinates are not in increasing order")

        # number of intervals
        m = len(xi) - 1

        # global poly order is k-1, local orders are <=k and can vary
        try:
            k = max(len(yi[i]) + len(yi[i+1]) for i in range(m))
        except TypeError:
            raise ValueError("Using a 1D array for y? Please .reshape(-1, 1).")

        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (integer_types, np.integer)):
                orders = [orders] * m
            k = max(k, max(orders))

            if any(o <= 0 for o in orders):
                raise ValueError("Orders must be positive.")

        c = []
        for i in range(m):
            y1, y2 = yi[i], yi[i+1]
            if orders[i] is None:
                n1, n2 = len(y1), len(y2)
            else:
                n = orders[i]+1
                n1 = min(n//2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1+n2 != n:
                    raise ValueError("Point %g has %d derivatives, point %g"
                            " has %d derivatives, but order %d requested" %
                            (xi[i], len(y1), xi[i+1], len(y2), orders[i]))
                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError("`order` input incompatible with"
                            " length y1 or y2.")

            b = BPoly._construct_from_derivatives(xi[i], xi[i+1], y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)

        c = np.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        """Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on `[xa, xb]` and having the values and derivatives at the
        endpoints ``xa`` and ``xb`` as specified by ``ya`` and ``yb``.
        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of ``ya`` and ``yb`` are ``na`` and ``nb``, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at ``xa``. ``ya[0]`` is the value of the function, and
            ``ya[i]`` for ``i > 0`` is the value of the ``i``-th derivative.
        yb : array_like
            Derivatives at ``xb``.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

          .. math:: Q_a = \sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At `x = xb` it's the same with `a = n - q`.

        """
        ya, yb = np.asarray(ya), np.asarray(yb)
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError('ya and yb have incompatible dimensions.')

        dta, dtb = ya.dtype, yb.dtype
        if (np.issubdtype(dta, np.complexfloating) or
               np.issubdtype(dtb, np.complexfloating)):
            dt = np.complex_
        else:
            dt = np.float_

        na, nb = len(ya), len(yb)
        n = na + nb

        c = np.empty((na+nb,) + ya.shape[1:], dtype=dt)

        # compute coefficients of a polynomial degree na+nb-1
        # walk left-to-right
        for q in range(0, na):
            c[q] = ya[q] / special.poch(n - q, q) * (xb - xa)**q
            for j in range(0, q):
                c[q] -= (-1)**(j+q) * special.comb(q, j) * c[j]

        # now walk right-to-left
        for q in range(0, nb):
            c[-q-1] = yb[q] / special.poch(n - q, q) * (-1)**q * (xb - xa)**q
            for j in range(0, q):
                c[-q-1] -= (-1)**(j+1) * special.comb(q, j+1) * c[-q+j]

        return c

    @staticmethod
    def _raise_degree(c, d):
        """Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1D
        d : integer

        Returns
        -------
        array
            coefficient array, 1D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \sum_{j=0}^{d} b_{a+j, k+d} \
                                comb(d, j) / comb(k+d, a+j)

        """
        if d == 0:
            return c

        k = c.shape[0] - 1
        out = np.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)

        for a in range(c.shape[0]):
            f = c[a] * comb(k, a)
            for j in range(d+1):
                out[a+j] += f * comb(d, j) / comb(k+d, a+j)
        return out

def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for scipy argument validation.
    Many scipy linear algebra functions do support arbitrary array-like
    input arguments.  Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.
    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.
    Returns
    -------
    ret : ndarray
        The converted validated array.
    """
    # if not sparse_ok:
    #     import scipy.sparse
    #     if scipy.sparse.issparse(a):
    #         msg = ('Sparse matrices are not supported by this function. '
    #                'Perhaps one of the scipy.sparse.linalg functions '
    #                'would work instead.')
    #         raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a

class _Interpolator1D(object):
    """
    Common features in univariate interpolation
    Deal with input data type and interpolation axis rolling.  The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.
    Attributes
    ----------
    _y_axis
        Axis along which the interpolation goes in the original array
    _y_extra_shape
        Additional trailing shape of the input arrays, excluding
        the interpolation axis.
    dtype
        Dtype of the y-data arrays. Can be set via _set_dtype, which
        forces it to be float or complex.
    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate
    """

    __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """
        Evaluate the interpolant
        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.
        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator.
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to n-d array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        yi = np.rollaxis(np.asarray(yi), self._y_axis)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "%r + (N,) + %r" % (self._y_extra_shape[-self._y_axis:],
                                           self._y_extra_shape[:-self._y_axis])
            raise ValueError("Data must be of shape %s" % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")

        self._y_axis = (axis % yi.ndim)
        self._y_extra_shape = yi.shape[:self._y_axis]+yi.shape[self._y_axis+1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex_
        else:
            if not union or self.dtype != np.complex_:
                self.dtype = np.float_

def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError('%s argument must be able to broadcast up '
                     'to shape %s but had shape %s'
                     % (name, shape_to, shape_from))


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    string_types = str
    return (isinstance(fill_value, string_types) and
            fill_value == 'extrapolate')

class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.
    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.
    Note that calling `interp1d` with NaNs present in input values results in
    undefined behaviour.
    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
        refer to a spline interpolation of zeroth, first, second or third
        order; 'previous' and 'next' simply return the previous or next value
        of the point) or as an integer specifying the order of the spline
        interpolator to use.
        Default is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``.
          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.
          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.
    Attributes
    ----------
    fill_value
    Methods
    -------
    __call__
    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)
    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=np.nan,
                 assume_sorted=False):
        """ Initialize a 1D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)

        self.bounds_error = bounds_error  # used by fill_value setter
        self.copy = copy

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1,
                     'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest', 'previous', 'next'):
            raise NotImplementedError("%s is unsupported: Use fitpack "
                                      "routines for other types." % kind)
        x = np.array(x, copy=self.copy)
        y = np.array(y, copy=self.copy)

        if not assume_sorted:
            ind = np.argsort(x)
            x = x[ind]
            y = np.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float_)

        # Backward compatibility
        self.axis = axis % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x  # clean up namespace to prevent misuse; use attributes
        self._kind = kind
        self.fill_value = fill_value  # calls the setter, can modify bounds_err

        # Adjust to interpolation kind; store reference to *unbound*
        # interpolation methods, in order to avoid circular references to self
        # stored in the bound instance methods, and therefore delayed garbage
        # collection.  See: https://docs.python.org/reference/datamodel.html
        if kind in ('linear', 'nearest', 'previous', 'next'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            minval = 2
            if kind == 'nearest':
                # Do division before addition to prevent possible integer
                # overflow
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'previous':
                # Side for np.searchsorted and index for clipping
                self._side = 'left'
                self._ind = 0
                # Move x by one floating point value to the left
                self._x_shift = np.nextafter(self.x, -np.inf)
                self._call = self.__class__._call_previousnext
            elif kind == 'next':
                self._side = 'right'
                self._ind = 1
                # Move x by one floating point value to the right
                self._x_shift = np.nextafter(self.x, np.inf)
                self._call = self.__class__._call_previousnext
            else:
                # Check if we can delegate to numpy.interp (2x-10x faster).
                cond = self.x.dtype == np.float_ and self.y.dtype == np.float_
                cond = cond and self.y.ndim == 1
                cond = cond and not _do_extrapolate(fill_value)

                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1

            rewrite_nan = False
            xx, yy = self.x, self._y
            if order > 1:
                # Quadratic or cubic spline. If input contains even a single
                # nan, then the output is all nans. We cannot just feed data
                # with nans to make_interp_spline because it calls LAPACK.
                # So, we make up a bogus x and y with no nans and use it
                # to get the correct shape of the output, which we then fill
                # with nans.
                # For slinear or zero order spline, we just pass nans through.
                if np.isnan(self.x).any():
                    xx = np.linspace(min(self.x), max(self.x), len(self.x))
                    rewrite_nan = True
                if np.isnan(self._y).any():
                    yy = np.ones_like(self._y)
                    rewrite_nan = True

            self._spline = make_interp_spline(xx, yy, k=order,
                                              check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline

        if len(self.x) < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)

    @property
    def fill_value(self):
        """The fill value."""
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        # extrapolation only works for nearest neighbor and linear methods
        if _do_extrapolate(fill_value):
            if self.bounds_error:
                raise ValueError("Cannot extrapolate and raise "
                                 "at the same time.")
            self.bounds_error = False
            self._extrapolate = True
        else:
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # it's either a pair (_below_range, _above_range) or a single value
            # for both above and below range
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]),
                               np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def _call_linear_np(self, x_new):
        # Note that out-of-bounds values are taken care of in self._evaluate
        return np.interp(x_new, self.x, self.y)

    def _call_linear(self, x_new):
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1.  Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbour interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbour
        x_new_indices = searchsorted(self.x_bds, x_new, side='left')

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices]

        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbour of x_new, y_new = f(x_new)."""

        # 1. Get index of left/right value
        x_new_indices = searchsorted(self._x_shift, x_new, side=self._side)

        # 2. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(1-self._ind,
                                           len(self.x)-self._ind).astype(intp)

        # 3. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices+self._ind-1]

        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = np.asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.
        Parameters
        ----------
        x_new : array
        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                             "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                             "range.")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds
