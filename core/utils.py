import numpy as np
import operator
from scipy.signal import find_peaks, peak_widths


def composite_simpson(pdf_slice, interval, sample):
    """Use composite Simpson to integrate pdf.

    Parameters
    ----------
    pdf_slice : ArrayLike
        a pdf list/array/...
    """
    result = pdf_slice[0] + pdf_slice[-1]
    odd_sum = sum(pdf_slice[i] for i in range(1, sample, 2))
    even_sum = sum(pdf_slice[i] for i in range(2, sample, 2))
    result += 4 * odd_sum + 2 * even_sum
    result *= interval / 3
    return result


def isInBound(param: float | int, bound: tuple[None | float | int]) -> bool:
    """Check if a parameter is within a given bound.

    Parameters
    ----------
    param : float or int
        The parameter to be checked.
    bound : tuple[float or int or None, float or int or None]
        A two-element tuple specifying the lower and upper bounds.

    Returns
    -------
    bool
        True if the parameter is within the bounds, else False.
    """
    assert len(bound) == 2, "Illegal bound!"
    if None not in bound:
        lower, upper = bound
        assert lower <= upper, "Illegal order of bound!"
        return (param >= lower) & (param <= upper)
    elif bound == (None, None):
        return True
    elif bound[0] is None:
        return param < bound[1]
    else:
        return param > bound[0]


def isParamsInBound(params, bounds):
    """Check if multiple parameters are all within their respective bounds.

    Parameters
    ----------
    params : list or array-like
        List of parameter values.
    bounds : list of tuple
        List of bounds for each parameter.

    Returns
    -------
    bool
        True if all parameters are within bounds.
    """
    flag = True
    for p, b in zip(params, bounds):
        flag = flag & isInBound(param=p, bound=b)
        if not flag:
            return False
    return flag


def isParamsWithinConstraints(args, constraints):
    """Check if a list of parameters satisfy all user-defined constraints.

    Parameters
    ----------
    args : list or array
        Parameter values.
    constraints : list of dict
        Constraints in the form of coefficient maps and thresholds.

    Returns
    -------
    bool
        True if all constraints are satisfied.
    """
    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }

    for constraint in constraints:
        if isinstance(constraint, dict):
            lhs = sum(coeff * args[idx] for idx, coeff in constraint["coeffs"])
            rhs = constraint["threshold"]
            op = ops.get(constraint.get("op", ">"))
            if not op(lhs, rhs):
                return False
        else:
            raise ValueError(f"Unknown constraint format: {constraint}")

    return True


def merge_bins(hist, y, threshold=5):
    """
    Merge bins with low counts.

    Parameters
    ----------
    hist : ArrayLike
        Histogram of counts.
    y : ArrayLike
        Counts.
    threshold : int
        Threshold of counts to merge.

    Return
    ------
    hist_ : ArrayLike
        Merged histogram.
    y_ : ArrayLike
        Merged counts.

    Notes
    -----
    Merge the bins below `threshold` from both sides to the middle.
    """
    hist_ = hist.copy()
    y_ = y.copy()

    while True:
        peak_idx = np.argmax(hist_)
        idx = np.where(hist_ <= threshold)[0]
        if idx.size == 0:
            break

        idx = idx[0]
        merged = False

        if idx < peak_idx:
            hist_tmp = np.append(hist_[:idx], sum(hist_[idx : idx + 2]))
            hist_ = np.append(hist_tmp, hist_[idx + 2 :])

            y_tmp = np.append(y_[:idx], sum(y_[idx : idx + 2]))
            y_ = np.append(y_tmp, y_[idx + 2 :])

            merged = True
        elif idx > peak_idx:
            idx = np.where(hist_ <= threshold)[0][-1]
            hist_tmp = np.append(hist_[: idx - 1], sum(hist_[idx - 1 : idx + 1]))
            hist_ = np.append(hist_tmp, hist_[idx + 1 :])

            y_tmp = np.append(y_[: idx - 1], sum(y_[idx - 1 : idx + 1]))
            y_ = np.append(y_tmp, y_[idx + 1 :])

            merged = True
        if not merged:
            break

    return hist_, y_


_FIND_PEAKS_KW = {
    "height",
    "threshold",
    "distance",
    "prominence",
    "width",
    "wlen",
    "rel_height",
    "plateau_size",
}


def compute_init(
    hist, edges, *, peak_idx=0, distance=15, width=5, rel_height=1, **kwargs
):
    """
    Compute initial values (mean, std) for a given peak in a histogram.

    Parameters
    ----------
    hist : ndarray
        Histogram counts.
    edges : ndarray
        Bin edges (length = len(hist) + 1).
    peak_idx : int
        Which peak to extract (0 = first prominent, 1 = second, ...).
    distance : float
        Minimum distance from the peak to be considered a peak.
    width : float
        Minimum width required to be considered a peak.
    rel_height : float
        Relative height for FWHM computation.

    Returns
    -------
    gp_init : float
        Estimated peak position (Gaussian mean).
    sigma_init : float
        Estimated standard deviation.
    """
    fp_params = {
        "distance": distance,
        "width": width,
        "rel_height": rel_height,
    }
    # override / append
    for k, v in kwargs.items():
        if k in _FIND_PEAKS_KW:
            fp_params[k] = v

    edges = np.array(edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Find prominent peaks
    peaks, props = find_peaks(hist, **fp_params)
    if len(peaks) <= peak_idx:
        raise ValueError(f"Only found {len(peaks)} peaks with prominence ≥ {width}")

    peak = peaks[peak_idx : peak_idx + 1]
    _, _, left_ips, right_ips = peak_widths(hist, peak, rel_height=0.5)

    # Interpolate position for FWHM
    def interpolate(idx):
        base = np.floor(idx).astype(int)
        frac = idx - base
        return bin_centers[base] + frac * (bin_centers[1] - bin_centers[0])

    gp_init = interpolate(peak)[0]
    x_left = interpolate(left_ips)[0]
    x_right = interpolate(right_ips)[0]
    fwhm = x_right - x_left
    sigma_init = fwhm / (2 * np.sqrt(2 * np.log(2)))

    return gp_init, sigma_init


def merged_pearson_chi2(hist, y, zero, z, dof):
    """Compute merged Pearson chi-square.

    Parameters
    ----------
    hist : ArrayLike
        Histogram of counts.(
    y : ArrayLike
        Estimated counts.
    zero : int
        Zero count.
    z : float
        Estimated zero count.
    dof : int
        Degrees of freedom.

    Returns
    -------
    chi_sq : float
        Merged Pearson chi-square.
    ndf : int

    Notes
    -----
    Hmm, I don't recommend this.
    """
    y_reg, hist_reg = merge_bins(y, hist, threshold=5)
    # actually len(hist_reg) + 1 - dof - 1
    ndf = len(hist_reg) - dof
    return sum((y_reg - hist_reg) ** 2 / y_reg) + (z - zero) ** 2 / z, ndf


def modified_neyman_chi2_A(hist, y, zero, z, dof):
    """Compute modified Neyman chi-square.

    Parameters
    ----------
    hist : ArrayLike
        Histogram of counts.
    y : ArrayLike
        Estimated counts.
    zero : int
        Zero count.
    z : float
        Estimated zero count.
    dof : int
        Degrees of freedom.

    Returns
    -------
    chi_sq : float
        Modified Neyman chi-square.
    ndf : int

    Notes
    -----
    This is from Cressie-Read family.
    I believe this is the case when lambda -> -2.
    """
    hist_ = np.append(hist, zero)
    y_ = np.append(y, z)
    nonZeroIdx = hist_ != 0
    hist_ = hist_[nonZeroIdx]
    y_ = y_[nonZeroIdx]
    ndf = len(hist_) - dof - 1
    return sum(y_**2 / hist_ - hist_), ndf


def modified_neyman_chi2_B(hist, y, zero, z, dof):
    """Compute modified Neyman chi-square.

    Parameters
    ----------
    hist : ArrayLike
        Histogram of counts.
    y : ArrayLike
        Estimated counts.
    zero : int
        Zero count.
    z : float
        Estimated zero count.
    dof : int
        Degrees of freedom.

    Returns
    -------
    chi_sq : float
        Modified Neyman chi-square.
    ndf : int

    Notes
    -----
    This is to avoid O = 0.
    """
    hist_ = np.append(hist, zero)
    y_ = np.append(y, z)
    nonZeroIdx = hist_ != 0
    hist_ = hist_[nonZeroIdx]
    y_ = y_[nonZeroIdx]
    ndf = len(hist_) - dof - 1
    return sum((y_ - hist_) ** 2 / hist_), ndf


def mighell_chi2(hist, y, zero, z, dof):
    """Compute Mighell chi-square.

    Parameters
    ----------
    hist : ArrayLike
        Histogram of counts.
    y : ArrayLike
        Estimated counts.
    zero : int
        Zero count.
    z : float
        Estimated zero count.
    dof : int
        Degrees of freedom.

    Returns
    -------
    chi_sq : float
        Mighell chi-square.
    ndf : int

    Notes
    -----
    Hmm, this might be awkward because we have many bins with zero counts.
    """
    hist_ = np.append(hist, zero)
    y_ = np.append(y, z)
    nonZeroIdx = hist_ != 0
    hist_ = hist_[nonZeroIdx]
    y_ = y_[nonZeroIdx]
    ndf = len(hist_) - dof - 1
    return sum((hist_ + 1 - y_) ** 2 / (hist_ + 1)), ndf
