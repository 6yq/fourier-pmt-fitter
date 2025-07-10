import numpy as np
from scipy.fft import fft, ifft


def roll_and_pad(pdf, shift, pad_safe):
    """
    Symmetrically pad and roll the input PDF. Also return the slice to recover original portion.

    Returns
    -------
    pdf_shifted : ndarray
        The padded and shifted PDF.
    recover_slice : slice
        Slice object to extract the original unpadded portion after IFFT.
    """
    pdf_padded = np.pad(pdf, (0, pad_safe), mode="constant")
    shift_padded = 2 * shift if shift < 0 else 0

    # Get slice to recover the center part of original length
    recover_slice = slice(0, len(pdf))

    return np.roll(pdf_padded, shift_padded), shift_padded, recover_slice


def fft_and_ifft(pdf, shift, dx, pad_safe, processor, const=0):
    """
    Perform FFT-based convolution and return real-space PDF after inverse transform.

    Parameters
    ----------
    pdf : ndarray
        Real-space PDF.
    shift : int
        Number of bins to roll so that x=0 aligns to index 0.
    dx : float
        Bin width.
    pad_safe : int
        Padding size.
    processor : callable
        Function that operates on FFT of input PDF.
    const : float
        Constant to add to the output PDF.
        This is contributed by zero PDF in SPE response.

    Returns
    -------
    ifft_pdf : ndarray
        Inverse transformed PDF (cropped to original size).
    """
    pdf_shifted, shift_padded, recover_slice = roll_and_pad(pdf, shift, pad_safe)
    fft_pdf = (1 - const) * fft(pdf_shifted) * dx + const
    fft_processed = processor(fft_pdf)
    ifft_pdf = np.roll(np.real(ifft(fft_processed)) / dx, -shift_padded)
    return ifft_pdf[recover_slice]
