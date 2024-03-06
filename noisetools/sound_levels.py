"""
Functions related to the levels of measured or simulated sound signals.
"""
import numpy as np

from .weighting_functions import weigh_signal


__all__ = ['ospl',
           ]


def ospl(signal: list | np.ndarray, fs: int | float | np.number, weighting: str = None) -> float:
    """
    Calculate the Overall Sound Pressure Level of the input sound signal.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.

    Returns
    -------
    float
        Overall sound pressure level in dB (weighted to selected weighting)
    """
    # Convert signal to numpy array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Check that this signal is 1d array
    if len(signal.shape) > 1:
        raise ValueError('noisetools.ospl only supports 1d signal arrays.')

    # Apply the selected weighting
    if weighting is not None:
        signal = weigh_signal(signal, fs, curve=weighting)

    # Determine the corresponding time for proper integration
    t = np.linspace(0, signal.size / fs, signal.size)

    # Determine the equivalent pressure
    pe2 = 1 / t[-1] * np.trapz(signal ** 2, t)

    return 10 * np.log10(pe2 / (2e-5 ** 2))
