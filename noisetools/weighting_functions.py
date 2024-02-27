"""
Spectral weighting functions for acoustic processing.
"""
import numpy as np


__all__ = ['a_weighting_simons', 'a_weighting', 'c_weighting', 'weighting_poles']


def a_weighting_simons(f: np.ndarray, ) -> np.ndarray:
    """
    A frequency weighting network A(f) defined by Equation 6.7 in Simons and Snellen [1]_.

    Parameters
    ----------
    f : numpy.ndarray
        N-dimensional array of frequencies over which to calculate the c-weighting.

    Returns
    -------
    numpy.ndarray
        Array of A-weighting values in dB, corresponding to the input frequencies f.

    References
    ----------
    .. [1] D. G. Simons and M. Snellen, ‘Aircraft Noise and Emissions - Part A: Introduction to general acoustics and
        aircraft noise’, Delft University of Technology, Delft, Netherlands, Course notes AE4431, Aug. 2021.

    """
    delta_l_a = np.zeros(f.shape)
    delta_l_a[f > 0] = (-145.528 +
                        98.262 * np.log10(f[f > 0]) -
                        19.509 * np.log10(f[f > 0]) ** 2 +
                        0.975 * np.log10(f[f > 0]) ** 3
                        )
    return delta_l_a


def weighting_poles() -> tuple[float, float, float, float]:
    """
    Calculates the pole frequencies for the A- and C-weighting networks in IEC 61672-1:2013 [1]_.

    Returns
    -------
    float, float, float, float
        f1, f2, f3 and f4 from Equations E.2, E.7, E.8, E.3 from IEC 61672-1:2013 [1]_.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.
    """
    # Reference, and low- and high-pass cutoff frequencies of the c-weighting
    fr, fl, fh = 1e3, 10. ** 1.5, 10. ** 3.9
    # Bi-quadratic coefficients to determine c-weighting poles
    d = np.sqrt(.5)
    b = 1 / (1 - d) * (fr ** 2 + (fl * fh / fr) ** 2 - d * (fl ** 2 + fh ** 2))
    c = (fl * fh) ** 2
    # Poles of the c-weighting
    f1: float = np.sqrt((-b - np.sqrt(b ** 2 - 4 * c)) / 2)
    f4: float = np.sqrt((-b + np.sqrt(b ** 2 - 4 * c)) / 2)

    # A-weighting high-pass filter cutoff frequency
    fa: float = 10 ** 2.45
    # Poles of the a-weighting
    f2: float = fa * (3 - np.sqrt(5)) / 2
    f3: float = fa * (3 + np.sqrt(5)) / 2

    return f1, f2, f3, f4


def a_weighting(f, a_1000: float = None) -> np.ndarray:
    """
    A frequency weighting network A(f) defined by Equation E.6 in the IEC 61672-1:2013 standard. [1]_

    Parameters
    ----------
    f : numpy.ndarray
        N-dimensional array of frequencies over which to calculate the a-weighting.
    a_1000 : float, optional
        Manually set the gain of the weighting function. Defaults to the a_weighting([1000, ], a_1000=0.).

    Returns
    -------
    numpy.ndarray
        Array of A-weighting values in dB, corresponding to the input frequencies f.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.
    """
    # Set a_1000 to default, such that a_weighting([1000, ], ) = [0, ]. According to E.3.2
    if a_1000 is None:
        a_1000 = a_weighting(np.array([1000., ]), a_1000=0.)
    # Get the pole frequencies from equations E.2, E.7, E.8 and E.3
    f1, f2, f3, f4 = weighting_poles()
    # Use equation E.6 to obtain the final attenuation
    delta_l_a = np.zeros(f.shape)
    delta_l_a[f > 0] = 10 * np.log10(
        ((
         (f4 * f[f > 0] ** 2) ** 2
         ) / (
         (f[f > 0] ** 2 + f1 ** 2) * np.sqrt((f[f > 0] ** 2 + f2 ** 2)) *
         np.sqrt((f[f > 0] ** 2 + f3 ** 2)) * (f[f > 0] ** 2 + f4 ** 2)
         )) ** 2
    )

    return delta_l_a - a_1000


def c_weighting(f: np.ndarray, c_1000: float = None) -> np.ndarray:
    """
    C frequency weighting network C(f) defined by Equation E.1 in the IEC 61672-1:2013 standard. [1]_

    Parameters
    ----------
    f : numpy.ndarray
        N-dimensional array of frequencies over which to calculate the c-weighting.
    c_1000 : float, optional
        Manually set the gain of the weighting function. Defaults to the c_weighting([1000, ], c_1000=0.).

    Returns
    -------
    numpy.ndarray
        Array of C-weighting values in dB, corresponding to the input frequencies f.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.
    """
    # Set c_1000 to default, such that c_weighting([1000, ], ) = [0, ], according to E.2.2
    if c_1000 is None:
        c_1000 = c_weighting(np.array([1000., ]), c_1000=0.)
    # Get the pole frequencies from equations E.2 and E.3
    f1, _, _, f4 = weighting_poles()
    # Use equation E.1 to obtain the final attenuation
    delta_l_c = np.zeros(f.shape)
    delta_l_c[f > 0] = 10 * np.log10(
        ((
         (f4 * f[f > 0]) ** 2
         ) / (
         (f[f > 0] ** 2 + f1 ** 2) * (f[f > 0] ** 2 + f4 ** 2)
         )) ** 2
    )

    return delta_l_c - c_1000
