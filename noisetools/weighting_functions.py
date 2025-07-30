"""Spectral weighting functions for acoustic processing.
"""

# Copyright 2025 Josephine Pockelé
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy.signal as spsig
import numpy as np


__all__ = ['a_weighting', 'c_weighting', 'weigh_signal', ]


def a_weighting_simons(f: np.ndarray,
                       ) -> np.ndarray:
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


def a_weighting(f: float | int | np.number | list | np.ndarray,
                a_1000: float = None,
                ) -> float | np.ndarray:
    """
    A frequency weighting network A(f) defined by Equation E.6 in the IEC 61672-1:2013 standard. [1]_

    Parameters
    ----------
    f : number | array_like
        Frequenc(y)(ies) over which to calculate the c-weighting.
    a_1000 : float, optional
        Manually set the gain of the weighting function. Defaults to the a_weighting(1000, a_1000=0.).

    Returns
    -------
    float | np.ndarray
        A-weighting value(s) in dB, corresponding to the input frequenc(y)(ies) f.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.

    """
    f_is_num = False
    if isinstance(f, list):
        f = np.array(f)
    elif not issubclass(type(f), np.ndarray):
        f = np.array([f, ])
        f_is_num = True

    # Set a_1000 to default, such that a_weighting([1000, ], ) = [0, ]. According to E.3.2
    if a_1000 is None:
        a_1000 = a_weighting(1000., a_1000=0.)
    # Get the pole frequencies from equations E.2, E.7, E.8 and E.3
    f1, f2, f3, f4 = weighting_poles()
    # Use equation E.6 to obtain the final attenuation
    delta_l_a = 1e-18 * np.ones(f.shape)

    num = (f4 * f[f > 0] ** 2) ** 2
    p1 = f[f > 0] ** 2 + f1 ** 2
    p2 = f[f > 0] ** 2 + f2 ** 2
    p3 = f[f > 0] ** 2 + f3 ** 2
    p4 = f[f > 0] ** 2 + f4 ** 2
    delta_l_a[f > 0] = num / (p1 * np.sqrt(p2) * np.sqrt(p3) * p4)

    if f_is_num:
        delta_l_a = float(delta_l_a[0])

    return 20 * np.log10(delta_l_a) - a_1000


def c_weighting(f: float | int | np.number | list | np.ndarray,
                c_1000: float = None,
                ) -> float | np.ndarray:
    """
    C frequency weighting network C(f) defined by Equation E.1 in the IEC 61672-1:2013 standard. [1]_

    Parameters
    ----------
    f : number | array_like
        Frequenc(y)(ies) over which to calculate the c-weighting.
    c_1000 : float, optional
        Manually set the gain of the weighting function. Defaults to the c_weighting(1000, c_1000=0.).

    Returns
    -------
    float | np.ndarray
        C-weighting value(s) in dB, corresponding to the input frequenc(y)(ies) f.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.

    """
    f_is_num = False
    if isinstance(f, list):
        f = np.array(f)
    elif not isinstance(f, np.ndarray):
        f = np.array([f, ])
        f_is_num = True

    # Set c_1000 to default, such that c_weighting([1000, ], ) = [0, ], according to E.2.2
    if c_1000 is None:
        c_1000 = c_weighting(1000., c_1000=0.)
    # Get the pole frequencies from equations E.2 and E.3
    f1, _, _, f4 = weighting_poles()
    # Use equation E.1 to obtain the final attenuation
    delta_l_c = 1e-18 * np.ones(f.shape)

    num = (f4 * f[f > 0]) ** 2
    p1 = f[f > 0] ** 2 + f1 ** 2
    p4 = f[f > 0] ** 2 + f4 ** 2

    delta_l_c[f > 0] = num / (p1 * p4)

    if f_is_num:
        delta_l_c = float(delta_l_c[0])

    return 20 * np.log10(delta_l_c) - c_1000


def weighting_filter(curve: str = 'A',
                     analog: bool = False,
                     output: str = 'ba',
                     fs: float = 51.2e3,
                     ):
    """
    Returns the filter design for the weighting filters defined in IEC 61672-1:2013. [1]_

    Parameters
    ----------
    curve: str, optional
        The name of the weighting curve to be used in this filter. Can be 'A' or 'C'.
    analog: bool, optional
        When True, return an analog filter, otherwise a digital filter is returned.
    output: str, optional
        Type of output: numerator/denominator (‘ba’), pole-zero (‘zpk’), or second-order sections (‘sos’).
        Default is ‘ba’ for backwards compatibility, but ‘sos’ should be used for general-purpose filtering.
    fs: float, optional
        The sampling frequency of the digital system. Defaults to 51.2 kHz.

    Returns
    -------
    b, a : numpy.ndarray, numpy.ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter. Only returned if output='ba'.
    z, p, k : numpy.ndarray, numpy.ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer function. Only returned if output='zpk'.
    sos: numpy.ndarray
        Second-order sections representation of the IIR filter. Only returned if output='sos'.

    Notes
    -----
    A-weighting complies with performance class 2 and C-weighting with performance class 1.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1:
        Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.

    """
    # Get the pole frequencies from equations E.2, E.7, E.8 and E.3
    f1, f2, f3, f4 = weighting_poles()

    # Set up the A-weighting curve
    if curve.upper() == 'A':
        # There are 4 zeroes at 0 Hz
        zeros = np.zeros((4, ))
        # There are 8 poles, 1 pair per pole frequency
        poles = np.array([-2 * np.pi * f1, -2 * np.pi * f1,
                          -2 * np.pi * f4, -2 * np.pi * f2,
                          -2 * np.pi * f4, -2 * np.pi * f3,
                          -2 * np.pi * f4, -2 * np.pi * f4,
                          ])

    # Set up the C-weighting curve
    elif curve.upper() == 'C':
        # There are 2 zeroes at 0 Hz
        zeros = np.zeros((2, ))
        # There are 4 poles, 1 pair per pole frequency
        poles = np.array([-2 * np.pi * f1, -2 * np.pi * f1,
                          -2 * np.pi * f4, -2 * np.pi * f4,
                          ])

    # Error if the curve type is not understood
    else:
        raise ValueError(f'{curve} not understood as weighting curve type.')

    # Get the gain to normalise the weighting s.t. weighting(1000 Hz) = 0 dB
    b, a = spsig.zpk2tf(zeros, poles, 1.)
    gain = 1 / abs(spsig.freqs(b, a, [2 * np.pi * 1000])[1][0])

    # Convert to a digital filter if so desired
    if not analog:
        zeros, poles, gain = spsig.bilinear_zpk(zeros, poles, gain, fs)

    # Convert to ba/tf format
    if output.lower() in ('ba', 'tf', ):
        return spsig.zpk2tf(zeros, poles, gain)
    # Return the zpk form
    elif output.lower() == 'zpk':
        return zeros, poles, gain
    # Convert to sos format
    elif output.lower() == 'sos':
        return spsig.zpk2sos(zeros, poles, gain)
    # Error if the output form is not valid.
    else:
        raise ValueError(f'{output} is not a valid output form.')


def weigh_signal(signal: list | np.ndarray, fs: int | float | np.number, curve: str = 'A'):
    """
    Apply signal weighting in the time domain.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: float
        The sampling frequency of the digital signal
    curve: str, optional
        The name of the weighting curve to be used in this filter. Can be 'A' or 'C'
    """
    weighting_sos = weighting_filter(curve, output='sos', fs=fs)
    return spsig.sosfilt(weighting_sos, signal)


if __name__ == '__main__':
    """
    Compliance checks for IEC 61672-1:2013.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    weighting_table = pd.read_csv('weighting_table.csv',
                                  header=0, index_col=0, delimiter=';').replace(np.inf, 1e12)

    for weighting in ('a', 'c'):
        plt.figure(f'{weighting.upper()}-weighting')
        if weighting == 'a':
            plt.semilogx(weighting_table.index, a_weighting(weighting_table.index.to_numpy()), 'k', label='reference')
        elif weighting == 'c':
            plt.semilogx(weighting_table.index, c_weighting(weighting_table.index.to_numpy()), 'k', label='reference')

        plt.errorbar(x=weighting_table.index, y=weighting_table[weighting],
                     yerr=(weighting_table['limit 1-'], weighting_table['limit 1+'], ),
                     fmt='k.', capsize=3)
        plt.errorbar(x=weighting_table.index, y=weighting_table[weighting],
                     yerr=(weighting_table['limit 2-'], weighting_table['limit 2+'], ),
                     fmt='k.', capsize=1.5, elinewidth=1.)

        # ba/tf type
        ba = weighting_filter(weighting, analog=True, output='ba')
        w, h = spsig.freqs(*ba)
        plt.semilogx(w / (2 * np.pi), 20 * np.log10(np.abs(h)), ':', label='ba/tf analog', color='tab:blue')
        # zpk type
        zpk = weighting_filter(weighting, analog=True, output='zpk')
        w, h = spsig.freqs_zpk(*zpk)
        plt.semilogx(w / (2 * np.pi), 20 * np.log10(np.abs(h)), ':', label='zpk analog', color='tab:orange')

        # ba/tf digital type
        ba = weighting_filter(weighting, output='ba', fs=51.2e3)
        w, h = spsig.freqz(*ba, fs=51.2e3)
        plt.semilogx(w[w > 0], 20 * np.log10(np.abs(h[w > 0])), label='ba/tf digital', color='tab:blue')
        # zpk digital type
        zpk = weighting_filter(weighting, output='zpk', fs=51.2e3)
        w, h = spsig.freqz_zpk(*zpk, fs=51.2e3)
        plt.semilogx(w[w > 0], 20 * np.log10(np.abs(h[w > 0])), label='zpk digital', color='tab:orange')
        # sos digital type
        sos = weighting_filter(weighting, output='sos', fs=51.2e3)
        w, h = spsig.freqz_sos(sos, fs=51.2e3)
        plt.semilogx(w[w > 0], 20 * np.log10(np.abs(h[w > 0])), label='sos digital', color='tab:red')

        plt.legend()
        plt.ylim(-42, 6)
        plt.yticks(np.arange(-42, 6 + 6, 6))
        plt.ylabel('Attenuation (dB)')
        plt.xlim(10, 25e3)
        plt.xlabel('f (Hz)')
        plt.grid()

    plt.show()
