"""Functions related to the levels of measured or simulated sound signals.
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

import numpy as np

from .weighting_functions import weigh_signal


__all__ = ['equivalent_pressure', 'ospl', 'ospl_t', ]


def equivalent_pressure(signal: list | np.ndarray,
                        fs: int | float | np.number,
                        weighting: str = None,
                        ) -> float:
    """
    Calculate the equivalent pressure (Pe^2) of the input sound signal.

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
    Equivalent pressure in Pa^2 (weighted to selected weighting)

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

    # Determine the equivalent pressure
    return 1 / signal.size * np.trapezoid(signal ** 2)


def ospl(signal: list | np.ndarray,
         fs: int | float | np.number,
         weighting: str = None,
         ) -> float:
    """
    Calculate the Overall Sound Pressure Level of the input sound signal.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal.
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.

    Returns
    -------
    Overall sound pressure level in dB (weighted to selected weighting)

    """
    pe2 = equivalent_pressure(signal, fs, weighting, )

    return 10 * np.log10(pe2 / (2e-5 ** 2))


def ospl_t(signal: list | np.ndarray,
           fs: int | float | np.number,
           weighting: str = None,
           t: list | np.ndarray = None,
           delta_t: float | np.number = 1.,
           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the OSPL over time, of a digital signal.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal.
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.
    t: list | np.ndarray, optional
        Optional input of the time series of the signal.
    delta_t: float | np.number, optional (default=1.)
        Desired timestep in the OSPL output, in seconds.

    Returns
    -------
    Two arrays:
        - Time (seconds) at which the OSPL is calculated. Determined as the central timestamp in the
            sections of length delta_t.
        - OPSL (dB) (weighted to selected weighting) at the timestamps defined in the time array.

    """
    # Convert signal to numpy array
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Ensure the time series is correct.
    if t is not None and not isinstance(t, np.ndarray):
        t = np.array(t)
    elif t is None:
        t = np.linspace(0, signal.size / fs, signal.size)
    if t.size != signal.size:
        raise ValueError(f'Shape mismatch: t and signal should have the same length: {t.size} != {signal.size}')

    t_out = np.arange(t[0], t[-1], delta_t)
    ospl_out = np.zeros(t_out.size)

    for ti, t0 in enumerate(t_out):
        select = (t0 <= t) & (t < t0 + delta_t)

        ospl_out[ti] = ospl(signal[select], fs, weighting=weighting)

    return t_out + delta_t / 2, ospl_out
