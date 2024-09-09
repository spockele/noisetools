"""
Functions related to the levels of measured or simulated sound signals.
"""

# Copyright 2024 Josephine Pockelé
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


__all__ = ['equivalent_pressure',
           'ospl',
           ]


def equivalent_pressure(signal: list | np.ndarray, fs: int | float | np.number, weighting: str = None) -> float:
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
    float
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

    # Determine the corresponding time for proper integration
    t = np.linspace(0, signal.size / fs, signal.size)

    # Determine the equivalent pressure
    return 1 / t[-1] * np.trapezoid(signal ** 2, t)


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
    pe2 = equivalent_pressure(signal, fs, weighting)

    return 10 * np.log10(pe2 / (2e-5 ** 2))
