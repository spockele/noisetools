"""Functions to generate tonal signals and write them to wav files
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

import scipy.io as spio
import numpy as np


__all__ = ['generate_tone', 'generate_tones', ]


def generate_tone(f: int | float | np.number,
                  a: int | float | np.number,
                  phase: int | float | np.number,
                  length: int | float | np.number,
                  fs: int | float | np.number = 48e3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a tonal signal of a single frequency.

    Parameters
    ----------
    f : number
        Frequency of the tone in Hz.
    a : number
        Amplitude of the tone.
    phase : number
        Phase of the tone in radians.
    length : number
        Length of the tonal signal in seconds. Final signal will have duration int(fs * length), which may not match the
        given length.
    fs : number, optional
        Sampling frequency of the tonal signal in Hz. Defaults to 48 kHz.

    Returns
    -------
    Time signal and generated tonal signal.

    """
    # Determine the time series
    t = np.linspace(0, length, int(fs * length))
    # Generate the signal
    sig = a * np.sin(2 * np.pi * f * t + phase)

    return t, sig


def generate_tones(f: int | float | np.number | np.ndarray | list,
                   length: int | float | np.number,
                   a: int | float | np.number | np.ndarray | list = None,
                   phase: int | float | np.number | np.ndarray | list = None,
                   fs: int | float | np.number = 48e3,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a signal composed of a number of tones.

    Parameters
    ----------
    f : number or array_like
        Frequency or list of frequencies at which to generate tones in Hz.
    length : number
        Length of the signal in seconds. Final signal will have duration int(fs * length), which may not match the
        given length.
    a : number or array_like, optional
        Amplitude(s) of the generated tone(s). Defaults to 1.
    phase : number or array_like, optional
        Sets the relative phases of the different tones in the signal. Defaults to zero phase.
    fs : number, optional
        Sampling frequency of the signal in Hz. Defaults to 48 kHz.

    Returns
    -------
    Time signal and generated tonal signal.

    """
    # Check if the parameter f is valid and do necessary conversions to numpy.ndarray
    if isinstance(f, int) or isinstance(f, float) or isinstance(f, np.number):
        f = np.array([f,])
    elif isinstance(f, list):
        f = np.array(f)

    def check_var_f(var: int | float | np.number | np.ndarray | list | None,
                    name: str) -> np.ndarray:
        """
        Internal check function for the optional parameters in relation to f.

        Parameters
        ----------
        var : number or array_like
            The input parameter to check against f.
        name : str
            Name of the parameter under scrutiny
        """
        if var is None:
            if name == 'a':
                var = np.ones(f.shape)
            elif name == 'phase':
                var = np.zeros(f.shape)

        elif isinstance(var, int) or isinstance(var, float) or isinstance(f, np.number):
            var = np.array([var, ])
        elif isinstance(var, list):
            var = np.array(var)

        if var.shape != f.shape:
            raise ValueError(f'Shape mismatch between f and {name}. Got {f.shape} and {var.shape}.')

        return var

    a = check_var_f(a, 'a')
    phase = check_var_f(phase, 'phase')

    tme = np.linspace(0, length, int(fs * length))
    sig_total = np.empty((int(fs * length), ))

    for fi, frequency in enumerate(f.flatten()):
        frequency: float = float(frequency)
        amplitude: float = float(a.flatten()[fi])
        ph: float = float(phase.flatten()[fi])

        t, sig = generate_tone(frequency, amplitude, ph, length, fs)

        if t.shape != tme.shape:
            raise RuntimeError(f'Shape mismatch between overall time series and tone generator time series.')

        sig_total += sig

    return tme, sig_total
