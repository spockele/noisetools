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

import pandas as pd
import numpy as np

from .weighting_functions import weigh_signal
from .octave_band import OctaveBand


__all__ = ['equivalent_pressure', 'ospl', 'ospl_t', 'octave_spectrum', 'octave_spectrogram']


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


def octave_index(fs: int | float | np.number,
                 octave: OctaveBand = None,
                 order: int = 3,
                 ) -> tuple[pd.MultiIndex, OctaveBand]:
    """
    Set up the OctaveBand instance and create the MultiIndex for the octave band OSPL functions.

    Parameters
    ----------
    fs: number
        The sampling frequency of the digital signal.
    octave: OctaveBand, optional
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.
    order: int, optional (default = 3)
        Order b of the octave bands. This will result in 1/b octave bands.
        This parameter will be ignored if parameter ```octave``` is provided.

    Returns
    -------
    A MultiIndex that contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    if octave is None:
        octave = OctaveBand(order=order)

    fs_fltr = octave.f.loc[:, 'f2'] < fs / 2
    out_index = pd.MultiIndex.from_arrays([octave.f.index[fs_fltr][:-1],
                                           octave.f.loc[fs_fltr, 'fm'][:-1],
                                           octave.f.loc[fs_fltr, 'df'][:-1],
                                           ],
                                          names=['band', 'fm', 'df'])

    return out_index, octave


def octave_spectrum(signal: list | np.ndarray,
                    fs: int | float | np.number,
                    weighting: str = None,
                    octave: OctaveBand = None,
                    order: int = 3,
                    ) -> pd.Series:
    """
    Calculate the OSPL of a signal per octave band.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal.
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.
    octave: OctaveBand, optional
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.
    order: int, optional (default = 3)
        Order b of the octave bands. This will result in 1/b octave bands.
        This parameter will be ignored if parameter ```octave``` is provided.

    Returns
    -------
    A pandas Series containing the sound pressure levels per octave band.
    The index contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    out_index, octave = octave_index(fs, octave, order)

    out_spectrum = pd.Series(index=out_index, dtype=float)

    for band_select in out_index.get_level_values('band'):
        band_signal = octave.filter_signal(signal, fs, band_select)
        out_spectrum.loc[band_select] = ospl(band_signal, fs, weighting)

    return out_spectrum


def octave_spectrogram(signal: list | np.ndarray,
                       fs: int | float | np.number,
                       weighting: str = None,
                       t: list | np.ndarray = None,
                       delta_t: float | np.number = 1.,
                       octave: OctaveBand = None,
                       order: int = 3,
                       ) -> pd.DataFrame:
    """
    Calculate the OSPL over time, of a digital signal, per octave band.

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
    octave: OctaveBand, optional
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.
    order: int, optional (default = 3)
        Order b of the octave bands. This will result in 1/b octave bands.
        This parameter will be ignored if parameter ```octave``` is provided.

    Returns
    -------
    A pandas DataFrame containing the sound pressure levels per octave band, over time.
    The columns are the time dimension. The index contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    out_index, octave = octave_index(fs, octave, order)

    out_t, _ = ospl_t(signal, fs, weighting, t, delta_t)
    out_spectrogram = pd.DataFrame(index=out_index, columns=out_t, dtype=float)

    for band_select in out_index.get_level_values('band'):
        band_signal = octave.filter_signal(signal, fs, band_select)
        out_spectrogram.loc[band_select, :] = ospl_t(band_signal, fs, weighting, t, delta_t)[1]

    return out_spectrogram


def amplitude_modulation(signal: list | np.ndarray,
                         fs: int | float | np.number,
                         weighting: str = None,
                         ) -> None:
    pass
