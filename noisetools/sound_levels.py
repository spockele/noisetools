"""Functions related to the levels of measured or simulated sound signals.
"""
from fontTools.ufoLib.utils import deprecated
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

from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import scipy.signal as spsig
import pandas as pd
import numpy as np
import warnings

from .weighting_functions import weigh_signal
from .octave_band import OctaveBand

tableau = list(TABLEAU_COLORS.keys())
__all__ = ['p2eq', 'ospl', 'ospl_t', 't_out',
           'octave_spectrum', 'octave_spectrogram',
           'amplitude_modulation', ]

@deprecated
def equivalent_pressure(*args,
                        **kwargs,
                        ) -> float:
    """
    DEPRECATED

    Calculate the equivalent pressure (Pe^2) of the input sound signal.

    Parameters
    ----------
    *args
        See p2eq.
    **kwargs
        See p2eq

    Returns
    -------
    Equivalent pressure in Pa^2 (weighted to selected weighting)

    """
    warnings.warn("The function 'equivalent_pressure()' is renamed to 'p2eq'. "
                  "equivalent_pressure will be removed in a future version of noisetools",
                  DeprecationWarning
                  )
    return p2eq(*args, **kwargs)


def p2eq(signal: list | np.ndarray,
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
        sig = np.array(signal).copy()
    else:
        sig = signal.copy()

    # Check that this signal is 1d array
    if sig.ndim > 1:
        raise ValueError('noisetools.p2eq only supports 1d signal arrays.')

    # Apply the selected weighting
    if weighting is not None:
        sig = weigh_signal(sig, fs, curve=weighting)

    # Determine the equivalent pressure
    return 1 / sig.size * np.sum(sig ** 2)


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
    pe2 = p2eq(signal, fs, weighting, )

    return 10 * np.log10(pe2 / (2e-5 ** 2))


@deprecated
def ospl_t_out(*args,
               **kwargs,
               ) -> np.ndarray:
    """
    DEPRECATED

    Calculate the matching time series for the ospl_t function.

    Parameters
    ----------
    *args
        See t_out.
    **kwargs
        See t_out.

    Returns
    -------
    Time (seconds) at which the OSPL is calculated. Determined as the central timestamp
        in the sections of length delta_t.
    """
    warnings.warn("The function 'ospl_t_out()' is renamed to 't_out'."
                  "ospl_t_out will be removed in a future version of noisetools",
                  DeprecationWarning
                  )
    return t_out(*args, **kwargs)


def t_out(signal_size: int,
          fs: int | float | np.number,
          delta_t: float | np.number = 1.,
          complete: bool = True,
          centered: bool = True,
          ) -> np.ndarray:
    """
    Calculate the matching time series for the ospl_t function.

    Parameters
    ----------
    signal_size: int
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal.
    delta_t: float | np.number, optional (default=1.)
        Desired timestep in the OSPL output, in seconds.
    complete: bool, optional (default=True)
        In case the final timestep does not cover the full delta_t, this parameter indicates whether to still
        calculate the last step. Example: signal.size = 95500, fs=48000, delta_t = 1., then complete=True will result
        in two OSPL timesteps, while complete=False will result in only one OSPL timestep.
    centered: bool, optional (default=True)
        Centres the timestamps to the average time in each bin.
        Effectively, this shifts the time indication to t + delta_t / 2.

    Returns
    -------
    Time (seconds) at which the OSPL is calculated. Determined as the central timestamp
        in the sections of length delta_t.
    """
    if complete:
        t_array = np.arange(0, signal_size / fs, delta_t)
    else:
        t_array = np.arange(0, np.floor(signal_size / fs), delta_t)

    t_array = t_array + delta_t / 2 if centered else t_array

    return t_array


def p2eq_t(signal: list | np.ndarray,
           fs: int | float | np.number,
           weighting: str = None,
           delta_t: float | np.number = 1.,
           complete: bool = True,
           ) -> float:
    """
    Calculate the equivalent pressure (Pe^2) over time, of the input sound signal.

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.
    delta_t: float | np.number, optional (default=1.)
        Desired timestep in the OSPL output, in seconds.
    complete: bool, optional (default=True)
        In case the final timestep does not cover the full delta_t, this parameter indicates whether to still
        calculate the last step. Example: signal.size = 95500, fs=48000, delta_t = 1., then complete=True will result
        in two OSPL timesteps, while complete=False will result in only one OSPL timestep.

    Returns
    -------
    Equivalent pressure in Pa^2 (weighted to selected weighting) at the timestamps defined by delta_t.

    """
    # Convert signal to numpy array
    if not isinstance(signal, np.ndarray):
        sig = np.array(signal).copy()
    else:
        sig = signal.copy()

    # Check that this signal is 1d array
    if sig.ndim > 1:
        raise ValueError('noisetools.p2eq_t only supports 1d signal arrays.')

    # Apply the selected weighting
    if weighting is not None:
        sig = weigh_signal(sig, fs, curve=weighting)

    # Determine the timestep length in terms of samples.
    step = int(delta_t * fs)
    # Determine the number of full timesteps to compute.
    n_step = int(signal.size / step)

    # Separate the remainder from the signal.
    remainder = sig[n_step * step:]
    sig = sig[:n_step * step]

    # Integrate the signal per timestep.
    sig = sig.reshape((n_step, -1))
    pe2 = 1 / step * np.sum(sig ** 2, axis=1)

    # Add the equivalent pressure of the remainder if required.
    if complete and remainder.size > 0:
        pe2 = np.append(pe2, 1 / remainder.size * np.sum(remainder ** 2))

    return pe2


def ospl_t(signal: list | np.ndarray,
           fs: int | float | np.number,
           weighting: str = None,
           delta_t: float | np.number = 1.,
           complete: bool = True,
           ) -> np.ndarray:
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
    delta_t: float | np.number, optional (default=1.)
        Desired timestep in the OSPL output, in seconds.
    complete: bool, optional (default=True)
        In case the final timestep does not cover the full delta_t, this parameter indicates whether to still
        calculate the last step. Example: signal.size = 95500, fs=48000, delta_t = 1., then complete=True will result
        in two OSPL timesteps, while complete=False will result in only one OSPL timestep.

    Returns
    -------
    OPSL (dB) (weighted to selected weighting) at the timestamps defined by delta_t.

    """
    pe2 = p2eq_t(signal, fs, weighting, delta_t, complete)

    return 10 * np.log10(pe2 / (2e-5 ** 2))


def octave_index(fs: int | float | np.number,
                 octave: OctaveBand = None,
                 ) -> tuple[pd.MultiIndex, OctaveBand]:
    """
    Set up the OctaveBand instance and create the MultiIndex for the octave band OSPL functions.

    Parameters
    ----------
    fs: number
        The sampling frequency of the digital signal.
    octave: OctaveBand, optional
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.

    Returns
    -------
    A MultiIndex that contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    if octave is None:
        octave = OctaveBand()

    fs_fltr = octave.f.loc[:, 'f2'] < fs / 2
    out_index = pd.MultiIndex.from_arrays([octave.f.index[fs_fltr],
                                           octave.f.loc[fs_fltr, 'fm'],
                                           octave.f.loc[fs_fltr, 'df'],
                                           ],
                                          names=['band', 'fm', 'df'])

    return out_index, octave


def octave_spectrum(signal: list | np.ndarray,
                    fs: int | float | np.number,
                    weighting: str = None,
                    octave: OctaveBand = None,
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
    octave: OctaveBand, optional (default = OctaveBand())
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.

    Returns
    -------
    A pandas Series containing the sound pressure levels per octave band.
    The index contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    out_index, octave = octave_index(fs, octave)

    out_spectrum = pd.Series(index=out_index, dtype=float)

    for band_select in out_index.get_level_values('band'):
        band_signal = octave.filter_signal(signal, fs, band_select)
        out_spectrum.loc[band_select] = ospl(band_signal, fs, weighting)

    return out_spectrum


def octave_spectrogram(signal: list | np.ndarray,
                       fs: int | float | np.number,
                       weighting: str = None,
                       delta_t: float | np.number = 1.,
                       octave: OctaveBand = None,
                       complete: bool = True,
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
    delta_t: float | np.number, optional (default=1.)
        Desired timestep in the OSPL output, in seconds.
    octave: OctaveBand, optional (default = OctaveBand())
        Instance of the noisetools.octave_band.OctaveBand class to base the octave band spectrum on.
    complete: bool, optional (default=True)
        In case the final timestep does not cover the full delta_t, this parameter indicates whether to still
        calculate the last step. Example: signal.size = 95500, fs=48000, delta_t = 1., then complete=True will result
        in two OSPL timesteps, while complete=False will result in only one OSPL timestep.

    Returns
    -------
    A pandas DataFrame containing the sound pressure levels per octave band, over time.
    The columns are the time dimension. The index contains:
        - ```band```: the band number (-)
        - ```fm```: the corresponding central frequency (Hz)
        - ```df```: the octave band width (Hz)

    """
    # Convert signal to numpy array
    if not isinstance(signal, np.ndarray):
        sig = np.array(signal).copy()
    else:
        sig = signal.copy()

    # Check that this signal is 1d array
    if sig.ndim > 1:
        raise ValueError('noisetools.p2eq_t only supports 1d signal arrays.')

    # Apply the selected weighting
    if weighting is not None:
        sig = weigh_signal(signal, fs, curve=weighting)

    out_index, octave = octave_index(fs, octave)

    out_t = t_out(sig.size, fs, delta_t, complete=complete)
    out_spectrogram = pd.DataFrame(index=out_index, columns=out_t, dtype=float)

    for band_select in out_index.get_level_values('band'):
        band_signal = octave.filter_signal(sig, fs, band_select)
        out_spectrogram.loc[band_select, :] = ospl_t(band_signal, fs, None, delta_t, complete=complete)

    return out_spectrogram


def amplitude_modulation(signal: list | np.ndarray,
                         fs: int | float | np.number,
                         expected_bpf: tuple[int | float | np.number, int | float | np.number],
                         weighting: str = 'A',
                         freq_range: str | tuple[float, float] = 'reference',
                         verbose: bool = False,
                         ) -> pd.Series:
    """
    Implementation of the amplitude modulation algorithm described by Bass et al. [1]_

    Parameters
    ----------
    signal: array_like
        Array with the digital signal.
    fs: number
        The sampling frequency of the digital signal.
    expected_bpf: tuple[number, number]
        Range of frequencies between which the blade-pass frequency of the wind turbine is expected (Hz).
    weighting: str, optional
        The name of the optional weighting curve to be used. Can be 'A' or 'C'.
    freq_range: str | tuple[float, float], optional (default = 'reference')
        Indication of which frequency range from the IOA report to use (see also section 4.3.1 [1]_):
            - 'low': 50 - 200 Hz
            - 'reference': 100 - 400 Hz
            - 'high': 200 - 800 Hz
        An integer can be used to calculate the amplitude modulation in a single octave band.
    verbose: bool, optional
        Turn on print statements about the detection/acceptance of the BPF and harmonics.
        Also turns on plotting of the detrended SPL(t) and filtered SPL(t).

    Returns
    -------
    A pandas Series with the amplitude modulation in dB, every 10s (in accordance with Bass et al. [1]_).

    References
    ----------
    .. [1] J. Bass et al., ‘A Method for Rating Amplitude Modulation in Wind Turbine Noise’, Institute of Acoustics,
        Noise Working Group (Wind Turbine Noise), United Kingdom, Amplitude Modulation Working Group Final Report,
        Aug. 2016.

    """
    if isinstance(freq_range, str):
        # The relevant bands for the selected range of frequency.
        band_range = {'low': (-13, -7), 'reference': (-10, -4), 'high': (-7, -1)}[freq_range.lower()]
        octave = OctaveBand(3, band_range)
    elif isinstance(freq_range, tuple):
        octave = OctaveBand(3, freq_range=freq_range)
    else:
        raise TypeError(f'Invalid type for parameter freq_range. '
                        f'Expected str or tuple[float, float], got {type(freq_range)}.')

    # 0) Calculate the 1/3 octave band spectrogram with timestep 100ms.
    oct_spectrogram = octave_spectrogram(signal, fs, weighting, .1, octave)

    # Define the FFT frequencies of the FFT{SPL(t)}.
    fft_f = np.fft.rfftfreq(100, .1)
    # Create empty pandas Series to store the resulting modulation depths.
    modulation_depth = pd.Series()
    modulation_depth.index.name = 'Time (s)'

    ii = 0
    while ii + 100 <= oct_spectrogram.columns.size:
        # Gather a 10-second interval from the full signal spectrogram.
        oct_spectrogram_select = oct_spectrogram.iloc[:, ii:ii+100]
        # Sum logarithmically over the frequency bands to obtain one SPL(t) signal.
        spl_sig = 20 * np.log10(np.sum(10 ** (oct_spectrogram_select.to_numpy() / 20), axis=0))
        spl_t = oct_spectrogram_select.columns.copy()

        # 1) Detrend with a 3rd order polynomial.
        pp3 = np.polynomial.Polynomial.fit(spl_t, spl_sig, deg=3)
        spl_sig_det = spl_sig - pp3(spl_t)

        # 2) Calculate the FFT of the detrended SPL(t).
        spl_fft = np.fft.rfft(spl_sig_det, )

        # 3) Calculate the spectral density.
        spl_spectrum = np.abs(spl_fft) / (100**2)

        # 4) Step 1: Find peak frequencies of SPL(f).
        spl_peak_idx, _ = spsig.find_peaks(spl_spectrum, threshold=np.std(spl_spectrum))

        if verbose:
            plt.figure('Amplitude modulation')
            plt.plot(spl_t, spl_sig_det, color=tableau[ii//100], linewidth=.5)

            plt.figure(f'Spectrum {ii // 100}')
            plt.fill_betweenx((0, 1), 1 * expected_bpf[0], 1 * expected_bpf[1], color='k', alpha=.250, zorder=-100)
            plt.fill_betweenx((0, 1), 2 * expected_bpf[0], 2 * expected_bpf[1], color='k', alpha=.125, zorder=-100)
            plt.fill_betweenx((0, 1), 3 * expected_bpf[0], 3 * expected_bpf[1], color='k', alpha=.125, zorder=-100)
            plt.bar(fft_f, spl_spectrum, color=tableau[ii//100], width=.05, alpha=.25)
            plt.plot(fft_f[spl_peak_idx], spl_spectrum[spl_peak_idx], '|', color=tableau[ii//100])

        # Already calculate all local maxima for 6).
        spl_local_max = spsig.argrelmax(spl_spectrum)[0]

        # 7a) Create an array of zeroes to store the FFT for inversion.
        reconstruct_fft = np.zeros(spl_fft.shape, dtype=complex)

        found_bpf = False

        for peak_idx in spl_peak_idx:
            # 4) Step 2: Find a peak inside the given expected range of the BPF.
            if expected_bpf[0] <= fft_f[peak_idx] and fft_f[peak_idx] <= expected_bpf[1]:
                found_bpf = True

                # 5) Calculate the prominence of the peak and check if it is >= 4..
                level_peak = spl_spectrum[peak_idx]
                side_idx = [peak_idx - 3, peak_idx - 2, peak_idx + 2, peak_idx + 3]
                side_idx = [idx for idx in side_idx if idx >= 0]
                level_side = sum(spl_spectrum[side_idx]) / len(side_idx)

                if verbose:
                    plt.figure(f'Spectrum {ii // 100}')
                    plt.plot(fft_f[side_idx], spl_spectrum[side_idx], '_', color=tableau[ii//100])
                    plt.plot(fft_f[side_idx], len(side_idx) * [level_side, ], color=tableau[ii//100])

                if level_peak / level_side >= 4:
                    if verbose:
                        print(f'BPF prominent!')

                    # 7b) Add the BPF peak to the reconstruction FFT
                    reconstruct_fft[peak_idx - 1:peak_idx + 2] = spl_fft[peak_idx - 1:peak_idx + 2]

                    # Loop over the possible harmonics.
                    for harmonic in (2, 3):
                        # 6a) Set the index of the harmonic.
                        harmonic_idx = harmonic * peak_idx
                        harmonic_fft = np.zeros(fft_f.size, dtype=complex)

                        # 6b) Determine if harmonic is local maximum.
                        if harmonic_idx in spl_local_max:
                            if verbose:
                                print(f'\tHarmonic {harmonic - 1} is local max!')

                            # 6c,i) Generate time-series from harmonic.
                            harmonic_fft[harmonic_idx - 1:harmonic_idx + 2] = spl_fft[harmonic_idx - 1:harmonic_idx + 2]
                            harmonic_sig = np.fft.irfft(harmonic_fft)

                            # 6c,ii) Check if harmonic peak-to-peak amplitude is >= 1.5 dB.
                            if np.max(harmonic_sig) - np.min(harmonic_sig) >= 1.5:
                                if verbose:
                                    print(f'\t > Harmonic {harmonic - 1} accepted!')
                                # 7c) Add harmonic to the reconstruction FFT.
                                reconstruct_fft[harmonic_idx - 1:harmonic_idx + 2] = spl_fft[harmonic_idx - 1:harmonic_idx + 2]
                            elif verbose:
                                # 6d) Reject harmonic if not all conditions are met.
                                print(f'\t > Harmonic {harmonic - 1} rejected...')

                        # 6b,bis) Determine if lines next to harmonic are local maxima.
                        else:
                            # Initialisers for the loop.
                            harmonic_mag = 0
                            n_accept = None

                            # Go over allowable N.
                            for n in range(-(harmonic-1), harmonic):
                                n_idx = harmonic_idx + n
                                # Check if this step next to the harmonic is a local maximum,
                                #  and if it is larger than a previously detected local maximum.
                                if n and (n_idx in spl_local_max and spl_spectrum[n_idx] > harmonic_mag):
                                    n_accept = n

                            if n_accept is not None:
                                if verbose:
                                    print(f'\t Harmonic {harmonic - 1} (step {n_accept}) is local max!')
                                n_idx = harmonic_idx + n_accept

                                # 6c,i) Generate time-series from shifted harmonic.
                                harmonic_fft[n_idx - 1:n_idx + 2] = spl_fft[n_idx - 1:n_idx + 2]
                                harmonic_sig = np.fft.irfft(harmonic_fft)

                                # 6c,ii) Check if harmonic peak-to-peak amplitude is >= 1.5 dB.
                                if np.max(harmonic_sig) - np.min(harmonic_sig) >= 1.5:
                                    if verbose:
                                        print(f'\t > Harmonic {harmonic - 1} accepted!')
                                    # 7c) Add harmonic to the reconstruction FFT.
                                    reconstruct_fft[n_idx - 1:n_idx + 2] = spl_fft[n_idx - 1:n_idx + 2]
                                elif verbose:
                                    # 6d) Reject harmonic if not all conditions are met.
                                    print(f'\t > Harmonic {harmonic - 1} rejected...')

                    # Once the BPF is found, no more peaks should be explored.
                    continue

                # 5,bis) In case prominence < 4:
                elif verbose:
                    print('BPF not prominent...')

        # 4,bis) The BPF is not detected in the expected range.
        if not found_bpf and verbose:
            print('BPF not found...')

        # 7d) Inverse FFT of the newly constructed array.
        spl_reconstruct = np.fft.irfft(reconstruct_fft, )

        # 9) Determine the modulation depth.
        l5, l95 = np.percentile(spl_reconstruct, [5, 95])
        modulation_depth.loc[ii // 10] = abs(l95 - l5)

        # 8) Resulting signal.
        if verbose:
            plt.figure(f'Spectrum {ii // 100}')
            plt.bar(fft_f, np.abs(reconstruct_fft) / (100**2), color=tableau[ii//100], width=.05)
            plt.figure('Amplitude modulation')
            plt.plot(spl_t, spl_reconstruct, color=tableau[ii//100])

            plt.plot(spl_t, len(spl_t) * [l5, ], ':', color=tableau[ii // 100])
            plt.plot(spl_t, len(spl_t) * [l95, ], ':', color=tableau[ii // 100])

        ii += 100

    if verbose:
        plt.figure('Amplitude modulation')
        plt.ylim(-5, 5)

        for ii in range(ii // 100):
            plt.figure(f'Spectrum {ii}')
            plt.ylim(0, .01)
        plt.show()

    return modulation_depth
