"""Beamforming functions based on the descriptions by Merino-Martinez [1]_.

References
----------
.. [1] R. Merino-Martínez, ‘Microphone arrays for imaging of aerospace noise sources’, Doctoral thesis,
    Delft University of Technology, Delft, Netherlands, 2018. doi: 10.4233/uuid:a3231ea9-1380-44f4-9a93-dbbd9a26f1d6.
"""

# Copyright 2026 Josephine Pockelé
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
from typing import Literal
import pandas as pd
import numpy as np


def _cross_spectral_matrix(dat: np.ndarray | pd.DataFrame,
                           stft: spsig.ShortTimeFFT,
                           f_range: tuple[int | float | np.number, int | float | np.number] | None = None,
                           cal: np.ndarray | pd.DataFrame | None = None,
                           f_cal: np.ndarray | None = None,
                           ):
    """

    Parameters
    ----------
    dat: numpy.ndarray | pandas.DataFrame
        Microphone signal data in a 2-dimensional array or DataFrame. Microphones on axis 0, and time on axis 1.
    stft: scipy.signal.ShortTimeFFT
        Instance of the ShortTimeFFT class that matches the sampling frequency of the entered data. This instance will
        be used to calculate the STFT of the signals in dat.
    f_range: tuple[int | float | np.number, int | float | np.number], optional (default = None)
        Tuple with minimum and maximum frequency (Hz) for which to calculate the CSM and return a masking array.
    cal: numpy.ndarray | pandas.DataFrame, optional (default = None)
        Calibration data in a 2-dimensional array or DataFrame. Microphones on axis 0, and frequency on axis 1.
        Frequencies do not have to match the instance of ShortTimeFFT.
    f_cal: numpy.ndarray, optional (default = None)
        Frequencies of the calibration data. Should match axis 1 of cal.

    Returns
    -------
    fltr: 1D numpy.ndarray containing a mask to filter future results to match f_range.
    csm: 3D numpy.ndarray containing the CSMs of the entered data, at all frequencies of the ShortTimeFFT instance.
        Axis 0 matches the filtered frequencies of stft.
        Axis 1 and 2 have a shape matching the CSM for the number of microphones in the data array.
        To select the CSM of a given frequency at index fi, use csm[fi].
    """

    # Convert the data and calibration dataframes to Numpy arrays
    if isinstance(dat, pd.DataFrame):
        dat = dat.values
    if isinstance(cal, pd.DataFrame):
        cal = cal.values

    # Create the filter array to limit the frequency range.
    if f_range is None:
        fltr = np.ones(stft.f.size, dtype=bool)
    else:
        fltr = (f_range[0] <= stft.f) & (stft.f <= f_range[1])

    # Prepare the array to store the STFT.
    x_fft = np.zeros((stft.f[fltr].size, dat.shape[0], stft.t(dat.shape[1]).size, ), dtype=complex)

    # Loop over all microphones
    for mi in range(dat.shape[0]):
        # Determine the STFT of the signal.
        x_fft[:, mi, :] = stft.stft(dat[mi, :])[fltr, :]

        # Apply the calibration values (linearly interpolated to match STFT frequencies).
        if cal is not None:
            x_fft[:, mi, :] *= np.interp(stft.f[fltr], f_cal, cal[mi, :]).reshape((-1, 1))

    # Determine the time-averaged CSM <XX*>
    # x_fft has 3 axes: (frequency (f), microphone (m), time (t)).
    # This is expanded with an extra axis to: (f, m, 1, t) and (f, 1, m, t)
    # The multiplication results in axes: (f, m, m, t), where the (m, m) axes contain the CSMs. Thus (f, CSM(f,t), t).
    # This is time averaged to obtain the time averaged CSM for all frequencies. Thus (f, m, m) -> (f, CSM(f))
    csm = np.mean(x_fft[:, :, np.newaxis, :] * x_fft[:, np.newaxis, :, :].conj(), axis=3)

    return fltr, csm


def _steering_vector_1d(f: int | float | np.number,
                        scan_xy: tuple[int | float | np.number, int | float | np.number],
                        scan_z: int | float | np.number,
                        mic_xy: tuple[int | float | np.number, int | float | np.number],
                        mic_z: np.ndarray,
                        norm: bool = False,
                        ):
    """
    Calculate the steering vector for 1D beamforming of a singular scan point.

    Parameters
    ----------
    f: int | float | numpy.number
        Single frequency to determine the steering vector for.
    scan_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the scan line.
    scan_z: int | float | numpy.number
        z coordinate (in metres) of the scan point on the scan line
    mic_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the microphone array line.
    mic_z: numpy.ndarray
        z coordinates (in metres) of the individual microphones on the array line.
    norm: bool, optional (default = False)
        Boolean to toggle normalisation of the steering vector

    Returns
    -------
    1D Numpy.ndarray containing the requested array steering vector.
    """

    # Calculate the distance(s) between the scan point and the microphone(s).
    r = np.sqrt((mic_xy[0] - scan_xy[0]) ** 2 + (mic_xy[1] - scan_xy[1]) ** 2 + (mic_z - scan_z) ** 2)
    # Determine the distance-corrected steering vector.
    g = (np.exp(-2j * np.pi * f * r / 343.) / r).reshape((-1, 1))

    # Return normalised vector if this is selected.
    if norm:
        return g / (g.T.conj() @ g).item()
    else:
        return g


def cfdbf_1d(dat: np.ndarray | pd.DataFrame,
             stft: spsig.ShortTimeFFT,
             scan_xy: tuple[int | float | np.number, int | float | np.number],
             scan_z: np.ndarray,
             mic_xy: tuple[int | float | np.number, int | float | np.number],
             mic_z: np.ndarray,
             f_range: tuple[int | float | np.number, int | float | np.number] | None = None,
             cal: np.ndarray | pd.DataFrame | None = None,
             f_cal: np.ndarray | None = None,
             dz_spi: int | float | np.number | None = None,
             nz_spi: int | np.integer = 11,
             verbose: bool = False,
             ):
    """

    Parameters
    ----------
    dat: numpy.ndarray | pandas.DataFrame
        Microphone signal data in a 2-dimensional array or DataFrame. Microphones on axis 0, and time on axis 1.
    stft: scipy.signal.ShortTimeFFT
        Instance of the ShortTimeFFT class that matches the sampling frequency of the entered data. This instance will
        be used to calculate the STFT of the signals in dat.
    scan_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the scan line.
    scan_z: int | float | numpy.number
        z coordinate (in metres) of the scan point on the scan line
    mic_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the microphone array line.
    mic_z: numpy.ndarray
        z coordinates (in metres) of the individual microphones on the array line.
    f_range: tuple[int | float | np.number, int | float | np.number], optional (default = None)
        Tuple with minimum and maximum frequency (Hz) for which to calculate the CSM and return a masking array.
    cal: numpy.ndarray | pandas.DataFrame, optional (default = None)
        Calibration data in a 2-dimensional array or DataFrame. Microphones on axis 0, and frequency on axis 1.
        Frequencies do not have to match the instance of ShortTimeFFT.
    f_cal: numpy.ndarray, optional (default = None)
        Frequencies of the calibration data. Should match axis 1 of cal.
    dz_spi
    nz_spi
    verbose

    Returns
    -------

    """
    fltr, csm = _cross_spectral_matrix(dat, stft, f_range, cal, f_cal)

    beamform = np.zeros((stft.f[fltr].size, scan_z.shape[0]), dtype=complex)

    for fi, f in enumerate(stft.f[fltr]):
        if not fi % 50 and fi and verbose:
            print(f'Beamforming (CFDBF) @ {str(round(f, 1)).rjust(8, ' ')} Hz')

        if scan_z.ndim == 2:
            scan_z_f = scan_z[:, fi]
        else:
            scan_z_f = scan_z.copy()

        for si, sz in enumerate(scan_z_f):
            if dz_spi is not None:
                scan_z_spi = np.linspace(sz - dz_spi, sz + dz_spi, nz_spi, dtype=float)

                wcw = np.empty(nz_spi, dtype=complex)
                wgw = np.empty(nz_spi, dtype=complex)

                rk = np.sqrt((mic_xy[0] - scan_xy[0]) ** 2 + (mic_xy[1] - scan_xy[0]) ** 2 + (mic_z - sz) ** 2)
                gk = (np.exp(-2j * np.pi * f * rk / 343.) / rk).reshape((-1, 1))

                for szi, szs in enumerate(scan_z_spi):
                    w = _steering_vector_1d(f, scan_xy, szs, mic_xy, mic_z, norm=True)
                    wcw[szi] = (w.T.conj() @ csm[fi] @ w).item()
                    wgw[szi] = (w.T.conj() @ (gk @ gk.T.conj()) @ w).item()

                beamform[fi, si] = np.sum(wcw) / np.sum(wgw)

            else:
                w = _steering_vector_1d(f, scan_xy, sz, mic_xy, mic_z, norm=True)
                beamform[fi, si] = (w.T.conj() @ csm[fi] @ w).item()

    return stft.f[fltr], scan_z, 10 * np.log10(np.abs(beamform) / 4e-10)


def functional_1d(dat: np.ndarray | pd.DataFrame,
                  stft: spsig.ShortTimeFFT,
                  nu: int | float | np.number,
                  scan_xy: tuple[int | float | np.number, int | float | np.number],
                  scan_z: np.ndarray,
                  mic_xy: tuple[int | float | np.number, int | float | np.number],
                  mic_z: np.ndarray,
                  f_range: tuple[int | float | np.number, int | float | np.number] | None = None,
                  cal: np.ndarray | pd.DataFrame | None = None,
                  f_cal: np.ndarray | None = None,
                  verbose: bool = False,
                  ):
    """

    Parameters
    ----------
    dat: numpy.ndarray | pandas.DataFrame
        Microphone signal data in a 2-dimensional array or DataFrame. Microphones on axis 0, and time on axis 1.
    stft: scipy.signal.ShortTimeFFT
        Instance of the ShortTimeFFT class that matches the sampling frequency of the entered data. This instance will
        be used to calculate the STFT of the signals in dat.
    nu
    scan_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the scan line.
    scan_z: int | float | numpy.number
        z coordinate (in metres) of the scan point on the scan line
    mic_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the microphone array line.
    mic_z: numpy.ndarray
        z coordinates (in metres) of the individual microphones on the array line.
    f_range: tuple[int | float | np.number, int | float | np.number], optional (default = None)
        Tuple with minimum and maximum frequency (Hz) for which to calculate the CSM and return a masking array.
    cal: numpy.ndarray | pandas.DataFrame, optional (default = None)
        Calibration data in a 2-dimensional array or DataFrame. Microphones on axis 0, and frequency on axis 1.
        Frequencies do not have to match the instance of ShortTimeFFT.
    f_cal: numpy.ndarray, optional (default = None)
        Frequencies of the calibration data. Should match axis 1 of cal.
    verbose

    Returns
    -------

    """
    fltr, csm = _cross_spectral_matrix(dat, stft, f_range, cal, f_cal)

    beamform = np.zeros((stft.f[fltr].size, scan_z.shape[0]), dtype=complex)

    for fi, f in enumerate(stft.f[fltr]):
        if not fi % 50 and fi and verbose:
            print(f'Beamforming (Functional nu = {nu}) @ {str(round(f, 1)).rjust(8, ' ')} Hz')

        sm, um = np.linalg.eig(csm[fi])
        mat = um @ np.diag(sm ** (1 / nu)) @ um.T.conj()

        for si, sz in enumerate(scan_z):
            g = _steering_vector_1d(f, scan_xy, sz, mic_xy, mic_z)
            gn2 = (g.T.conj() @ g).item()

            beamform[fi, si] = (((g.T.conj() @ mat @ g).item() / gn2) ** nu) / gn2

    return stft.f[fltr], scan_z, 10 * np.log10(np.abs(beamform) / 4e-10)


def clean_1d(dat: np.ndarray | pd.DataFrame,
             stft: spsig.ShortTimeFFT,
             mode: Literal['psf', 'sc'],
             scan_xy: tuple[int | float | np.number, int | float | np.number],
             scan_z: np.ndarray,
             mic_xy: tuple[int | float | np.number, int | float | np.number],
             mic_z: np.ndarray,
             f_range: tuple[int | float | np.number, int | float | np.number] | None = None,
             cal: np.ndarray | pd.DataFrame | None = None,
             f_cal: np.ndarray | None = None,
             loop_gain: float | np.floating | None = None,
             csm_tol: float | np.floating = 1e-4,
             max_iter: int | np.integer = 100,
             clean_beam: float | np.floating | None = None,
             verbose: bool = False,
             ):
    """

    Parameters
    ----------
    dat: numpy.ndarray | pandas.DataFrame
        Microphone signal data in a 2-dimensional array or DataFrame. Microphones on axis 0, and time on axis 1.
    stft: scipy.signal.ShortTimeFFT
        Instance of the ShortTimeFFT class that matches the sampling frequency of the entered data. This instance will
        be used to calculate the STFT of the signals in dat.
    mode
    scan_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the scan line.
    scan_z: int | float | numpy.number
        z coordinate (in metres) of the scan point on the scan line
    mic_xy: tuple[int | float | np.number, int | float | np.number]
        x and y coordinates (in metres) of the microphone array line.
    mic_z: numpy.ndarray
        z coordinates (in metres) of the individual microphones on the array line.
    f_range: tuple[int | float | np.number, int | float | np.number], optional (default = None)
        Tuple with minimum and maximum frequency (Hz) for which to calculate the CSM and return a masking array.
    cal: numpy.ndarray | pandas.DataFrame, optional (default = None)
        Calibration data in a 2-dimensional array or DataFrame. Microphones on axis 0, and frequency on axis 1.
        Frequencies do not have to match the instance of ShortTimeFFT.
    f_cal: numpy.ndarray, optional (default = None)
        Frequencies of the calibration data. Should match axis 1 of cal.
    loop_gain
    csm_tol
    max_iter
    clean_beam
    verbose

    Returns
    -------

    """
    fltr, csm = _cross_spectral_matrix(dat, stft, f_range, cal, f_cal)

    if loop_gain is None:
        loop_gain = 1. if mode == 'psf' else .99

    diag_mask = 1 - np.diag(np.ones(csm.shape[1]))
    source_map = np.zeros((stft.f[fltr].size, scan_z.shape[0]), dtype=complex)

    ci = 0
    for fi, f in enumerate(stft.f[fltr]):
        ci -= 1
        beamform = np.zeros((scan_z.shape[0]), dtype=complex)

        csm_norm = np.linalg.norm(csm[fi] * diag_mask)
        csm_deg = csm[fi].copy()
        crit = True
        ii = -1
        while crit and ii < int(max_iter / loop_gain):
            for si, sz in enumerate(scan_z):
                w = _steering_vector_1d(f, scan_xy, sz, mic_xy, mic_z, norm=True)
                beamform[si] = (w.T.conj() @ csm_deg @ w).item()

            idx_max = np.argmax(beamform)
            if clean_beam is None:
                source_map[fi, idx_max] += loop_gain * beamform[idx_max]
            else:
                source_map[fi, :] += loop_gain * beamform[idx_max] * 2 ** (-((scan_z - scan_z[idx_max]) / clean_beam) ** 2)

            if mode == 'sc':
                w_max = _steering_vector_1d(f, scan_xy, scan_z[idx_max], mic_xy, mic_z, norm=True)
                h_max = csm_deg @ w_max / beamform[idx_max]

            elif mode == 'psf':
                h_max = _steering_vector_1d(f, scan_xy, scan_z[idx_max], mic_xy, mic_z, norm=False)

            csm_deg_norm = np.linalg.norm(csm_deg * diag_mask)
            csm_deg = csm_deg - loop_gain * beamform[idx_max] * (h_max @ h_max.T.conj())

            crit = (abs(np.linalg.norm(csm_deg * diag_mask) - csm_deg_norm) / csm_norm) > csm_tol * loop_gain

            ii += 1
            ci += 1

        n_skip = round(50 * loop_gain)
        if not fi % n_skip and fi and verbose:
            print(f'Beamforming (CLEAN-{mode.upper()}) @ {str(round(f, 1)).rjust(8, ' ')} Hz (~{str(round(ci / n_skip, 1)).rjust(6, ' ')} iter. / freq.)')
            ci = 0

        for si, sz in enumerate(scan_z):
            w = _steering_vector_1d(f, scan_xy, sz, mic_xy, mic_z, norm=True)
            source_map[fi, si] += (w.T.conj() @ csm_deg @ w).item()

    return stft.f[fltr], scan_z, 10 * np.log10(np.abs(source_map) / 4e-10)
