"""Octave band related operations.
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

from typing import Iterable, Literal
import pandas as pd
import numpy as np


class OctaveBand:
    """
    Class to handle operations with Octave bands. Compliant with the IEC 61260-1:2014 standard. [1]_

    Parameters
    ----------
    order: int, optional (default = 3)
        Order b of the octave bands. This will result in 1/b octave bands.
    band_range: tuple[int, int], optional
        Range of band numbers to use for this instance. Required parameter for order not in [1, 3, 6, 12]!
        Defaults to None, which results in a band range to cover frequencies from 2 Hz to 26 kHz.

    Attributes
    ----------
    f: pd.DataFrame
        DataFrame containing the frequency information of this OctaveBand object.
        Index contains the band numbers. Columns 'fc', 'fl', 'fu' contain the central, lower, and upper frequencies of
        the bands, respectively. 'df' defines the bandwidth of each band.

    References
    ----------
    .. [1] International Electrotechnical Commission (IEC), ‘Electroacoustics - Octave-band and fractional-octave-band
        filters - Part 1: Specifications’, Geneva, Switzerland, International Standard IEC 61260-1:2013, Feb. 2014.

    """
    def __init__(self,
                 order: int = 3,
                 band_range: tuple[int, int] = None,
                 ) -> None:
        if band_range is None:
            if order == 1:
                bands = np.arange(-9, 5)
            elif order == 3:
                bands = np.arange(-27, 15)
            elif order == 6:
                bands = np.arange(-54, 29)
            elif order == 12:
                bands = np.arange(-108, 61)
            else:
                raise ValueError(f'Default band ranges only defined for orders (1, 3, 6, 12). For order={order}, '
                                 f'define band_range.')
        else:
            bands = np.arange(*band_range)

        self.order = order
        self.band_range = band_range
        self.f = pd.DataFrame(index=pd.Index(bands, name='band nr.'), columns=['fl', 'fc', 'fu', 'df'])

        g = 10 ** (3 / 10)
        self.f.loc[:, 'fc'] = 1e3 * g ** (self.f.index / order)
        self.f.loc[:, 'fl'] = self.f.loc[:, 'fc'] * g ** (-1 / (2 * order))
        self.f.loc[:, 'fu'] = self.f.loc[:, 'fc'] * g ** (1 / (2 * order))
        self.f.loc[:, 'df'] = self.f.loc[:, 'fu'] - self.f.loc[:, 'fl']

    def f_to_band(self,
                  f: np.ndarray | Iterable,
                  ) -> np.ndarray:
        """
        Convert an array of frequencies to their band numbers.

        Parameters
        ----------
        f: numpy.ndarray | Iterable
            Array-like object containing the frequencies (Hz) to convert to band numbers.

        Returns
        -------
        A numpy array containing the band numbers. Array will have same shape as the input array.

        """
        f: np.ndarray = np.array(f, dtype=float)
        bands = np.empty(f.shape, dtype=float)
        bands[:] = np.nan

        for bi, band in enumerate(self.f.index):
            band_select = (self.f.loc[band, 'fl'] < f) & (f < self.f.loc[band, 'fu'])
            bands[band_select] = band

            bands[f == self.f.loc[band, 'fl']] = band - .5

        return bands.astype(int)

    def interp1d_to_narrowband(self,
                               f_interpolate: np.ndarray | Iterable,
                               f_octave: np.ndarray | Iterable,
                               dat_octave: np.ndarray | Iterable,
                               extrapolate: Literal['zero', 'constant', 'slope_to_zero'] = 'zero',
                               ) -> np.ndarray:
        """
        1D interpolation of an octave band spectrum to a narrowband spectrum, to constant values per band.
        Note: this function is designed to interpolated attenuation/amplification spectra.
        For interpolation from octave band dB data to narrowband dB/Hz data, convert dat_octave to dB/Hz beforehand.

        Parameters
        ----------
        f_interpolate: numpy.ndarray | Iterable
            Array-like object containing the frequencies (Hz) to interpolate the octave band data to.
        f_octave: numpy.ndarray | Iterable
            Array-like object containing frequencies (Hz) which represent the octave bands of the input data.
            These frequencies are converted to octave band numbers using OctaveBand.f_to_band(f_octave).
        dat_octave: numpy.ndarray | Iterable
            Array-like object containing the octave band data to be interpolated.
        extrapolate: str, Literal, optional (default = 'zero')
            Indicator for the extrapolation method to be used beyond f_octave and inside the range defined
            by OctaveBand.band_range. This option will fill bands beyond those defined by f_octave:
             - 'zero': with zeroes.
             - 'constant': with the extreme values.
             - 'slope_to_zero': with values linearly interpolated from the extreme values to zero at the edges of
                                the band range. This creates a 'staircase' in the bands outside f_octave.

        Returns
        -------
        Interpolated values.

        """
        f_interpolate: np.ndarray = np.array(f_interpolate)
        f_octave: np.ndarray = np.array(f_octave)
        dat_octave: np.ndarray = np.array(dat_octave)

        if f_interpolate.ndim > 1 or f_octave.ndim > 1 or dat_octave.ndim > 1:
            raise ValueError(f'This function only supports 1D interpolation to narrowband.')

        # Initialise a results array.
        result = np.zeros(f_interpolate.size, dtype=float)
        # Convert the octave band frequencies to band numbers.
        bands_octave = self.f_to_band(f_octave).astype(float)

        # slope_to_zero extrapolation mode.
        if extrapolate.lower() == 'slope_to_zero':
            # In case there are undefined bands in the input data in the low range:
            if bands_octave[0] > self.f.index[0]:
                # Add a zero point before the first band.
                bands_octave = np.append([self.f.index[0] - 1, ], bands_octave)
                dat_octave = np.append([0., ], dat_octave)
            # In case there are undefined bands in the input data in the high range:
            if bands_octave[-1] < self.f.index[-1]:
                # Add a zero point after the last band.
                bands_octave = np.append(bands_octave, self.f.index[-1] + 1)
                dat_octave = np.append(dat_octave, 0.)
            # Interpolate the input data.
            dat_intermediate = np.interp(self.f.index, bands_octave, dat_octave)

        # constant extrapolation mode.
        elif extrapolate.lower() == 'constant':
            # Just use the basic numpy 1d interpolation, because this has the desired extrapolation behaviour.
            dat_intermediate = np.interp(self.f.index, bands_octave, dat_octave)

        # zero extrapolation mode.
        elif extrapolate.lower() == 'zero':
            # Initialise the interpolated data array.
            dat_intermediate = np.zeros(self.f.index.size)
            # Define the non-zero area for interpolation.
            non_zero = (f_octave[0] <= self.f['fu']) & (self.f['fl'] <= f_octave[-1])
            # Interpolate the non-zero area linearly.
            dat_intermediate[non_zero] = np.interp(self.f.index[non_zero], bands_octave, dat_octave)

        # Complain about any other selected mode.
        else:
            raise ValueError(f"Unknown extrapolation method '{extrapolate}'. "
                             f"Select from ('zero', 'constant', 'slope_to_zero').")

        # TODO: check if this interpolation results in undesired behaviour in case there are two datapoints per band
        #       in f_octave and dat_octave. Hypothesis: no, because intermediate linear interpolation to self.f.index.
        for bi, band in enumerate(self.f.index):
            # Select the band based on the frequency.
            band_select = (self.f.loc[band, 'fl'] <= f_interpolate) & (f_interpolate < self.f.loc[band, 'fu'])
            # Add the data from this band to the result array.
            result[band_select] = dat_intermediate[bi]

        return result
