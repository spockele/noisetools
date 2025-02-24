"""Octave band related operations.
"""
import matplotlib.pyplot as plt
from typing import Iterable, Literal
import pandas as pd
import numpy as np


class OctaveBand:
    def __init__(self, order: int = 3, band_range: tuple = None):
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

        self.f = pd.DataFrame(index=pd.Index(bands, name='band nr.'), columns=['fl', 'fc', 'fu', 'df'])

        self.f.loc[:, 'fc'] = 1e3 * 2 ** (self.f.index / order)
        self.f.loc[:, 'fl'] = self.f.loc[:, 'fc'] / 2 ** (1 / (2 * order))
        self.f.loc[:, 'fu'] = self.f.loc[:, 'fc'] * 2 ** (1 / (2 * order))
        self.f.loc[:, 'df'] = self.f.loc[:, 'fu'] - self.f.loc[:, 'fl']

    def f_to_band(self, f: np.ndarray | Iterable):
        bands = np.empty(f.shape, dtype=float)
        bands[:] = np.nan
        for bi, band in enumerate(self.f.index):
            band_select = (self.f.loc[band, 'fl'] < f) & (f < self.f.loc[band, 'fu'])
            bands[band_select] = float(band)

            bands[f == self.f.loc[band, 'fl']] = band - .5

        return bands

    def interp1d_to_narrowband(self,
                               f_interpolate: np.ndarray | Iterable,
                               f_octave: np.ndarray | Iterable,
                               dat_octave: np.ndarray | Iterable,
                               extrapolate: Literal['zero', 'constant', 'slope_to_zero'] = 'zero',
                               ):
        f_interpolate: np.ndarray = np.array(f_interpolate)
        f_octave: np.ndarray = np.array(f_octave)
        dat_octave: np.ndarray = np.array(dat_octave)

        if f_interpolate.ndim > 1 or f_octave.ndim > 1 or dat_octave.ndim > 1:
            raise ValueError(f'This function only supports 1D interpolation to narrowband.')

        result = np.zeros(f_interpolate.size)
        bands_interpolate = self.f_to_band(f_interpolate)
        bands_octave = self.f_to_band(f_octave)

        if extrapolate.lower() == 'slope_to_zero':
            if bands_octave[0] > self.f.index[0]:
                bands_octave = np.append([self.f.index[0] - 1, ], bands_octave)
                dat_octave = np.append([0., ], dat_octave)
            if bands_octave[-1] < self.f.index[-1]:
                bands_octave = np.append(bands_octave, self.f.index[-1] + 1)
                dat_octave = np.append(dat_octave, 0.)

            dat_intermediate = np.interp(self.f.index, bands_octave, dat_octave)

        elif extrapolate.lower() == 'constant':
            dat_intermediate = np.interp(self.f.index, bands_octave, dat_octave)

        elif extrapolate.lower() == 'zero':
            dat_intermediate = np.zeros(self.f.index.size)
            non_zero = (f_octave[0] <= self.f['fc']) & (self.f['fc'] <= f_octave[-1])
            dat_intermediate[non_zero] = np.interp(self.f.index, bands_octave, dat_octave)

        else:
            raise ValueError(f"Unknown extrapolation method '{extrapolate}'. "
                             f"Select from ('zero', 'constant', 'slope_to_zero').")

        for bi, band in enumerate(self.f.index):
            band_select = (self.f.loc[band, 'fl'] <= f_interpolate) & (f_interpolate < self.f.loc[band, 'fu'])
            result[band_select] = dat_intermediate[bi]

        return result
