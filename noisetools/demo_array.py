"""Functions for processing data from the DEMO 64-microphone array, owned by the Operations and Environment section of
the TU Delft Aerospace Engineering faculty.
"""

from typing import Literal
from nptdms import TdmsFile
from datetime import datetime
import pandas as pd
import numpy as np
import os


class ArrayData:
    """
    Subclass of nptdms.TdmsFile for using data from our DEMO 64-microphone array and DAQ.

    Parameters
    ----------
    directory: str | os.PathLike
        File directory with all the files from the array output.
    read: bool, optional (default=False)

    Attributes
    ----------
    fs: float

    n_mic: int

    start_time: int

    start_timestamp: datetime.datetime

    length: float

    comments: str

    read: bool

    tdms: TdmsFile
    """
    calibration = pd.read_csv(os.path.join(os.path.dirname(__file__), 'array_calibration.csv'),
                              index_col=(0, 1, 2, 3, ),
                              )
    calibration.columns = calibration.columns.astype(float)

    def __init__(self,
                 directory: str | os.PathLike,
                 read: bool = False,
                 configuration: Literal['normal', 'ge_exp'] = 'normal',
                 calibration_unit: Literal['dB', 'nd'] = 'nd',
                 ) -> None:

        if configuration == 'normal':
            select = self.calibration.index.get_level_values('connector') == self.calibration.index.get_level_values('microphone')
            self.calibration = self.calibration.loc[select, :]
            self.calibration.index = self.calibration.index.droplevel('microphone')
            self.uncertainty = self.calibration.loc[(slice(None), calibration_unit, 'uncertainty'), :]
            self.calibration = self.calibration.loc[(slice(None), calibration_unit, 'calibration')]

        elif configuration == 'ge_exp':
            select1 = self.calibration.index.get_level_values('connector') != self.calibration.index.get_level_values('microphone')
            select2 = self.calibration.index.get_level_values('connector') <= 32
            self.calibration = self.calibration.loc[np.logical_xor(select1, select2), :]
            self.calibration.index = self.calibration.index.droplevel('microphone')
            self.uncertainty = self.calibration.loc[(slice(None), calibration_unit, 'uncertainty'), :]
            self.calibration = self.calibration.loc[(slice(None), calibration_unit, 'calibration')]

        elif configuration == 'calibration':
            self.calibration = None

        self.directory = os.path.abspath(directory)
        with open(os.path.join(self.directory, 'info.txt')) as f:
            lines = f.readlines()

        lines = [line.replace('\n', '').split() for line in lines]
        lines = {line[0]: line[1:] for line in lines}

        self.fs = float(lines['acoustic_sample_frequency'][0])
        self.n_mic = int(lines['number_of_microphones'][0])
        self.start_time = int(lines['start_time'][0])
        self.start_timestamp = datetime.strptime(f'{lines['start_timestamp'][0]} {lines['start_timestamp'][1]}',
                                                 '%d/%m/%Y %H:%M:%S.%f')
        self.comments = ' '.join(lines['comments'])

        self.read = read
        self.open = True

        if self.read:
            self.tdms = TdmsFile.read(os.path.join(self.directory, 'acoustic_data.tdms'))
            dat = self.tdms['Microphones Data'][f'Microphone 1'].data * 1.57 / (2 ** (16 - 1)) / (12.589 / 1e3)
        else:
            self.tdms = TdmsFile.open(os.path.join(self.directory, 'acoustic_data.tdms'))
            dat = self.tdms['Microphones Data'][f'Microphone 1'][:] * 1.57 / (2 ** (16 - 1)) / (12.589 / 1e3)

        self.length = dat.size
        self.duration = self.length / self.fs
        self.time = pd.Index(np.linspace(0, self.duration, self.length, endpoint=False), name='t (s)')

    def close(self) -> None:
        """
        Close the tdms file instance.
        """
        if self.open:
            self.tdms.close()

        self.read = False
        self.open = False

    def tdms_open(self) -> None:
        """

        """
        if not self.open:
            self.open = True
            self.tdms = TdmsFile.open(os.path.join(self.directory, 'acoustic_data.tdms'))

    def tdms_read(self) -> None:
        """

        """
        if not self.read:
            self.open = True
            self.read = True
            self.tdms = TdmsFile.read(os.path.join(self.directory, 'acoustic_data.tdms'))

    def read_mic(self,
                 mic: int,
                 ) -> pd.Series:
        """
        Read data from a single microphone channel into a pandas Series.

        Parameters
        ----------
        mic: int
            Microphone number to read.

        Returns
        -------
        A pandas Series with the microphone data. Index contains time axis.

        """
        if self.read:
            dat = self.tdms['Microphones Data'][f'Microphone {mic}'].raw_data * np.float32(1.57 / (2**(16-1)) / (12.589 / 1e3))
        else:
            dat = self.tdms['Microphones Data'][f'Microphone {mic}'].read_data(scaled=False) * np.float32(1.57 / (2**(16-1)) / (12.589 / 1e3))

        return pd.Series(dat, index=self.time, name=mic, ) - np.mean(dat)

    def read_mics(self,
                      mics: list[int] | tuple[int],
                      ) -> pd.DataFrame:
        """
        Read data from multiple microphones into a pandas DataFrame.

        Parameters
        ----------
        mics: list[int] | tuple[int]
            List with microphone numbers to read.

        Returns
        -------
        A pandas DataFrame with the microphone data. Index contains time axis, columns are the microphones.

        """

        dfs = []
        for mic in mics:
            dfs.append(self.read_mic(mic))

        df = pd.concat(dfs, axis='columns')

        return df
