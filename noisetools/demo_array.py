"""Functions for processing data from the DEMO 64-microphone array, owned by the Operations and Environment section of
the TU Delft Aerospace Engineering faculty.
"""

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

    def __init__(self,
                 directory: str | os.PathLike,
                 read: bool = False,
                 ) -> None:

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
        self.length = float(lines['measurement_length'][0])
        self.comments = ' '.join(lines['comments'])

        self.read = read
        self.open = True
        if self.read:
            self.tdms = TdmsFile.read(os.path.join(self.directory, 'acoustic_data.tdms'))
        else:
            self.tdms = TdmsFile.open(os.path.join(self.directory, 'acoustic_data.tdms'))

    def close(self) -> None:
        """
        Close the tdms file instance.
        """
        if self.open:
            self.tdms.close()

        self.read = False
        self.open = False

    def open(self) -> None:
        """

        """
        if not self.open:
            self.open = True
            self.tdms = TdmsFile.open(os.path.join(self.directory, 'acoustic_data.tdms'))

    def read(self) -> None:
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
            dat = self.tdms['Microphones Data'][f'Microphone {mic}'].data * 1.57 / (2**(16-1)) / (12.589 / 1e3)
        else:
            dat = self.tdms['Microphones Data'][f'Microphone {mic}'][:] * 1.57 / (2**(16-1)) / (12.589 / 1e3)

        idx = pd.Index(np.linspace(0, dat.size / self.fs, dat.size), name='t (s)')
        return pd.Series(dat, index=idx, name=mic, dtype=float, ) - np.mean(dat)

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
