"""Translation layer to run SQAT [1]_ through Python.

Written for the following combination of MATLAB, Python and matlabengine versions:
- MATLAB release R2023b
- Python 3.11
- matlabengine 23.2.2

Other combinations will probably break the installation noisetools, but are not guaranteed to work anyway.

References
----------
.. [1] Gil Felix Greco, Roberto Merino-Martínez, and Alejandro Osses, ‘SQAT: a sound quality analysis toolbox for MATLAB’.
    Zenodo, Jan. 29, 2024. doi: 10.5281/ZENODO.10580337.

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

import scipy.interpolate as spint
import pandas as pd
import numpy as np
import errno
import os

try:
    import matlab.engine as mateng
except ImportError as e:
    raise ImportError('A problem is found while importing matlabengine. '
                      'Make sure you install noisetools[pysqat], to be able to use the PySQAT package.')


SQAT_PATH = os.path.join(os.path.dirname(__file__), 'SQAT-1.2')


class PySQAT:
    """
    Translation layer class to run SQAT functions through Python.

    NOTE: Initialisation of this class may take a while, as it involves starting a MATLAB session through matlabengine.

    Raises
    ------
    RuntimeError
        If the SQAT directory is not where it is expected to be.
    """
    def __init__(self,
                 ) -> None:
        if not os.path.isdir(SQAT_PATH):
            raise RuntimeError(f'SQAT directory is missing in the noisetools installation! '
                               f'Make sure noisetools was installed correctly.'
                               f'Expected location: {SQAT_PATH}.')

        self.eng: mateng.MatlabEngine = mateng.start_matlab()
        self.eng.cd(SQAT_PATH)
        self.eng.startup_SQAT(nargout=0)

    def stop_sqat(self
                  ) -> None:
        """
        Stop the internal MATLAB engine. Calling this function renders this instance of PySQAT useless.
        """
        self.eng.quit()
        del self

    @staticmethod
    def extract_instantaneous_sqms(pa_res: dict,
                                   ) -> pd.DataFrame:
        """
        Extract the instantaneous values of the SQMs from the dictionary with results from
        the PySQAT.psychoacoustic_annoyance_x() functions.

        Parameters
        ----------
        pa_res : dict
            Dictionary as it is returned by one of the three PySQAT.psychoacoustic_annoyance_x() functions.

        Returns
        -------
        Pandas DataFrame with the time series of the instantaneous SQMs.

        """
        # Define the metrics and the names of their instantaneous SQM vector.
        metrics = {'L': 'Loudness', 'S': 'Sharpness', 'R': 'Roughness', 'FS': 'FluctuationStrength', 'K': 'Tonality', }
        # Define the column names for the pandas DataFrame
        columns = ['PA'] + [metric for metric in metrics.keys() if metric in pa_res.keys()]

        # Create the DataFrame and fill the PA column.
        df = pd.DataFrame(index=np.array(pa_res['time'], dtype=float).flatten(), columns=columns)
        df.loc[:, 'PA'] = np.array(pa_res[f'InstantaneousPA'], dtype=float).flatten()

        # Loop over the metrics to be extracted.
        for metric, name in metrics.items():
            if metric in df.columns:
                # Extract time and SQM vectors.
                instantaneous_tme = np.array(pa_res[metric]['time'], dtype=float).flatten()
                instantaneous_sqm = np.array(pa_res[metric][f'Instantaneous{name}'], dtype=float).flatten()
                # Create interpolation function.
                f = spint.interp1d(instantaneous_tme, instantaneous_sqm, kind='cubic', fill_value='extrapolate', )
                # Interpolate the values to the PA time vector and store.
                df.loc[:, metric] = f(df.index)

        return df

    @staticmethod
    def extract_instantaneous_loudness(loudness_res: dict,
                                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the instantaneous values of the loudness from the output dictionary of PySQAT.loudness_iso532_1.

        Parameters
        ----------
        loudness_res: dict
            Dictionary that is returned by PySQAT.loudness_iso532_1()

        Returns
        -------
        Two Pandas DataFrames:
            - DataFrame with time (s) as index and columns containing:
                'N': Instantaneous Loudness (sone)
                'Ln': Instantaneous Loudness Level (phon)
                'OSPL': Instantaneous overall SPL from 1/3 octave band levels (dB(SPL))
            - DataFrame with time (s) as index, and barks as columns,
               containing the instantaneous Specific Loundess levels (sone/bark).
        """
        df_is_loudness = pd.DataFrame(loudness_res['InstantaneousSpecificLoudness'],
                                      index=np.array(loudness_res['time']).flatten(),
                                      columns=loudness_res['barkAxis'][0])

        df_instantaneous = pd.DataFrame(index=np.array(loudness_res['time']).flatten(),
                                        columns=['N', 'Ln', 'OSPL'])

        df_instantaneous.loc[:, 'N'] = np.array(loudness_res['InstantaneousLoudness']).flatten()
        df_instantaneous.loc[:, 'Ln'] = np.array(loudness_res['InstantaneousLoudnessLevel']).flatten()
        df_instantaneous.loc[:, 'OSPL'] = np.array(loudness_res['InstantaneousSPL']).flatten()[::4]

        return df_instantaneous, df_is_loudness

    @staticmethod
    def _check_pa_parameters(wavfilename: str,
                             dbfs: int | float,
                             time_skip: int | float,
                             loudness_field: int | float
                             ) -> None:
        """
        Check the input parameters of the psychoacoustic_annoyance functions in this class

        Parameters
        ----------
        wavfilename : str
            Specifies the file name of a wav file to be processed.

        dbfs : number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        time_skip : number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        loudness_field : number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info.
            NOTE: value should be convertible to an integer.

        """
        if not isinstance(dbfs, int) and not dbfs.is_integer():
            raise ValueError('Value of parameter "dbfs" should be an integer.')
        if not isinstance(time_skip, int) and not time_skip.is_integer():
            raise ValueError('Value of parameter "time_skip" should be an integer.')
        if not isinstance(loudness_field, int) and not loudness_field.is_integer():
            raise ValueError('Value of parameter "loudness_field" should be an integer.')

        if not os.path.exists(wavfilename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), wavfilename)

    def psychoacoustic_annoyance_zwicker1999(self,
                                             wavfilename: str,
                                             dbfs: int | float = 94.,
                                             time_skip: int | float = 0.,
                                             loudness_field: int | float = 0.
                                             ) -> dict:
        """
        Wrapper for the SQAT function PsychoacousticAnnoyance_Zwicker1999_from_wavfile.

        Only supports the use of .wav files, since the use of an array has unpredictable behaviour in the MATLAB engine.
        The MATLAB parameters 'showPA' and 'show' are always set to false, since MATLAB plotting breaks python.

        Parameters
        ----------
        wavfilename : str
            Specifies the file name of a wav file to be processed.

        dbfs : number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        time_skip : number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        loudness_field : number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info.
            NOTE: value should be convertible to an integer.

        Returns
        -------
        Equivalent to the struct returned by the MATLAB equivalent function. Contains the following key-value pairs
        with the PA results:
            - InstantaneousPA: instantaneous quantity (unity) vs time
            - ScalarPA : PA (scalar value) computed using the percentile values of each metric. NOTE: if the
              signal's length is smaller than 2s, this is the only output as no time-varying PA is calculated.
            - time : time vector in seconds
            - wfr : fluctuation strength and roughness weighting function (not squared)
            - ws : sharpness and loudness weighting function (not squared)
            - Statistics: dict with the following keys:
                - PAmean : mean value of instantaneous fluctuation strength (unit)
                - PAstd : standard deviation of instantaneous fluctuation strength (unit)
                - PAmax : maximum of instantaneous fluctuation strength (unit)
                - PAmin : minimum of instantaneous fluctuation strength (unit)
                - PAx : x percentile of the PA metric exceeded during x percent of the time

        Also contains the following dicts under keys:
            -  L : dict with Loudness results, similar structure to the PA results
            -  S : dict with Sharpness, similar structure to the PA results
            -  R : dict with roughness results, similar structure to the PA results
            - FS : dict with fluctuation strength results, similar structure to the PA results

        NOTE: values will be of a type according to the MATLAB help pages:
        https://nl.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html

        Raises
        ------
        ValueError
            If the input parameters dbfs, time_skip or loudness_field are not interpretable as integer values.
            This would break MATLAB, but is checked beforehand to give a clear error message.

        """
        self._check_pa_parameters(wavfilename, dbfs, time_skip, loudness_field)

        dbfs, time_skip, loudness_field = float(dbfs), float(time_skip), float(loudness_field)
        return self.eng.PsychoacousticAnnoyance_Zwicker1999_from_wavfile(wavfilename, dbfs, time_skip, loudness_field, )

    def psychoacoustic_annoyance_di2016(self,
                                        wavfilename: str,
                                        dbfs: int | float = 94.,
                                        time_skip: int | float = 0.,
                                        loudness_field: int | float = 0.
                                        ) -> dict:
        """
        Wrapper for the SQAT function PsychoacousticAnnoyance_Di2016_from_wavfile.

        Only supports the use of .wav files, since the use of an array has unpredictable behaviour in the MATLAB engine.
        The MATLAB parameters 'showPA' and 'show' are always set to false, since MATLAB plotting breaks python.

        Parameters
        ----------
        wavfilename : str
            Specifies the file name of a wav file to be processed.

        dbfs : number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        time_skip : number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        loudness_field : number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info.
            NOTE: value should be convertible to an integer.

        Returns
        -------
        Equivalent to the struct returned by the MATLAB equivalent function. Contains the following key-value pairs
        with the PA results:
            - InstantaneousPA: instantaneous quantity (unity) vs time
            - ScalarPA : PA (scalar value) computed using the percentile values of each metric. NOTE: if the
              signal's length is smaller than 2s, this is the only output as no time-varying PA is calculated.
            - time : time vector in seconds
            - wt : tonality and loudness weighting function (not squared)
            - wfr : fluctuation strength and roughness weighting function (not squared)
            - ws : sharpness and loudness weighting function (not squared)
            - Statistics: dict with the following keys:
                - PAmean : mean value of instantaneous fluctuation strength (unit)
                - PAstd : standard deviation of instantaneous fluctuation strength (unit)
                - PAmax : maximum of instantaneous fluctuation strength (unit)
                - PAmin : minimum of instantaneous fluctuation strength (unit)
                - PAx : x percentile of the PA metric exceeded during x percent of the time

        Also contains the following dicts under keys:
            -  L : dict with Loudness results, similar structure to the PA results
            -  S : dict with Sharpness, similar structure to the PA results
            -  R : dict with roughness results, similar structure to the PA results
            - FS : dict with fluctuation strength results, similar structure to the PA results
            -  K : dict with tonality results, similar structure to the PA results

        NOTE: values will be of a type according to the MATLAB help pages:
        https://nl.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html

        Raises
        ------
        ValueError
            If the input parameters dbfs, time_skip or loudness_field are not interpretable as integer values.
            This would break MATLAB, but is checked beforehand to give a clear error message.

        """
        self._check_pa_parameters(wavfilename, dbfs, time_skip, loudness_field)

        dbfs, time_skip, loudness_field = float(dbfs), float(time_skip), float(loudness_field)
        return self.eng.PsychoacousticAnnoyance_Di2016_from_wavfile(wavfilename, dbfs, time_skip, loudness_field, )

    def psychoacoustic_annoyance_more2010(self,
                                          wavfilename: str,
                                          dbfs: int | float = 94.,
                                          time_skip: int | float = 0.,
                                          loudness_field: int | float = 0.
                                          ) -> dict:
        """
        Wrapper for the SQAT function PsychoacousticAnnoyance_More2010_from_wavfile.

        Only supports the use of .wav files, since the use of an array has unpredictable behaviour in the MATLAB engine.
        The MATLAB parameters 'showPA' and 'show' are always set to false, since MATLAB plotting breaks python.

        Parameters
        ----------
        wavfilename : str
            Specifies the file name of a wav file to be processed.

        dbfs : number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        time_skip : number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        loudness_field : number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info.
            NOTE: value should be convertible to an integer.

        Returns
        -------
        Equivalent to the struct returned by the MATLAB equivalent function. Contains the following key-value pairs
        with the PA results:
            - InstantaneousPA: instantaneous quantity (unity) vs time
            - ScalarPA : PA (scalar value) computed using the percentile values of each metric. NOTE: if the
              signal's length is smaller than 2s, this is the only output as no time-varying PA is calculated.
            - time : time vector in seconds
            - wt : tonality and loudness weighting function (not squared)
            - wfr : fluctuation strength and roughness weighting function (not squared)
            - ws : sharpness and loudness weighting function (not squared)
            - Statistics: dict with the following keys:
                - PAmean : mean value of instantaneous fluctuation strength (unit)
                - PAstd : standard deviation of instantaneous fluctuation strength (unit)
                - PAmax : maximum of instantaneous fluctuation strength (unit)
                - PAmin : minimum of instantaneous fluctuation strength (unit)
                - PAx : x percentile of the PA metric exceeded during x percent of the time

        Also contains the following dicts under keys:
            -  L : dict with Loudness results, similar structure to the PA results
            -  S : dict with Sharpness, similar structure to the PA results
            -  R : dict with roughness results, similar structure to the PA results
            - FS : dict with fluctuation strength results, similar structure to the PA results
            -  K : dict with tonality results, similar structure to the PA results

        NOTE: values will be of a type according to the MATLAB help pages:
        https://nl.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html

        Raises
        ------
        ValueError
            If the input parameters dbfs, time_skip or loudness_field are not interpretable as integer values.
            This would break MATLAB, but is checked beforehand to give a clear error message.

        """
        self._check_pa_parameters(wavfilename, dbfs, time_skip, loudness_field)

        dbfs, time_skip, loudness_field = float(dbfs), float(time_skip), float(loudness_field)
        return self.eng.PsychoacousticAnnoyance_More2010_from_wavfile(wavfilename, dbfs, time_skip, loudness_field, )

    def loudness_iso532_1(self,
                          wavfilename: str,
                          dbfs: int | float = 94.,
                          field: int | float = 0.,
                          method: int | float = 2.,
                          time_skip: int | float = 0.
                          ) -> dict:
        """
        Wrapper for the SQAT function Loudness_ISO532_1_from_wavfile.

        Only supports the use of .wav files, since the use of an array has unpredictable behaviour in the MATLAB engine.
        The MATLAB parameters 'show' are always set to false, since MATLAB plotting breaks python.

        Parameters
        ----------
        wavfilename: str
            Specifies the file name of a wav file to be processed.

        dbfs: number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        field: number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info.
            NOTE: value should be convertible to an integer.

        method: number, optional
            0 = stationary (from input 1/3 octave unweighted SPL)
            1 = stationary (from audio file)
            2 = time varying (from audio file)
            NOTE: value should be convertible to an integer.

        time_skip: number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        Returns
        -------
        Equivalent to the struct returned by the MATLAB equivalent function. Contains the following key-value pairs
        with the Loudness results:
            - barkAxis - vector of Bark band numbers used for specific loudness computation
            - time - time vector of the final loudness calculation, in seconds
            - time_insig - time vector of insig, in seconds
            - InstantaneousLoudness - instantaneous loudness (sone) vs time
            - InstantaneousSpecificLoudness - specific loudness (sone/Bark) vs time
            - InstantaneousLoudnessLevel - instantaneous loudness level (phon) vs time
            - SpecificLoudness - time-averaged specific loudness (sone/Bark)
            - InstantaneousSPL - overall SPL (1/3 octave bands) for each time step, in dBSPL
            - Several statistics based on the InstantaneousLoudness
                - Nmean : mean value of InstantaneousLoudness (sone)
                - Nstd : standard deviation of InstantaneousLoudness (sone)
                - Nmax : maximum of InstantaneousLoudness (sone)
                - Nmin : minimum of InstantaneousLoudness (sone)
                - Nx : percentile loudness exceeded during x percent of the signal (sone)
                - N_ratio : ratio between N5/N95 ( 1.1 (stationary)> N_ratio > 1.1 (time varying) )
             HINT:  loudness calculation takes some time to have a steady-response
                    therefore, it is a good practice to consider a time_skip to compute the statistics
                    due to transient effects in the beginning of the loudness calculations
        """
        if not isinstance(dbfs, int) and not dbfs.is_integer():
            raise ValueError('Value of parameter "dbfs" should be an integer.')
        if not isinstance(field, int) and not field.is_integer():
            raise ValueError('Value of parameter "field" should be an integer.')
        if not isinstance(method, int) and not method.is_integer():
            raise ValueError('Value of parameter "method" should be an integer.')
        if not isinstance(time_skip, int) and not time_skip.is_integer():
            raise ValueError('Value of parameter "time_skip" should be an integer.')

        if not os.path.exists(wavfilename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), wavfilename)

        dbfs, field, method, time_skip = float(dbfs), float(field), float(method), float(time_skip)
        return self.eng.Loudness_ISO532_1_from_wavfile(wavfilename, dbfs, field, method, time_skip, )
