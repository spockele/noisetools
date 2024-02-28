"""
Translation layer to run SQAT [1]_ through Python.

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
import matlab.engine as mateng
import numpy as np
import errno
import os


SQAT_PATH = os.path.join(os.path.dirname(__file__), 'SQAT-1.1')


class PySQAT:
    """
    Translation layer class to run SQAT functions through Python.

    NOTE: Initialisation of this class may take a while, as it involves starting a MATLAB session through matlabengine.

    Raises
    ------
    RuntimeError
        If the SQAT directory is not where it is expected to be.
    """
    def __init__(self, ) -> None:
        if not os.path.isdir(SQAT_PATH):
            raise RuntimeError(f'SQAT directory is missing in the noisetools installation! '
                               f'Make sure noisetools was installed correctly.'
                               f'Expected location: {SQAT_PATH}.')

        self.eng: mateng.MatlabEngine = mateng.start_matlab()
        self.eng.cd(SQAT_PATH)
        self.eng.startup_SQAT(nargout=0)

    def stop_sqat(self) -> None:
        """
        Stop the internal MATLAB engine. Calling this function renders this instance of PySQAT useless.
        """
        self.eng.quit()

    def psychoacoustic_annoyance_zwicker1999(self, wavfilename: str, dbfs: int | float = 94.,
                                             time_skip: int | float = 0., loudness_field: int | float = 0.) -> dict:
        """
        Wrapper for the SQAT function PsychoacousticAnnoyance_Zwicker1999_from_wavfile.

        Only supports the use of .wav files, since the use of an array has unpredictable behaviour in the MATLAB engine.
        The MATLAB parameters 'showPA' and 'show' are always set to false, since MATLAB plotting breaks python.

        Parameters
        ----------
        wavfilename : str
            wavfilename specifies the file name of a wav file to be processed.

        dbfs : number, optional
            Full scale convention. Internally this algorithm works with a convention of full scale being equal to 94 dB
            SPL, or dBFS=94. If the specified dBFS is different from 94 dB SPL, then a gain factor will be applied.
            NOTE: value should be convertible to an integer.

        time_skip : number, optional
            Skip start of the signal in <time_skip> seconds for statistics calculations.
            NOTE: value should be convertible to an integer.

        loudness_field : number, optional
            Choose field for loudness calculation; free field = 0; diffuse field = 1. See Loudness_ISO532_1 for
            more info. NOTE: value should be convertible to an integer.

        Returns
        -------
        dict :
            Equivalent to the struct returned by the MATLAB equivalent function. Contains the following key-value pairs with the PA results:
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

            NOTE: values will be of a type according to the MATLAB help pages: https://nl.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html

        Raises
        ------
        ValueError
            If the input parameters dbfs, time_skip or loudness_field are not interpretable as integer values.
            This would break MATLAB, but is checked beforehand to give a clear error message.

        """
        if not isinstance(dbfs, int) and not dbfs.is_integer():
            raise ValueError('Value of parameter "dbfs" should be an integer.')
        if not isinstance(time_skip, int) and not time_skip.is_integer():
            raise ValueError('Value of parameter "time_skip" should be an integer.')
        if not isinstance(loudness_field, int) and not loudness_field.is_integer():
            raise ValueError('Value of parameter "loudness_field" should be an integer.')

        if not os.path.exists(wavfilename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), wavfilename)

        dbfs, time_skip, loudness_field = float(dbfs), float(time_skip), float(loudness_field)
        return self.eng.PsychoacousticAnnoyance_Zwicker1999_from_wavfile(wavfilename, dbfs, time_skip, loudness_field, )
