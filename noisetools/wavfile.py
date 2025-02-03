"""Functions to interact with wav files.
"""


import scipy.signal as spsig
import scipy.io as spio
import numpy as np
import os


__all__ = ['WavFile', ]


class WavFile:
    """
    Class for handling WAV files.

    Parameters
    ----------
    filename: str
        Filename for an existing WAV file or for a new WAV file.
    norm: int | float, optional (default=1.)
        Normalisation factor that was used to write the original WAV file. The values in the WAV file are divided by
        this factor directly after reading. This factor is not used for writing. The signals are converted to 32-bit
        floats and new WAV files are written as such.
    wav: numpy.ndarray, optional
        Optional entry for the creation of a new WAV file. For mono signals, this should be a 1D array.
        For stereo signals, this should be a 2D array with shape (length, 2).
        NOTE: when an array is given, a sampling frequency fs should be included.
    fs: int, optional
        Sampling frequency of input array 'wav'.
        NOTE: this parameter should be included when wav is defined.

    Attributes
    ----------
    filename: str
        Name of the associated WAV file.
    fs: int
        Sampling frequency of the stored singal(s).
    wav_left: numpy.ndarray
        Left audio signal (in case of a mono signal, this equals wav_right).
    wav_right: numpy.ndarray
        Right audio signal (in case of a mono signal, this equals wav_left).
    length: int
        Number of samples in the signal(s).
    duration: float
        Duration of the signal(s) in seconds.
    duration_string: str
        String representation of the signal(s) duration in mm:ss:ms_.
    t: numpy.ndarray
        Array containing the time vector of the WAV file in seconds.
    """
    def __init__(self,
                 filename: str,
                 norm: int | float = 1.,
                 wav: np.ndarray = None,
                 fs: int = None,
                 ) -> None:
        self.filename = filename if filename.endswith('.wav') else filename + '.wav'

        # Read from a file, if it exists.
        if os.path.isfile(self.filename):
            self.fs, wav = spio.wavfile.read(filename)
            self.fs: int
            wav: np.ndarray = wav / norm
        # Otherwise, a wav array is expected, so check for the existence.
        elif wav is None:
            raise FileNotFoundError(f'[Errno 2] No such file or directory: {self.filename}')
        # At this point, a wav array exists, so check for the presence of a sampling frequency.
        elif fs is None:
            raise SyntaxError(f'Initialising from an array requires defining the sampling frequency fs.')
        else:
            self.fs = fs

        # Convert to 32-bit floats, for saving to wavfile purposes.
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # It is now certain the variable wav is filled.
        # Check that wav has maximum two dimensions, since there is only a sample and channel dimension.
        if wav.ndim > 2:
            raise ValueError(f'wav input array has too many dimensions. wav.ndim = {wav.ndim} > 2.')

        # In case of a stereo array.
        elif wav.ndim == 2:
            # Check it has a correct shape for use.
            if wav.shape[1] > 2:
                raise ValueError(f'wav input array has too many channels. wav.shape[1] = {wav.shape[1]} > 2')
            # Split up the channels.
            self.wav_left: np.ndarray = wav[:, 0].copy()
            self.wav_right: np.ndarray = wav[:, 1].copy()

        # In case of a mono array, fill both channels with this array.
        else:
            self.wav_left: np.ndarray = wav.copy()
            self.wav_right: np.ndarray = wav.copy()

        # Determine the signal length, duration and create a time vector.
        self.length = self.wav_left.size
        self.duration = float(self.length / self.fs)
        self.duration_string = self.seconds_to_mmssms(self.duration)
        self.t = np.linspace(0, self.duration, self.length)

    def __repr__(self) -> str:
        mono_str = 'mono' if self.check_mono() else 'stereo'
        return f"<'{self.filename}' ({self.duration_string}) ({mono_str})>"

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('WavFile only supports addition between WavFile instances.')

        if self.fs != other.fs:
            other.resample(self.fs)

        if self.length != other.length:
            raise ValueError('Two WavFile instances require the same length for addition.')

        left_array = self.wav_left + other.wav_left
        right_array = self.wav_right + other.wav_right

        return self.from_two_channel(self.filename, left_array, right_array, self.fs)

    @staticmethod
    def _two_channel_to_wav(left_array: np.ndarray,
                            right_array: np.ndarray,
                            ) -> np.ndarray:
        """
        Convert a left and right signal to the correct array format for a stereo WAV file.

        Parameters
        ----------
        left_array: numpy.ndarray
            1D Numpy array containing the left signal.
        right_array: numpy.ndarray
            1D Numpy array containing the right signal.

        Returns
        -------
        An 2D array with shape (length, 2)
        """
        return np.concatenate([left_array.reshape(-1, 1), right_array.reshape(-1, 1)], axis=1)

    @classmethod
    def from_two_channel(cls,
                         filename: str,
                         left_array: np.ndarray,
                         right_array: np.ndarray,
                         fs: int,
                         ):
        """
        Create an instance of WavFile from a left and right signal array

        Parameters
        ----------
        filename: str
            Filename for the new WAV file.
        left_array: numpy.ndarray
            1D Numpy array containing the left signal.
        right_array: numpy.ndarray
            1D Numpy array containing the right signal.
        fs: int, optional
            Sampling frequency of input arrays.

        Returns
        -------
        An instance of WavFile with the given signal information.
        """
        wav = cls._two_channel_to_wav(left_array, right_array)
        return cls(filename, wav=wav, fs=fs)

    @staticmethod
    def seconds_to_mmssms(t: int | float) -> str:
        """
        Convert seconds to a mm:ss:ms_ format (e.g. 72.512s -> 01:12:512)

        Parameters
        ----------
        t: int | float
            Time in seconds.

        Returns
        -------
        The given time in a mm:ss:ms_ string format.
        """
        mm = str(int(t / 60)).zfill(2)
        ss = str(int(t % 60)).zfill(2)
        ms = str(int((t % 60) % 1 * 1e3)).zfill(3)

        return f'{mm}:{ss}:{ms}'

    @staticmethod
    def mmssms_to_seconds(t: str) -> float:
        """
        Convert a mm:ss:ms_ format to seconds (e.g. 01:12:512 -> 72.512s)

        Parameters
        ----------
        t: str
            The given time in a mm:ss:ms_ string format.
            NOTE: ms_ should be 3 digits, any further digits are ignored.

        Returns
        -------
        Float with the time in seconds.
        """
        mm, ss, ms = t.split(':')
        mm, ss, ms = int(mm), int(ss), int(ms[:3])

        return 60 * mm + ss + ms / 1e3

    def check_mono(self) -> bool:
        """
        Determine whether the signal in this WAV file is mono.
        """
        return np.all(np.isclose(self.wav_left, self.wav_right, atol=1e-12))

    def resample(self,
                 fs: int,
                 ) -> None:
        """
        Resample the signals in this wav file to a new sampling frequency.

        Parameters
        ----------
        fs: int
            New sampling frequency for the signal in Hertz (Hz)
        """
        # Don't do anything if the sampling frequency is equal.
        if self.fs == fs:
            return

        # Set the up- and down-sampling for scipy signal.
        up = fs
        down = self.fs

        # Resample the signal(s).
        self.wav_left: np.ndarray = spsig.resample_poly(self.wav_left, up, down)
        self.wav_right: np.ndarray = spsig.resample_poly(self.wav_right, up, down)
        # Update the wavfile information.
        self.fs = fs
        self.length = self.wav_left.size
        self.duration = self.length / self.fs
        self.duration_string = self.seconds_to_mmssms(self.duration)
        self.t = np.linspace(self.t[0], self.duration, self.length)

    def write(self,
              overwrite: bool = True,
              ) -> None:
        """
        Write the the information in this instance to a WAV file.

        Parameters
        ----------
        overwrite: bool, optional (default=True)
            Explicit indication whether to overwrite the WAV file of this instance.
        """
        if os.path.isfile(self.filename) and not overwrite:
            return

        if self.check_mono():
            spio.wavfile.write(self.filename, self.fs, self.wav_left)
        else:
            wav = np.concatenate([self.wav_left.reshape(-1, 1),
                                  self.wav_right.reshape(-1, 1)],
                                 axis=1)
            spio.wavfile.write(self.filename, self.fs, wav)

    def export(self,
               t0: float | str,
               t1: float | str,
               filename: str,
               fs: int = None,
               write: bool = False,
               ):
        """
        Export the signal section where t0 <= WavFile.t < t1, to a new WavFile instance.

        Parameters
        ----------
        t0: float | str
            Start time of the export in seconds, or in a mm:ss:_ms format.
        t1: float | str
            End time of the export in seconds, or in a mm:ss:_ms format.
            NOTE: In case t1==self.duration, the selection becomes t0 <= WavFile.t <= t1
        filename: str
            File name to export partial signal to. Defaults to filename_export.wav.
        fs: int, optional
            Sampling frequency for optional resampling before export.
        write: bool
            Write the information of the new instance to a WAV file immediately.

        Returns
        -------
        A new instance of WavFile with the signal information between t0 and t1.
        """
        if isinstance(t0, str):
            t0 = self.mmssms_to_seconds(t0)
        if isinstance(t1, str):
            t1 = self.mmssms_to_seconds(t1)

        select = (t0 <= self.t) & (self.t < t1) if t1 != self.duration else (t0 <= self.t) & (self.t <= t1)

        if filename is None:
            filename = self.filename.replace('.wav', '_export.wav')
        elif filename == self.filename and not write:
            raise FileExistsError(f"This action would overwrite the original WAV file, which is not allowed. "
                                  f"Use 'overwrite=True' if you really want to overwrite the original WAV file.")

        if fs is not None:
            self.resample(fs)

        if self.check_mono():
            wavfile = WavFile(filename, wav=self.wav_left[select], fs=self.fs)
        else:
            wavfile = WavFile.from_two_channel(filename, self.wav_left[select], self.wav_right[select], self.fs)

        if write:
            wavfile.write()

        return wavfile
