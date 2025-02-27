"""Functions to interact with wav files.
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

import scipy.signal as spsig
import scipy.io as spio
import numpy as np
import warnings
import os


__all__ = ['WavFile', ]
pcm_table = {'s32': (-2147483648, +2147483647, np.int32),
             's24': (-2147483648, +2147483392, np.int32),
             's16': (-32768, +32767, np.int16),
             'u8': (0, 255, np.uint8),
             }


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
                 pcm: str = None,
                 ) -> None:
        self.filename = filename if filename.endswith('.wav') else filename + '.wav'

        # Read from a file, if it exists.
        if os.path.isfile(self.filename) and wav is None:
            self.fs, wav = spio.wavfile.read(filename)
            self.fs: int
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
            if np.issubdtype(wav.dtype, np.floating):
                wav = wav.astype(np.float32)
            elif pcm is None:
                raise ValueError(f'{filename} is not in PCM f32 format, provide the PCM format to WavFile.')
            else:
                pcm = pcm.lower()

                if pcm not in pcm_table.keys():
                    raise ValueError('Given WV file PCM format not supported by noisetools.')
                elif pcm_table[pcm][2] != wav.dtype:
                    raise ValueError(f'Provided PCM format ({pcm}) does not match the obtained dtype ({wav.dtype}).')

                wav = wav.astype(np.float32)
                wav = 2 * (wav - pcm_table[pcm][0]) / (pcm_table[pcm][1] - pcm_table[pcm][0]) - 1.

        wav: np.ndarray = wav / norm

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
            raise NotImplementedError('WavFile only supports addition between WavFile instances.')

        if self.fs != other.fs:
            other.resample(self.fs)

        if self.length != other.length:
            raise ValueError('Two WavFile instances require the same length for addition.')

        left_array = self.wav_left + other.wav_left
        right_array = self.wav_right + other.wav_right

        return self.from_two_channel(self.filename, left_array, right_array, self.fs)

    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float) or isinstance(other, np.number)):
            raise NotImplementedError('WavFile only support multiplication by a constant value.')

        left_array = self.wav_left * other
        right_array = self.wav_right * other

        return self.from_two_channel(self.filename, left_array, right_array, self.fs)

    __rmul__ = __mul__

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
        A 2D array with shape (length, 2)
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
            warnings.warn('Requested resampling frequency is equal to current sampling frequency.', stacklevel=2)
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
              filename: str = None,
              ) -> None:
        """
        Write the the information in this instance to a WAV file.

        Parameters
        ----------
        overwrite: bool, optional (default=True)
            Explicit indication whether to overwrite the WAV file of this instance.
        filename: str, optional
            Optional filename, if writing to a different file is desired.
            This will change the WavFile objects self.filename.
        """
        self.filename = self.filename if filename is None else filename

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
               filename: str = None,
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
        filename: str, optional
            File name to export partial signal to. Defaults to filename_export.wav.
        fs: int, optional
            Sampling frequency for optional resampling before export.
        write: bool, optional
            Write the information of the new instance to a WAV file immediately.
            NOTE: This will overwrite previous wav files with the same filename!

        Returns
        -------
        A new instance of WavFile with the signal information between t0 and t1.
        """
        if fs is not None:
            self.resample(fs)

        if isinstance(t0, str):
            t0 = self.mmssms_to_seconds(t0)
        if isinstance(t1, str):
            t1 = self.mmssms_to_seconds(t1)

        select = (t0 <= self.t) & (self.t < t1) if t1 != self.duration else (t0 <= self.t) & (self.t <= t1)

        if filename is None:
            filename = self.filename.replace('.wav', '_export.wav')
        elif filename == self.filename and write:
            raise FileExistsError(f"Filename of the WavFile export equals the original filename ({self.filename}). "
                                  f"Please choose a different filename or set write=False.")
        elif filename == self.filename:
            warnings.warn(f"Filename of the WavFile export equals the original filename ({self.filename}). It is "
                          f"recommended to choose a different filename for exporting sections of the signal.")

        if self.check_mono():
            wavfile = WavFile(filename, wav=self.wav_left[select], fs=self.fs)
        else:
            wavfile = WavFile.from_two_channel(filename, self.wav_left[select], self.wav_right[select], self.fs)

        if write:
            wavfile.write(overwrite=True)

        return wavfile
