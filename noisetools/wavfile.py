"""Functions to interact with wav files.
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
import scipy.io as spio
import numpy as np
import warnings
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
        IMPORTANT: both ```norm``` and ```cal``` are applied to the data.
    cal: int | float, optional (default=1.)
        Calibration factor for the wav file signal. The values in the WAV file are multiplied by this factor directly
        after reading. This factor is not used for writing. The signals are converted to 32-bit floats and new WAV files
        are written as such.
        IMPORTANT: both ```norm``` and ```cal``` are applied to the data.
    wav: numpy.ndarray, optional
        Optional entry for the creation of a new WAV file. This should be a 2D array with shape (size, 1) or (size, 2).
        NOTE: when an array is given, a sampling frequency fs should be included.
    fs: int, optional
        Sampling frequency of input array 'wav'.
        NOTE: this parameter should be included when wav is defined.
    pcm: str, optional


    Attributes
    ----------
    filename: str
        Name of the associated WAV file.
    fs: int
        Sampling frequency of the stored singal(s).
    size: int
        Number of samples in the signal(s).
    duration: float
        Duration of the signal(s) in seconds.
    duration_string: str
        String representation of the signal(s) duration in mm:ss:ms_.
    time: numpy.ndarray
        Array containing the time vector of the WAV file in seconds.

    """
    pcm_table = {'s32': (-2147483648, +2147483647, np.int32),
                 's24': (-2147483648, +2147483392, np.int32),
                 's16': (-32768, +32767, np.int16),
                 'u8': (0, 255, np.uint8),
                 }

    def __init__(self,
                 filename: str,
                 norm: int | float = 1.,
                 cal: int | float = 1.,
                 wav: np.ndarray | None = None,
                 fs: int | None = None,
                 pcm: Literal['s32', 's24', 's16', 'u8'] | None = None,
                 ) -> None:
        self.filename = filename if filename.endswith('.wav') else filename + '.wav'

        # Read from a file, if it exists, and if wav is not defined.
        if os.path.isfile(self.filename) and wav is None:
            self.fs, wav = spio.wavfile.read(filename)
            if wav.ndim == 1:
                wav = wav.reshape((-1, 1))

        # Warn the user if the WAV file exists when a wav array is given.
        elif os.path.isfile(self.filename):
            warnings.warn(f'WAV file with name {self.filename} already exists. Created WavFile instance based on given wav array.')
        # Otherwise, a wav array is expected, so check for the existence.
        elif wav is None:
            raise FileNotFoundError(f'[Errno 2] No such file or directory: {self.filename}')
        # At this point, a wav array exists, so check for the presence of a sampling frequency.
        elif fs is None:
            raise SyntaxError(f'Initialising from an array requires defining the sampling frequency fs.')
        else:
            self.fs = fs

        # It is now certain the variable _wav is filled.
        self._wav: np.ndarray = wav.copy()
        self.fs: int

        # Convert to 32-bit floats, for saving to wavfile purposes.
        if self._wav.dtype != np.float32:
            if np.issubdtype(self._wav.dtype, np.floating):
                self._wav = self._wav.astype(np.float32)
            elif pcm is None:
                raise ValueError(f'{filename} is not in PCM f32 format, provide the PCM format to WavFile.')
            else:
                pcm = pcm.lower()

                if pcm not in self.pcm_table.keys():
                    raise ValueError('Given WAV file PCM format not supported by noisetools.')
                elif self.pcm_table[pcm][2] != self._wav.dtype:
                    raise ValueError(f'Provided PCM format ({pcm}) does not match the obtained dtype ({self._wav.dtype}).')

                self._wav = self._wav.astype(np.float32)
                self._wav = 2 * (self._wav - self.pcm_table[pcm][0]) / (self.pcm_table[pcm][1] - self.pcm_table[pcm][0]) - 1.

        # Apply both calibration and normalisation factors to the wav file data.
        self._wav: np.ndarray = cal * self._wav / norm

        # Check that wav has exactly two dimensions.
        if self._wav.ndim != 2:
            raise ValueError(f'Parameter wav should be a 2D numpy array.')
        else:
            # Determine if it is mono or stereo.
            self.mono = self._wav.shape[1] == 1
            # Check if the array has the correct shape.
            if self._wav.shape[1] > 2:
                raise ValueError(f'Axis 1 of parameter wav should have size 2, got {self._wav.shape[1]}')

        # Set the signal size.
        self.size = self._wav.shape[0]
        # Determine the signal duration and create a time vector.
        self.duration = float(self.size / self.fs)
        self.time = np.linspace(0, self.duration, self.size, endpoint=False)
        # Create a human-readable string of the signal duration.
        self.duration_string = f'{str(int(self.duration / 60)).zfill(2)}:{str(int(self.duration % 60)).zfill(2)}:{str(int((self.duration % 60) % 1 * 1e3)).zfill(3)}'

    @property
    @warnings.deprecated('The attribute WavFile.length will be replaced with WavFile.size.')
    def length(self):
        return self.size

    @property
    @warnings.deprecated('The attribute WavFile.wav_left will be replaced with WavFile.left for stereo signals, and WavFile.sig for mono signals.')
    def wav_left(self):
        if self.mono:
            return self._wav.flatten()
        else:
            return self._wav[:, 0]

    @property
    @warnings.deprecated('The attribute WavFile.wav_right will be replaced with WavFile.right for stereo signals, and WavFile.sig for mono signals.')
    def wav_right(self):
        if self.mono:
            return self._wav.flatten()
        else:
            return self._wav[:, 1]

    @property
    def left(self):
        if self.mono:
            raise AttributeError('Mono WavFile objects do not contain attribute WavFile.left.')
        else:
            return self._wav[:, 0]

    @property
    def right(self):
        if self.mono:
            raise AttributeError('Mono WavFile objects do not contain attribute WavFile.right.')
        else:
            return self._wav[:, 1]

    @property
    def sig(self):
        if self.mono:
            return self._wav.flatten()
        else:
            raise AttributeError('Stereo WavFile objects do not contain attribute WavFile.sig.')

    def __repr__(self) -> str:
        mono_str = 'mono' if self.mono else 'stereo'
        return f"<'{self.filename}' ({self.duration_string}) ({mono_str})>"

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError('WavFile only supports addition between WavFile instances.')

        if self.fs != other.fs:
            other.resample(self.fs)

        if self.size != other.size:
            raise ValueError('Two WavFile instances require the same length for addition.')

        if self.mono and not other.mono:
            return self.from_two_channel(self.filename, self.sig + other.left, self.sig + other.right, self.fs)
        else:
            return self.__class__(self.filename, wav=self._wav + other._wav, fs=self.fs)


    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float) or isinstance(other, np.number)):
            raise NotImplementedError('WavFile only support multiplication by a constant value.')

        return self.__class__(self.filename, wav=self._wav * other, fs=self.fs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not (isinstance(other, int) or isinstance(other, float) or isinstance(other, np.number)):
            raise NotImplementedError('WavFile only support division by a constant value.')

        return self.__class__(self.filename, wav=self._wav / other, fs=self.fs)

    def __rtruediv__(self, other):
        if not (isinstance(other, int) or isinstance(other, float) or isinstance(other, np.number)):
            raise NotImplementedError('WavFile only support division by a constant value.')

        return self.__class__(self.filename, wav=other / self._wav, fs=self.fs)

    @staticmethod
    @warnings.deprecated('Static method WavFile._two_channel_to_wav is deprecated.')
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
        if left_array.ndim != 1:
            raise ValueError('Left signal array must be 1D.')
        if right_array.ndim != 1:
            raise ValueError('Right signal array must be 1D.')
        if left_array.size != right_array.size:
            raise ValueError('Left and right signal arrays must have the same size.')

        wav = np.concatenate([left_array.reshape((-1, 1)), right_array.reshape((-1, 1))], axis=1)
        return cls(filename, wav=wav, fs=fs)

    @classmethod
    def from_one_channel(cls,
                         filename: str,
                         sig_array: np.ndarray,
                         fs: int,
                         ):
        """
        Create an instance of WavFile from a mono signal array

        Parameters
        ----------
        filename: str
            Filename for the new WAV file.
        sig_array: numpy.ndarray
            1D Numpy array containing the mono signal.
        fs: int, optional
            Sampling frequency of input arrays.

        Returns
        -------
        An instance of WavFile with the given signal information.

        """
        if sig_array.ndim != 1:
            raise ValueError('Signal array must be 1D.')

        return cls(filename, wav=sig_array.reshape((-1, 1)), fs=fs)

    @staticmethod
    @warnings.deprecated('Static method WavFile.seconds_to_mmssms is deprecated.')
    def seconds_to_mmssms(t: int | float,
                          ) -> str:
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
    @warnings.deprecated('Static method WavFile.mmssms_to_seconds is deprecated.')
    def mmssms_to_seconds(t: str,
                          ) -> float:
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

    @warnings.deprecated('WavFile.check_mono is deprecated and replaced by the attribute mono.')
    def check_mono(self,
                   ) -> bool:
        """
        Determine whether the signal in this WAV file is mono.
        """
        return self.mono

    def resample(self,
                 fs: int,
                 filename: str | None = None,
                 ):
        """
        Resample this wav file to a new sampling frequency.

        Parameters
        ----------
        fs: int
            New sampling frequency for the signal in Hertz (Hz)
        filename: str, optional
            Optional filename, if the resampled signal requires a different file is desired.

        Returns
        -------
        A WavFile instance with the resampled signal.

        """
        # Don't do anything if the sampling frequency is equal.
        if self.fs == fs:
            warnings.warn('Requested resampling frequency is equal to current sampling frequency.', stacklevel=2)
            return self

        # Resample the signal(s).
        _wav: np.ndarray = spsig.resample_poly(self._wav, fs, self.fs, axis=1)

        filename = self.filename if filename is None else filename

        return self.__class__(filename, wav=_wav, fs=fs)

    def write(self,
              overwrite: bool = True,
              filename: str | None = None,
              ) -> None:
        """
        Write the information in this instance to a WAV file.

        Parameters
        ----------
        overwrite: bool, optional (default=True)
            Explicit indication whether to overwrite the WAV file of this instance.
        filename: str, optional
            Optional filename, if writing to a different file is desired.

        """
        filename = self.filename if filename is None else filename

        if os.path.isfile(self.filename) and not overwrite:
            return

        if self.mono:
            spio.wavfile.write(filename, self.fs, self._wav.flatten())
        else:
            spio.wavfile.write(filename, self.fs, self._wav)

    def export(self,
               t0: float,
               t1: float,
               filename: str | None = None,
               fs: int | None = None,
               write: bool = False,
               ):
        """
        Export the signal section where t0 <= WavFile.t < t1, to a new WavFile instance.

        Parameters
        ----------
        t0: float
            Start time of the export in seconds.
        t1: float
            End time of the export in seconds.
            Note: In case t1==self.duration, the selection becomes t0 <= WavFile.t <= t1
        filename: str, optional
            File name to export partial signal to. Defaults to filename_export.wav.
        fs: int, optional
            Sampling frequency for optional resampling before export.
        write: bool, optional
            Write the information of the new instance to a WAV file immediately.
            IMPORTANT: This will ALWAYS overwrite previous wav files with the same filename!

        Returns
        -------
        A new instance of WavFile with the signal information between t0 and t1.

        """
        if fs is not None:
            warnings.warn('Resampling of signals in WavFile.export with parameter fs is deprecated.',
                          DeprecationWarning
                          )
            # Temporary patch to keep this working through the deprecation.
            resampled = self.resample(fs)
            self.time = resampled.time
            self._wav = resampled._wav

        if isinstance(t0, str):
            warnings.warn('Use of a string for WavFile.export parameter t0 is deprecated.',
                          DeprecationWarning
                          )
            t0 = self.mmssms_to_seconds(t0)
        if isinstance(t1, str):
            warnings.warn('Use of a string for WavFile.export parameter t1 is deprecated.',
                          DeprecationWarning
                          )
            t1 = self.mmssms_to_seconds(t1)

        select = (t0 <= self.time) & (self.time < t1) if t1 != self.duration else (t0 <= self.time) & (self.time <= t1)

        if filename is None:
            filename = self.filename.replace('.wav', '_export.wav')
        elif filename == self.filename and os.path.isfile(self.filename):
            warnings.warn(f"Filename of the WavFile export equals the original filename ({self.filename}). It is "
                          f"recommended to choose a different filename for exporting sections of the signal."
                          )

        wavfile = self.__class__(filename, wav=self._wav[select, :], fs=self.fs)

        if write:
            wavfile.write(overwrite=True)

        return wavfile
