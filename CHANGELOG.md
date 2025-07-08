# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[//]: # (## [Unreleased])

[//]: # (### Added)

[//]: # (### Changed)

[//]: # (### Deprecated)

[//]: # (### Removed)

[//]: # (### Fixed)

## [v0.1.0]
Public version of this repository. No functional differences from v0.0.3.


## [v0.0.3]

### Added
- ```wavfile``` module with ```WavFile``` class.\
  - Functions to resample, write to file, export part of the audio signal.
  - Stores information like the sampling frequency, number of samples, duration in seconds, and a time series corresponding to the samples.
- ```PySQAT``` class in ```pysqat```: 
  - Added ```loudness_iso532_1``` function.
  - Added functions to process whole directories.
- ```pysqat``` package: Added function to convert loudness to loudness level.
- Function to extract all loudness information from result dictionary added to ```PySQAT```.
- ```octave_band``` module with ```OctaveBand``` class.
  - Functions to work with frequency domain data in Octave Bands.
  - Designed to comply with IEC 61260-1:2014.


- Created Data Acquisition scripts for NI cDAQ devices.

### Changed
- Updated included ```SQAT``` version to v1.2.
- Default sampling frequency of WinTAur Lite output changed to 48 kHz to match ```SQAT```.

### Removed
- Removed the ```tone_generation``` module, because it was not very usefull.


## [v0.0.2]
### Added
- Addition of WinTAur Lite code to noisetools.
  - Main goal: allow for easy creation and modification of ```.aurlite``` files outside WinTAur.
  - Includes Case and Project classes, without the ```.run()``` functions.


## [v0.0.1]

### Added
- Addition of the CHANGELOG :)
- Function in PySQAT to obtain a dataframe with only the time series of the instantaneous SQMs.

### Changed
- Switching to a slightly different semantic versioning number to move away from the '-dev0' notation that is just hideous.


## [v.0.1.0-dev0]
Very first version. Very pre-release...
This is a composition of a collection of functions from my MSc. Thesis, that were used for signal analysis. Also some functions developed to comply with the noise analysis standards in IEC 61400-11:2012.

### Added
- Functions to enable spectral weighting in time and frequency domain (weighting_functions.py)
- Tone generation function to allow for the creation of tonal signals with ease (tone_generation.py)
- Determination of the sound levels, with and without weighting (sound_levels.py)
- Created a basic translation layer to easily run SQAT through Python. 
  - Currently supports the direct computation of the three formulations of Pyschoacoustic annoyance. 
  - Other metrics are also obtained through these functions.
