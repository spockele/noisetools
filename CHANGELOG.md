# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [v0.2.1.dev2] - UNRELEASED

### Added

- ```WavFile``` class:
  - ```from_one_channel()``` function to create a new WavFile instance from a mono signal array, similar to ```from_two_channel()```.

- ```sound_levels``` module:
  - Added parameter ```centered: bool = True``` to ```octave_spectrogram()``` to allow the same time series control of the ```t_out()``` function.


### Changed

- ```WavFile``` class:
  - Changed typehint of ```pcm``` parameter to a Literal with the accepted values.
  - Move the ```pcm_table``` to the class itself, as it is only useful for ```WavFile.__init__()```


[//]: # (### Deprecated)

[//]: # (### Removed)

[//]: # (### Fixed)

[//]: # (### Security)


## [v0.2.1.dev1] - 4 August 2026

### Added
- ```wintaur``` module: new parameters in the configspec:
  - ```[hawc2_noise] aerosections```: the number of HAWC2 aerodynamic calculation points.
  - ```[hawc2_noise] aero_distribution```: the distribution method of HAWC2 aerodynamic calculation points.
  - New section: ```[propagation]```
  - ```[propagation] mode```: selects the mode of calculating the propagation effects.
  - ```[propagation] constant```: sets the constant *x dB* amplification if ```mode = 'constant'```. 
  - ```[propagation] hub_diameter```: the rotor diameter used for the propagation effect calculations.
  - ```[propagation] n_sources```: the number of point sources used when ```mode = 'distribute```.

### Changed
- ```wintaur``` module: changed parameters in the configspec:
  - Moved ```rotor_diameter``` from ```[hawc2_noise]``` to ```[propagation]```
  - Moved ```source_rr``` from ```[hawc2_noise]``` to ```[propagation]```


## [v0.2.0] - 29 July 2026
Major update with many added tools and functions. 

**NOT backwards compatible with v0.1.0.**

### Added
- ```OctaveBand``` class in ```octave_band``` module:
  - Added class 1 IEC 61260-1:2013 compliant octave band filter design function.
  - Option to define a frequency range between which the octave bands should be defined.
  - Added a general frequency range for fractional octave bands other than those previously defined.


- ```sound_levels``` module:
  - ```octave_spectrum``` function to calculate an average SPL spectrum in octave bands, of a signal .
  - ```octave_spectrogram``` function to calculate SPL over time in octave bands, of a signal.
  - ```amplitude_modulation``` function to calculate the amplitude modulation depth, with the method by Bass et al. (2016).
  - ```t_out``` function to obtain the timesteps matching the ```ospl_t``` function.
  - ```p2eq_t``` function to calculate equivalent pressure over time (similar to ```ospl_t```).


- ```wintaur``` module (configspec):
  - Added ```mech``` parameter to select the noise generation mechanism.
  - Added ```seed``` parameter to set the seed for the random phase generator.
  - Added ```rotor_diameter``` parameter to define the rotor diameter of the turbine.
  - Added ```source_rr``` parameter to define the virtual source location as r/R.


- ```hawc2``` module:
  - Added ```read_hawc2_aedata``` function to read the aerodynamic blade layout files for HAWC2.
  - ```numbered_columns``` optional parameter in ```read_hawc2_res()``` to replicate old column name behaviour.


- ```PySQAT``` sub-package:
  - Added the optional ```overwrite``` variable to the ```process_directory``` functions, so it doesn't do double work after an interruption.


- ```WavFile``` class in ```wavfile``` module:
  - Added option for a multiplicative calibration value ```cal``` in class initialiser. To avoid double divisions.


- ```demo_array``` module to work with data from our section's in-house microphone array solution.


### Changed
- ```OctaveBand``` class in ```octave_band``` module:
  - Changed the column names of the frequencies in self.f to match the IEC standard.
  - The default band range for 1/1 and 1/3 octave bands now matches the nominal frequencies defined in the IEC standard. 1/6 and 1/12 band ranges have been modified to a similar range.


- ```ospl_t``` function in ```sound_levels``` module:
  - Changed the method for splitting in timesteps. This has significantly improved compute times for this function.
  - **The variable ```t``` has been removed!!**
  - **The function no longer returns the time output!!** This has been separated into a new function ```t_out``` (see the **Added** section)


- ```pySQAT``` sub-package: SQAT updated to v1.3.


- ```hawc2``` module:
  - Updated logic to read the whole final column of the sel file in ```read_hawc2_res()```.


### Deprecated
- ```equivalent_pressure``` in ```sound_levels``` has been renamed to ```p2eq```.
- Development function ```ospl_t_out``` in ```sound_levels``` renamed to ```t_out```


### Fixed
- Time compensation of the OSPL function was wrong when ```t``` was not provided. Changed the integration to use the discrete signal length instead.
- Constant extrapolation in ```OctaveBand.interp1d_to_narrowband``` resulted in zeroes outside the band range, which was not desired behaviour.
- Time array in ```wavfile.WavFile``` included the end time, which resulted in a slightly offset sampling frequency of this array.


## [v0.1.0] - 8 July 2025
Public version of this repository. No functional differences from v0.0.3.


## [v0.0.3] - 13 May 2025

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
- Removed the ```tone_generation``` module, because it was not very useful.


## [v0.0.2] - 8 January 2025
### Added
- Addition of WinTAur Lite code to noisetools.
  - Main goal: allow for easy creation and modification of ```.aurlite``` files outside WinTAur.
  - Includes Case and Project classes, without the ```.run()``` functions.


## [v0.0.1] - 3 January 2025

### Added
- Addition of the CHANGELOG :)
- Function in PySQAT to obtain a dataframe with only the time series of the instantaneous SQMs.

### Changed
- Switching to a slightly different semantic versioning number to move away from the '-dev0' notation that is just hideous.


## [v0.0.1.dev0] - 27 August 2024
Very first version. Very pre-release...
This is a composition of a collection of functions from my MSc. Thesis, that were used for signal analysis. Also some functions developed to comply with the noise analysis standards in IEC 61400-11:2012.

### Added
- Functions to enable spectral weighting in time and frequency domain (weighting_functions.py)
- Tone generation function to allow for the creation of tonal signals with ease (tone_generation.py)
- Determination of the sound levels, with and without weighting (sound_levels.py)
- Created a basic translation layer to easily run SQAT through Python. 
  - Currently supports the direct computation of the three formulations of Pyschoacoustic annoyance. 
  - Other metrics are also obtained through these functions.
