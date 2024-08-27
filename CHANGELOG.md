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

## [v0.0.1]

### Added
- Addition of the CHANGELOG :)
- Function in PySQAT to obtain a dataframe with only the time series of the instantaneous SQMs.

### Changed
- Switching to a slightly different semantic versioning number to move away from the '-dev0' notation that is just hideous.

### Deprecated

### Removed

### Fixed

## [v.0.1.0-dev0]
Very first version. Very pre-release...
This is a composition of a collection of functions from my MSc. Thesis, that were used for signal analysis. Also some functions developed to comply with the noise analysis standards in IEC 61400-11:2012.

### Added
- Functions to enable spectral weighting in time and frequency domain (weighting_functions.py)
- Tone generation function to allow for the creation of tonal signals with ease (tone_generation.py)
- Determination of the sound levels, with and without weighting (sound_levels.py)
- Created a basic translation layer to easily run SQAT through Python. Currently supports the direct computation of the three formulations of Pyschoacoustic annoyance. Other metrics are also obtained through these functions.
