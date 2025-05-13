# Python Noise Toolbox (Noisetools)
A Python noise analysis and research toolbox developed for my PhD.

![GitHub Release](https://img.shields.io/github/v/release/spockele/noisetools?logo=github) ![GitHub License](https://img.shields.io/github/license/spockele/noisetools)

---

### Developed by
[**_Josephine Pockelé_**](https://orcid.org/0009-0002-5152-9986)\
PhD Candidate, [TU Delft, faculty of Aerospace Engineering](https://www.tudelft.nl/lr/).\
Email: [j.s.pockele@tudelft.nl](mailto:j.s.pockele@tudelft.nl)\
LinkedIn: [Josephine Pockelé](https://www.linkedin.com/in/josephine-pockele)

## Noisetools Package

The primary component of this repository is the *noisetools* package. Wheels and tarballs of this package are included in the GitHub releases.

- ***octave_band.py***

    Module containing a class to handle operations with octave band sound/attenuation/amplification spectra.
    It is recommended to only use 1/1, 1/3, 1/6 and 1/12.
    
    This module is designed to comply with IEC 61260-1:2014 [[1](#iec61260)]


- ***sound_levels.py***
    
    Module with functions to determine equivalent sound pressure levels of sound signals.


- ***wavfile.py***

    Contains a class to handle operations with WAV audio files. 


- ***weighting_functions.py***

    Functions related to A- and C- weighting of sound signals.
    This module is designed to compy with IEC 61672-1:2013 [[2](#iec61672)]


- ***wintaur.py***

    Functions to make working with WinTAur Lite easier. Currently compatible with latest version.


- ***hawc2.py***

    Module with functions to interact with [HAWC2](http://www.hawc2.dk/).


### pySQAT
Translation layer to run SQAT in Python. This sub-package includes a licensed copy of SQAT v1.2 [[3](#greco2023)].

Currently only supports the following SQMs: 

IMPORTANT: to use pySQAT, and the required [matlabengine](https://pypi.org/project/matlabengine/) package, an activated installation of [MATLAB](https://www.mathworks.com/products/matlab.html) is required.\
The authors recommend manually installing the matlabengine version compatible with your MATLAB and Python versions. See the [PyPI: matlabengine version history](https://pypi.org/project/matlabengine/#history) for more information about compatibility of the releases.


## Data Acquisition scripts
Two Python scripts for data acquisition using a NI cDAQ-9171 with NI 9234 module. Recording to different file format(s) is work in progress. Recording to CSV is very storage-intensive!!

A pip _requirements_ file is included in the directory with these scripts.


- ***francois.py***

    Script with two-channel recording to WAV and CSV format. Specifically designed for use with a Head- And Torso Simulator.
    Named after the HATS of the Operations and Environment section, François.
    
    IMPORTANT: this script requires the noisetools package.


- ***4-channel.py***

    Script with (up to) 4 channel recording to CSV format. This script can be modified to record using any configuration of a [NI DAQmx](https://nidaqmx-python.readthedocs.io/en/stable/) compatible device.


## Copyright Notices

Copyright 2025 Josephine Pockelé

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

---

Technische Universiteit Delft hereby disclaims all copyright interest in the program “noisetools”, a Python toolbox for noise analysis in scientific research, written by the Author(s).<br>
Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

SQAT 1.2 [(available on GitHub)](https://github.com/ggrecow/SQAT) [[3](#greco2023)], is included in this work under the terms of the CC BY-NC 4.0 license.


## References
<a id="iec61260">[1]</a> International Electrotechnical Commission (IEC), ‘Electroacoustics - Octave-band and fractional-octave-band filters - Part 1: Specifications’, International Standard IEC 61260-1:2014, Feb. 2014.

<a id="iec61672">[2]</a> International Electrotechnical Commission (IEC), ‘Electroacoustics - Sound Level Meters - Part 1: Specifications’, International Standard IEC 61672-1:2013, Sep. 2013.

<a id="greco2023">[3]</a> G. F. Greco, R. Merino-Martínez, and A. Osses, SQAT: a sound quality analysis toolbox for MATLAB. (Jan. 13, 2025). Zenodo. DOI: [10.5281/ZENODO.14641811](https://doi.org/10.5281/ZENODO.14641811)
