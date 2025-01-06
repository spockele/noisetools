""" Functions for interaction with WinTAur Lite
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

from configobj import ConfigObj, flatten_errors, get_extra_values
from configobj.validate import Validator, VdtMissingValue
import warnings
import os

from . import hawc2 as h2

# Define the WinTAurCase configspec.
configspec_path = os.path.join(os.path.dirname(__file__), 'configspec.aurlite')
configspec = ConfigObj(configspec_path, interpolation=False, list_values=False, _inspec=True)
# Create a configobj.validate.Validator instance.
validator = Validator()

__all__ = ['WinTAurProject', 'WinTAurCase', ]


class WinTAurCase(ConfigObj):
    """
    Subclass of ConfigObj to manage a WinTAur Lite case.

    Parameters
    ----------
    project_path: str | os.PathLike
        Path to the WinTAur Lite project folder.
    case_name: str | os.PathLike
        Name of the case file as a path relative to the project folder. Without the .aurlite file extension.

    """
    def __init__(self,
                 project_path: str | os.PathLike,
                 case_name: str | os.PathLike,
                 new: bool = False
                 ) -> None:
        self.project_path = project_path
        self.case_name = case_name
        # Define the path of this specific case.
        self.case_path: str = os.path.join(project_path, case_name + '.aurlite')
        self.wav_path = os.path.join(self.project_path, 'wav')
        self.base_wav_path = os.path.join(self.wav_path, case_name)
        # Initialise the ConfigObj class.
        if new:
            # From scratch.
            super().__init__(configspec=configspec, create_empty=True)
            self.filename = self.case_path
        else:
            # Read existing. (will start new one when it does not exist yet).
            super().__init__(self.case_path, configspec=configspec)
            self.validate_case()

    def validate_case(self):
        """
        Function to do the validation of a new WinTAur case.
        """
        # Run the validation of the case file.
        validation = self.validate(validator, preserve_errors=True)

        # Check for HAWC2 output files.
        if (not os.path.exists(h2.fname_hawc2_noise_psd(self.project_path, self.case_name, 1))
                and not self['hawc2_noise']['run']):
            raise FileNotFoundError(f"{self.case_name}, Section hawc2_noise, entry 'run': "
                                    f"No HAWC2 noise output found for this case. Set hawc2_noise.run to True.")

        # Go over the errors in the validation.
        for section, key, error in flatten_errors(self, validation):
            # If there is an actual Exception, change the message and raise it.
            if issubclass(type(error), Exception):
                msg = error.args[0]
                msg = f"{self.case_name}, Section {section}, entry '{key}': " + msg
                error.args = (msg,)
                raise error

            # If the error is a boolean, and it is False, a section entry is missing. Raise a VdtMissingValue.
            # Extra check for the hawc2_noise section, as this may not be required if hawc2_noise.run == False.
            elif isinstance(error, bool) and not (section[0] == 'hawc2_noise' and not self['hawc2_noise']['run']):
                if not error:
                    raise VdtMissingValue(f"{self.case_name}, Section {section}, entry '{key}': no value provided.")

        # Initialise collection of warnings for extra values.
        extra_warn = []
        # Get the extra entries outside the configspec.
        extra_values = get_extra_values(self, )
        # Loop over these extra entries.
        for section, key in extra_values:
            extra_warn.append(f"Section {list(section)}, entry '{key}': this entry is not used by WinTAur.")

        # First raise a VdtMissingValue in case no observers are defined.
        if not self['observers'] and self['hawc2_noise']['run']:
            raise VdtMissingValue(f"{self.case_path}: Section ['observers']: no observers defined.")

        # At the end, clear the HAWC2 section if run=False.
        if not self['hawc2_noise']['run']:
            self['hawc2_noise'] = {'run': False}
        # Only after that, warn the user of extra entries in the case file.
        for warn_msg in extra_warn:
            warnings.warn(warn_msg, category=UserWarning, stacklevel=2)

    def __repr__(self) -> str:
        return f"<WinTAurCase: '{self.case_name}' in project '{self.project_path}'>"


class WinTAurProject:
    """
    Class to manage a full WinTAur lite project.

    Parameters
    ----------
    project_path: str | os.PathLike
        Path to the WinTAur Lite project folder.
    """

    def __init__(self,
                 project_path: str | os.PathLike,
                 new: bool = False,
                 ) -> None:
        self.project_path = project_path

        if new:
            self.case_names = []
            self.cases = []
            if not os.path.exists(self.project_path):
                os.mkdir(self.project_path)

        else:
            self.case_names = [case.replace('.aurlite', '') for case in os.listdir(self.project_path)
                               if case.endswith('.aurlite')]
            self.cases = [WinTAurCase(self.project_path, case) for case in self.case_names]

    def new_case(self,
                 case_name: str,
                 temp: float = 15.,
                 pres: float = 101325.,
                 dens: float = 1.225,
                 humi: float = 80.,
                 grnd: str = 'grass',
                 run: bool = False,
                 base_htc: str = None,
                 time: tuple[float, float] = None,
                 simulation_dt: float = .01,
                 noise_dt: float = .5,
                 hub_height: float = None,
                 ws: float = None,
                 shear: list[int, float] = (3, 0.2),
                 wdir: list[float, float, float] = (0., 0., 0.),
                 ti: float = None,
                 z0: float = 1.0,
                 bldata: str = None,
                 observers: list[list[str, float, float, float]] = None,
                 fs: int = 44100,
                 overlap: int = 3
                 ) -> None:
        """
        Creates a new case in this project.

        Parameters
        ----------
        case_name: str | os.PathLike
            Name of the case file as a path relative to the project folder. Without the .aurlite file extension.

        Conditions Parameters
        ---
        temp: float, optional (default = 15.)
            Atmospheric temperature used for atmospheric attenuation.
            Default value  is ISA temperature at 0m [1]_ .
        pres: float, optional (default = 101325.)
            Atmospheric pressure used for atmospheric attenuation and HAWC2.
            Default value  is ISA pressure at 0m [1]_ .
        dens: float, optional (default = 1.225)
            Atmospheric density used for atmospheric attenuation and HAWC2.
            Default value is ISA density at 0m [1]_ .
        humi: float, optional (default = 80.)
            Atmospheric relative humidity used for atmospheric attenuation
            Default value is based on KNMI climate information [2]_ .
        grnd: str, optional (default = 'grass')
            Ground type used for the ground effect calculation.

        HAWC2 Parameters
        ---
        run: bool, optional (default = False)
            Indication to run the HAWC2 simulation.
        base_htc: str, optional (default = None)
            Name of the base htc input file, which defines the turbine structure, aerodynamics, and control.
            Required parameter when run = True.
        time: list[float, float], optional (default = None)
            Start and end time for the HAWC2 noise calculations.
            Required parameter when run = True.
        simulation_dt: float, optional (default = 0.01)
            Time step for the HAWC2 turbine aeroelastic simulation.
            Optional parameter when run = True.
        noise_dt: float, optional (default = 0.5)
            Time step for the HAWC2 turbine noise calculations.
            Optional parameter when run = True.
        hub_height: float, optional (default = None)
            Hub height of the wind turbine.
            Required parameter when run = True.
        ws: float, optional (default = None)
            Mean wind speed for the HAWC2 simulation.
            Required parameter when run = True.
        shear: list[int, float], optional (default = (3, 0.2))
            Wind shear parameters (see HAWC2 manual (wind -> shear_format) for more information [3]_).
            Optional parameter when run = True.
        wdir: list[float, float, float], optional (default = (0., 0., 0.)
            Wind direction input (see HAWC2 manual (wind -> windfield_rotations) for more information [3]_).
            Optional parameter when run = True.
        ti: float, optional (default = None)
            Turbulence intensity in percent for the HAWC2 simulation.
            Required parameter when run = True.
        z0: float, optional (default = None)
            Blade surface roughness z0 (see HAWC2 manual (aero_noise -> surface_roughness) for more information [3]_).
            Required parameter when run = True.
        bldata: str, optional (default = None)
            Filename of the boundary layer data file, relative to the project path.
            Required parameter when run = True.

        Observer parameters
        ---
        observers: list[list[str, float, float, float]], optional (default = None)
            List of observation points in the HAWC2 global coordinate system.
            Each point is a list containing: [name string, x, y, z].
            Required parameter when run = True.

        Reconstruction parameters
        ---
        fs: int, optional (default = 44100)
            Sampling frequency of the output wav files.
        overlap: int, optional (default = 3)
            Amount of overlap in the inverse short-time Fourier transform.

        References
        ----------
        .. [1] International Organisation for Standardisation (ISO), ‘Standard Atmosphere’, Geneva, Switzerland,
            International Standard ISO 2533:1975, May 1975.
        .. [2] Koninklijk Nederlands Meteorologisch Instituut (KNMI), ‘KNMI - Klimaatviewer’. Accessed: Nov. 13, 2024.
            [Online]. Available: https://www.knmi.nl/klimaat-viewer/kaarten/
        .. [3] T. J. Larsen and A. M. Hansen, ‘How 2 HAWC2, the user’s manual’, DTU, Department of Wind Energy,
            Roskilde, Denmark, Technical Report Risø-R-1597(ver. 13.0)(EN), May 2023. Accessed: Jan. 31, 2023.
            [Online]. Available: http://tools.windenergy.dtu.dk/HAWC2/manual/

        """
        new_case = WinTAurCase(self.project_path, case_name, new=True)

        new_case['conditions'] = {}
        new_case['conditions']['temp'] = temp
        new_case['conditions']['pres'] = pres
        new_case['conditions']['dens'] = dens
        new_case['conditions']['humi'] = humi
        new_case['conditions']['grnd'] = grnd

        if run:
            new_case['hawc2_noise'] = {}
            new_case['hawc2_noise']['run'] = run
            new_case['hawc2_noise']['base_htc'] = base_htc
            new_case['hawc2_noise']['time'] = time
            new_case['hawc2_noise']['simulation_dt'] = simulation_dt
            new_case['hawc2_noise']['noise_dt'] = noise_dt
            new_case['hawc2_noise']['hub_height'] = hub_height
            new_case['hawc2_noise']['ws'] = ws
            new_case['hawc2_noise']['shear'] = shear
            new_case['hawc2_noise']['wdir'] = wdir
            new_case['hawc2_noise']['ti'] = ti
            new_case['hawc2_noise']['z0'] = z0
            new_case['hawc2_noise']['bldata'] = bldata

        if observers is not None:
            new_case['observers'] = {}
            for observer in observers:
                new_case['observers'][observer[0]] = {}
                new_case['observers'][observer[0]]['name'] = observer[0]
                new_case['observers'][observer[0]]['pos'] = observer[1:4]

        new_case['reconstruction'] = {}
        new_case['reconstruction']['fs'] = fs
        new_case['reconstruction']['overlap'] = overlap

        new_case.validate_case()
        new_case.write()

        self.case_names.append(case_name)
        self.cases.append(new_case)
