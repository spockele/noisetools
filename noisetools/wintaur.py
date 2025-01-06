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
import numpy as np
import warnings
import os

from . import hawc2 as h2

# Define the WinTAurCase configspec.
configspec_path = os.path.join(os.path.dirname(__file__), 'wintaur_lite_configspec.aurlite')
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
                 ) -> None:
        self.project_path = project_path
        self.case_names = [case.replace('.aurlite', '') for case in os.listdir(self.project_path)
                           if case.endswith('.aurlite')]
        self.cases = [WinTAurCase(self.project_path, case) for case in self.case_names]
