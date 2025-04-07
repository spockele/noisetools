""" Code for data acquisition of our KEMAR HATS, Francois.
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

from nidaqmx.constants import READ_ALL_AVAILABLE
from noisetools.wavfile import WavFile
import nidaqmx as ni
import pandas as pd
import numpy as np
import time
import os


# Define NI cDAQ parameters.
daq_id = 'cDAQ1Mod1'
channels = ['ai0', 'ai1', 'ai2', 'ai3', ]
sensitivity = [1., 1., 1., 1., ]

# Define recording parameters.
fs = 51.2e3
t_record = 10.


if __name__ == '__main__':
    # Determine number of samples.
    n_samples = int(fs * t_record)

    # Create measurements folder if required.
    if not os.path.isdir('measurements'):
        print(f'Created folder ./measurements.')
        os.mkdir('measurements')

    # Initialise the DAQ and add both ears' channels
    with ni.Task() as task:
        for ci, channel in enumerate(channels):
            task.ai_channels.add_ai_microphone_chan(f'{daq_id}/{channel}',
                                                    mic_sensitivity=sensitivity[ci], )

        # Loop until user break.
        repeat = True
        while repeat:
            # Ask for a filename
            out_filename = input('Enter output filename (defaults to <timestamp>.csv): ')
            # Set a default timestamp filename
            if not out_filename:
                timestamp = '_'.join(time.ctime(time.time()).replace(":", "-").split())
                out_filename = f'{timestamp}.csv'
                print(f'\tDefaulted to {out_filename}')

            # Make sure the filename has the '.wav' extension.
            elif not out_filename.endswith('.csv'):
                out_filename = out_filename + '.csv'
            # Set the full output path.
            out_path = os.path.join('measurements', out_filename)

            # Measure after user start.
            input('Press enter to start measurement: ')
            task.timing.cfg_samp_clk_timing(fs, samps_per_chan=n_samples, )
            dat = task.read(READ_ALL_AVAILABLE, timeout=t_record)
            print('\tDone!')

            dat = np.array(dat).T
            # Create a pandas DataFrame and store it as csv.
            df = pd.DataFrame(dat, index=np.linspace(0., t_record, num=dat.shape[0]), columns=channels)
            df.to_csv(out_path.replace('.wav', '.csv'))

            # Let user break the loop.
            repeat = not bool(input(f'\tMeasurement succesfully stored as: {out_path}.\n'
                                    f'Press enter to measure again: '))
            print()
