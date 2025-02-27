from nidaqmx.constants import READ_ALL_AVAILABLE
from noisetools.wavfile import WavFile
import nidaqmx as ni
import pandas as pd
import numpy as np
import os


daq_id = 'cDAQ1Mod1'
channels = ['ai0', 'ai1', ]
sensitivity = [4.596, 4.331, ]

fs = 51.2e3
t_record = 12.


if __name__ == '__main__':
    if not os.path.isdir('measurements'):
        print(f'Created folder ./measurements.')
        os.mkdir('measurements')

    n_samples = int(fs * t_record)

    with ni.Task() as task:
        for ci, channel in enumerate(channels):
            task.ai_channels.add_ai_microphone_chan(f'{daq_id}/{channel}',
                                                    mic_sensitivity=sensitivity[ci], )

        repeat = True
        while repeat:
            out_filename = input('Enter output filename (defaults to out_xxx.wav): ')
            if not out_filename:
                outfiles = [fname for fname in os.listdir('measurements') if fname.endswith('.wav')]
                outfiles_nrs = [int(fname[:-5].split('_')[-1]) for fname in outfiles if fname.startswith('out')]
                print(outfiles_nrs)
                top_nr = str(max(outfiles_nrs) + 2).zfill(3) if outfiles_nrs else '001'
                print(top_nr)

                out_filename = f'out_{top_nr}.wav'

            out_path = os.path.join('measurements', out_filename)

            input('Press enter to start measurement: ')
            task.timing.cfg_samp_clk_timing(fs, samps_per_chan=n_samples, )
            dat = task.read(READ_ALL_AVAILABLE, timeout=t_record)

            print('Done!')
            dat = np.array(dat).T

            wav = WavFile(out_path, norm=1., wav=dat, fs=int(fs))
            wav.write()

            pd.DataFrame(dat, index=np.linspace(0, t_record, wav.length), columns=['left', 'right']).to_csv(out_path.replace('.wav', '.csv'))
            repeat = not bool(input(f'Measurement succesfully stored as: {out_path}.\n'
                                    f'Press enter to repeat: '))
