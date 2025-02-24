from nidaqmx.constants import READ_ALL_AVAILABLE
from noisetools.wavfile import WavFile
import matplotlib.pyplot as plt
import nidaqmx as ni
import numpy as np
import os


daq_id = 'cDAQ2Mod1'
channels = ['ai0', 'ai1', ]
sensitivity = [4.596, 4.331, ]

fs = 51.2e3
t_record = 1.


out_filename = input('Output filename: ')
if not out_filename:
    outfiles = [fname for fname in os.listdir('measurements') if fname.endswith('.wav')]
    outfiles_nrs = [float(fname[:-5].split('_')[-1]) for fname in outfiles if fname.startswith('out')]
    top_nr = str(max(outfiles_nrs)).zfill(3) if outfiles_nrs else '001'

    out_filename = f'out_{top_nr}.wav'

out_path = os.path.join('measurements', out_filename)


if __name__ == '__main__':
    n_samples = int(fs * t_record)

    with ni.Task() as task:
        for ci, channel in enumerate(channels):
            task.ai_channels.add_ai_microphone_chan(f'{daq_id}/{channel}', mic_sensitivity=sensitivity[ci])

        task.timing.cfg_samp_clk_timing(fs, samps_per_chan=n_samples, )
        dat = task.read(READ_ALL_AVAILABLE)

    dat = np.array(dat).T

    # plt.plot(dat)
    # plt.show()

    wav = WavFile(out_path, norm=1., wav=dat, fs=int(fs))
    wav.write()
