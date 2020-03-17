import numpy as np
import os

indir = "/home/sdevgupta/mine/Text2Style/logs_coords_attention/outputs"

files = [ os.path.join(indir, i) for i in sorted(os.listdir(indir)) ]

stacked = np.array([ np.load(i) for i in files])
print(stacked.shape)
np.save("/home/sdevgupta/mine/Text2Style/logs_coords_attention/stacked_mels_arch_coordconv_attention.npy", stacked)
