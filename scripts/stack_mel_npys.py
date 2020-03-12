import numpy as np
import os

indir = "/home/sdevgupta/mine/Text2Style/logs2/outputs"

files = [ os.path.join(indir, i) for i in os.listdir(indir) ]

stacked = np.array([ np.load(i) for i in files])
print(stacked.shape)
np.save("stacked_mels.npy", stacked)