# load a file with cleanGravity
import cleanGravity as cg
oi = cg.GravityDualfieldAstrored("./data/GRAVI.2019-11-11T06:29:03.495_astroreduced.fits", extension = 10)

# print the shape of visibility data
print("Visibility data array is of shape {}".format(oi.visOi.visData.shape))

# plot the flux on each telescope
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 8))
for k in range(oi.fluxOi.nchannel):
    ax = fig.add_subplot(2, 2, k+1)
    ax.plot(oi.wav*1e6, oi.fluxOi.flux[0, k, :])
    ax.set_xlabel("Wavelength ($\mu$m)")
    ax.set_ylabel("Flux (ADU/s)")
    ax.set_title("Flux of DIT=0 for telescope {}".format(oi.telnames[k]))
plt.show()

# plot visibilities using cleanGravity.gravityPlot
from cleanGravity import gravityPlot as gplot
import numpy as np
gplot.reImPlot(np.tile(oi.wav, (oi.visOi.nchannel, 1)), oi.visOi.visData[0, :, :], subtitles = oi.basenames)

# for purely real data (like the flux)
gplot.baselinePlot(np.tile(oi.wav, (oi.fluxOi.nchannel, 1)), oi.fluxOi.flux[0, :, :], subtitles = oi.telnames)

# apply phase corrections when loading the file
oi = cg.GravityDualfieldAstrored("./data/GRAVI.2019-11-11T06:29:03.495_astroreduced.fits", extension = 10, corrDisp = "drs", corrMet = "drs")
