import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
plt.ion()


hdu = fits.open('/home/mnowak/Documents/betapic_gravity/cds/spectrum.fit')

data = np.loadtxt("highres/contrast.txt", skiprows=2)

plt.plot(hdu[1].data, hdu[3].data)
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('Wavelength ($\mu\mathrm{m}$)')
plt.ylabel("Contrast")
