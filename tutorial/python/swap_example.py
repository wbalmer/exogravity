# useful imports
import numpy as np
import matplotlib.pyplot as plt
import cleanGravity as cg
from cleanGravity import gravityPlot as gplot
import glob

# load a file with cleanGravity
oi = cg.GravityDualfieldAstrored("./data/HD25535/GRAVI.2019-11-11T04:18:15.164_astroreduced_s.fits", extension = 10)

# difference between raw and FT referenced visbilities
fig = plt.figure()
gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visData[0, :, :], subtitles = oi.basenames, fig = fig)
gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visRef[0, :, :], fig = fig)

# see how the visData shift with FT reference at each DIT
fig = plt.figure()
gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visData[0, :, :], subtitles = oi.basenames, fig = fig)
for k in range(1, 4):
    gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visData[k, :, :], fig = fig)

# visRef is much more stable
fig = plt.figure()
gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visRef[0, :, :], subtitles = oi.basenames, fig = fig)
for k in range(1, 4):
    gplot.reImPlot(np.tile(oi.wav, (6, 1)), oi.visOi.visRef[k, :, :], fig = fig)

# correct for OPD dispersion
oi.visOi.visRef = oi.visOi.visRef * np.exp(-1j*2*np.pi*oi.visOi.opdDisp/oi.visOi.wav)

# OPD correction when loading the file
oi = cg.GravityDualfieldAstrored("./data/HD25535/GRAVI.2019-11-11T04:18:15.164_astroreduced_s.fits", extension = 10, corrDisp = "drs")

# an alternative OPD correction, for '_s.fits' file
oi = cg.GravityDualfieldAstrored("./data/HD25535/GRAVI.2019-11-11T04:18:15.164_astroreduced_s.fits", extension = 10, corrDisp = "sylvestre")

# metrology correction when loading the file
oi = cg.GravityDualfieldAstrored("./data/HD25535/GRAVI.2019-11-11T04:18:15.164_astroreduced_s.fits", extension = 10, corrDisp = "drs", corrMet = "drs")
                
# matrix to link telescopes and baselines
telCorrToBaseCorr = np.array([[1, -1, 0, 0],
                               [1, 0, -1, 0],
                               [1, 0, 0, -1],
                               [0, 1, -1, 0],
                               [0, 1, 0, -1],
                               [0, 0, 1, -1]])

# manual metrology correction
phaseCorr = 2*np.pi/oi.wav*(np.dot(telCorrToBaseCorr, (oi.fluxOi.telFcCorr + oi.fluxOi.fcCorr).T).T[:, :, None])
oi.visOi.visRef = oi.visOi.visRef*np.exp(-1j*phaseCorr)

# average the visbilities over the DITs
oi = cg.GravityDualfieldAstrored("./data/HD25535/GRAVI.2019-11-11T04:26:33.186_astroreduced_s.fits", extension = 10, corrMet = "drs", corrDisp = "drs")
oi.visOi.recenterPhase(oi.sObjX, oi.sObjY) # shift to 0 OPD (or close)
oi.visOi.visData = np.tile(np.copy(oi.visOi.visData).mean(axis = 0), (1, 1, 1))
oi.visOi.visRef = np.tile(np.copy(oi.visOi.visRef).mean(axis = 0), (1, 1, 1))
oi.visOi.uCoord = np.tile(np.copy(oi.visOi.uCoord).mean(axis = 0), (1, 1, 1))
oi.visOi.vCoord = np.tile(np.copy(oi.visOi.vCoord).mean(axis = 0), (1, 1, 1))
oi.visOi.u = np.tile(np.mean(oi.visOi.u, axis = 0), (1, 1))
oi.visOi.v = np.tile(np.mean(oi.visOi.v, axis = 0), (1, 1))
oi.visOi.ndit = 1
oi.ndit = 1
oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY) # back to FT reference position


# load the files, and separate in two groups for 2 swap positions
datafiles = glob.glob("./data/HD25535/*_s.fits")
datafiles.sort()
oi_stand, oi_swap = [], []
for filename in datafiles:
    oi = cg.GravityDualfieldAstrored(filename, extension = 10, corrDisp = "drs", corrMet = "drs")
    if oi.swap:
        oi_swap.append(oi)
    else:
        oi_stand.append(oi)

# average each group properly
visStand = np.zeros([oi.visOi.nchannel, oi.visOi.nwav])
visSwap = np.zeros([oi.visOi.nchannel, oi.visOi.nwav])
for oi in oi_stand:
    oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
    visStand = visStand+oi.visOi.visRef.mean(axis = 0)/len(oi_stand)
    oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)    
for oi in oi_swap:
    oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
    visSwap = visSwap+oi.visOi.visRef.mean(axis = 0)/len(oi_swap)
    oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)

# get the phase reference and remove it from each OI
phaseRef = 0.5*(np.angle(visStand)+np.angle(visSwap))
for oi in oi_stand+oi_swap:
    oi.visOi.addPhase(-phaseRef)


# chi2 map to extract the astrometry
oi = oi_stand[0] # we fit only the first oi in this example
raValues = np.linspace(820, 834, 10) # the ra grid
decValues = np.linspace(745, 755, 10) # the dec grid
chi2Map = np.zeros([oi.visOi.ndit, len(raValues), len(decValues)]) # one map for each dit
Y, X = np.zeros([1, 2*oi.nwav]), np.zeros([1, 2*oi.nwav]) # Y will contain data, X will contain model
for m in range(len(raValues)):
    for n in range(len(decValues)): # loop through grid
        ra, dec = raValues[m]/1000.0/3600.0/180.0*np.pi, decValues[n]/1000.0/3600.0/180.0*np.pi # convert to rad
        wavelet = np.exp(-1j*2*np.pi*(ra*oi.visOi.uCoord+dec*oi.visOi.vCoord)) # the visibility model, without the amplitude
        for dit in range(oi.visOi.ndit):
            for c in range(oi.visOi.nchannel): 
                # here we need to explicitly separate the real/imag parts and do the fitting for the
                # scaling coefficient in real numbers. Otherwise we would get a complex coeff, which
                # means it would also be fitting for the phase offset. 
                X[0, 0:oi.nwav] = np.real(wavelet[dit, c, :]) # model
                X[0, oi.nwav:] = np.imag(wavelet[dit, c, :]) # model               
                Y[0, 0:oi.nwav] = np.real(oi.visOi.visRef[dit, c, :]) # data 
                Y[0, oi.nwav:] = np.imag(oi.visOi.visRef[dit, c, :]) # data
                coeff = np.dot(Y, np.linalg.pinv(X)) # linear fit
                if coeff[0, 0] < 0: # if coeff is negative, set model to 0 and redo the fit
                    coeff[0, 0] = 0
                    X[0, :] = 0
                    coeff = np.dot(Y, np.linalg.pinv(X))
                Yfit = np.dot(coeff, X)
                chi2Map[dit, m, n] = chi2Map[dit, m, n]+np.sum((Y-Yfit)**2) # sum of squared differences
# show the map
plt.figure()
plt.imshow(np.sum(chi2Map, axis = 0).T, origin="lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
        

