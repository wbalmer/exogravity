# import
import numpy as np
import cleanGravity as cg
import glob

# load files
datafiles = glob.glob("./data/onaxis/*.fits")
datafiles.sort()
ois = []
for filename in datafiles:
    oi = cg.GravityDualfieldAstrored(filename, extension = 10, corrDisp = "drs", corrMet = "drs")
    # print some info
    print("At mjd={:.4f}, fiber was at RA={:.2f} mas, DEC={:.2f} mas, flux was {:.2f} ADU/s".format(oi.mjd, oi.sObjX, oi.sObjY, oi.fluxOi.flux.mean()))
    ois.append(oi)

# separate on-star from on-planet
starOis = [ois[0], ois[2]]
plaOi = ois[1]

# calculate the visbility reference
visReference = 0.5*(starOis[0].visOi.visRef.mean(axis = 0) + starOis[1].visOi.visRef.mean(axis = 0))

# get the reference amplitude and phase
ampRef = np.abs(visReference)
phaseRef = np.angle(visReference)

# phase-reference the on-planet observation
plaOi.visOi.addPhase(-phaseRef)

# plot the phase-referenced visibility
import matplotlib.pyplot as plt
from cleanGravity import gravityPlot as gplot
fig = plt.figure()
gplot.reImPlot(np.tile(plaOi.wav*1e6, (plaOi.visOi.nchannel, 1)), plaOi.visOi.visRef.mean(axis = 0), subtitles = plaOi.basenames, fig = fig)

# fit the stellar component with a polynomial
polyOrder = 4
Y = np.zeros([oi.nwav, 1], 'complex')
X = np.zeros([oi.nwav, polyOrder+1])
starFit = np.zeros([plaOi.visOi.ndit, plaOi.visOi.nchannel, plaOi.nwav], 'complex')
for c in range(plaOi.visOi.nchannel):
    for k in range(polyOrder+1):
        X[:, k] = ampRef[c, :]*((plaOi.wav - plaOi.wav.mean(axis = 0))*1e6)**k
    for dit in range(plaOi.visOi.ndit):
        Y[:, 0] = plaOi.visOi.visRef[dit, c, :]
        A = np.dot(np.linalg.pinv(X), Y)
        starFit[dit, c, :] = np.dot(X, A)[:, 0]        

# overplot the result of the fit to the previous fig
gplot.reImPlot(np.tile(oi.wav*1e6, (oi.visOi.nchannel, 1)), starFit.mean(axis = 0), fig = fig)

# subtract the stellar component for the planet OI and plot the result
plaOi.visOi.visRef = plaOi.visOi.visRef - starFit
fig = plt.figure()
gplot.reImPlot(np.tile(oi.wav*1e6, (oi.visOi.nchannel, 1)), plaOi.visOi.visRef[0, :, :], subtitles = plaOi.basenames, fig = fig)


# extract the astrometry using a chi2 map approach
raValues = np.linspace(66, 70, 20) # the ra grid
decValues = np.linspace(124, 128, 20) # the dec grid
chi2Map = np.zeros([plaOi.visOi.ndit, len(raValues), len(decValues)]) # one map for each dit
Y, X = np.zeros([1, 2*plaOi.nwav]), np.zeros([1, 2*plaOi.nwav]) # Y will contain data, X will contain model
planetFit = np.zeros([plaOi.visOi.ndit, plaOi.visOi.nchannel, plaOi.nwav], 'complex') # to save the fit
chi2Min = np.inf
for m in range(len(raValues)):
    for n in range(len(decValues)): # loop through grid
        ra, dec = raValues[m]/1000.0/3600.0/180.0*np.pi, decValues[n]/1000.0/3600.0/180.0*np.pi # convert to rad
        wavelet = np.exp(-1j*2*np.pi*(ra*plaOi.visOi.uCoord+dec*plaOi.visOi.vCoord)) # the visibility model, without the amplitude
        for dit in range(plaOi.visOi.ndit):
            for c in range(plaOi.visOi.nchannel): 
                # here we need to explicitly separate the real/imag parts and do the fitting for the
                # scaling coefficient in real numbers. Otherwise we would get a complex coeff, which
                # means it would also be fitting for the phase offset. 
                X[0, 0:plaOi.nwav] = np.real(wavelet[dit, c, :]) # model
                X[0, plaOi.nwav:] = np.imag(wavelet[dit, c, :]) # model               
                Y[0, 0:plaOi.nwav] = np.real(plaOi.visOi.visRef[dit, c, :]) # data 
                Y[0, plaOi.nwav:] = np.imag(plaOi.visOi.visRef[dit, c, :]) # data
                coeff = np.dot(Y, np.linalg.pinv(X)) # linear fit
                if coeff[0, 0] < 0: # if coeff is negative, set model to 0 and redo the fit
                    coeff[0, 0] = 0
                    X[0, :] = 0
                    coeff = np.dot(Y, np.linalg.pinv(X))
                Yfit = np.dot(coeff, X)
                planetFit[dit, c, :] = Yfit[0, 0:plaOi.nwav] + 1j*Yfit[0, plaOi.nwav:]
                chi2Map[dit, m, n] = chi2Map[dit, m, n]+np.sum((Y-Yfit)**2) # sum of squared differences
        if chi2Map[0, m, n] < chi2Min: # just for the example, we look for best fit on first dit
            bestPlanetFit = np.copy(planetFit)
            chi2Min = chi2Map[dit, m, n]
gplot.reImPlot(np.tile(oi.wav*1e6, (oi.visOi.nchannel, 1)), bestPlanetFit[0, :, :], fig = fig)        
# show the map
plt.figure()
plt.imshow(np.sum(chi2Map, axis = 0).T, origin="lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])


