import numpy as np
import matplotlib.pyplot as plt
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity import gravityPlot
import glob

datadir = "/data/mcn35/gravity/51Eri/reduced/"
filenames = glob.glob(datadir+"/*_s.fits")
filenames.sort()

objOis = []
starOis = []

ois1 = []
ois2 = []

raGuess = 824.77
decGuess = 748.80

for k in range(len(filenames)*0+8):
    filename = filenames[k]
    oi = gravity.GravityDualfieldAstrored(filename, extension = 10, corrMet = "sylvestre", opdDispCorr = "sylvestre")
    if oi.sObjX < 0:
        ois1.append(oi)
    if oi.sObjX > 0:
        ois2.append(oi)        
    objOis.append(oi)

w = np.zeros([6, oi.nwav])
for k in range(6):
    w[k, :] = oi.wav*1e6

for oi in ois1+ois2:
    oi.computeMean()
#    oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
    if oi.sObjX < 0:
        oi.visOi.recenterPhase(-raGuess, -decGuess)
    else:
        oi.visOi.recenterPhase(raGuess, decGuess)        
    
    
visRefS1 = 0*oi.visOi.visRef.mean(axis = 0)
visRefS2 = 0*oi.visOi.visRef.mean(axis = 0)
for k in range(len(ois1)):
    oi = ois1[k]
    visRefS1 = visRefS1+oi.visOi.visRef.mean(axis = 0)
visRefS1 = visRefS1/len(ois1)    
for k in range(len(ois2)):
    oi = ois2[k]
    visRefS2 = visRefS2+oi.visOi.visRef.mean(axis = 0)
visRefS2 = visRefS2/len(ois2)

phaseRef = 0.5*(np.angle(visRefS1)+np.angle(visRefS2))

# unwrap
testCase = np.angle(ois1[0].visOi.visRef.mean(axis = 0))-phaseRef
unwrapped = 0.5*np.unwrap(2*testCase)
correction = unwrapped - testCase
phaseRef = phaseRef - correction

stop
for oi in ois1+ois2:
    oi.visOi.addPhase(-phaseRef)
#    oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)
    if oi.sObjX < 0:
        oi.visOi.recenterPhase(raGuess, decGuess)
    else:
        oi.visOi.recenterPhase(-raGuess, -decGuess)        

# chi2Map
nRa = 100
nDec = 100
ra = np.linspace(824.76-1, 824.76+1, nRa)
dec = np.linspace(748.84-1, 748.84+1, nDec)

for k in range(len(ois2)):
    oi = ois2[k]
    oi.visOi.computeChi2Map(ra, dec, nopds = 100, visInst = np.abs(oi.visOi.visRef.mean(axis = 0)))

for k in range(len(ois1)):
    oi = ois1[k]
    oi.visOi.computeChi2Map(-ra[::-1], -dec[::-1], nopds = 100, visInst = np.abs(oi.visOi.visRef.mean(axis = 0)))    

raBest = np.array([oi.visOi.fit['xBest'] for oi in ois2]+[-oi.visOi.fit['xBest'] for oi in ois1])
decBest = np.array([oi.visOi.fit['yBest'] for oi in ois2]+[-oi.visOi.fit['yBest'] for oi in ois1])

plt.figure()
plt.imshow(oi.visOi.fit["chi2Map"].T, origin = "lower", extent = [np.min(ra), np.max(ra), np.min(dec), np.max(dec)])

fig = plt.figure()
gravityPlot.reImPlot(w, oi.visOi.visRef[:, :, :].mean(axis = 0), fig = fig, subtitles = oi.basenames)
gravityPlot.reImPlot(w, oi.visOi.visRefFit[:, :, :].mean(axis = 0), fig = fig)

plt.figure()
plt.plot(raBest, decBest, "+")


stop

        
fig = plt.figure()
gravityPlot.reImPlot(w, ois1[0].visOi.visRef[:, :, :].mean(axis = 0), fig = fig, subtitles = oi.basenames)
gravityPlot.reImPlot(w, ois2[0].visOi.visRef[:, :, :].mean(axis = 0), fig = fig)


"""
for c in range(oi.visOi.nchannel):
    for k in range(1, oi.nwav):
        if phaseRef[c, k]-phaseRef[c, k-1] > np.pi/2:
            phaseRef[c, k:]=phaseRef[c, k:] - np.pi
        elif phaseRef[c, k]-phaseRef[c, k-1] < -np.pi/2:
            phaseRef[c, k:]=phaseRef[c, k:] + np.pi
"""
    
#ois1.visOi.recenterPhase(-ois1.sObjX, -ois1.sObjY)
#ois2.visOi.recenterPhase(-ois2.sObjX, -ois2.sObjY)

stop

"""
fig = plt.figure()
gravityPlot.reImPlot(w, ois1.visOi.visRef[0, :, :]*np.exp(1j*np.angle(ois2.visOi.visRef[0, :, :])), fig = fig, subtitles = oi.basenames)
"""


fig = plt.figure()
gravityPlot.reImPlot(w, ois1.visOi.visRef[0, :, :], fig = fig, subtitles = oi.basenames)

ois1.visOi.addPhase(-phaseRef)

gravityPlot.reImPlot(w, ois1.visOi.visRef[:, :, :].mean(axis = 0), fig = fig)

u = ois1.visOi.uCoord.mean(axis = 0)
v = ois1.visOi.vCoord.mean(axis = 0)
ra =  ois1.sObjX/3600.0/360.0/1000.0*2*np.pi
dec = ois1.sObjY/3600.0/360.0/1000.0*2*np.pi
cosine = np.exp(1j*2*np.pi*(ra*u+dec*v))

#gravityPlot.reImPlot(w, 5000*cosine, fig = fig)
