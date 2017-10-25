import numpy as np
from matplotlib import pyplot as plt
import pyfits

LCC_path = "/home/annaf/BlazarNeutrino/data/sourceListAll2283_1GeV.fits"

hdulist = pyfits.open(LCC_path)
tbdata = hdulist[1].data

fluxArray = []
tsArray = []
fluxErrorArray = []
bSource = ''
maxEFlux = 0
for i in tbdata:
    fluxHist = i['eflux']
    tsHist = i['ts']
    fluxErrorHist = i['eflux_err']

    maxSource = max(i['eflux'])
    if maxSource>maxEFlux:
        maxEFlux = maxSource
        bSource = i['Source_Name']

    for fi in range(len(fluxHist)):
        fluxArray.append(fluxHist[fi])
        tsArray.append(tsHist[fi])
        fluxErrorArray.append(fluxErrorHist[fi])

print bSource, maxEFlux

fluxErrorArray = np.asarray(fluxErrorArray)
fluxArray = np.asarray(fluxArray)
tsArray = np.asarray(tsArray)

mask = np.isnan(fluxArray)
tsArray = tsArray[~mask]
fluxArray = fluxArray[~mask]
fluxErrorArray = fluxErrorArray[~mask]


#maskZero = fluxArray<=0
#tsMask = tsArray>25
#tsMask = np.asarray(tsMask)
#fluxArray[maskZero] = 1e-8
#fluxArray[tsMask] = 1e-8

plt.figure()
bins = np.linspace(-14,-2,100)
n, b, a = plt.hist(np.log10(fluxArray), bins=bins, log=True)
plt.xlabel('monthly flux >1GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxDist.png",bbox_inches='tight')

TXSFlux = 0.00012919
TXSFluxErr = 2.89975827e-05

plt.figure()
bins = np.linspace(-14,-2,200)
c, b, a = plt.hist(np.log10(fluxArray), bins=bins,cumulative=-1, normed=True, log=True)
plt.plot([np.log10(TXSFlux),np.log10(TXSFlux)],[0,1],color='red',lw=2)
plt.xlabel('monthly energy flux >1GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxDistCum.png",bbox_inches='tight')

plt.figure()
bins = np.linspace(-16,-2,200)
c, b, a = plt.hist(np.log10(fluxErrorArray), bins=bins,cumulative=False, normed=True, log=True)
plt.plot([np.log10(TXSFluxErr),np.log10(TXSFluxErr)],[0,1],color='red',lw=2)
plt.xlabel('monthly energy flux >1GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxErrDistCum.png",bbox_inches='tight')

plt.figure()
bins = np.linspace(0,1,200)
c, b, a = plt.hist(fluxErrorArray/fluxArray, bins=bins,cumulative=True, normed=True, log=True)
plt.plot([TXSFluxErr/TXSFlux,TXSFluxErr/TXSFlux],[0,1],color='red',lw=2)
plt.xlabel('monthly energy flux >1GeV [ph/cm$^2$/s]')
plt.xscale('log')
plt.savefig("plots/FluxErrOverFlux.png",bbox_inches='tight')


def fluxMax(flux, fluxE):
    return flux*0.5 + np.sqrt((flux * 0.5) ** 2 + 0.5* fluxE ** 2)

def nuisance(flux, fluxMax, fluxE):
    return 1./np.sqrt(2.*np.pi*fluxE ** 2) * np.exp(-(fluxMax-flux)**2 /(2*fluxE**2))

TXSFluxMax = fluxMax(TXSFlux, TXSFluxErr)
TXSNuis = nuisance(TXSFlux, TXSFluxMax, TXSFluxErr)

print "TXS flux ", TXSFlux, TXSFluxErr
print "TXS max flux ", TXSFluxMax
print "nuisance ", TXSNuis

fluxMaxArray = fluxMax(fluxArray, fluxErrorArray)
nuisanceTerm = nuisance(fluxArray, fluxMaxArray, fluxErrorArray)

plt.figure()
bins = np.linspace(2,9,100)
a,b,c=plt.hist(np.log10(nuisanceTerm),bins=bins)
plt.plot([np.log10(TXSNuis), np.log10(TXSNuis)],[0,max(a)],color='red')
plt.xlabel('nuisance term')
plt.savefig('plots/nuisance.png')

plt.figure()
a,b,c=plt.hist(np.log10(fluxMaxArray),bins=50)
plt.plot([np.log10(TXSFluxMax), np.log10(TXSFluxMax)],[0,max(a)],color='red')
plt.xlabel('fluxMax')
plt.savefig('plots/FluxMax.png')

plt.figure()
bins = np.linspace(0,1,100)
a,b,c=plt.hist(nuisanceTerm*fluxMaxArray,bins=bins,log=True)
plt.plot([TXSFluxMax*TXSNuis, TXSFluxMax*TXSNuis],[0,max(a)],color='red')
plt.xlabel('nuisance term * fluxMax')
plt.savefig('plots/nuisanceTimesFlux.png')

print fluxMax[0:10]*nuisanceTerm[0:10]
print nuisanceTerm[0:10]

#add stat. errors
#bcenter = b[1:] - (b[1]-b[0])*0.5
#plt.figure()
#plt.plot(bcenter,n/n[0], color='blue')
#a = (n+np.sqrt(n))/n[0]
#b = (n-np.sqrt(n))/n[0]

#plt.plot(bcenter,n,color='blue')
#plt.plot(bcenter,a, color='blue',ls=':')
#plt.plot(bcenter,b, color='blue',ls=':')


#plt.yscale('log')

#plt.savefig('plots/FluxDistCumError.png',bbox_inches='tight')

