import numpy as np
from matplotlib import pyplot as plt
import pyfits
import llh
import os
from astropy.table import Column
from astropy.io import fits

tscut = 10

#LCC_path = "sourceListAll2280_1GeV_fixedSpec_EFlux.fits"
LCC_path = "sourceListAll2280_1GeV_fixedSpec_EFlux_VHE.fits"

hdulist = pyfits.open(LCC_path)
tbdataLong = hdulist[1].data
#hdulist = fits.open(LCC_path)
#tbdata = hdulist[1].data


sMask = tbdataLong['Source_Name'] == '3FGL_J0509.4+0541'
sTXS = tbdataLong[sMask]
sTXS = llh.getSourceAverage(sTXS, tscut)[0]

#print sTXS

tbdata = tbdataLong[:] #np.random.choice(tbdata, replace=False, size=500)




name = 'testCat.npy'
if os.path.exists(name):
    tbdata = np.load(name)
else:
    tbdata = llh.getSourceAverage(tbdata, tscut)
    np.save(name,tbdata)

fluxRatio = []


fluxArray = []
fluxArray100 = []
fluxArray10 = []

fluxArrayHE = []
efluxArrayHE = []

fluxArrayAverage = []
photonFluxArray = []
photonFluxErrorArray = []

tsArray = []
fluxErrorArray = []
bSource = ''
maxEFlux = 0

npredArray = []

nameArray = []

for i in range(len(tbdata)):
    fluxHist = tbdata[i]['eflux']
    fluxHist10 =  tbdata[i]['eflux1000_10000']
    fluxHist100 = tbdata[i]['eflux1000_100000']

    #fluxHistHE = tbdata[i]['eflux_vhe_EBL_100000_1000000']
    efluxHistHE = tbdata[i]['eflux_vhe_100000_1000000']
    fluxHistHE = tbdata[i]['flux_vhe_100000_1000000']

    photonFluxHist = tbdata[i]['flux']
    photonFluxErrorHist = tbdata[i]['flux_err']    
    photonFluxHist = np.asarray(photonFluxHist)
    tsHist = tbdata[i]['ts']
    fluxErrorHist = tbdata[i]['eflux_err']
    fluxR = tbdata[i]['flux100']
    flux9y = tbdata[i]['Energy_Flux']

    npred = tbdata[i]['npred']
    nname = tbdata[i]['Source_Name']
    
    maxSource = max(tbdata[i]['eflux'])
    if maxSource>maxEFlux:
        maxEFlux = maxSource
        bSource = tbdata[i]['Source_Name']

    for fi in range(len(fluxHist)):
        fluxArray.append(fluxHist[fi])
        fluxArray10.append(fluxHist10[fi])
        fluxArray100.append(fluxHist100[fi])
        fluxArrayAverage.append(flux9y)
        tsArray.append(tsHist[fi])
        fluxErrorArray.append(fluxErrorHist[fi])
        photonFluxArray.append(photonFluxHist[fi])
        photonFluxErrorArray.append(photonFluxErrorHist[fi])
        fluxRatio.append(fluxR[fi])
        npredArray.append(npred[fi])
        nameArray.append(nname)
	fluxArrayHE.append(fluxHistHE)
        efluxArrayHE.append(efluxHistHE)
        

print bSource, maxEFlux

fluxArrayHE = np.asarray(fluxArrayHE)
efluxArrayHE = np.asarray(efluxArrayHE)

fluxErrorArray = np.asarray(fluxErrorArray)
fluxArray = np.asarray(fluxArray)
tsArray = np.asarray(tsArray)
fluxRatio = np.asarray(fluxRatio)
fluxArrayAverage = np.asarray(fluxArrayAverage)

npredArray = np.asarray(npredArray)
nameArray = np.asarray(nameArray)

mask = np.isnan(fluxArray)
tsArray = tsArray[~mask]
fluxArray = fluxArray[~mask]
fluxErrorArray = fluxErrorArray[~mask]
fluxRatio = fluxRatio[~mask]
fluxArrayAverage = fluxArrayAverage[~mask]

photonFluxArray = np.asarray(photonFluxArray)
photonFluxErrorArray = np.asarray(photonFluxErrorArray)

fluxArray10 = np.asarray(fluxArray10)
fluxArray100 = np.asarray(fluxArray100)


mask = np.isnan(fluxRatio)
fluxRatio = fluxRatio[~mask]
mask = np.isfinite(fluxRatio)
fluxRatio = fluxRatio[mask]


plt.figure()
bins = np.linspace(-10,-4,100)
n, b, a = plt.hist(np.log10(fluxArrayHE), bins=bins, log=True, label='flux 0.1-1TeV')
n, b, a = plt.hist(np.log10(efluxArrayHE), bins=bins, log=True, label='eflux 0.1-1TeV')
plt.legend()
plt.xlabel('monthly flux $>$100GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxDistHE.png",bbox_inches='tight')



plt.figure()
bins = np.linspace(-10,-4,100)
n, b, a = plt.hist(np.log10(photonFluxArray), bins=bins, log=True, label='all')
n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>4) & (photonFluxErrorArray<0.5*photonFluxArray)]), bins=bins, log=True, label='TS$>$4', histtype='step', color='blue')
#n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>9) & (photonFluxErrorArray<0.5*photonFluxArray)]), bins=bins, log=True, label='TS$>$9', histtype='step')
n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>10) & (photonFluxErrorArray<0.5*photonFluxArray)]), bins=bins, log=True, label='TS$>$10', histtype='step', color='green')
n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>16) & (photonFluxErrorArray<0.5*photonFluxArray)]), bins=bins, log=True, label='TS$>$16', histtype='step', color='orange')
n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>25) & (photonFluxErrorArray<0.5*photonFluxArray)]), bins=bins, log=True, label='TS$>$25', histtype='step', color='black')
#n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>4)]), bins=bins, log=True, label='TS$>$4', histtype='step')
#n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>9)]), bins=bins, log=True, label='TS$>$9', histtype='step')
#n, b, a = plt.hist(np.log10(photonFluxArray[(tsArray>25)]), bins=bins, log=True, label='TS$>$25', histtype='step')

plt.legend()
plt.xlabel('monthly flux $>$1GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxDist.png",bbox_inches='tight')


brightSources = ["3FGL_J1104.4+3812", "3FGL_J1653.9+3945"]
maskBright = [i in brightSources for i in tbdata['Source_Name']]
maskBright = np.asarray(maskBright)

print len(tbdata), len(tbdata[maskBright])

print "9 year ratio ", sTXS['Energy_Flux'] / np.sum(tbdata[~maskBright]['Energy_Flux'])

TXSRatio = sTXS['flux100'][-1]

print "TXSRatio ", TXSRatio
print "TXS flux ", sTXS['flux'][-1]

TXSFlux = sTXS['eflux'][-1] #0.00012919
TXSFlux10 = sTXS['eflux1000_10000'][-1] #0.00012919
TXSFlux100 = sTXS['eflux1000_100000'][-1] #0.00012919

TXSFluxErr = sTXS['eflux_err'][-1] #2.89975827e-05
print "TXS EFlux ", TXSFlux, TXSFluxErr

TXSPhotonFlux = sTXS['eflux'][-1]

TXSFlux_HE = sTXS['flux_vhe_100000_1000000'][-1] #0.00012919
TXSEFlux_HE = sTXS['eflux_vhe_100000_1000000'][-1] #0.00012919


#plt.figure()
#bins = np.linspace(0,100,100)
#plt.hist(npredArray,bins=bins,log=True)

#plt.figure()
#mask = tbdata['Source_Name'] == "3FGL_J2254.0+1608"
#s = tbdata[mask][0]
#print s['SpectrumType']
#print s['param_names']
#print s['param_values'][0:8]
#plt.plot(s['tmin_mjd'],s['eflux'],'.',color='red',label='1TeV')
#plt.plot(s['tmin_mjd'],s['eflux1000_100000'],'.',color='blue', label='100GeV')
#plt.plot(s['tmin_mjd'],s['eflux1000_10000'],'.',color='blue', label='10GeV')
#plt.legend()

#plt.figure()
#mask = tbdata['Source_Name'] == "3FGL_J0509.4+0541"
#s = tbdata[mask][0]
#plt.plot(s['tmin_mjd'],s['eflux'],'.',color='red',label='1TeV')
#plt.plot(s['tmin_mjd'],s['eflux1000_100000'],'.',color='blue', label='100GeV')
#plt.plot(s['tmin_mjd'],s['eflux1000_10000'],'.',color='blue', label='10GeV')
#plt.legend()


#plt.figure()
#fluxArray = np.asarray(fluxArray)
#tsArray = np.asarray(tsArray)
#a = plt.plot(fluxArray[(npredArray>9) & (tsArray>25)],fluxArray100[(npredArray>9) & (tsArray>25)],'.')
#a = plt.hist(fluxArray[fluxArray100>0.00001]-fluxArray100[fluxArray100>0.00001])

#a = plt.hist(fluxArray-fluxArray10)/fluxArray)
#plt.xlabel('eflux $>$1TeV')
#plt.ylabel('eflux $>$100GeV')

#plt.figure()
#print 'bad sources ', len(tsArray[(fluxArray100<0.00001) & (fluxArray>0.0001)])
#a = plt.loglog(tsArray[(fluxArray100<0.00001) & (fluxArray>0.0001)],npredArray[(fluxArray100<0.00001) & (fluxArray>0.0001)],'.')
#plt.xlabel('ts')
#plt.ylabel('npred')

#plt.figure()
#a = plt.loglog(fluxArray[(fluxArray100<0.00001) & (fluxArray>0.0001)],fluxErrorArray[(fluxArray100<0.00001) & (fluxArray>0.0001)],'.')
#plt.xlabel('eflux')
#plt.ylabel('eflux error')
#print nameArray[(fluxArray100<0.00001) & (fluxArray>0.0001)]


plt.figure()
X2 = np.sort(np.log10(photonFluxArray))
F2 = np.ones(len(np.log10(photonFluxArray))) - np.array(range(len(np.log10(photonFluxArray)))) / float(len(np.log10(photonFluxArray)))
plt.plot(X2, F2, lw=3, label='All monthly time bins in LC catalog')
plt.plot([np.log10(TXSPhotonFlux),np.log10(TXSPhotonFlux)],[0.5e-5,1],color='red',lw=2, label='TXS')
print np.log10(TXSPhotonFlux)
plt.yscale('log')
plt.xlim(-9,)
plt.legend()
plt.xlabel(r'monthly photon flux $>$1GeV [ph/cm$^2$/s]')
plt.savefig("plots/FluxDistCum.png",bbox_inches='tight')


plt.figure()
X2 = np.sort(np.log10(fluxArray))
F2 = np.ones(len(np.log10(fluxArray))) - np.array(range(len(np.log10(fluxArray)))) / float(len(np.log10(fluxArray)))
plt.plot(X2, F2, lw=3, label='1TeV', color='red')

X210 = np.sort(np.log10(fluxArray10))
F210 = np.ones(len(np.log10(fluxArray10))) - np.array(range(len(np.log10(fluxArray10)))) / float(len(np.log10(fluxArray10)))
plt.plot(X210, F210, lw=3, label='10GeV',color='blue')

X2100 = np.sort(np.log10(fluxArray100))
F2100 = np.ones(len(np.log10(fluxArray100))) - np.array(range(len(np.log10(fluxArray100)))) / float(len(np.log10(fluxArray100)))
plt.plot(X2100, F2100, lw=3, label='100GeV', color='green')

plt.plot([np.log10(TXSFlux),np.log10(TXSFlux)],[0.5e-5,1],color='red',lw=2, label='TXS, 1TeV', ls=':')
plt.plot([np.log10(TXSFlux10),np.log10(TXSFlux10)],[0.5e-5,1],color='blue',lw=2, label='TXS, 10GeV', ls=':')
plt.plot([np.log10(TXSFlux100),np.log10(TXSFlux100)],[0.5e-5,1],color='green',lw=2, label='TXS, 100GeV', ls=':')


print np.log10(TXSFlux)
plt.yscale('log')
plt.xlim(-7,)
plt.legend()
plt.xlabel(r'monthly energy flux $>$1GeV [ph/cm$^2$/s]')
plt.savefig("plots/EFluxDistCum.png",bbox_inches='tight')

print 'larger than eflux 1TeV ', len(X2[X2>np.log10(TXSFlux)])/float(len(X2))
print 'larger than eflux 100 GeV ', len(X2100[X2100>np.log10(TXSFlux100)])/float(len(X2100))
print 'larger than eflux 10 GeV', len(X210[X210>np.log10(TXSFlux10)])/float(len(X210))


plt.figure()
X2 = np.sort(np.log10(fluxArray_HE))
F2 = np.ones(len(np.log10(fluxArray_HE))) - np.array(range(len(np.log10(fluxArray_HE)))) / float(len(np.log10(fluxArray_HE)))
plt.plot(X2, F2, lw=3, label='flux 0.1-1TeV', color='blue')
plt.plot([np.log10(TXSFlux_HE),np.log10(TXSFlux_HE)],[0.5e-5,1],color='red',lw=2, label='TXS', ls=':')

plt.figure()
X2 = np.sort(np.log10(efluxArray_HE))
F2 = np.ones(len(np.log10(efluxArray_HE))) - np.array(range(len(np.log10(efluxArray_HE)))) / float(len(np.log10(efluxArray_HE)))
plt.plot(X2, F2, lw=3, label='eflux 0.1-1TeV', color='blue')
plt.plot([np.log10(TXSEFlux_HE),np.log10(TXSEFlux_HE)],[0.5e-5,1],color='red',lw=2, label='TXS', ls=':')



plt.figure()
bins = np.linspace(0,20,100)
n, b, a = plt.hist(fluxRatio, bins=bins, normed=1, color='blue', log=True)
plt.plot([TXSRatio,TXSRatio],[0.5e-5,1],color='red',lw=2, label='TXS')
plt.legend()
plt.xlabel('flux ratio')
plt.savefig("plots/FluxRatioDist.png",bbox_inches='tight')

plt.figure()
X2 = np.sort(fluxRatio)
F2 = np.ones(len(fluxRatio)) - np.array(range(len(fluxRatio))) / float(len(fluxRatio))
plt.plot(X2, F2, lw=3, label='All monthly time bins in LC catalog', color='blue')
plt.plot([TXSRatio,TXSRatio],[0.5e-5,1],color='red',lw=2, label='TXS')
print np.log10(TXSFlux)
plt.yscale('log')
plt.xlim(0,5)
plt.legend()
plt.xlabel('flux ratio')
plt.legend()
plt.savefig("plots/FluxRatioCum.png",bbox_inches='tight')
print 'larger flux ratio (no TS cut)', len(X2[X2>TXSRatio])/float(len(X2))


plt.figure()
fluxRatioW = fluxRatio*fluxArray
TXSRatioW = TXSRatio*TXSFlux
print TXSRatioW
X2 = np.sort(fluxRatioW)
F2 = np.ones(len(fluxRatioW)) - np.array(range(len(fluxRatioW))) / float(len(fluxRatioW))
plt.plot(X2, F2, lw=3, label='All monthly time bins in LC catalog', color='blue')
plt.plot([TXSRatioW,TXSRatioW],[0.5e-5,1],color='red',lw=2, label='TXS')
plt.yscale('log')
plt.xlim(0,0.005)
plt.legend()
plt.xlabel('flux ratio * eflux')
plt.legend()
plt.savefig("plots/FluxRatioCum.png",bbox_inches='tight')
print 'larger flux ratio *flux', len(X2[X2>TXSRatioW])/float(len(X2))


def fluxMax(flux, fluxE):
    return flux*0.5 + np.sqrt((flux * 0.5) ** 2 + 0.5* fluxE ** 2)

def nuisance(flux, fluxMax, fluxE):
    return 1./np.sqrt(2.*np.pi*fluxE ** 2) * np.exp(-(fluxMax-flux)**2 /(2*fluxE**2))

#TXSFluxMax = fluxMax(TXSFlux, TXSFluxErr)
#TXSNuis = nuisance(TXSFlux, TXSFluxMax, TXSFluxErr)

#print "TXS flux ", TXSFlux, TXSFluxErr
#print "TXS max flux ", TXSFluxMax
#print "nuisance ", TXSNuis

#fluxMaxArray = fluxMax(fluxArray, fluxErrorArray)
#nuisanceTerm = nuisance(fluxArray, fluxMaxArray, fluxErrorArray)

#fluxArray = fluxArray[tsArray>25]
#fluxArrayAverage = fluxArrayAverage[tsArray>25]

#print np.sum(fluxArray)
#print np.sum(fluxArray[fluxArray>fluxArrayAverage])


plt.show()


