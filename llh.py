# coding: utf-8

#from astropy.coordinates import SkyCoord
#from astropy import units as u
import matplotlib.mlab as mlab
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import pyfits as fits
import datetime
import sys

nugen_path = '/data/user/tglauch/EHE/processed/combined.npy'
# input in rad
def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2, use_astro=False):
    '''Compute the great circle distance between two events'''
    '''SkyCoord is super slow ..., don't use it'''
    if use_astro:
        coords1 = SkyCoord(ra=ra_1 * u.rad, dec=dec_1 * u.rad)
        coords2 = SkyCoord(ra=ra_2 * u.rad, dec=dec_2 * u.rad)
        sep = (coords1.separation(coords2)).rad
        return sep
    else:
        delta_dec = np.abs(dec_1 - dec_2)
        delta_ra = np.abs(ra_1 - ra_2)
        x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
            np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
        return 2. * np.arcsin(np.sqrt(x))


def pull_EHE(zen, lognpe, cr_dzen, cr_dazi):
    '''from realtime_ehe'''
    # From pull fit
    slope = -0.59247735857
    inter = 2.05973576429

    # define minimum value allowed (0.25 \deg) in radians
    # min_corrected_pull = 0.0043633
    # Calculate sigma
    sin2 = np.sin(zen) * np.sin(zen)
    sigma = np.sqrt(cr_dzen * cr_dzen + cr_dazi * cr_dazi * sin2 / 2.)

    # Return corrected value
    pull_corrtd = sigma / np.power(10, inter + slope * lognpe)
    pull_corrtd = np.asarray(pull_corrtd)
    return pull_corrtd


def read3FGL():
    file_name = "data/gll_psc_v16.fit"
    hdulist = fits.open(file_name)
    tbdata = hdulist[1].data

    # select only extra gal sources
    eGal3FGL = ['css', 'BLL', 'bll', 'fsrq', 'FSRQ', 'agn', 'RDG',
                'rdg', 'sey', 'BCU', 'bcu', 'GAL', 'gal', 'NLSY1',
                'nlsy1', 'ssrq']

    # get rid of extra spaces at the end
    mask = [c.strip() in eGal3FGL for c in tbdata['Class1']]
    mask = np.asarray(mask)
    tbdata = tbdata[mask]

    timeBins = []

    for i in range(len(hdulist['Hist_start'].data)):
        timeBins.append(hdulist['Hist_start'].data[i][0])

    return tbdata, np.asarray(timeBins)

def readLCCat():
    file_name = "data/myCat.fits"
    hdulist = fits.open(file_name)
    tbdata = hdulist[1].data

    timeBins = []

    for i in range(len(hdulist[2].data['Hist_start'])):
        timeBins.append(hdulist[2].data['Hist_start'][i])

    return tbdata, np.asarray(timeBins)

def getTBin(testT, timeBins):
    ind = np.searchsorted(timeBins, testT, side='right')
    return ind - 1


def getDataEHE():
    # load data
    f = np.load(nugen_path)
    cr_theta = f['cramer_zen']
    cr_phi = f['cramer_az']
    cr = pull_EHE(f['mpe_zen'], np.log10(f['NPE']), cr_theta, cr_phi)
    mask = np.isnan(cr)
    cr = cr[~mask]
    f = f[~mask]
    return f['mpe_az'], f['mpe_zen'], f['muex'], f['energy'], f['ow'], f['atmo'], cr, cr_theta, cr_phi, f['NPE']


def BGRatePDF(zen, AtmWeight, plot=False):
    # get BG rate as function of zenith PDF
    bins = np.linspace(-1, 1, 20)
    nBG, binsBG, patches = plt.hist(np.cos(zen), bins=bins,
                                    weights=AtmWeight,
                                    normed=True)  # bins=bins)
    binCenter = (binsBG[1:] - binsBG[:-1]) * 0.5 + binsBG[:-1]
    pBG = interp1d(binCenter, nBG, kind='nearest',
                   fill_value=0.0, bounds_error=False)
    return pBG


def EnergyPDF(zen, recoE, AtmWeight, OneWeight, gammaSig=2.1):
    # get energy PDF
    cosZenBins = [-1, -0.5, -0.25, -0.1, 1]
    cosZenBins = np.asarray(cosZenBins)
    EPDFSig = []
    EPDFBG = []
    for zi in range(len(cosZenBins) - 1):
        zMask = [(np.cos(zen) > cosZenBins[zi]) & (np.cos(zen) <= cosZenBins[zi + 1])]
        print "zenith band ", cosZenBins[zi], cosZenBins[zi + 1]

        bins = np.linspace(3, 7, 15)
        nSig, binsSig, patches = plt.hist(np.log10(recoE[zMask]),
                                          weights=OneWeight[zMask] / (trueE[zMask]**gammaSig),
                                          normed=True, bins=bins, alpha=0.5,
                                          label='E$^{-2.1}$')
        nBG, binsBG, patches = plt.hist(np.log10(recoE[zMask]),
                                        weights=AtmWeight[zMask],
                                        normed=True, bins=bins,
                                        alpha=0.5, label='E$^{-2.7}$')

        binCenter = (binsSig[1:] - binsSig[:-1]) * 0.5 + binsSig[:-1]
        EPDFSig.append(interp1d(binCenter, nSig, kind='cubic',
                                fill_value=0.0, bounds_error=False))
        EPDFBG.append(interp1d(binCenter, nBG, kind='cubic',
                               fill_value=0.0, bounds_error=False))
    return cosZenBins, EPDFSig, EPDFBG


# get extragal. 3FGL sources and flux
# ra, dec in rad, neuTime in Fermi MET
def get3FGL(ra, dec, sigma, neuTime, tbdata, timeBins):
    circ = sigma * 3.
    if circ > np.deg2rad(5):
        circ = np.deg2rad(5)
    
    dist = GreatCircleDistance(np.deg2rad(tbdata['RAJ2000']),
                                np.deg2rad(tbdata['DEJ2000']),
                                ra, dec)
    mask = dist<circ
    foundSources = tbdata[mask]
    print "found %i sources close by"%len(foundSources)
    if len(foundSources) == 0:
        return None
    # this is a hack to get the flux of the measured EHE event
    # which is outside of the catalog time
    if neuTime < 0:
        fluxNeuTime = [0.39e-6]
        mask = dist<np.deg2rad(0.5)
        foundSources = tbdata[mask]
    else:
        fluxHist = foundSources['Flux_History']
        #ts = foundSources['TS']
        fluxNeuTime = [f[getTBin(neuTime, timeBins)] for f in fluxHist]
        #tsNeuTime = [f[getTBin(neuTime, timeBins)] for f in ts]
        #tsMask = tsNeuTime < 4
        #tsMask = np.asarray(tsMask)
        #fluxNeuTime = fluxNeuTime[tsMask]
        #foundSources = foundSources[tsMask]
    retval = np.deg2rad(foundSources['RAJ2000']), \
        np.deg2rad(foundSources['DEJ2000']), fluxNeuTime
    return retval


def likelihood(en, ra, dec, sigma, neuTime,
               cosZenBins, EPDFSig, EPDFBG,
               pBG, tbdata, timeBins):
    foundSources = get3FGL(ra, dec, sigma, neuTime, tbdata, timeBins)
    if foundSources is None:
        return -99

    raS, decS, fluxS = foundSources

    fluxS = np.asarray(fluxS)
    mask = fluxS < 1e-12
    fluxS[mask] = 1e-12

    fluxNorm = 1e-6
    sourceTerm = fluxS / fluxNorm * np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2. * sigma**2))

    cosZenBins = np.asarray(cosZenBins)
    zi = cosZenBins[cosZenBins < np.cos(dec + 0.5 * np.pi)].size - 1

    if EPDFBG[zi](np.log10(en)) <= 0 or EPDFSig[zi](np.log10(en)) <= 0:
        energyTerm = 1e-12
    else:
        energyTerm = EPDFSig[zi](np.log10(en)) / EPDFBG[zi](np.log10(en))

    BGRateTerm = pBG(np.cos(dec + 0.5 * np.pi))
    if BGRateTerm < 1e-12:
        BGRateTerm = 1e-12

    llh = -2 * np.log(sigma) + np.log(np.sum(sourceTerm)) - np.log(BGRateTerm) + np.log(energyTerm)
    print 'likelihhod ', llh
    return 2 * llh


def simulate(zen, azi, recoE, cr, AtmWeight, timeBins,
             filename, cosZenBins, EPDFSig, EPDFBG, pBG,
             tbdata, NSim=1000):

    # draw random index weighted with E^-2
    #draw = np.random.choice(range(len(trueE)), NSim, p=OneWeight/(trueE**2)/np.sum(OneWeight/(trueE**2)))
    # draw random index weighted with atmo
    draw = np.random.choice(range(len(trueE)),
                            NSim,
                            p=AtmWeight / np.sum(AtmWeight))

    enSim = recoE[draw]
    crSim = cr[draw]
    decSim = zen[draw] - 0.5 * np.pi
    raSim = azi[draw]
    # atmoSim = AtmWeight[draw]
    tmin = timeBins[0]
    tmax = timeBins[-1]
    # draw uniform neutrino times within catalog time span
    neuTimeSim = np.random.uniform(tmin, tmax, size=NSim)

    llh = []
    for i in range(NSim):
        if i % 1 == 0:
            print i, raSim[i], decSim[i], crSim[i], neuTimeSim[i]
        llh.append(likelihood(enSim[i], raSim[i],
                              decSim[i], crSim[i],
                              neuTimeSim[i], cosZenBins,
                              EPDFSig, EPDFBG,
                              pBG, tbdata, timeBins))

    llh = np.asarray(llh)
    np.save(filename, llh)


def plotLLH(llhfile, outfile, cosZenBins, EPDFSig, EPDFBG, pBG, tbdata):
    llh = np.load(llhfile)
    plt.figure()
    bins = np.linspace(-20, 25, 50)
    a, b, c = plt.hist(llh, bins=bins, cumulative=-1, normed=1, log=True)
    llhNu = likelihood(120000,
                       np.deg2rad(77.43),
                       np.deg2rad(5.72),
                       np.deg2rad(0.25), -9,
                       cosZenBins, EPDFSig, EPDFBG,
                       pBG, tbdata, timeBins)
    print llhNu
    plt.plot([llhNu, llhNu], [0, max(a)])
    plt.xlabel('log(likelihood)')
    plt.grid()
    plt.savefig(outfile)
    plt.show()


if __name__ == '__main__':

    jobN = int(sys.argv[1])
    # get Data
    azi, zen, recoE, trueE, OneWeight, AtmWeight, cr, cr_zen, cr_azi, npe = getDataEHE()
    # read 3FGL catalog
    #tbdata, timeBins = read3FGL()
    tbdata, timeBins = readLCCat()

    gammaSig = 2.1
    cosZenBins, EPDFSig, EPDFBG = EnergyPDF(zen, recoE, AtmWeight,
                                            OneWeight, gammaSig)
    pBG = BGRatePDF(zen, AtmWeight)

    NSim = 1000
    filename = 'output/llh_%i_%.1f_%i.npy' % (NSim, gammaSig, jobN)
    simulate(zen, azi, recoE, cr,
             AtmWeight, timeBins, filename,
             cosZenBins, EPDFSig, EPDFBG,
             pBG, tbdata, NSim)

    plotLLH(filename,
            filename.replace('output', 'plots').replace('npy', 'png'),
            cosZenBins, EPDFSig, EPDFBG, pBG, tbdata)
