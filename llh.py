# coding: utf-8

# from astropy.coordinates import SkyCoord
# from astropy import units as u
# from plot_conf import *

from fancy_plot import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyfits as fits
import sys
from numpy.lib.recfunctions import rec_append_fields


def setNewEdges(edges):
    newEdges = []
    for i in range(0, len(edges) - 1):
        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
        newEdges.append(newVal)
    return np.array(newEdges)


# ------------------------------- Settings ---------------------------- #

nugen_path = '/data/user/tglauch/EHE/processed/combined.npy'

settings = {'E_reco': 'NPE',
            'zen_reco': 'mpe_zen',
            'az_reco': 'mpe_az',
            'sigma': 'cr',
            'gamma': 2.1,
            'ftypes': ['astro', 'atmo', 'prompt'],  # atmo = conv..sry for that
            'Nsim': 1000,
            'Phi0': 0.91}

dtype = [("en", np.float64),
         ("ra", np.float64),
         ("dec", np.float64),
         ("sigma", np.float64),
         ("neuTime", np.float64)]

# IC170911
# NPE: 5784.9552
# MuEX: 120000 GeV

EHE_event = np.array((5784.9552,
                      np.deg2rad(77.43),
                      np.deg2rad(5.72),
                      np.deg2rad(0.25),
                      -9),
                     dtype=dtype)

E_spline = np.load('spline.npy')[()]
coszen_spline = np.load('coszen_spl.npy')[()]

# --------------------------------------------------------------------------- #


@np.vectorize
def powerlaw(trueE, ow):
    return ow * settings['Phi0'] * 1e-18 * (trueE * 1e-5) ** (- settings['gamma'])


dtype = [("en", np.float64),
         ("ra", np.float64),
         ("dec", np.float64),
         ("sigma", np.float64),
         ("neuTime", np.float64)]


def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2, use_astro=False):
    '''Compute the great circle distance between two events'''
    '''SkyCoord is super slow ..., don't use it'''
    # if use_astro:
    #     coords1 = SkyCoord(ra=ra_1 * u.rad, dec=dec_1 * u.rad)
    #     coords2 = SkyCoord(ra=ra_2 * u.rad, dec=dec_2 * u.rad)
    #     sep = (coords1.separation(coords2)).rad
    #     return sep
    # else:
    delta_dec = np.abs(dec_1 - dec_2)
    delta_ra = np.abs(ra_1 - ra_2)
    x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
        np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
    return 2. * np.arcsin(np.sqrt(x))


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


def BGRatePDF(f, plot=False):
    # get BG rate as function of zenith PDF
    zen = f[settings['zen_reco']]
    AtmWeight = f['atmo']
    bins = np.linspace(-1, 1, 20)
    nBG, binsBG, patches = plt.hist(np.cos(zen), bins=bins,
                                    weights=AtmWeight,
                                    normed=True)  # bins=bins)
    binCenter = (binsBG[1:] - binsBG[:-1]) * 0.5 + binsBG[:-1]
    pBG = interp1d(binCenter, nBG, kind='nearest',
                   fill_value=0.0, bounds_error=False)
    return pBG


def EnergyPDF(f, gammaSig=2.1):
    # get energy PDF
    zen = f[settings['zen_reco']]
    recoE = f[settings['E_reco']]
    AtmWeight = f['atmo']
    cosZenBins = [-1, -0.5, -0.25, -0.1, 1]
    cosZenBins = np.asarray(cosZenBins)
    EPDFSig = []
    EPDFBG = []
    for zi in range(len(cosZenBins) - 1):
        zMask = [(np.cos(zen) > cosZenBins[zi]) & (np.cos(zen) <= cosZenBins[zi + 1])]
        print "zenith band ", cosZenBins[zi], cosZenBins[zi + 1]

        bins = np.linspace(3, 7, 15)
        nSig, binsSig, patches = plt.hist(np.log10(recoE[zMask]),
                                          weights=f['astro'][zMask],
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
    mask = dist < circ
    foundSources = tbdata[mask]
    if len(foundSources) != 0:
        print "found %i sources close by" % len(foundSources)
    if len(foundSources) == 0:
        return None
    # this is a hack to get the flux of the measured EHE event
    # which is outside of the catalog time
    if neuTime < 0:
        fluxNeuTime = [0.39e-6]
        mask = dist < np.deg2rad(0.5)
        foundSources = tbdata[mask]
    else:
        fluxHist = foundSources['Flux_History']
        # ts = foundSources['TS']
        fluxNeuTime = [f[getTBin(neuTime, timeBins)] for f in fluxHist]
        # tsNeuTime = [f[getTBin(neuTime, timeBins)] for f in ts]
        # tsMask = tsNeuTime < 4
        # tsMask = np.asarray(tsMask)
        # fluxNeuTime = fluxNeuTime[tsMask]
        # foundSources = foundSources[tsMask]
    retval = np.deg2rad(foundSources['RAJ2000']), \
        np.deg2rad(foundSources['DEJ2000']), fluxNeuTime
    return retval


# def likelihood(sim, cosZenBins, EPDFSig, EPDFBG,
#                pBG, tbdata, timeBins):

def likelihood(sim, tbdata, timeBins):

    en = sim['en']
    ra = sim['ra']
    dec = sim['dec']
    sigma = sim['sigma']
    neuTime = sim['neuTime']

    foundSources = get3FGL(ra, dec, sigma, neuTime, tbdata, timeBins)
    if foundSources is None:
        return -99

    raS, decS, fluxS = foundSources

    fluxS = np.asarray(fluxS)
    mask = fluxS < 1e-12
    fluxS[mask] = 1e-12

    fluxNorm = 1e-6
    sourceTerm = fluxS / fluxNorm * np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2. * sigma**2))

    # cosZenBins = np.asarray(cosZenBins)
    # zi = cosZenBins[cosZenBins < np.cos(dec + 0.5 * np.pi)].size - 1

    # if EPDFBG[zi](np.log10(en)) <= 0 or EPDFSig[zi](np.log10(en)) <= 0:
    #     energyTerm = 1e-12
    # else:
    #     energyTerm = EPDFSig[zi](np.log10(en)) / EPDFBG[zi](np.log10(en))

    # BGRateTerm = pBG(np.cos(dec + 0.5 * np.pi))
    # if BGRateTerm < 1e-12:
    #     BGRateTerm = 1e-12

    coszen = np.cos(dec + 0.5 * np.pi)
    E_ratio = np.log(10 ** E_spline(coszen, np.log10(en))[0])
    coszen_prob = np.log(10**(coszen_spline(coszen)/ (2 * np.pi)))
    print('E: {} ra: {} coszen: {} \n \
           sigma: {} time : {}'.format(en, ra, coszen,
                                       sigma, neuTime,))

    # llh = -2 * np.log(sigma) + np.log(np.sum(sourceTerm)) - np.log(BGRateTerm) + np.log(energyTerm)
    llh = -2 * np.log(sigma) + np.log(np.sum(sourceTerm)) + E_ratio - coszen_prob
    print('Likelihood: {} \n'.format(llh))
    return 2 * llh


def simulate(f, timeBins, filename, tbdata, NSim=1000):
# zen, azi, recoE, cr, AtmWeight,

    enSim = []
    crSim = []
    zenSim = []
    raSim = []
    tot_rate = np.sum([np.sum(f[flux]) for
                       flux in settings['ftypes']])
    for flux in settings['ftypes']:
        print('Frac of {} : {:.2f}'.format(flux, np.sum(f[flux]) / tot_rate))
        N_events = int(NSim * np.sum(f[flux]) / tot_rate)
        draw = np.random.choice(range(len(f)),
                                N_events,
                                p=f[flux] / np.sum(f[flux]))
        enSim.extend(f[draw][settings['E_reco']])
        crSim.extend(f[draw][settings['sigma']])
        zenSim.extend(f[draw][settings['zen_reco']])
        raSim.extend(f[draw][settings['az_reco']])

    sim = dict()
    sim['en'] = np.array(enSim)
    sim['ra'] = np.array(raSim)
    sim['dec'] = np.array(zenSim) - 0.5 * np.pi
    sim['sigma'] = np.array(crSim)

    # atmoSim = AtmWeight[draw]
    tmin = timeBins[0]
    tmax = timeBins[-1]
    # draw uniform neutrino times within catalog time span
    neuTimeSim = np.random.uniform(tmin, tmax, size=len(sim['en']))
    sim['neuTime'] = neuTimeSim

    sim = np.array(
        zip(*[sim[ty[0]] for ty in dtype]), dtype=dtype)

    llh = []
    for i in range(len(sim)):
        # if i % 1 == 0:
        #     print i, sim[i]['en'], sim[i]['ra'], sim[i]['dec'], sim[i]['sigma'], sim[i]['neuTime']

        llh.append(likelihood(sim[i], tbdata, timeBins))

    llh = np.asarray(llh)
    np.save(filename, llh)


def plotLLH(llhfile, outfile, tbdata):
    llh = np.load(llhfile)
    bins = np.linspace(-20, 25, 50)
    fig, ax = newfig(0.9)
    X2 = np.sort(llh)
    F2 = np.ones(len(llh)) - np.array(range(len(llh)))/float(len(llh))
    ax.plot(X2, F2)
    ax.set_xlabel(r'$LLH$')
    ax.set_ylabel('Prob')
    ax.set_yscale('log')
    ax.set_xlim(-20)
    llhNu = likelihood(EHE_event, tbdata, timeBins)
    plt.axvline(llhNu)
    plt.grid()
    plt.savefig(outfile)
    # plt.show()


if __name__ == '__main__':

    jobN = int(sys.argv[1])
    # get Data
    f = np.load(nugen_path)
    astro = powerlaw(f['energy'], f['ow'])
    f = rec_append_fields(f, 'astro',
                          astro,
                          dtypes=np.float64)
    mask = np.isnan(f['cr'])
    f = f[~mask]
    # read 3FGL catalog
    tbdata, timeBins = read3FGL()
    # tbdata, timeBins = readLCCat()
    print('Read Cataloge...Finished')

    # cosZenBins, EPDFSig, EPDFBG = EnergyPDF(f, settings['gamma'])
    # pBG = BGRatePDF(f)
    print('Generating PDFs..Finished')

    filename = 'output/llh_%i_%.1f_%i.npy' % (settings['Nsim'],
                                              settings['gamma'],
                                              jobN)
    simulate(f, timeBins, filename, tbdata, settings['Nsim'])

    plotLLH(filename,
            filename.replace('output', 'plots').replace('npy', 'png'), tbdata)
