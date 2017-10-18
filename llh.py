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
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os


def setNewEdges(edges):
    newEdges = []
    for i in range(0, len(edges) - 1):
        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
        newEdges.append(newVal)
    return np.array(newEdges)


# ------------------------------- Settings ---------------------------- #

nugen_path = 'combined.npy' #/data/user/tglauch/EHE/processed/combined.npy'
#LCC_path = "/home/annaf/BlazarNeutrino/data/myCat.fits"
LCC_path = 'myCat2747.fits'#"/home/annaf/BlazarNeutrino/data/myCat2747.fits"

settings = {'E_reco': 'NPE',
            'zen_reco': 'mpe_zen',
            'az_reco': 'mpe_az',
            'sigma': 'cr',
            'gamma': 2.1,
            'ftypes': ['astro', 'atmo', 'prompt'],  # atmo = conv..sry for that
            'Nsim': 100000,
            'Phi0': 0.91}

dtype = [("en", np.float64),
         ("ra", np.float64),
         ("dec", np.float64),
         ("sigma", np.float64),
         ("neuTime", np.float64)]

# IC170911
# NPE: 5784.9552
# MuEX: 120000 GeV
# MJD : 58018.87118553)

EHE_event = np.array((5784.9552,
                      np.deg2rad(77.43),
                      np.deg2rad(5.72),
                      np.deg2rad(0.25),
                      58014), #-9 #MJD of EHE event #shift event by 4 days to put it in last LC bin (new shifted LC is in the catalog!)
                     dtype=dtype)



# --------------------------------------------------------------------------- #


@np.vectorize
def powerlaw(trueE, ow):
    return ow * settings['Phi0'] * 1e-18 * (trueE * 1e-5) ** (- settings['gamma'])


dtype = [("en", np.float64),
         ("ra", np.float64),
         ("dec", np.float64),
         ("sigma", np.float64),
         ("neuTime", np.float64)]


def getNormInBin(tbdata):
    binNorms = np.zeros_like(tbdata[0]['EFlux_History'])
    for ti in range(len(binNorms)):
       for si in range(len(tbdata)):
          if tbdata[si]['TS'][ti]>4:# and tbdata[si]['npred'][ti]>3:
             binNorms[ti]+=tbdata[si]['EFlux_History'][ti]
    np.save('data/binNorms.npy',binNorms)
    return binNorms

def norm_hist(h):
    h = np.array([i / np.sum(i) if np.sum(i) > 0 else i / 1. for i in h])
    return h


def create_splines(f):
    Hs = dict()
    mask = np.isnan(f['mpe_zen'])

    # energy ratio 2D spline
    print('Create Energy Spline..check yourself whether it is ok')
    zenith_bins=list(np.linspace(-1.,0.,15, endpoint=False)) + list(np.linspace(0.,1.,20))
    tot_weight = np.sum([f[flux][~mask] for flux in settings['ftypes']], axis=0)
    x = np.cos(f['mpe_zen'][~mask])
    y = np.log10(f['NPE'][~mask])
    H_tot, xedges, yedges = np.histogram2d(x, y,
                                       weights=tot_weight,
                                       bins=(zenith_bins,25), normed=True)

    H_tot = np.ma.masked_array(norm_hist(H_tot))
    H_tot.mask = (H_tot <= 0)

    H_astro, xedges, yedges = np.histogram2d(x, y,
                                       weights=f['astro'][~mask],
                                       bins=(zenith_bins,25), normed=True)
    H_astro = np.ma.masked_array(norm_hist(H_astro))
    H_astro.mask = (H_astro <= 0)

    spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 H_astro/H_tot ,
                                 kx=1, ky=1, s=1)
    np.save('E_spline.npy', spline)
    print(spline(-0.1, np.linspace(3.6,6.5,20)))

    # zenith dist 1D spline
    print('Create Zenith Spline...Check if ok..')
    vals, edges = np.histogram(np.cos(f['mpe_zen'][~mask]),
                               weights=tot_weight,
                               bins=30, density=True)
    zen_spl = InterpolatedUnivariateSpline(setNewEdges(edges),
                                           np.log10(vals), k=3)
    print(10**zen_spl(setNewEdges(edges)))
    np.save('coszen_spl.npy', zen_spl)


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


def readLCCat():
    hdulist = fits.open(LCC_path)
    tbdata = hdulist[1].data

    eGal3FHL = ['BLL', 'FSRQ', 'RDG','sbg', 'NLSY1'
                'bll', 'fsrq','agn', 'rdg', 'bcu',
                '', 'unknown']
    eGal3FGL = ['css','BLL','bll','fsrq','FSRQ', 'agn','RDG',
                'rdg','sey','BCU','bcu','GAL','gal','NLSY1',
                'nlsy1','ssrq','']
    eGal2FAV = ['bll','none',
                'bcu','fsrq',
                'rdg', 'nlsy1',
                'agn']
    eGal = eGal3FHL + eGal3FGL + eGal2FAV

    # get rid of extra spaces at the end
    mask = [c.strip() in eGal for c in tbdata['Class1']]
    mask = np.asarray(mask)
    print "N sources: ", len(tbdata)
    tbdata = tbdata[mask]
    print "N sources after cuts: ", len(tbdata)

    timeBins = []

    for i in range(len(hdulist[2].data['Hist_start'])):
        timeBins.append(hdulist[2].data['Hist_start'][i])

    return tbdata, np.asarray(timeBins)


def getTBin(testT, timeBins):
    ind = np.searchsorted(timeBins, testT, side='right')
    return ind - 1

# get extragal. sources and flux
# ra, dec in rad, neuTime in Fermi MET
def get_sources(ra, dec, sigma, neuTime, tbdata, timeBins, binNorms):
    circ = sigma * 3.
    if circ > np.deg2rad(5):
        circ = np.deg2rad(5)

    dist = GreatCircleDistance(np.deg2rad(tbdata['RAJ2000']),
                               np.deg2rad(tbdata['DEJ2000']),
                               ra, dec)
    mask = dist < circ
    foundSources = tbdata[mask]
    if len(foundSources) == 0:
        return None
        
    fluxHist = foundSources['EFlux_History']
    ts = foundSources['TS']
    tbin = getTBin(neuTime, timeBins)
    fluxNeuTime = [f[tbin] for f in fluxHist]
    tsNeuTime = [f[tbin] for f in ts]
    fluxNeuTime = np.asarray(fluxNeuTime)
    tsNeuTime = np.asarray(tsNeuTime)
    tsMask = tsNeuTime > 4
    tsMask = np.asarray(tsMask)
    fluxNeuTime = fluxNeuTime[tsMask]
    foundSources = foundSources[tsMask]
    if (foundSources) == 0:
       return None

    retval = np.deg2rad(foundSources['RAJ2000']), \
        np.deg2rad(foundSources['DEJ2000']), fluxNeuTime/binNorms[tbin]
    if not len(foundSources) == 0:
        print "found %i sources close by" % len(foundSources)
        print "flux weight", fluxNeuTime/binNorms[tbin]
        print "ts ", tsNeuTime
    return retval

def likelihood(sim, tbdata, timeBins, binNorms):

    en = sim['en']
    ra = sim['ra']
    dec = sim['dec']
    sigma = sim['sigma']
    neuTime = sim['neuTime']

    foundSources = get_sources(ra, dec, sigma, neuTime, tbdata, timeBins, binNorms)
    if foundSources is None:
        return 0, -99, -99
    if len(foundSources[0]) == 0:
        return 0, -99, -99

    raS, decS, fluxS = foundSources

    fluxS = np.asarray(fluxS)
    mask = fluxS < 1e-12
    fluxS[mask] = 1e-12

    sourceTerm = fluxS * np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2. * sigma**2))

    coszen = np.cos(dec + 0.5 * np.pi)
    E_ratio = np.log(10 ** E_spline(coszen, np.log10(en))[0])
    coszen_prob = np.log(10 ** (coszen_spline(coszen)) / (2 * np.pi))
    print('E: {} ra: {} coszen: {} \n \
           sigma: {} time : {}'.format(en, ra, coszen,
                                       sigma, neuTime,))

    llh = -2 * np.log(sigma) + np.log(np.sum(sourceTerm)) + E_ratio - coszen_prob
    if llh<0:
       llh = 0
    print('Likelihood: {} \n'.format(llh))
    print '----'
    print (2 * llh), E_ratio, coszen_prob
    return (2 * llh), E_ratio, coszen_prob


def simulate(f, timeBins, filename, tbdata, binNorms, NSim=1000):
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
    sim['ra'] =  np.random.uniform(0., 2*np.pi, len(enSim)) #np.array(raSim)
    sim['dec'] = np.array(zenSim) - 0.5 * np.pi
    sim['sigma'] = np.array(crSim)

    tmin = timeBins[0]
    tmax = timeBins[-1]

    # draw uniform neutrino times within catalog time span
    neuTimeSim = np.random.uniform(tmin, tmax, size=len(sim['en']))
    sim['neuTime'] = neuTimeSim

    sim = np.array(
        zip(*[sim[ty[0]] for ty in dtype]), dtype=dtype)

    llh = []
    e_rat = []
    cosz_prob = []
    for i in range(len(sim)):
        # if i % 1 == 0:
        #     print i, sim[i]['en'], sim[i]['ra'], sim[i]['dec'], sim[i]['sigma'], sim[i]['neuTime']
        l, er, cosz = likelihood(sim[i], tbdata, timeBins,binNorms)
        llh.append(l)
        e_rat.append(er)
        cosz_prob.append(cosz)

    llh = np.asarray(llh)
    e_rat = np.asarray(e_rat)
    cosz_prob = np.asarray(cosz_prob)

    np.save(filename, llh)
    np.save(filename.replace('llh','ERatio'), e_rat)
    np.save(filename.replace('llh','CosZProb'), cosz_prob)


def plotLLH(llhfile, outfile, tbdata,timeBins,binNorms):
    llh = np.load(llhfile)
    bins = np.linspace(-1, 25, 50)
    fig, ax = newfig(0.9)
    X2 = np.sort(llh)
    F2 = np.ones(len(llh)) - np.array(range(len(llh))) / float(len(llh))
    ax.plot(X2, F2)
    ax.set_xlabel(r'$LLH Ratio$')
    ax.set_ylabel('Prob')
    ax.set_yscale('log')
    ax.set_xlim(-1)
    print('Eval Event')
    llhNu, enNu, coszNu = likelihood(EHE_event, tbdata, timeBins,binNorms)
    #plt.axvline(llhNu)
    plt.grid()
    plt.savefig(outfile)

    plt.figure()
    eratio = np.load(llhfile.replace('llh','ERatio'))
    plt.hist(eratio[eratio>-90],bins=50)
    plt.axvline(enNu)
    plt.savefig(outfile.replace('llh','ERatio'))

    plt.figure()
    coszp = np.load(llhfile.replace('llh','CosZProb'))
    plt.hist(coszp[coszp>-90],bins=50)
    plt.axvline(coszNu)
    plt.savefig(outfile.replace('llh','CosZProb'))

    #plt.show()


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
    # read light curve catalog
    tbdata, timeBins = readLCCat()
    print('Read Cataloge...Finished')
    if not os.path.exists('coszen_spl.npy') or \
        not os.path.exists('E_spline.npy'):
            print('Create New Splines..')
            create_splines(f)
    E_spline = np.load('E_spline.npy')[()]
    coszen_spline = np.load('coszen_spl.npy')[()]

    binNorms = getNormInBin(tbdata)

    print('Generating PDFs..Finished')

    filename = 'output/llh_%i_%.1f_%i.npy' % (settings['Nsim'],
                                              settings['gamma'],
                                              jobN)
    simulate(f, timeBins, filename, tbdata, binNorms, settings['Nsim'])

    plotLLH(filename,
            filename.replace('output', 'plots').replace('npy', 'png'), tbdata, timeBins, binNorms)
