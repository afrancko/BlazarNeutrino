#! /usr/bin/env python
# coding: utf-8


# from astropy.coordinates import SkyCoord
# from astropy import units as u
# from plot_conf import *

from fancy_plot import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyfits as fits
#from astropy.table import Column
import sys
from numpy.lib.recfunctions import rec_append_fields
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os
import utils

from scipy.optimize import minimize


# ------------------------------- Settings ---------------------------- #

nugen_path = '/data/user/tglauch/EHE/processed/combined.npy'
muon_path = '/data/user/tglauch/EHE/processed/corsika_combined.npy'
hese_path = 'data/nugen-hese.npy'
#LCC_path = "/home/annaf/BlazarNeutrino/data/myCat.fits"
#LCC_path =  #'myCat2747.fits' #"/home/annaf/BlazarNeutrino/data/myCat2747.fits"
#LCC_path =  "sourceListAll2283_1GeV.fits" #/home/annaf/BlazarNeutrino/data/
#LCC_path = "data/sourceListAll2280_1GeV_fixedSpec_EFlux.fits"#"/home/annaf/BlazarNeutrino/data/sourceListAll2283_1GeV.fits"
LCC_path = "data/sourceListAll2280_1GeV_fixedSpec_EFlux_MAGIC_EBL.fits"
#LCC_path = "sourceListAll2280_1GeV.fits"#"/home/annaf/BlazarNeutrino/data/sourceListAll2283_1GeV.fits"

BOX = True

NOWEIGHT = True

MAGIC = False

HESE = False

# now included in pullCorr function
CR_corr = 1 #1./1.1774


if HESE:
    settings = {'E_reco': 'EReco_millipede',#'muex',
                'zen_reco': 'zen_reco',
                'az_reco': 'az_reco',
                'sigma': 'DirReco_err50',
                'gamma': 2.1,
                'dec_true' : 'TrueDec',
                'ra_true' : 'TrueRA',
                'ra_reco': 'DirReco_splinempe_ra',
                'dec_reco': 'DirReco_splinempe_dec',
                'ftypes': ['Conventional', 'Prompt', 'astro'],  # atmo = conv..sry for that
                'ftype_muon': 'GaisserH3a', #???????
                'Nsim': 50000,
                'Phi0': 0.91,
                'TXS_ra': np.deg2rad(77.36061776),
                'TXS_dec': np.deg2rad(5.69683419),
                'sys_err_corr': 1.21,
                'E_weights': False,
                'distortion': False} 

else:
    settings = {'E_reco': 'truncatedE',#'muex',
                'zen_reco': 'mpe_zen',
                'az_reco': 'mpe_az',
                'sigma': 'cr',
                'gamma': 2.1,
                'ftypes': ['astro', 'atmo', 'prompt'],  # atmo = conv..sry for that
                'ftype_muon': 'GaisserH3a', #???????
                'Nsim': 5000,
                'Phi0': 0.91,
                'TXS_ra': np.deg2rad(77.36061776),
                'TXS_dec': np.deg2rad(5.69683419),
                'sys_err_corr': 1.21,
                'E_weights': False,
                'distortion': False}
    
#addinfo = 'with_E_weights_HE'
addinfo = 'wo_E_weights_increasing_radius_fitNuis_pull_100GeV'
#addinfo = 'wo_E_weights_increasing_radius_noNuis'

CHIBA = False

if CHIBA:
    addinfo = '%s_CHIBA'%addinfo

if NOWEIGHT:
    addinfo = '%s_NOWEIGHT'%addinfo

if HESE==True:
    addinfo = '%s_HESE'%addinfo

if MAGIC==True:
    addinfo = '%s_MAGIC_EBL'%addinfo

if BOX==True:
    addinfo = '%s_BOX'%addinfo

dtype = [("en", np.float64),
         ("ra", np.float64),
         ("dec", np.float64),
         ("sigma", np.float64),
         ("nuTime", np.float64)]


# event = {'LF_zen' : 1.7015,
#          'LF_az' : 5.82828,
#          'mpe_zen' : 1.67165,
#          'mpe_az' : 5.72487,
#          'truncatedE' : 230990}

# IC170911
# NPE: 5784.9552
# MuEX: 120000 GeV
# MJD : 58018.87118553)

EHE_event_best = np.array((5784.9552,
                           #np.deg2rad(77.43),
                           #np.deg2rad(5.72),
                           settings['TXS_ra'],
                           settings['TXS_dec'],
                           utils.pullCorr(np.deg2rad(0.25),5784.9552),
                           58014), #-9 #MJD of EHE event #shift event by 4 days to put it in last LC bin (new shifted LC is in the catalog!)
                          dtype=dtype)

EHE_event = np.array((5784.9552,
                      #np.deg2rad(77.43),
                      #np.deg2rad(5.72),
                      np.deg2rad(77.285),
                      np.deg2rad(5.7517),
                      utils.pullCorr(np.deg2rad(0.25),5784.9552),
                      58014), #-9 #MJD of EHE event #shift event by 4 days to put it in last LC bin (new shifted LC is in the catalog!)
                     dtype=dtype)



# --------------------------------------------------------------------------- #

@np.vectorize
def powerlaw(trueE, ow):
    return ow * settings['Phi0'] * 1e-18 * (trueE * 1e-5) ** (- settings['gamma'])

def getNormInBin(tbdata):
    outfile = LCC_path.replace('.fits','_binNorms.npy')
    if os.path.exists(outfile):
       binNorms = np.load(outfile)
    else:
       binNorms = np.zeros_like(tbdata[0]['eflux1000_100000'])
       for ti in range(len(binNorms)):
          for si in range(len(tbdata)):
             if not np.isnan(tbdata[si]['eflux1000_100000'][ti]):# tbdata[si]['ts'][ti]>4:# and tbdata[si]['npred'][ti]>3:
                binNorms[ti]+=tbdata[si]['eflux1000_100000'][ti]
       np.save(outfile,binNorms)
    return binNorms


# get <I> for Chiba weighting
def getSourceAverage(tbdata, tscut=10, npredcut=9):

    print 'tscut ', tscut
    print 'npredcut ', npredcut

    flux = []
    ts = []
    flux_err = []
    npred = []

    for si in tbdata: # loop over sources
        flux = np.concatenate((flux, si['flux']))
        flux_err = np.concatenate((flux_err, si['flux_err']))
        ts = np.concatenate((ts, si['ts']))
        npred = np.concatenate((npred, si['npred']))
        
    baseline = np.min(flux[(~np.isnan(flux)) & (ts>tscut) & (flux_err<0.5*flux) & (npred>npredcut)])

    print len(flux[(~np.isnan(flux)) & (ts>tscut) & (flux_err<0.5*flux) & (npred>npredcut)]), len(flux)
    
    print "baseline ", baseline
  
    for s in range(len(tbdata)):
        mask = tbdata[s]['flux']<baseline
        mask = np.asarray(mask)
        tbdata[s]['flux'][mask] = baseline
        tbdata[s]['flux'][~((~np.isnan(tbdata[s]['flux'])) & (tbdata[s]['ts']>tscut) & (tbdata[s]['flux_err']<0.5*tbdata[s]['flux']) & (tbdata[s]['npred']>npredcut))] = baseline
        # just overwrite eflux with ratio, because I'm too stupid to add a column
        sumEFlux = np.sum(tbdata[s]['flux'])/float(len(tbdata[0]['flux']))
        tbdata[s]['flux100'] = tbdata[s]['flux']/sumEFlux
        tbdata[s]['flux100_err'] = tbdata[s]['flux_err']/sumEFlux
        
    return tbdata

def getTotalNorm(tbdata):
    if not CHIBA:
        sumEFlux = [np.sum(s['eflux1000_100000'][~np.isnan(s['eflux1000_100000'])]) for s in tbdata]
    else:
        sumEFlux = [np.sum(s['flux100'][~np.isnan(s['flux100'])]) for s in tbdata]
    totSum = np.sum(sumEFlux)
    print "total norm ", totSum
    return totSum
    
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

    eGal3FHL = ['BLL', 'FSRQ', 'RDG', 'sbg', 'NLSY1'
                'bll', 'fsrq', 'agn', 'rdg', 'bcu',
                '', 'unknown']
    eGal3FGL = ['css', 'BLL', 'bll', 'fsrq', 'FSRQ', 'agn', 'RDG',
                'rdg', 'sey', 'BCU', 'bcu', 'GAL', 'gal', 'NLSY1',
                'nlsy1', 'ssrq', '']
    eGal2FAV = ['bll', 'none',
                'bcu', 'fsrq',
                'rdg', 'nlsy1',
                'agn']
    eGal = eGal3FHL + eGal3FGL + eGal2FAV

    timeBins = np.append(tbdata[0]['tmin_mjd'], tbdata[0]['tmax_mjd'][-1])
    return tbdata, np.asarray(timeBins)


def getTBin(testT, timeBins):
    ind = np.searchsorted(timeBins, testT, side='right')
    return ind - 1

# get extragal. sources and flux
# ra, dec in rad, nuTime in Fermi MET
def get_sources(ra, dec, sigma, nuTime, tbdata, timeBins):
 
    if BOX:
        circ = np.deg2rad(0.5)

    else:
        # for testing set to 3
        circ = sigma * 3.#5
        if circ > np.deg2rad(10):
            circ = np.deg2rad(10)

    dist = GreatCircleDistance(np.deg2rad(tbdata['RAJ2000']),
                               np.deg2rad(tbdata['DEJ2000']),
                               ra, dec)
    mask = dist < circ
    foundSources = tbdata[mask]
    #while len(foundSources) == 0:
    #    circ = circ+np.deg2rad(1)
    #    mask = dist < circ
    #    foundSources = tbdata[mask]
        
    if (len(foundSources)) == 0:
       return None

   
    if CHIBA:
        # this is already the ratio!!!
        fluxHist = foundSources['flux100']
        fluxHistErr = foundSources['flux100_err']
    else:
        fluxHist = foundSources['eflux1000_100000']
        fluxHistErr = foundSources['eflux_err']
  

    ts = foundSources['ts']
        
    tbin = getTBin(nuTime, timeBins)
    tsNuTime = np.asarray([f[tbin] for f in ts])
    fluxNuTime = np.asarray([f[tbin] for f in fluxHist])
    fluxNuTimeError = np.asarray([f[tbin] for f in fluxHistErr])
    detByMAGIC = np.asarray([f[tbin] for f in foundSources['MAGIC_EBL_det']]) #MAGIC_noEBL_det, this is 1 or 0 if source in bin is detectable by MAGIC or not
    
    if (len(foundSources)) == 0:
       return None

    maskNan = np.isnan(fluxNuTime)


    if MAGIC:
        retval = np.deg2rad(foundSources['RAJ2000'][~maskNan]), \
                 np.deg2rad(foundSources['DEJ2000'][~maskNan]),\
                 detByMAGIC[~maskNan], \
                 detByMAGIC[~maskNan] # return dummy error
    else:
        retval = np.deg2rad(foundSources['RAJ2000'][~maskNan]), \
                 np.deg2rad(foundSources['DEJ2000'][~maskNan]), fluxNuTime[~maskNan], fluxNuTimeError[~maskNan]
    if 0:#len(foundSources) > 1:
        print "found %i sources close by" % len(foundSources)
        print "flux weight", fluxNuTime
        print "eflux, error ", fluxNuTime, fluxNuTimeError#, binNorms[tbin] 
        print "ts ", tsNuTime
        print foundSources['Source_Name'], foundSources['RAJ2000'], timeBins[tbin], foundSources['DEJ2000'], np.rad2deg(dist[dist<circ])
    return retval


#def negLogLike(fluxMax, fluxS, fluxError, sourceTerm):
#    nuisanceTermLog = -(fluxMax-fluxS)**2 /(2*fluxError**2)
#    sourceSum = np.sum(sourceTerm*fluxMax)
#    if (sourceSum<=1e-25):
#        sourceSum=1e-25
#    llh =   np.log(sourceSum) + np.sum(nuisanceTermLog)
#    return -2*llh


def negLogLike(fluxMax, fluxS, fluxError, spatialTerm, Nw):

    if NOWEIGHT:
        return -2*np.log(np.sum(spatialTerm))
    
    if MAGIC:
        sourceSum = np.dot(spatialTerm,fluxS)
        if (sourceSum<=1e-25):
            sourceSum=1e-25
        return -2*np.log(np.sum(sourceSum))

    nuisanceTermLog = (fluxMax-fluxS)**2 /(fluxError**2)
    sourceSum = np.dot(spatialTerm,fluxMax)

    #print "spatialTerm*fluxMax", spatialTerm*fluxMax
    #print "sum ", sourceSum
    #print "nuisanceTermLog ", nuisanceTermLog
    
    #if (sourceSum<=1e-25):
    #    sourceSum=1e-25

    #print "Nw ", Nw
    #print "fluxS.sum ", fluxS.sum()
    #print "fluxMax.sum ", fluxMax.sum()
    #print "Nw - fluxS.sum() + fluxMax.sum()", (Nw - fluxS.sum() + fluxMax.sum())
    llh =   2.*np.log(sourceSum) - np.sum(nuisanceTermLog) - 2*np.log(Nw - fluxS.sum() + fluxMax.sum())
    #print "llh ", llh
    return -llh


#def likelihood(sim, tbdata, timeBins, totNorm, distortion=False, E_weights=True):
def likelihood(sim, tbdata, timeBins, Nw, distortion=False):
    
    en = sim['en']
    ra = sim['ra']
    dec = sim['dec']
    sigma = sim['sigma']
    nuTime = sim['nuTime']

    foundSources = get_sources(ra, dec, sigma, nuTime, tbdata, timeBins)
    if foundSources is None:
        return -99
    if len(foundSources[0]) == 0:
        return -99

    raS, decS, fluxS, fluxError = foundSources

    fluxS = np.asarray(fluxS)
    fluxError = np.asarray(fluxError)

    # if the fit fails, the flux is nan - however in the latest light curve catalog
    # that should not happen any more
    fluxS = np.nan_to_num(fluxS)
    fluxError = np.nan_to_num(fluxError) 
    
    if not distortion:
        coszen = np.cos(utils.dec_to_zen(dec))
        coszen_prob = np.log(10 ** (coszen_spline(coszen)) / (2 * np.pi))
    else:
        coszen_prob = np.random.uniform(0,5)

    coszenS = np.cos(utils.dec_to_zen(decS))
    acceptance = 10**coszen_signal_reco_spline(coszenS)

    sigma *= CR_corr*settings['sys_err_corr']
    spatialTerm = np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2.*sigma**2)) * acceptance

    if BOX:
       dist = GreatCircleDistance(ra, dec, raS, decS)
       spatialTerm[dist<=np.deg2rad(0.5)] = 1
       spatialTerm[dist>np.deg2rad(0.5)] = 0



    #print "sigma ", sigma
    #print "spatialTerm ", spatialTerm
    #print "dist ", GreatCircleDistance(ra, dec, raS, decS)
    
    bounds = []
    for s in range(len(fluxS)):
        lowFlux = fluxS[s]-fluxError[s]
        if lowFlux<0:
            lowFlux = 0
        highFlux = fluxS[s]+fluxError[s]
        bounds.append((lowFlux, highFlux))

    if not NOWEIGHT and not MAGIC:
        res = minimize(negLogLike,x0=fluxS,
                       args=(fluxS, fluxError,spatialTerm,Nw), method='Nelder-Mead',#'SLSQP',
                       bounds=bounds, options={'maxiter':50,'disp':False,'ftol':1e-11})
    
        fluxMax = res.x
    else:
        fluxMax = fluxS

    print fluxS
        
    #fluxM = fluxS*0.5 + np.sqrt((fluxS*0.5)**2 + fluxError**2 )
    #print fluxMax, fluxS, fluxError, Nw, coszen_prob, acceptance
    
    # signal likelihood
    #print 'likelihood before/after min. ', -negLogLike(fluxS, fluxS, fluxError,spatialTerm, Nw), -negLogLike(fluxMax, fluxS, fluxError,spatialTerm, Nw)
    
    if BOX:
       llh = -negLogLike(fluxMax, fluxS, fluxError,spatialTerm, Nw)  - 2*coszen_prob
    else:
       llh = -negLogLike(fluxMax, fluxS, fluxError,spatialTerm, Nw)  - 2*coszen_prob - 4*np.log(sigma)
    
    print('Likelihood: {} \n'.format(llh))
    print '----------------------------'
    if 0: #len(fluxS)>1:#np.isnan(llh) or np.isinf(llh):
        print "************************************************"
        print 'ra, dec ', np.rad2deg(ra), np.rad2deg(dec)
        print 'coszen ', coszen
        print 'zen spline ', coszen_signal_reco_spline(coszen)
        print "spatialTerm ", spatialTerm
        print "coszen_prob ", coszen_prob
        print "sigma ", np.rad2deg(sigma), -4 * np.log(sigma)
        print "flux ", fluxS, fluxError
        print "fluxMax ", fluxMax
        exit()
        print "************************************************"
        
    return llh


def inject_pointsource(f, tbdata, timeBins, totNorm, raS, decS, nuTime, filename='', gamma=2.1,
                       Nsim=1000, distortion=False, sourceList=None):

    enSim = []
    crSim = []
    zenSim = []
    raSim = []
    timeSim = []
    distTrue = []
    
    # check if single source or source list
    if not raS is None:
        zen_mask = np.abs(np.cos(f['zenith'])-np.cos(utils.dec_to_zen(decS)))<0.05
        fSource = f[zen_mask]

        print "selected %i events in given zenith range"%len(fSource)
        for i in range(len(fSource)):
            rotatedRa, rotatedDec = utils.rotate(fSource['azimuth'][i],
                                                 utils.zen_to_dec(fSource['zenith'][i]),
                                                 raS, decS, 
                                                 fSource[settings['az_reco']][i],
                                                 utils.zen_to_dec(fSource[settings['zen_reco']][i]))
            fSource[i][settings['az_reco']] = rotatedRa
            fSource[i][settings['zen_reco']] = utils.dec_to_zen(rotatedDec)

        weight = fSource['ow']/fSource['energy']**gamma
        draw = np.random.choice(range(len(fSource)),
                                Nsim,
                                p=weight / np.sum(weight))

        enSim.extend(fSource[draw][settings['E_reco']])
        crSim.extend(fSource[draw][settings['sigma']])
        zenSim.extend(fSource[draw][settings['zen_reco']])
        raSim.extend(fSource[draw][settings['az_reco']])
        timeSim = np.ones_like(np.asarray(enSim))*nuTime
        distTrue.append(GreatCircleDistance(rotatedRa, rotatedDec, raS, decS))
    else:
        print "sample from gamma-ray brightness distribution"
        # mask nan fluxes, happens for 48 time bins (fit failed)
        maskNan = np.isnan(sourceList['eflux1000_100000'])
        sourceList = sourceList[~maskNan]
     
        # remove sources close to cos(zen)=1 because there are no signal events there
        coszen = np.cos(utils.dec_to_zen(np.deg2rad(sourceList['DEJ2000'])))
        #zenMask = coszen<0.825710148394
        #sourceList = sourceList[zenMask]
        #coszen = coszen[zenMask]        
        coszenWeight = (10 ** (coszen_signal_spline(coszen)))

        # assign eflux as weight, weight with zen dist of neutrinos
        if CHIBA:
            weight = sourceList['flux']*coszenWeight
        else:
            weight = sourceList['eflux1000_100000']*coszenWeight

            
        # draw from list of monthly bins, weighted with eflux
        draw = np.random.choice(range(len(sourceList)),
                                Nsim,
                                p=weight / np.sum(weight))
        sourceList = sourceList[draw]    
        fSource = np.zeros_like(f[0:Nsim])
      
        i = 0
        for s in sourceList:
            # select a neutrino and rotate to source direction
            coszen = np.cos(utils.dec_to_zen(np.deg2rad(s['DEJ2000'])))
            zen_mask = np.abs(np.cos(f['zenith'])-coszen)<0.05
            fDist = f[zen_mask]
            if len(fDist)<50:
                zen_mask = np.abs(np.cos(f['zenith'])-coszen)<0.1
                fDist = f[zen_mask]
                if len(fDist)<10:
                    zen_mask = np.abs(np.cos(f['zenith'])-coszen)<0.15
                    fDist = f[zen_mask]
                    if len(fDist)<1:
                        print "drop source. there are no MC events in that zen bin, %f. Exit."%s['DEJ2000']
                        exit()
                        #continue

            #coszenWeight = (10 ** (coszen_signal_spline(np.cos(fDist['zenith']))))
            weightAstro = fDist['astro']#*coszenWeight
            # pick an event following signal weight distribution
            rind = np.random.choice(range(len(fDist)),
                                    1,
                                    p=weightAstro / np.sum(weightAstro))
            rotatedRa, rotatedDec = utils.rotate(fDist['azimuth'][rind],
                                                 utils.zen_to_dec(fDist['zenith'][rind]),
                                                 np.deg2rad(s['RAJ2000']), np.deg2rad(s['DEJ2000']),
                                                 fDist[settings['az_reco']][rind],
                                                 utils.zen_to_dec(fDist[settings['zen_reco']][rind]))
    
            fSource[i][settings['az_reco']] = rotatedRa
            fSource[i][settings['zen_reco']] = utils.dec_to_zen(rotatedDec)
            fSource[i][settings['E_reco']] = fDist[rind][settings['E_reco']]
            fSource[i][settings['sigma']] = fDist[rind][settings['sigma']]
            timeSim.append(s['binCenterMJD'])
            distTrue.append(GreatCircleDistance(rotatedRa, rotatedDec, np.deg2rad(s['RAJ2000']), np.deg2rad(s['DEJ2000'])))
            i = i+1

        enSim.extend(fSource[settings['E_reco']])
        crSim.extend(fSource[settings['sigma']])
        zenSim.extend(fSource[settings['zen_reco']])
        raSim.extend(fSource[settings['az_reco']])
        
        
    sim = dict()
    sim['en'] = np.array(enSim)
    sim['ra'] =  np.array(raSim)
    sim['dec'] = utils.zen_to_dec(np.array(zenSim))
    sim['sigma'] = np.array(crSim)
    sim['nuTime'] = np.array(timeSim)
 
    sim = np.array( zip(*[sim[ty[0]] for ty in dtype]), dtype=dtype)

    llh = []
    print_after = round(Nsim/10)
    if distortion:
        print('Be careful you are running script with distortion set to TRUE!!!')
    for i in range(len(sim)):
        if i % print_after == 0:
            print('{}/{}'.format(i, Nsim))
        #print sourceList['Source_Name'][i], sourceList['eflux'][i], sim['nuTime'][i]
        l = likelihood(sim[i], tbdata, timeBins, totNorm, distortion=distortion)
        llh.append(l)

    if not filename == '':
        np.save(filename, llh)
    
    return llh



def calc_p_value(TS_dist, llh_vals, name='' ,save=True):
    if not isinstance(llh_vals, float):
        pvals = []
        for val in llh_vals:
            pval = float(len(np.where(TS_dist > val)[0])) / float(len(TS_dist))
            pvals.append(pval)
        if save:
            np.save('./output/pvals_{}.npy'.format(name), np.array(pvals))
        return pvals
    else:
        pval = float(len(np.where(TS_dist > llh_vals)[0])) / float(len(TS_dist))
        return pval
    return 


# add f_muon
def simulate(f, f_m, timeBins, tbdata, totNorm, NSim=1000, filename='', distortion=False, E_weights=True):
# zen, azi, recoE, cr, AtmWeight,

    enSim = []
    crSim = []
    zenSim = []
    raSim = []
    tot_rate = np.sum([np.sum(f[flux]) for
                       flux in settings['ftypes']])
    # add f_muons to tot_rate ftype is here muon
    if not f_m==None:
        flux_m = settings['ftype_muon']
        tot_rate += np.sum(f_m[flux_m])
        
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

    if not f_m==None:
        print('Frac of {} : {:.2f}'.format(flux_m, np.sum(f_m[flux_m]) / tot_rate))
        N_events = int(NSim * np.sum(f_m[flux_m]) / tot_rate)

        draw = np.random.choice(range(len(f_m)),
                                N_events,
                                p=f_m[flux_m] / np.sum(f_m[flux_m]))
        enSim.extend(f_m[draw][settings['E_reco']])
        crSim.extend(f_m[draw][settings['sigma']])
        zenSim.extend(f_m[draw][settings['zen_reco']])
        raSim.extend(f_m[draw][settings['az_reco']])
        
    sim = dict()
    sim['en'] = np.array(enSim)
    sim['ra'] =  np.random.uniform(0., 2 * np.pi, len(enSim))  #np.array(raSim)
    sim['dec'] = utils.zen_to_dec(np.array(zenSim))
    sim['sigma'] = np.array(crSim)

    tmin = timeBins[0]
    tmax = timeBins[-1]

    # draw uniform neutrino times within catalog time span
    nuTimeSim = np.random.uniform(tmin, tmax, size=len(sim['en']))
    sim['nuTime'] = nuTimeSim

    sim = np.array(
        zip(*[sim[ty[0]] for ty in dtype]), dtype=dtype)

    llh = []
    print_after = round(NSim/10)
    if distortion:
        print('Be careful you are running script with distortion set to TRUE!!!')
    for i in range(len(sim)):
        if i % print_after == 0:
            print('{}/{}'.format(i, NSim))
        l = likelihood(sim[i], tbdata, timeBins, totNorm, distortion=distortion)
        llh.append(l)

    llh = np.asarray(llh)
    if not filename == '':
        np.save(filename, llh)
    return llh


def plotLLH(llhfile, tbdata, timeBins, totNorm, EHE_event, distortion=False):
    outfile = llhfile.replace('output', 'plots').replace('npy', 'png')
    llh = np.load(llhfile)
    bins = np.linspace(-100, 25, 100)
    fig, ax = newfig(0.9)
    X2 = np.sort(llh)
    F2 = np.ones(len(llh)) - np.array(range(len(llh))) / float(len(llh))
    ax.plot(X2, F2, color='blue',lw=3)
    ax.set_xlabel(r'$LLH Ratio$')
    ax.set_ylabel('Prob')
    ax.set_yscale('log')
    ax.set_xlim(-10,5)
    print('Eval Event')
    llhNu = likelihood(EHE_event, tbdata, timeBins, totNorm, distortion=False)
    print "llh ", llhNu
    plt.axvline(llhNu)
    plt.grid()
    plt.savefig(outfile)

    return llhNu




def readHESE(fname):
    f = np.load(fname)
    mask = f['SignalTrackness']>0.1
    f = f[mask]
    f = rec_append_fields(f,'zenith',
                          #utils.dec_to_zen(f[settings['dec_true']]),
                          f[settings['dec_true']],
                          dtypes=np.float64)
    f = rec_append_fields(f,'azimuth',
                          f[settings['ra_true']],
                          dtypes=np.float64)
    f = rec_append_fields(f,'zen_reco',
                          #utils.dec_to_zen(f['DirReco_splinempe_dec']),
                          f['DirReco_splinempe_dec'],
                          dtypes=np.float64)
    f = rec_append_fields(f,'az_reco',
                          f['DirReco_splinempe_ra'],
                          dtypes=np.float64)
    f = rec_append_fields(f,'astro',
                          f['EMinus2']/f['TrueEnergy']**(settings['gamma']-2.),
                          dtypes=np.float64)
    f = rec_append_fields(f,'ow',
                          f['EMinus2']*f['TrueEnergy']**2,
                          dtypes=np.float64)
    f = rec_append_fields(f,'energy',
                          f['TrueEnergy'],
                          dtypes=np.float64)
    return f
 
    
if __name__ == '__main__':

    jobN = int(sys.argv[1])
    # get Data
    if not HESE:
        f = np.load(nugen_path)
        astro = powerlaw(f['energy'], f['ow'])
        f = rec_append_fields(f, 'astro',
                              astro,
                              dtypes=np.float64)
        delta_mask = np.degrees(utils.delta_psi(f['zenith'], f['azimuth'], f[settings['zen_reco']], f[settings['az_reco']]))<5
        mask = np.isfinite(f['cr'])
        f = f[mask&delta_mask]
        f['cr'][f['cr']<np.deg2rad(0.25)] = np.deg2rad(0.25)
        f['cr'] = utils.pullCorr(f['cr'],f['NPE'])
        # get muon Data
        f_m = np.load(muon_path)
        mask = np.isfinite(f_m['cr'])
        f_m = f_m[mask]
        f_m['cr'][f_m['cr']<np.deg2rad(0.25)] = np.deg2rad(0.25)
        f_m['cr'] = utils.pullCorr(f_m['cr'],f_m['NPE'])
        spline_name = ''
    else:
        f = readHESE(hese_path)
        spline_name = '_hese'
        f_m = None


    print 'splinename', spline_name
    # read light curve catalog
    tbdata, timeBins = readLCCat()

    if CHIBA:
        # add average flux to catalog table
        tbdata = getSourceAverage(tbdata)
    
    print('Read Cataloge...Finished')
    if not os.path.exists('coszen_spl%s.npy'%spline_name) or \
        not os.path.exists('E_spline.npy%s'%spline_name) or \
        not os.path.exists('coszen_signal_spl%s.npy'%spline_name):
            print('Create New Splines..')
            utils.create_splines(f,f_m,settings['ftypes'], settings['ftype_muon'],
                                 settings['zen_reco'],
                                 settings['az_reco'], 
                                 settings['E_reco'], spline_name)
    E_spline = np.load('E_spline%s.npy'%spline_name)[()]
    coszen_spline = np.load('coszen_spl%s.npy'%spline_name)[()]
    coszen_signal_spline = np.load('coszen_signal_spl%s.npy'%spline_name)[()]
    coszen_signal_reco_spline = np.load('coszen_signal_reco_spl%s.npy'%spline_name)[()]

    
    #binNorms = getNormInBin(tbdata)
    totNorm = getTotalNorm(tbdata)
    
    print('Generating PDFs..Finished')

    if HESE:
        which_sample = 'HESE'
    else:
        which_sample = 'EHE'

    filename = './output/{}/{}_llh_{}_{:.2f}_{}.npy'.format(which_sample,addinfo, settings['Nsim'],
                                              settings['gamma'],
                                              jobN)

    if not os.path.exists('./output/{}/'.format(which_sample)):
        os.makedirs('./output/{}/'.format(which_sample))
    print('##############Create BG TS Distrbution##############')
    if not os.path.exists(filename):
        llh_bg_dist= simulate(f, f_m, timeBins, tbdata,
                              totNorm, settings['Nsim'], filename=filename)
    else:
        print('Load Trials...')
        llh_bg_dist = np.load(filename)

    # print('##############Generate Background Trials##############')
    # llh_trials = simulate(f, f_m, timeBins, tbdata,
    #                      totNorm, settings['Nsim'], E_weights=True)

    # print('calculate p-values')
    # print len(llh_bg_dist), len(llh_trials)
    # calc_p_value(llh_bg_dist, llh_trials, name=addinfo)

    print('##############Generate Signal Trials, single source##############')
    signal_gamma = 2.1
    signal_trials = inject_pointsource(f, tbdata, timeBins, totNorm, settings['TXS_ra'], settings['TXS_dec'], EHE_event['nuTime'], 
                                       filename=filename.replace('%.2f'%settings['gamma'],'%.2f'%signal_gamma).replace('.npy','_signal.npy'), 
                                       gamma=signal_gamma, Nsim=settings['Nsim'],distortion=settings['distortion'],
                                       sourceList=None)
    print('calculate p-values signal')
    #print len(llh_bg_dist), len(signal_trials)
    #calc_p_value(llh_bg_dist, signal_trials, name='%s_signal'%addinfo)


    #print('##############Generate Signal Trials from brightness distribution##############')
    #sourceList = utils.makeSourceFluxList(tbdata, LCC_path)
    #mask = np.isnan(sourceList['eflux'])
    #print "time bins with nan ", len(sourceList[mask])
    #np.random.seed(42)
    #signal_trials = inject_pointsource(f, tbdata, timeBins, totNorm, None, None, None,
    #                                   filename=filename.replace('%.2f'%settings['gamma'],'%.2f'%signal_gamma).replace('.npy','_signal_sourceList.npy'),
    #                                   gamma=signal_gamma, Nsim=settings['Nsim'],distortion=settings['distortion'],E_weights=settings['E_weights'],
    #                                   sourceList=sourceList)


    #print('calculate p-values signal')
    #calc_p_value(llh_bg_dist, signal_trials, name='%s_signal_brightness'%addinfo)


    exp_llh = plotLLH(filename,
                      tbdata,
                      timeBins,
                      totNorm,
                      EHE_event,
                      distortion=settings['distortion'])

    print('Exp LLH : {}'.format(exp_llh))
    #print('Exp P-Val {}'.format(calc_p_value(llh_bg_dist, exp_llh, save=False)))

    #exp_llh = plotLLH(filename, tbdata, timeBins, totNorm, EHE_event, distortion=settings['distortion'], E_weights=settings['E_weights'])
    #print exp_llh
    #print('Exp P-Val {}'.format(calc_p_value(llh_bg_dist, exp_llh, save=False)))
    #print('best possible p-value')
    #exp_llh = plotLLH(filename, tbdata, timeBins, totNorm, EHE_event_best, distortion=settings['distortion'], E_weights=settings['E_weights'])
    #print('Exp P-Val {}'.format(calc_p_value(llh_bg_dist, exp_llh, save=False)))
    # print exp_llh

    #pastEvents = utils.getPastAlerts():
    #llhPastEvent = []
    #pvalPastEvents = []
    #for ev in pastEvents:
    #    llhPastEvent.append(likelihood(ev, tbdata, timeBins, totNorm, distortion=False, E_weights=True))
    #    pvalPastEvents.append(calc_p_value(llh_bg_dist, llhPastEvent[-1], save=False))

