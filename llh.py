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
import utils

#def setNewEdges(edges):
#    newEdges = []
#    for i in range(0, len(edges) - 1):
#        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
#        newEdges.append(newVal)
#    return np.array(newEdges)


# ------------------------------- Settings ---------------------------- #

nugen_path = 'combined.npy' #'/data/user/tglauch/EHE/processed/combined.npy'
muon_path = '/data/user/tglauch/EHE/processed/corsika_combined.npy'
#LCC_path = "/home/annaf/BlazarNeutrino/data/myCat.fits"
#LCC_path =  #'myCat2747.fits' #"/home/annaf/BlazarNeutrino/data/myCat2747.fits"
#LCC_path =  "sourceListAll2283_1GeV.fits" #/home/annaf/BlazarNeutrino/data/
LCC_path = "sourceListAll2283_1GeV.fits"#"/home/annaf/BlazarNeutrino/data/sourceListAll2283_1GeV.fits"

settings = {'E_reco': 'truncatedE',#'muex',
            'zen_reco': 'mpe_zen',
            'az_reco': 'mpe_az',
            'sigma': 'cr',
            'gamma': 2.1,
            'ftypes': ['astro', 'atmo', 'prompt'],  # atmo = conv..sry for that
            'Nsim': 10000,
            'Phi0': 0.91,
            'TXS_ra': np.deg2rad(77.36061776),
            'TXS_dec': np.deg2rad(5.69683419),
            'E_weights': True,
            'distortion': False}
#addinfo = 'with_E_weights_HE'
addinfo = 'with_E_weights_increasing_radius'


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

EHE_event = np.array((230990,
                      np.deg2rad(77.43),
                      np.deg2rad(5.72),
                      np.deg2rad(0.25),
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
       binNorms = np.zeros_like(tbdata[0]['eflux'])
       for ti in range(len(binNorms)):
          for si in range(len(tbdata)):
             if not np.isnan(tbdata[si]['eflux'][ti]):# tbdata[si]['ts'][ti]>4:# and tbdata[si]['npred'][ti]>3:
                binNorms[ti]+=tbdata[si]['eflux'][ti]
       np.save(outfile,binNorms)
    return binNorms


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

    # get rid of extra spaces at the end
    #mask = [c.strip() in eGal for c in tbdata['Class1']]
    #mask = np.asarray(mask)
    #print "N sources: ", len(tbdata)
    #tbdata = tbdata[mask]
    #print "N sources after cuts: ", len(tbdata)

    #timeBins = []

    #for i in range(len(hdulist[2].data['Hist_start'])):
    #    timeBins.append(hdulist[2].data['Hist_start'][i])
    timeBins = np.append(tbdata[0]['tmin_mjd'], tbdata[0]['tmax_mjd'][-1])
    #print timeBins
    return tbdata, np.asarray(timeBins)


def getTBin(testT, timeBins):
    ind = np.searchsorted(timeBins, testT, side='right')
    return ind - 1

# get extragal. sources and flux
# ra, dec in rad, nuTime in Fermi MET


def get_sources(ra, dec, sigma, nuTime, tbdata, timeBins, binNorms):
    # for testing set to 3
    circ = sigma * 5.
    if circ > np.deg2rad(10):
        circ = np.deg2rad(10)

    dist = GreatCircleDistance(np.deg2rad(tbdata['RAJ2000']),
                               np.deg2rad(tbdata['DEJ2000']),
                               ra, dec)
    mask = dist < circ
    foundSources = tbdata[mask]
    while len(foundSources) == 0:
        circ = circ+np.deg2rad(1)
        mask = dist < circ
        foundSources = tbdata[mask]
        
    if (len(foundSources)) == 0:
       return None
    
    fluxHist = foundSources['eflux']
    fluxHistErr = foundSources['eflux_err']

    ts = foundSources['ts']

    tbin = getTBin(nuTime, timeBins)
    tsNuTime = np.asarray([f[tbin] for f in ts])
    fluxNuTime = np.asarray([f[tbin] for f in fluxHist])
    fluxNuTimeError = np.asarray([f[tbin] for f in fluxHistErr])

    if (len(foundSources)) == 0:
       return None

    maskNan = np.isnan(fluxNuTime)
   
    retval = np.deg2rad(foundSources['RAJ2000'][~maskNan]), \
        np.deg2rad(foundSources['DEJ2000'][~maskNan]), fluxNuTime[~maskNan], fluxNuTimeError[~maskNan], binNorms[tbin]
    #if not len(foundSources) == 0:
    #    print "found %i sources close by" % len(foundSources)
    #    print "flux weight", fluxNuTime/binNorms[tbin]
    #    print "eflux, error, norm ", fluxNuTime, fluxNuTimeError, binNorms[tbin] 
    #    print "ts ", tsNuTime
    #    print foundSources['Source_Name'], foundSources['RAJ2000'], timeBins[tbin], foundSources['DEJ2000'], dist[dist<circ]
    return retval


def likelihood(sim, tbdata, timeBins, binNorms, distortion=False, E_weights=True):

    en = sim['en']
    ra = sim['ra']
    dec = sim['dec']
    sigma = sim['sigma']
    nuTime = sim['nuTime']

    foundSources = get_sources(ra, dec, sigma, nuTime, tbdata, timeBins, binNorms)
    if foundSources is None:
        return -99
    if len(foundSources[0]) == 0:
        return -99

    raS, decS, fluxS, fluxError, fluxNorm = foundSources

    fluxS = np.asarray(fluxS)
    fluxError = np.asarray(fluxError)
    
    # in 50 out 2200*119 cases, i.e. 0.02% the fit fails and returns nan.
    # in the following those are replaced by zeros
    fluxS = np.nan_to_num(fluxS)
    fluxError = np.nan_to_num(fluxError) 

    #mask = (fluxS<=0) | (fluxError<=0)
    #fluxS[mask] = 1e-25
    #fluxError[mask] =  1e-12
    
    #if fluxError<=0:
    #    fluxError = 1e
    
    if not distortion:
        coszen = np.cos(utils.dec_to_zen(dec))
        E_ratio = np.log(E_spline(coszen, np.log10(en),grid=False))
        coszen_prob = np.log(10 ** (coszen_spline(coszen)) / (2 * np.pi))
    else:
        coszen_prob = np.random.uniform(0,5)
        E_ratio = np.random.uniform(0,5)

    # account for flux error
    fluxMax = fluxS*0.5 + np.sqrt((fluxS * 0.5) ** 2 + 0.5* fluxError ** 2)
    
    nuisanceTerm = 1./np.sqrt(2.*np.pi*fluxError ** 2) * np.exp(-(fluxMax-fluxS)**2 /(2*fluxError**2))
    
    sourceTerm = fluxMax/fluxNorm * np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2. * sigma**2))
    #sourceTerm = fluxS/fluxNorm * np.exp(-GreatCircleDistance(ra, dec, raS, decS)**2 / (2. * sigma**2))

    sourceSum = np.sum(sourceTerm*nuisanceTerm)
    
    if (sourceSum<=1e-25):
        sourceSum=1e-25
    
    if E_weights:
        llh = -2 * np.log(sigma) + np.log(sourceSum) + E_ratio - coszen_prob 
    else:
        llh = -2 * np.log(sigma) + np.log(sourceSum) - coszen_prob 

    print('Likelihood: {} \n'.format(2*llh))
    print '----'
    if np.isnan(llh) or np.isinf(llh):
        print "sourceTerm ", sourceTerm
        print "nuisanceTerm ", nuisanceTerm
        print "E_ratio ", E_ratio
        print "coszen_prob ", coszen_prob
        print "sigma ", sigma
        print "flux ", fluxS, fluxError
        exit()
    return 2 * llh


def inject_pointsource(f, raS, decS, nuTime, filename='', gamma=2.1, Nsim=1000, distortion=False, E_weights=True, sourceList=None):

    enSim = []
    crSim = []
    zenSim = []
    raSim = []
    timeSim = []

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

    else:
        print "sample from gamma-ray brightness distribution"
        # mask nan fluxes, happens for 48 time bins (fit failed)
        maskNan = np.isnan(sourceList['eflux'])
        sourceList = sourceList[~maskNan]
     
        # remove sources close to cos(zen)=1 because there are no signal events there
        coszen = np.cos(utils.dec_to_zen(np.deg2rad(sourceList['DEJ2000'])))
        zenMask = coszen<0.825710148394
        sourceList = sourceList[zenMask]
        coszen = coszen[zenMask]        
        coszenWeight = (10 ** (coszen_signal_spline(coszen)))

        # assign eflux as weight, weight with zen dist of neutrinos
        weight = sourceList['eflux']*coszenWeight
     
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

            weightAstro = fDist['astro']
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
        l = likelihood(sim[i], tbdata, timeBins, binNorms, distortion=distortion, E_weights=E_weights)
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
    else:
        pval = float(len(np.where(TS_dist > llh_vals)[0])) / float(len(TS_dist))
        return pval
    return


# add f_muon
def simulate(f, timeBins, tbdata, binNorms, NSim=1000, filename='', distortion=False, E_weights=True):
# zen, azi, recoE, cr, AtmWeight,

    enSim = []
    crSim = []
    zenSim = []
    raSim = []
    tot_rate = np.sum([np.sum(f[flux]) for
                       flux in settings['ftypes']])
    # add f_muons to tot_rate ftype is here muon
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
        l = likelihood(sim[i], tbdata, timeBins, binNorms, distortion=distortion, E_weights=E_weights)
        llh.append(l)

    llh = np.asarray(llh)
    if not filename == '':
        np.save(filename, llh)
    return llh


def plotLLH(llhfile, tbdata, timeBins, binNorms, distortion=False, E_weights=True):
    outfile = llhfile.replace('output', 'plots').replace('npy', 'png')
    llh = np.load(llhfile)
    bins = np.linspace(-100, 25, 100)
    fig, ax = newfig(0.9)
    X2 = np.sort(llh)
    F2 = np.ones(len(llh)) - np.array(range(len(llh))) / float(len(llh))
    ax.plot(X2, F2)
    ax.set_xlabel(r'$LLH Ratio$')
    ax.set_ylabel('Prob')
    ax.set_yscale('log')
    ax.set_xlim(-100)
    print('Eval Event')
    llhNu = likelihood(EHE_event, tbdata, timeBins, binNorms, distortion=False, E_weights=True)
    #plt.axvline(llhNu)
    plt.grid()
    plt.savefig(outfile)

    return llhNu

    #plt.show()


if __name__ == '__main__':

    jobN = int(sys.argv[1])
    # get Data
    f = np.load(nugen_path)
    astro = powerlaw(f['energy'], f['ow'])
    f = rec_append_fields(f, 'astro',
                          astro,
                          dtypes=np.float64)
    delta_mask = np.degrees(utils.delta_psi(f['zenith'], f['azimuth'], f[settings['zen_reco']], f[settings['az_reco']]))<5
    mask = np.isfinite(f['cr'])
    f = f[mask&delta_mask]
    # read light curve catalog
    tbdata, timeBins = readLCCat()
    print('Read Cataloge...Finished')
    if not os.path.exists('coszen_spl.npy') or \
        not os.path.exists('E_spline.npy') or \
        not os.path.exists('coszen_signal_spl.npy'):
            print('Create New Splines..')
            utils.create_splines(f,settings['ftypes'],
                                 settings['zen_reco'],
                                 settings['az_reco'], 
                                 settings['E_reco'])
    E_spline = np.load('E_spline.npy')[()]
    coszen_spline = np.load('coszen_spl.npy')[()]
    coszen_signal_spline = np.load('coszen_signal_spl.npy')[()]

    
    binNorms = getNormInBin(tbdata)

    print('Generating PDFs..Finished')

    filename = './output/{}_llh_{}_{:.2f}_{}.npy'.format(addinfo, settings['Nsim'],
                                              settings['gamma'],
                                              jobN)
    print('##############Create BG TS Distrbution##############')
    if not os.path.exists(filename):
        llh_bg_dist= simulate(f, timeBins, tbdata,
                              binNorms, settings['Nsim'], filename=filename, E_weights=settings['E_weights'])
    else:
        llh_bg_dist = np.load(filename)

    print('##############Generate Background Trials##############')
    llh_trials = simulate(f, timeBins, tbdata,
                          binNorms, settings['Nsim'], E_weights=True)

    print('calculate p-values')
    print len(llh_bg_dist), len(llh_trials)
    calc_p_value(llh_bg_dist, llh_trials, name=addinfo)

    print('##############Generate Signal Trials, single source##############')
    signal_gamma = 2.1
    signal_trials = inject_pointsource(f, settings['TXS_ra'], settings['TXS_dec'], EHE_event['nuTime'], 
                                       filename=filename.replace('%.2f'%settings['gamma'],'%.2f'%signal_gamma).replace('.npy','_signal.npy'), 
                                       gamma=signal_gamma, Nsim=settings['Nsim'],distortion=settings['distortion'],E_weights=settings['E_weights'],
                                       sourceList=None)
    print('calculate p-values signal')
    print len(llh_bg_dist), len(signal_trials)
    calc_p_value(llh_bg_dist, signal_trials, name='%s_signal'%addinfo)


    print('##############Generate Signal Trials from brightness distribution##############')
    sourceList = utils.makeSourceFluxList(tbdata, LCC_path)
    signal_trials = inject_pointsource(f, None, None, None,
                                       filename=filename.replace('%.2f'%settings['gamma'],'%.2f'%signal_gamma).replace('.npy','_signal_sourceList.npy'),
                                       gamma=signal_gamma, Nsim=settings['Nsim'],distortion=settings['distortion'],E_weights=settings['E_weights'],
                                       sourceList=sourceList)


    print('calculate p-values signal')
    calc_p_value(llh_bg_dist, signal_trials, name='%s_signal_brightness'%addinfo)

    exp_llh = plotLLH(filename, tbdata, timeBins, binNorms,distortion=settings['distortion'], E_weights=settings['E_weights'])
    print('Exp P-Val {}'.format(calc_p_value(llh_bg_dist, exp_llh, save=False)))
