import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import llh

nugen_path = '/data/user/tglauch/EHE/processed/combined.npy'

f = np.load(nugen_path)

mask = np.isnan(np.log10(f['energy'])) | np.isnan(np.log10(f['muex'])) | np.isnan(np.log10(f['NPE']))

f = f[~mask]

indSig = 2.1


def plotBGSigHist(varArray, xlabel, binning, outname):
   plt.figure()
   a = plt.hist(varArray,
                weights=f['atmo'], normed=True,
                bins=binning, alpha=0.5, label='atmo')
   a = plt.hist(varArray,        
                weights=f['ow']/f['energy']**indSig, normed=True,   
                bins=binning, alpha=0.5, label='E$^{-%.1f}$'%indSig)
   plt.xlabel(xlabel)
   plt.legend()
   plt.savefig(outname,bbox_inches='tight')



plotBGSigHist(np.log10(f['energy']), 'log10(trueEnergy [GeV])', np.linspace(3,7.5,30),'plots/trueEnergy.png')
plotBGSigHist(np.log10(f['muex']), 'log10(trueEnergy [GeV])', np.linspace(3,7.5,30),'plots/muex.png')
plotBGSigHist(np.cos(f['zenith']), 'cos(trueZenith)', np.linspace(-1,1,30),'plots/trueZenith.png')
plotBGSigHist(np.cos(f['mpe_zen']), 'cos(mpeZenith)', np.linspace(-1,1,30),'plots/mpeZenith.png')
plotBGSigHist(np.log10(f['NPE']), 'log10(NPE)', np.linspace(3.,5.,30),'plots/NPE.png')
plotBGSigHist(np.rad2deg(f['cr']), 'cramer rao [deg]', np.linspace(0,3,30),'plots/cramerRao.png')


plt.figure()
a = plt.hist2d(np.cos(f['mpe_zen']), 
               np.log10(f['NPE']), 
               weights=f['atmo'], bins=20,norm=LogNorm())
plt.colorbar()
plt.savefig("plots/zenNPE.png",bbox_inches='tight')


