import numpy as np
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os

SLtype = [("RAJ2000", np.float64),
          ("DEJ2000", np.float64),
          ("binCenterMJD", np.float64),
          ("eflux", np.float64),
          ("eflux_err", np.float64),
          ("ts", np.float64),
          ("Source_Name",'U17')]


def makeSourceFluxList(tbdata, LCC_path):
    print "make source flux list"
    outfile = LCC_path.replace('.fits','_SourceList.npy')
    if os.path.exists(outfile):
       fluxList = np.load(outfile)
    else:
       tbins = len(np.zeros_like(tbdata[0]['eflux']))
       fluxList = []
       binCenterTime = tbdata[0]['tmin_mjd']+(tbdata[0]['tmax_mjd']-tbdata[0]['tmin_mjd'])*0.5 
       for si in tbdata: # loop over sources
          raj2000 = si['RAJ2000']
          dej2000 = si['DEJ2000']
          sourcen = si['Source_Name']
          eflux = si['eflux']
          eflux_err = si['eflux_err']
          ts = si['ts']
          fluxList.extend([np.array((raj2000, dej2000, binCenterTime[ti], eflux[ti], eflux_err[ti], ts[ti], sourcen),dtype=SLtype) for ti in range(tbins)])
       np.save(outfile,fluxList)
    return fluxList

def delta_psi(theta1,phi1, theta2, phi2):
    sp = np.sin(theta1)*np.cos(phi1)*np.sin(theta2)*np.cos(phi2) \
         + np.sin(theta1)*np.sin(phi1)*np.sin(theta2)*np.sin(phi2) \
         +np.cos(theta1)*np.cos(theta2)
    return np.arccos(sp)


def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).
    """
    def cross_matrix(x):
        r"""Calculate cross product matrix
        A[ij] = x_i * y_j - y_i * x_j
        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
        )

    alpha = np.arccos(np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2)
                      + np.sin(dec1) * np.sin(dec2))
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1.-np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec


def setNewEdges(edges):
    newEdges = []
    for i in range(0, len(edges) - 1):
        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
        newEdges.append(newVal)
    return np.array(newEdges)

def norm_hist(h):
    h = np.array([i / np.sum(i) if np.sum(i) > 0 else i / 1. for i in h])
    return h


def create_splines(f, ftypes, zen_reco, az_reco, en_reco):
    Hs = dict()
    mask = np.isfinite(f[zen_reco])

    delta_mask = np.degrees(delta_psi(f['zenith'], f['azimuth'], f[zen_reco], f[az_reco]))<5
    # energy ratio 2D spline
    print('Create Energy Spline..check yourself whether it is ok')
    zenith_bins=list(np.linspace(-1.,0.,6, endpoint=False)) + list(np.linspace(0.,1.,10))
    tot_weight = np.sum([f[flux][mask & delta_mask] for flux in ftypes], axis=0)
    x = np.cos(f[zen_reco][mask & delta_mask])
    y = np.log10(f[en_reco][mask & delta_mask])
    H_tot, xedges, yedges = np.histogram2d(x, y,
                                       weights=tot_weight,
                                       bins=(20,np.linspace(3.5, 11, 30)), normed=True)

    H_tot = np.ma.masked_array(norm_hist(H_tot))
    H_tot.mask = (H_tot <= 0)

    H_astro, xedges, yedges = np.histogram2d(x, y,
                                       weights=f['astro'][mask & delta_mask],
                                       bins=(20, np.linspace(3.5, 11, 30)),
                                       normed=True)
    print yedges
    H_astro = np.ma.masked_array(norm_hist(H_astro))
    H_astro.mask = (H_astro <= 0)

    spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 H_astro/H_tot,
                                 kx=1, ky=1, s=0)
    np.save('E_spline.npy', spline)
    print(spline(-0.1, np.linspace(3.6,6.5,20)))

    # zenith dist 1D spline
    print('Create Zenith Spline...Check if ok..')
    vals, edges = np.histogram(np.cos(f[zen_reco][mask & delta_mask]),
                               weights=tot_weight,
                               bins=30, density=True)
    zen_spl = InterpolatedUnivariateSpline(setNewEdges(edges),
                                           np.log10(vals), k=3)
    print(10**zen_spl(setNewEdges(edges)))
    np.save('coszen_spl.npy', zen_spl)

    # zenith dist 1D spline
    print('Create Zenith Spline Signal with true zenith...Check if ok..')
    vals_sig, edges_sig = np.histogram(np.cos(f['zenith'][mask & delta_mask]),
                                       weights=f['astro'],
                                       bins=30, density=True)
    zen_spl_sig = InterpolatedUnivariateSpline(setNewEdges(edges_sig),
                                               np.log10(vals_sig), k=3)
    print(10**zen_spl_sig(setNewEdges(edges_sig)))
    np.save('coszen_signal_spl.npy', zen_spl_sig)
    

@np.vectorize
def zen_to_dec(zen):
    return zen - 0.5*np.pi


@np.vectorize
def dec_to_zen(dec):
    return dec + 0.5*np.pi
