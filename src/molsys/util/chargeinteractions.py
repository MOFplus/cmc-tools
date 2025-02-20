"""
    
                chargeinteractions

    BFJ 2023 : implementation of interaction energies of various charge distributions

    used in chargemodels.py in classes that do not make use of the inter module

"""

import numpy as np
from scipy.special import erf
from molsys.util.constants import *

# define some extra constants here
CONVERT_EV_KCPM = electronvolt / kcalmol
R4PIE0 = 332.06371083674475
PI = np.pi
SQPI = np.sqrt(PI)
SQ2 = np.sqrt(2.0)


def calc_Jij_gauss(sig_i, sig_j, r):
    sigij = np.sqrt(sig_i * sig_i + sig_j * sig_j)
    erfrsig = erf(r / sigij)
    etmp = erfrsig / r
    return R4PIE0 * etmp


def calc_Jij_gauss_wolf(sig_i, sig_j, r, cutoff):
    sigij = np.sqrt(sig_i * sig_i + sig_j * sig_j)
    alpha = 1.0 / sigij
    erfalphar = erf(alpha * r)
    erfalphacut = erf(alpha * cutoff)
    etmp = erfalphar / r - erfalphacut / cutoff
    return R4PIE0 * etmp


def calc_Jij_gauss_dsf(sig_i, sig_j, r, cutoff):
    sigij = np.sqrt(sig_i * sig_i + sig_j * sig_j)
    alpha = 1.0 / sigij
    alpha2 = alpha * alpha
    cut2 = cutoff * cutoff
    erfalphar = erf(alpha * r)
    erfalphacut = erf(alpha * cutoff)
    etmp1 = erfalphar / r
    etmp2 = erfalphacut / cutoff
    etmp3 = erfalphacut / cut2 - 2.0 * alpha / SQPI * np.exp(-alpha2 * cut2) / cutoff
    etmp4 = etmp1 - etmp2 + etmp3 * (r - cutoff)
    return R4PIE0 * etmp4


def calc_Jij_slater_legacy(sig_i, sig_j, r):
    """
    old version using conditional statements incompatible with numpy
    """
    delta = sig_j - sig_i
    if np.absolute(delta) < 0.001:
        etmp, ra, ra2, ra3 = calc_Jij_slater_taylor_zero(sig_i, r)
        if sig_i != sig_j:
            etmp += calc_Jij_slater_taylor_o12(sig_i, delta, ra, ra2, ra3)
    else:
        etmp = calc_Jij_slater_exact(sig_i, sig_j, r)
    return R4PIE0 * etmp


def calc_Jij_slater(sig_i, sig_j, r):
    """
    new version exclusively usable with r of type np.ndarray
    """
    delta = sig_j - sig_i
    etmp = np.zeros_like(sig_i)
    id_taylor = np.absolute(delta) < 0.001
    id_exact = ~id_taylor
    etmp[id_exact] += calc_Jij_slater_exact(sig_i[id_exact], sig_j[id_exact], r[id_exact])
    t0, ra, ra2, ra3 = calc_Jij_slater_taylor_zero(sig_i[id_taylor], r[id_taylor])
    etmp[id_taylor] += t0
    id_o12 = delta[id_taylor] != 0.0
    id_o12_full = id_taylor.copy()
    id_o12_full[np.where(id_taylor == True)] = id_o12
    etmp[id_o12_full] += calc_Jij_slater_taylor_o12(sig_i[id_o12_full], delta[id_o12_full], ra[id_o12], ra2[id_o12], ra3[id_o12])
    return R4PIE0 * etmp


def calc_Jij_slater_exact(sig_i, sig_j, r):
    a2 = sig_i * sig_i
    b2 = sig_j * sig_j
    da2b2 = a2 - b2
    d2a2b2 = da2b2 * da2b2
    f1ab = (a2 * sig_i) / (2.0 * d2a2b2)
    f1ba = (b2 * sig_j) / (2.0 * d2a2b2)
    f0ab = f1ab * 2.0 * sig_i * (a2 - 3.0 * b2) / da2b2
    f0ba = f1ba * 2.0 * sig_j * (b2 - 3.0 * a2) / -da2b2
    era = np.exp(-r / sig_i)
    erb = np.exp(-r / sig_j)
    etmp = (1.0 - (f0ab + f1ab * r) * era - (f0ba + f1ba * r) * erb) / r
    return etmp


def calc_Jij_slater_taylor_zero(sig_i, r):
    ra = r / sig_i
    ra2 = ra * ra
    ra3 = ra2 * ra
    era = np.exp(-ra)
    t0 = (1.0 - (1.0 + 0.0208333333333333 * ra3 + 0.1875 * ra2 + 0.6875 * ra) * era) / r
    return t0, ra, ra2, ra3


def calc_Jij_slater_taylor_o12(sig_i, delta, ra, ra2, ra3):
    a2 = sig_i * sig_i
    a3 = sig_i * a2
    delta2 = delta * delta
    ra4 = ra2 * ra2
    era = np.exp(-ra)
    ra4 = ra2 * ra2
    t1 = -(0.0104166666666667 * ra3 + 0.0625 * ra2 + 0.15625 * ra + 0.15625) * era / a2
    t2 = -(- 0.003125 * ra4 - 0.005208333333333333 * ra3 + 0.015625 * ra2 + 0.0625 * ra + 0.0625) * era / a3
    return t1 * delta + t2 * delta2


def calc_Jij_point(r):
    etmp = 1.0 / r
    return R4PIE0 * etmp


def calc_Jii_self_gauss(sig_i):
    etmp = SQ2 / (SQPI * sig_i)
    return R4PIE0 * etmp


def calc_Jii_self_gauss_wolf(sig_i, cutoff):
    sq2sig_i = SQ2 * sig_i
    erfcutsig = erf(cutoff / sq2sig_i)
    etmp = SQ2 / (SQPI * sig_i) - erfcutsig / cutoff
    return R4PIE0 * etmp


def calc_Jii_self_gauss_dsf(sig_i, cutoff):
    sq2sig_i = SQ2 * sig_i
    alpha = 1.0 / sq2sig_i
    alpha2 = alpha * alpha
    erfalphacut = erf(alpha * cutoff)
    preexp = SQ2 / (SQPI * sig_i)
    expterm = np.exp(-alpha2 * cutoff**2.0)
    etmp = preexp * (1.0 + expterm) - 2.0 * erfalphacut / cutoff
    return R4PIE0 * etmp


def calc_Jii_self_slater(sig_i):
    return R4PIE0 * 0.3125 / sig_i


def calc_Jii_self_point(sig_i):
    return 0.0
