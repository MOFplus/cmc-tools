"""
    
                chargeinteractions

    BFJ 2023 : implementation of interaction energies of various charge distributions

    used in chargemodels.py in classes that do not make use of the inter module

"""

import numpy as np
from scipy.special import erf
from molsys.util.constants import *
from scipy.sparse import coo_array

# define some extra constants here
CONVERT_EV_KCPM = electronvolt / kcalmol
R4PIE0 = 332.06371083674475
PI = np.pi
SQPI = np.sqrt(PI)
SQ2 = np.sqrt(2.0)
# expansion coefficients for slater_taylor 
C01 = 0.0208333333333333
C02 = 0.1875
C03 = 0.6875
C11 = 0.0104166666666667
C12 = C24 = C25 = 0.0625
C13 = C14 = 0.15625
C21 = 0.003125
C22 = 0.005208333333333333
C23 = 0.015625


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


def calc_Jij_slater_wolf(sig_i, sig_j, r, cutoff):
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
    etmp[id_exact] -= calc_Jij_slater_exact(sig_i[id_exact], sig_j[id_exact], cutoff)
    t0, ra, ra2, ra3 = calc_Jij_slater_taylor_zero(sig_i[id_taylor], cutoff)
    etmp[id_taylor] -= t0
    etmp[id_o12_full] -= calc_Jij_slater_taylor_o12(sig_i[id_o12_full], delta[id_o12_full], ra[id_o12], ra2[id_o12], ra3[id_o12])
    return R4PIE0 * etmp


def calc_Jij_slater_dsf(sig_i, sig_j, r, cutoff):
    """
    makes use of new functions implemented for sparse variants
    """
    delta = sig_j - sig_i
    etmp = np.zeros_like(sig_i)
    id_taylor = np.absolute(delta) < 0.001
    id_exact = ~id_taylor
    id_o12 = delta[id_taylor] != 0.0
    id_o12_full = id_taylor.copy()
    id_o12_full[np.where(id_taylor == True)] = id_o12
    etmp[id_exact] += calc_Jij_slater_exact_efficient(sig_i[id_exact], sig_j[id_exact], r[id_exact])
    etmp[id_exact] -= calc_Jij_slater_exact_efficient(sig_i[id_exact], sig_j[id_exact], cutoff)
    etmp[id_exact] -= (r[id_exact] - cutoff) * force_Jij_slater_exact_efficient(sig_i[id_exact], sig_j[id_exact], cutoff)
    etmp[id_taylor] += calc_Jij_slater_taylor_efficient(sig_i[id_taylor], sig_j[id_taylor], r[id_taylor], delta[id_taylor], id_o12)
    etmp[id_taylor] -= calc_Jij_slater_taylor_efficient(sig_i[id_taylor], sig_j[id_taylor], cutoff, delta[id_taylor], id_o12)
    etmp[id_taylor] -= (r[id_taylor] - cutoff) * force_Jij_slater_taylor_efficient(sig_i[id_taylor], sig_j[id_taylor], cutoff, delta[id_taylor], id_o12)
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
    t0 = (1.0 - (1.0 + C01 * ra3 + C02 * ra2 + C03 * ra) * era) / r
    return t0, ra, ra2, ra3


def calc_Jij_slater_taylor_o12(sig_i, delta, ra, ra2, ra3):
    a2 = sig_i * sig_i
    a3 = sig_i * a2
    delta2 = delta * delta
    ra4 = ra2 * ra2
    era = np.exp(-ra)
    ra4 = ra2 * ra2
    t1 = -(C11 * ra3 + C12 * ra2 + C13 * ra + C14) * era / a2
    t2 = -(- C21 * ra4 - C22 * ra3 + C23 * ra2 + C24 * ra + C25) * era / a3
    return t1 * delta + t2 * delta2


def calc_Jij_slater_sparse(sig_i, sig_j, r):
    """
    sparse version
    """
    rows = []
    cols = []
    data = []
    i, j = r.nonzero()
    delta = sig_j - sig_i
    id_exact = (abs(delta)[i,j] >= 0.001)
    n_exact = np.count_nonzero(id_exact)
    if n_exact > 0:
        i_exact = i[id_exact]
        j_exact = j[id_exact]
        etmp_exact = calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                     sig_j[i_exact, j_exact],
                                                         r[i_exact, j_exact])
        etmp_exact *= R4PIE0
        rows.append(i_exact) 
        cols.append(j_exact) 
        data.append(etmp_exact)
    if n_exact < i.size:
        i_taylor = np.delete(i, id_exact)
        j_taylor = np.delete(j, id_exact)
        id_o12 = (delta[i_taylor, j_taylor] != 0.0)
        etmp_taylor = calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                       sig_j[i_taylor, j_taylor],
                                                           r[i_taylor, j_taylor],
                                                       delta[i_taylor, j_taylor],
                                                                          id_o12)
        etmp_taylor *= R4PIE0
        rows.append(i_taylor) 
        cols.append(j_taylor) 
        data.append(etmp_taylor)
    etmp = coo_array((np.concatenate(data + data), 
                     (np.concatenate(rows + cols),
                      np.concatenate(cols + rows))),
                      shape=r.shape)
    return etmp


def calc_Jij_slater_wolf_sparse(sig_i, sig_j, r, cutoff):
    """
    sparse version
    """
    rows = []
    cols = []
    data = []
    i, j = r.nonzero()
    delta = sig_j - sig_i
    id_exact = (abs(delta)[i,j] >= 0.001)
    n_exact = np.count_nonzero(id_exact)
    if n_exact > 0:
        i_exact = i[id_exact]
        j_exact = j[id_exact]
        etmp_exact = calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                     sig_j[i_exact, j_exact],
                                                         r[i_exact, j_exact])
        etmp_exact -= calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                      sig_j[i_exact, j_exact],
                                                                       cutoff)
        etmp_exact *= R4PIE0
        rows.append(i_exact) 
        cols.append(j_exact) 
        data.append(etmp_exact)
    if n_exact < i.size:
        i_taylor = np.delete(i, id_exact)
        j_taylor = np.delete(j, id_exact)
        id_o12 = (delta[i_taylor, j_taylor] != 0.0)
        etmp_taylor = calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                       sig_j[i_taylor, j_taylor],
                                                           r[i_taylor, j_taylor],
                                                       delta[i_taylor, j_taylor],
                                                                          id_o12)
        etmp_taylor -= calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                        sig_j[i_taylor, j_taylor],
                                                                           cutoff,
                                                        delta[i_taylor, j_taylor],
                                                                           id_o12)
        etmp_taylor *= R4PIE0
        rows.append(i_taylor) 
        cols.append(j_taylor) 
        data.append(etmp_taylor)
    etmp = coo_array((np.concatenate(data + data), 
                     (np.concatenate(rows + cols),
                      np.concatenate(cols + rows))),
                      shape=r.shape)
    return etmp


def calc_Jij_slater_dsf_sparse(sig_i, sig_j, r, cutoff):
    """
    sparse version
    """
    rows = []
    cols = []
    data = []
    i, j = r.nonzero()
    delta = sig_j - sig_i
    id_exact = (abs(delta)[i,j] >= 0.001)
    n_exact = np.count_nonzero(id_exact)
    if n_exact > 0:
        i_exact = i[id_exact]
        j_exact = j[id_exact]
        etmp_exact = calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                     sig_j[i_exact, j_exact],
                                                         r[i_exact, j_exact])
        etmp_exact -= calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                      sig_j[i_exact, j_exact],
                                                                       cutoff)
        etmp_exact -= (r[i_exact, j_exact] - cutoff) * force_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                                                        sig_j[i_exact, j_exact],
                                                                                                         cutoff)
        etmp_exact *= R4PIE0
        rows.append(i_exact) 
        cols.append(j_exact) 
        data.append(etmp_exact)
    if n_exact < i.size:
        i_taylor = np.delete(i, id_exact)
        j_taylor = np.delete(j, id_exact)
        id_o12 = (delta[i_taylor, j_taylor] != 0.0)
        etmp_taylor = calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                       sig_j[i_taylor, j_taylor],
                                                           r[i_taylor, j_taylor],
                                                       delta[i_taylor, j_taylor],
                                                                          id_o12)
        etmp_taylor -= calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                        sig_j[i_taylor, j_taylor],
                                                                           cutoff,
                                                        delta[i_taylor, j_taylor],
                                                                           id_o12)
        etmp_taylor -= (r[i_taylor, j_taylor] - cutoff) * force_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                                                            sig_j[i_taylor, j_taylor],
                                                                                                               cutoff,
                                                                                            delta[i_taylor, j_taylor],
                                                                                                               id_o12)
        etmp_taylor *= R4PIE0
        rows.append(i_taylor) 
        cols.append(j_taylor) 
        data.append(etmp_taylor)
    etmp = coo_array((np.concatenate(data + data), 
                     (np.concatenate(rows + cols),
                      np.concatenate(cols + rows))),
                      shape=r.shape)
    return etmp


def calc_Jij_slater_hybrid_sparse(sig_i, sig_j, r, cutoff):
    """
    sparse version
    """
    rows = []
    cols = []
    data = []
    i, j = r.nonzero()
    delta = sig_j - sig_i
    id_exact = (abs(delta)[i,j] >= 0.001)
    n_exact = np.count_nonzero(id_exact)
    if n_exact > 0:
        i_exact = i[id_exact]
        j_exact = j[id_exact]
        etmp_exact = calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                     sig_j[i_exact, j_exact],
                                                         r[i_exact, j_exact])
        etmp_exact -= calc_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                      sig_j[i_exact, j_exact],
                                                                       cutoff)
        etmp_exact -= (r[i_exact, j_exact] - cutoff) * force_Jij_slater_exact_efficient(sig_i[i_exact, j_exact],
                                                                                        sig_j[i_exact, j_exact],
                                                                                                         cutoff)
        etmp_exact *= R4PIE0
        rows.append(i_exact) 
        cols.append(j_exact) 
        data.append(etmp_exact)
    if n_exact < i.size:
        i_taylor = np.delete(i, id_exact)
        j_taylor = np.delete(j, id_exact)
        id_o12 = (delta[i_taylor, j_taylor] != 0.0)
        etmp_taylor = calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                       sig_j[i_taylor, j_taylor],
                                                           r[i_taylor, j_taylor],
                                                       delta[i_taylor, j_taylor],
                                                                          id_o12)
        etmp_taylor -= calc_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                        sig_j[i_taylor, j_taylor],
                                                                           cutoff,
                                                        delta[i_taylor, j_taylor],
                                                                           id_o12)
        etmp_taylor -= ((r[i_taylor, j_taylor] - cutoff) 
                       * (r[i_taylor, j_taylor]**2 * (3 * cutoff - 2 * r[i_taylor, j_taylor]) / cutoff**3) 
                       * force_Jij_slater_taylor_efficient(sig_i[i_taylor, j_taylor],
                                                           sig_j[i_taylor, j_taylor],
                                                                              cutoff,
                                                           delta[i_taylor, j_taylor],
                                                                              id_o12))
        etmp_taylor *= R4PIE0
        rows.append(i_taylor) 
        cols.append(j_taylor) 
        data.append(etmp_taylor)
    etmp = coo_array((np.concatenate(data + data), 
                     (np.concatenate(rows + cols),
                      np.concatenate(cols + rows))),
                      shape=r.shape)
    return etmp


def calc_Jij_slater_exact_efficient(sig_i, sig_j, r):
    return (1.0 - fij(sig_i, sig_j, r) -fij(sig_j, sig_i, r)) / r


def force_Jij_slater_exact_efficient(sig_i, sig_j, r):
    Jij = calc_Jij_slater_exact_efficient
    return -(Jij(sig_i, sig_j, r) + dfij(sig_i, sig_j, r) + dfij(sig_j, sig_i, r)) / r


def fij(sig_i, sig_j, r):
    fij_tmp = f1ij(sig_i, sig_j)
    return (f0ij(fij_tmp, sig_i, sig_j) + fij_tmp * r) * np.exp(-r / sig_i)


def f0ij(f1ij, sig_i, sig_j):
    return f1ij * 2.0 * sig_i * (sig_i**2 - 3.0 * sig_j**2) / (sig_i**2 - sig_j**2)


def f1ij(sig_i, sig_j):
    return (sig_i**3) / (2.0 * (sig_i**2 - sig_j**2)**2)


def dfij(sig_i, sig_j, r):
    fij_tmp = f1ij(sig_i, sig_j)
    return (fij_tmp * (sig_i - r) - f0ij(fij_tmp, sig_i, sig_j)) * np.exp(-r / sig_i) / sig_i


def calc_Jij_slater_taylor_efficient(sig_i, sig_j, r, delta, id_o12):
    etmp = np.zeros_like(sig_i)
    etmp += 0.5 * t0ij(sig_i, r / sig_i)
    etmp[~id_o12] *= 2.0
    if id_o12.size > 0:
        etmp[id_o12] -= 0.5 * t12ij(sig_i[id_o12], (r / sig_i)[id_o12], delta[id_o12])
        etmp[id_o12] += 0.5 * t0ij(sig_j[id_o12], (r / sig_j)[id_o12])
        etmp[id_o12] -= 0.5 * t12ij(sig_j[id_o12], (r / sig_j)[id_o12], -delta[id_o12])
    return etmp
        

def force_Jij_slater_taylor_efficient(sig_i, sig_j, r, delta, id_o12):
    etmp = np.zeros_like(sig_i)
    etmp += 0.5 * dt0ij(sig_i, r / sig_i)
    etmp[~id_o12] *= 2.0
    if id_o12.size > 0:
        etmp[id_o12] -= 0.5 * dt12ij(sig_i[id_o12], (r / sig_i)[id_o12], delta[id_o12])
        etmp[id_o12] += 0.5 * dt0ij(sig_j[id_o12], (r / sig_j)[id_o12])
        etmp[id_o12] -= 0.5 * dt12ij(sig_j[id_o12], (r / sig_j)[id_o12], -delta[id_o12])
    return etmp


def t0ij(sig_i, rs_i):
    return (1.0 - p0ij(rs_i) * np.exp(-rs_i)) / (rs_i * sig_i)


def t12ij(sig_i, rs_i, delta):
    return delta * np.exp(-rs_i) * (p1ij(rs_i) + delta * p2ij(rs_i) / sig_i) / sig_i**2


def p0ij(r):
    return ((C01 * r + C02) * r + C03) * r + 1.0


def p1ij(r):
    return ((C11 * r + C12) * r + C13) * r + C14


def p2ij(r):
    return (((- C21 * r - C22) * r + C23) * r + C24) * r + C25


def dt0ij(sig_i, rs_i):
    return (np.exp(-rs_i) * (p0ij(rs_i) - sdp0ij(rs_i)) / sig_i - t0ij(sig_i, rs_i)) / (rs_i * sig_i)


def dt12ij(sig_i, rs_i, delta):
    return delta * np.exp(-rs_i) * ((p1ij(rs_i) - sdp1ij(rs_i)) + delta * (p2ij(rs_i) - sdp2ij(rs_i)) / sig_i) / sig_i**3


def sdp0ij(r):
    return (3.0 * C01 * r + 2.0 * C02) * r + C03


def sdp1ij(r):
    return (3.0 * C11 * r + 2.0 * C12) * r + C13


def sdp2ij(r):
    return ((- 4.0 * C21 * r - 3.0 * C22) * r + 2.0 * C23) * r + C24


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


def calc_Jii_self_slater_wolf(sig_i, cutoff):
    return R4PIE0 * (0.3125 / sig_i - t0ij(sig_i, cutoff / sig_i))


def calc_Jii_self_slater_dsf(sig_i, cutoff):
    etemp = 0.3125 / sig_i
    etemp -= t0ij(sig_i, cutoff / sig_i)
    etemp += dt0ij(sig_i, cutoff / sig_i) * cutoff
    return R4PIE0 * etemp

def calc_Jii_self_point(sig_i):
    return 0.0
