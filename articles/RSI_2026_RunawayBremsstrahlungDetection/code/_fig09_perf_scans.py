import copy


import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpinteg
import scipy.constants as scpct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as asunits
import datastock as ds


from . import _load_spect_anis
from ._fig02_dist_type import _DDIST
from ._fig05_emiss import _RE, _DDMIX
from ._fig07_responsivities import _PFE_RESPONSIVITIES
from ._fig08_scores_1keV import _DANGLES, _LRESP
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


# #######################################
# #######################################
#           Main
# #######################################


def main(
    dmix=None,
    # cases
    Te_eV=None,
    # d2cross
    d2cross_phi=None,
    # dist
    ne_m3=None,
    jp_Am2=None,
    # RE
    re=None,
    dominant=None,
    jp_fraction_re=None,
    Efield_par_Vm=None,
    Ekin_min_eV=None,
    Ekin_max_eV=None,
    sigmap=None,
    pnormW=0,
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    # plot
    figsize=(8, 10),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):

    # -----------
    # inputs
    # -----------

    if dmix is None:
        dmix = _DDMIX[1]

    # Te
    if Te_eV is None:
        Te_eV = _TE

    # jp_fraction_re
    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    # Maxwell
    kwd_max = {'ne_m3': ne_m3, 'jp_Am2': jp_Am2}

    # RE
    re = ds._generic_check._check_var(
        re, 're',
        types=str,
        default=_RE,
        allowed=sorted(_DDIST['RE'].keys()),
    )

    lRE = [
        'dominant', 'jp_fraction_re', 'Efield_par_Vm',
        'Efield_par_Vm', 'Ekin_max_eV', 'Ekin_min_eV',
        'sigmap', 'pnormW'
    ]
    kwd_RE = {
        kk: _DDIST['RE'][re].get(kk) if vv is None else vv
        for kk, vv in locals().items()
        if kk in lRE
    }

    # --------------
    # load elements
    # --------------

    demiss = {}
    kwd = dict(kwd_max)
    kwd.update(**kwd_RE)
    demiss, ddist, dmix = _load_spect_anis.main(
        dmix=dmix,
        # d2cross
        d2cross_phi=d2cross_phi,
        # dist
        **kwd,
    )

    # extract
    # E_ph = demiss[0]['common']['E_photon']['data']
    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    # ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    # jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    # units = demiss['emiss']['maxwell']['ff']['units']

    indTe = np.argmin(np.abs(Teu - Te_eV))
    Te_eV = Teu[indTe]
    sli_emiss = (0, indTe, 0, slice(None), slice(None))

    # -------------------
    # load responsivities
    # -------------------
