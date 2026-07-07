import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from . import _load_spect_anis
from ._fig02_dist_type import _DDIST
from ._fig05_emiss import _RE, _DDMIX
from ._fig07_responsivities import _PFE_RESPONSIVITIES
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_TE = 1e3
_JP_FRAC = 0.5


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
    figsize=(15, 8),
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
    if re is None:
        re = _RE
    assert re in _DDIST['RE'].keys()
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

    # load
    dresp = {
        k0: v0.tolist()
        for k0, v0 in np.load(_PFE_RESPONSIVITIES, allow_pickle=True).items()
    }

    # -----------------------------
    # Integrate over responsivity
    # -----------------------------

    demiss_integ = {kk: {'maxwell': {}, 'RE': {}} for kk in dresp.keys()}
    for kresp, vresp in dresp.items():

        # Eph_resp
        iok_resp = np.isfinite(vresp['responsivity']['data'])
        iok_resp[iok_resp] = vresp['responsivity']['data'][iok_resp] > 0.
        Eph_resp = vresp['E_eV']['data'][iok_resp]

        # loop on dist
        for kdist, vdist in demiss['emiss'].items():

            # loop on emiss type
            for kemiss, vemiss in vdist.items():

                # skip
                if kemiss == 'ff_iso':
                    continue

                # Eph_emiss
                data = demiss['emiss'][kdist][kemiss]['data'][sli_emiss]
                iok_emiss = (
                    (demiss['E_ph']['data'] >= Eph_resp.min())
                    & (demiss['E_ph']['data'] <= Eph_resp.max())
                    & np.all(np.isfinite(data), axis=-1)
                )
                Eph_emiss = demiss['E_ph']['data'][iok_emiss]

                # Eph
                Eph = np.unique(np.r_[Eph_resp, Eph_emiss])

                # interpolate responsivity
                resp = scpinterp.make_interp_spline(
                    Eph_resp,
                    vresp['responsivity']['data'][iok_resp],
                    k=1,
                    axis=0,
                    check_finite=True,
                )(Eph)

                # interpolate emissivity
                sli = (iok_emiss, slice(None))

                emiss = np.power(
                    10,
                    scpinterp.make_interp_spline(
                        np.log10(Eph_emiss),
                        np.log10(data[sli]),
                        k=1,
                        axis=0,
                        check_finite=True,
                    )(np.log10(Eph)),
                )

                # integrate
                demiss_integ[kresp][kdist][kemiss] = scpinteg.trapezoid(
                    resp[:, None] * emiss,
                    x=Eph,
                    axis=-2,
                )

    # -----------------------------
    # Integrate over angles
    # -----------------------------

    dsignal = {}
    for kresp, vresp in dresp.items():
        pass

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.94,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return demiss, dresp, demiss_integ, dsignal


# #######################################
# #######################################
#           Subroutine
# #######################################

