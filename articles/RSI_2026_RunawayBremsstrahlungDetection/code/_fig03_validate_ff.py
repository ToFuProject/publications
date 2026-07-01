

import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as asunits
import datastock as ds


from . import _load_spect_anis
from ._savefig import main as savefig


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    d2cross_phi='EH1',
    ne_m3=1e19,
    Te_plot=None,
    # plot
    figsize=(6, 8),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Validate Bremsstrahlung vs SCRAM and FLYCHK

    """

    # --------------
    # demiss
    # --------------

    demiss, ddist = _load_spect_anis.main(
        dmix='H',
        # d2cross
        d2cross_phi=d2cross_phi,
        # dist
        ne_m3=ne_m3,
        pnormW=None,
        Ekin_max_eV=None,
        Te_eV=Te_eV,
        jp_fraction_re=0.,
    )

    units = demiss['emiss']['RE']['ff']['units']
    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    nEkin = demiss['emiss']['maxwell']['ff']['data'].shape[0]

    # diff
    emiss_min = np.minimum(interp, dplasma['emiss_tot']['ff']['data'])
    iok = emiss_min > 0.
    diff = np.full(interp.shape, np.nan)
    diff[iok] = 100 * (
        np.abs(interp - dplasma['emiss_tot']['ff']['data'])[iok]
        / emiss_min[iok]
    )

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.13, 'right': 0.99,
        'bottom': 0.10, 'top': 0.92,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - abs

    ax = fig.add_subplot(
        gs[0, 0],
        xscale='log',
        yscale='log',
        aspect='auto',
    )
    ax.set_xlabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon^{Max}_{ff}$" + f"  ({asunits.Unit(units)})",
        size=fontsize,
        fontweight='bold',
    )
    tit = (
        "Validation of Bremstrahlung implemented from EH cross-section\n"
        "Maxwellian distribution with  "
        + r"$j_{P}$" + " = 0 A/m2,  "
        + r"$n_e$" + f" = {ne_m3:1.0e}"
    )
    ax.set_title(
        tit,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['abs'] = {'handle': ax}

    # --------------
    # ax - diff

    ax0 = ax
    ax = fig.add_subplot(
        gs[1, 0],
        sharex=ax0,
        yscale='log',
        aspect='auto',
    )
    ax.set_xlabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "error  (\%)",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['diff'] = {'handle': ax}

    # standardize
    dax = ds._generic_check._check_dax(dax)

    # ---------------------
    # Absolute values
    # ---------------------

    kax = 'abs'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(dplasma['common']['Te']['data']):

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ii, slice(None))

            # loop on elements
            for kk, vv in dplasma['emiss'].items():

                # SCRAM / FLYCHK
                l0, = ax.loglog(
                    E_ph*1e-3,
                    vv['ff']['data'][sli],
                    ls='-',
                    lw=1,
                    label=lab,
                )
                dcolor[ii] = l0.get_color()

                # ff
                l0, = ax.loglog(
                    E_ph*1e-3,
                    interp[sli],
                    ls='--',
                    lw=1,
                    color=dcolor[ii],
                    # label=lab,
                )

        vmax = np.nanmax(np.maximum(vv['ff']['data'], interp))
        vmax_plot = 10**np.ceil(np.log10(vmax))

        ax.legend()
        ax.set_xlim(E_ph.min()*1e-3, E_ph.max()*1e-3)
        ax.set_ylim(vmax_plot/1e15, vmax_plot)

    # ---------------
    # plot diff
    # ---------------

    kax = 'diff'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(Te):

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ii, slice(None))

            # loop on elements
            for kk, vv in dplasma['emiss'].items():

                # SCRAM / FLYCHK
                l0, = ax.loglog(
                    E_ph*1e-3,
                    diff[sli],
                    ls='-',
                    lw=1,
                    color=dcolor[ii],
                    label=lab,
                )

        ax.set_ylim(1e-4, 1e4)

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return dax, demiss, ddist, dplasma
