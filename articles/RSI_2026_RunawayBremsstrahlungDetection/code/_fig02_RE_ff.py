import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _load_spect_anis
from ._fig02_dist_type import _DDIST, _DDIST_PLOT
from ._savefig import main as savefig


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    dmix='H',
    # plotting temp
    Te_eV=1e3,
    # d2cross
    d2cross_phi=None,
    # Eph
    Eph=np.r_[0.1, 2, 20]*1e3,
    # plot
    figsize=(5, 7),
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
    # inputs
    # --------------

    # --------------
    # load elements
    # --------------

    demiss = {}
    for kdist, vdist in _DDIST['RE'].items():

        kwd = {
            kk: vv for kk, vv in _DDIST['maxwell'].items()
            if 'Te' not in kk
        }
        kwd.update(**vdist)
        # kwd.update(**_DDIST['coords'])

        demiss[kdist], ddist = _load_spect_anis.main(
            dmix=dmix,
            # d2cross
            d2cross_phi=d2cross_phi,
            # dist
            **kwd,
        )

    # --------------
    # extract
    # --------------

    # Te, ne, units
    kdomref = list(_DDIST['RE'].keys())[0]
    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    units = demiss[kdomref]['emiss']['maxwell']['ff']['units']

    nE = Eph.size

    # vmax, vmin
    vmax = np.nanmax(demiss[kdomref]['emiss']['maxwell']['ff']['data'])
    vmax_log10 = np.ceil(np.log10(vmax))
    vmax = 10**vmax_log10
    vmin = 10**(vmax_log10 - 21)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.06, 'right': 0.98,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.25, 'hspace': 0.30,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=nE, nrows=3, **dmargin)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    ax = fig.add_subplot(
        gs[0, :],
        aspect='auto',
    )
    ax.set_ylabel(
        r"$\epsilon_{ff}$" + f' ({units})',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_xlabel(
        r"$E_{ph}$" + ' (keV)',
        fontsize=fontsize,
        fontweight='bold',
    )
    dax['spectra'] = ax

    # ----------------
    # ax - theta_rel
    # ----------------

    ax0 = None
    for ie, ee in enumerate(Eph):

        ax = fig.add_subplot(
            gs[1, ie],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        ax.set_xlabel(
            r"$\theta_{ph,B}$" + " (deg)",
            fontsize=fontsize,
            fontweight='bold',
        )
        if ie == 0:
            ax0 = ax
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_ylabel(
                r"$\epsilon_{ff}$" + f' ({units})',
                fontsize=fontsize,
                fontweight='bold',
            )

        dax[f'theta_{ie}'] = ax

    # ----------------
    # ax - abs
    # ----------------

    ax0 = None
    for ie, ee in enumerate(Eph):

        ax = fig.add_subplot(
            gs[1, ie],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        ax.set_xlabel(
            r"$\theta_{ph,B}$" + " (deg)",
            fontsize=fontsize,
            fontweight='bold',
        )
        if ie == 0:
            ax0 = ax
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_ylabel(
                r"$\epsilon_{ff}$" + f' ({units})',
                fontsize=fontsize,
                fontweight='bold',
            )

        dax[f'theta_{ie}'] = ax

    # ----------------
    # check dax format

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - spectra
    # --------------

    kax = 'spectra'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # slice
        indTe = np.argmin(np.abs(Teu - Te_eV))
        Te_eV = ddist['plasma']['Te_eV']['data'][(0, indTe, 0)]
        sli = (0, indTe, 0, slice(None), slice(None))

        # ----------
        # Maxwell

        data = demiss[kdomref]['emiss']['maxwell']['ff']['data'][sli]
        ax.loglog(
            demiss[kdomref]['E_ph']['data']*1e-3,
            np.mean(data, axis=-1),
            c='k',
            ls='-',
            label='Maxwellian',
        )

        # ----------
        # RE

        for kdist in demiss.keys():
            emiss_E = demiss[kdist]['emiss']['RE']['ff']['data'][sli]

            ax.fill_between(
                demiss[kdomref]['E_ph']['data']*1e-3,
                np.nanmin(emiss_E, axis=-1),
                np.nanmax(emiss_E, axis=-1),
                hatch=_DDIST_PLOT[kdist]['hatch'],
                facecolor='None',
                alpha=0.5,
                edgecolor=_DDIST_PLOT[kdist]['color'],
                ls='-',
                label=kdist,
            )

        # --------
        # vlines

        for ie, ee in enumerate(Eph):
            ax.axvline(
                ee*1e-3,
                c='k',
                ls='--',
                lw=1,
                label=f"E_ph = {ee*1e-3:3.1f} keV",
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(1e-3, 1e5)
        ax.grid(True)

    # --------------
    # plot vs theta
    # --------------

    for ie, ee in enumerate(Eph):

        kax = f"theta_{ie}"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # indTe
            indTe = np.argmin(np.abs(Teu - Te_eV))
            Te_eV = ddist['plasma']['Te_eV']['data'][(0, indTe, 0)]

            # indE
            indE = np.argmin(np.abs(demiss[kdomref]['E_ph']['data'] - ee))
            ee = demiss[kdomref]['E_ph']['data'][indE]
            sli = (0, indTe, 0, indE, slice(None))

            # plot
            for kdist in demiss.keys():

                # data
                emiss_RE = demiss[kdist]['emiss']['RE']['ff']['data'][sli]

                # plot
                ax.plot(
                    demiss[kdomref]['theta_ph_vsB']['data']*180/np.pi,
                    emiss_RE / np.nanmax(emiss_RE),
                    c=_DDIST_PLOT[kdist]['color'],
                    ls='-',
                    lw=1,
                    label=kdist,
                )

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return dax, demiss
