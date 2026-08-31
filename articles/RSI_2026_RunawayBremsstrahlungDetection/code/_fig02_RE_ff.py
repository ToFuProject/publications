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
    Eph=np.r_[0.05, 5, 50]*1e3,
    # plot
    figsize=(5, 7),
    fontsize=12,
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

        demiss[kdist], ddist, dmix, data_source = _load_spect_anis.main(
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

    indTe = np.argmin(np.abs(Teu - Te_eV))
    Te_eV = ddist['plasma']['Te_eV']['data'][(0, indTe, 0)]
    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    jp_frac = np.unique(ddist['plasma']['jp_fraction_re']['data'])[0]

    nE = Eph.size

    # vmax, vmin
    vmax = np.nanmax(demiss[kdomref]['emiss']['maxwell']['ff']['data'])
    vmax_log10 = np.ceil(np.log10(vmax))
    vmax = 10**vmax_log10
    vmin = 10**(vmax_log10 - 21)

    # title
    tit = (
        r"$n_e$" + f" = {ne:1.0e}" + r"$/m^3$,  "
        + r"$j_P$" + f" = {jp*1e-6:1.0f}" + r"$MA/m^2$" + "\n"
        + r"$T_e$" + f" = {Te_eV*1e-3:1.0f} keV,  "
        + r"$F_{RE}$" + f" = {jp_frac:2.1f}"
    )

    # --------------
    # prepare axes
    # --------------

    dmargin_spect = {
        'left': 0.13, 'right': 0.97,
        'bottom': 0.06, 'top': 0.94,
        'wspace': 0.25, 'hspace': 0.20,
    }
    dmargin_theta = {
        'left': 0.13, 'right': 0.97,
        'bottom': 0.07, 'top': 0.55,
        'wspace': 0.15, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs_spect = gridspec.GridSpec(ncols=1, nrows=3, **dmargin_spect)
    gs_theta = gridspec.GridSpec(ncols=nE, nrows=2, **dmargin_theta)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    ax = fig.add_subplot(
        gs_spect[0, :],
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
    ax.set_title(
        tit,
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.95,
        0.95,
        '(a)',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes,
    )

    dax['spectra'] = ax

    # ----------------
    # ax - theta_rel
    # ----------------

    ax0_rel = None
    for ie, ee in enumerate(Eph):

        if ee < 1e3:
            estr = f"{ee:3.0f} eV"
        elif ee < 1e6:
            estr = f"{ee*1e-3:3.0f} keV"
        else:
            estr = f"{ee*1e-6:3.0f} MeV"

        ax = fig.add_subplot(
            gs_theta[0, ie],
            aspect='auto',
            sharex=ax0_rel,
            sharey=ax0_rel,
        )
        ax.set_title(
            r"$E_{ph}$" + f" = {estr}",
            fontsize=fontsize,
            fontweight='bold',
        )
        if ie == 0:
            ax0_rel = ax
            ax.set_xlim(0, 180)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_ylabel(
                r"$\epsilon_{ff} / max(\epsilon_{ff})$",
                fontsize=fontsize,
                fontweight='bold',
            )
        else:
            ax.tick_params(labelleft=False)

        ax.text(
            0.95,
            0.95,
            ['(b)', '(c)', '(d)'][ie],
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
        )
        ax.tick_params(labelbottom=False)
        dax[f'theta_rel_{ie}'] = ax

    # ----------------
    # ax - theta_abs
    # ----------------

    ax0_abs = None
    for ie, ee in enumerate(Eph):

        ax = fig.add_subplot(
            gs_theta[1, ie],
            aspect='auto',
            sharex=ax0_rel,
            sharey=ax0_abs,
        )
        ax.set_xlabel(
            r"$\theta_{ph,B}$" + " (deg)",
            fontsize=fontsize,
            fontweight='bold',
        )
        if ie == 0:
            ax0_abs = ax
            ax.set_ylabel(
                r"$\epsilon_{ff}$" + f' ({units})',
                fontsize=fontsize,
                fontweight='bold',
            )
            ax.set_ylim(1e6, 1e13)
        else:
            ax.tick_params(labelleft=False)

        ax.text(
            0.95,
            0.95,
            ['(e)', '(f)', '(g)'][ie],
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
        )
        dax[f'theta_abs_{ie}'] = ax

    # ----------------
    # check dax format

    dax = ds._generic_check._check_dax(dax)

    # ticklabels size
    for kax, vax in dax.items():
        dax[kax]['handle'].tick_params(
            axis='both',
            which='major',
            labelsize=fontsize - 1,
        )

    # --------------
    # plot - spectra
    # --------------

    kax = 'spectra'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # slice
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
                # label=f"E_ph = {ee*1e-3:3.1f} keV",
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(1e-3, 1e5)
        ax.set_ylim(1e-3, 1e18)
        ax.set_yticks(np.logspace(-3, 18, 8))
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=fontsize - 2)

    # --------------
    # plot vs theta
    # --------------

    for ie, ee in enumerate(Eph):

        # indTe
        indTe = np.argmin(np.abs(Teu - Te_eV))
        Te_eV = ddist['plasma']['Te_eV']['data'][(0, indTe, 0)]

        # indE
        indE = np.argmin(np.abs(demiss[kdomref]['E_ph']['data'] - ee))
        ee = demiss[kdomref]['E_ph']['data'][indE]
        sli = (0, indTe, 0, indE, slice(None))

        # -----------
        # theta_rel

        kax = f"theta_rel_{ie}"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

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
        ax.grid(True)

        # -----------
        # theta_abs

        kax = f"theta_abs_{ie}"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # plot
            for kdist in demiss.keys():

                # data
                emiss_RE = demiss[kdist]['emiss']['RE']['ff']['data'][sli]

                # plot
                ax.semilogy(
                    demiss[kdomref]['theta_ph_vsB']['data']*180/np.pi,
                    emiss_RE,
                    c=_DDIST_PLOT[kdist]['color'],
                    ls='-',
                    lw=1,
                    label=kdist,
                )

            if ie == 0:
                ax.set_yticks(np.logspace(6, 13, 6))
            ax.grid(True)

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
