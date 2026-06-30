import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds
import tofu as tf


from ._load_spect_anis import main as load_spect_anis
from ._savefig import main as savefig


tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    d2cross_phi='EH1',
    dmix={
        0: 'H',
        1: {'H': 0.95, 'O': 0.04, 'Fe': 0.01},
    },
    ne_m3=1e19,
    # plot
    figsize=(15, 8),
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

    nmix = len(dmix)

    # --------------
    # load elements
    # --------------

    demiss = {}
    for ii in dmix.keys():
        demiss[ii] = load_spect_anis(
            dmix=dmix[ii],
            ne_m3=ne_m3,
            d2cross_phi=d2cross_phi,
        )

    # extract
    E_ph = demiss[0]['common']['E_photon']['data']
    Te = demiss[0]['common']['Te']['data']
    units = demiss[0]['emiss_tot']['ff']['units']

    # --------------
    # integrated cross-section
    # --------------

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.06, 'right': 0.98,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.25, 'hspace': 0.30,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=3, nrows=nmix, **dmargin)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    ax0_spect = None
    ax0_map = None
    for ii in sorted(dmix.keys()):

        ax = fig.add_subplot(
            gs[ii, :2],
            sharex=ax0_spect,
            sharey=ax0_spect,
            aspect='auto',
        )
        ax.set_ylabel(
            r"$\epsilon$" + f' ({units})',
            fontsize=fontsize,
            fontweight='bold',
        )
        if ii == 0:
            ax0_spect = ax
        elif ii == nmix - 1:
            ax.set_xlabel(
                r"$E_{ph}$" + ' (keV)',
                fontsize=fontsize,
                fontweight='bold',
            )
        dax[f'spect_{ii}'] = ax

        # ----------------
        # ax - Elim
        # ----------------

        ax = fig.add_subplot(
            gs_map[ii, 2:],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        if ii == 0:
            ax0_map = ax
            ax.set_title(
                tit,
                fontsize=fontsize,
                fontweight='bold',
            )
        elif ii == nmix - 1:
            ax.set_xlabel('Te (keV)', fontsize=fontsize, fontweight='bold')
        ax.set_ylabel('jp_frac', fontsize=fontsize, fontweight='bold')

        dax[f'Elim_{ii}'] = ax

    # ----------------
    # check dax format

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - spectra
    # --------------

    for ii, vplasma in sorted(dplasma.keys()):

        kax = f"spect_{ii}"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # ----------------------
            # total emiss Maxwellian

            emiss_tot = np.sum(
                [vv['data'] for vv in plasma['emiss_tot'].values()],
                axis=0,
            )

            ax.plot(
                vplasma['common']['E_photon']['data'],
                emiss_tot,
                color=None,
                ls='-',
                lw=1,
            )

            # ---------
            # ff - RE

            # plot
            ax.fill_between(
                demiss['E_ph_eV']['data']*1e-3,
                np.nanmin(emiss_E, axis=-1),
                np.nanmax(emiss_E, axis=-1),
                hatch=v0['hatch'],
                facecolor='None',
                alpha=0.5,
                edgecolor=v0['color'],
                ls='--' if kdist == 'RE' else '-',
                label=f'{kdist}_{ic}',
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True)

            # Elim
            iE = np.argmin(np.abs(demiss['E_ph_eV']['data'] - Elim[ic]))
            ax.plot(
                np.r_[Elim[ic], Elim[ic]] * 1e-3,
                [emiss_E[iE, 0], 1e16],
                color=v0['color'],
                ls='--',
                lw=1,
                label=f"E_lim = {Elim[ic]*1e-3:3.1f} keV",
            )

            # text
            trans = transforms.blended_transform_factory(
                ax.transData,
                ax.transAxes,
            )
            ax.text(
                Elim[ic] * 1e-3,
                1,
                r"$E_{ph,lim}$" + f"\n = {Elim[ic] * 1e-3:2.1f} keV",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=fontsize,
                fontweight='bold',
                color=v0['color'],
                transform=trans,
            )

            ax.set_ylim(1e0, 1e16)
            ax.set_xlim(1e-2, 2e4)

    # --------------
    # plot - Elim
    # --------------

    for ii in range(nEkin):
        kax = f'Elim_{ii}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            sli = (ii, slice(None), slice(None))
            cs = ax.contour(
                ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                ddist['plasma']['jp_fraction_re']['data'][sli],
                Elim[sli] * 1e-3,
                cmap=plt.cm.viridis,
                levels=np.r_[1, 2, 5, 7.5, 10, 15],
                vmin=0.1,
                vmax=20,
            )

            # cases
            for i0, (k0, v0) in enumerate(cases['case'].items()):
                ax.plot(
                    v0['Te']*1e-3,
                    v0['jp_frac'],
                    marker='*',
                    markersize=8,
                    markerfacecolor=v0['color'],
                    color=v0['color'],
                )

            ax.clabel(cs, cs.levels, fontsize=12)

            ax.set_xlim(0, ddist['plasma']['Te_eV']['data'].max()*1e-3)
            ax.set_ylim(0, 1)

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return
