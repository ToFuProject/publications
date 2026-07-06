import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _load_spect_anis
from ._fig04_bremsstrahlung import _CASES
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
    dmix={
        0: 'O',
        1: {'O': 0.90, 'Fe': 0.01},
    },
    # cases
    cases=None,
    # d2cross
    d2cross_phi='EH1',
    # dist
    ne_m3=None,
    pnormW=None,
    Ekin_max_eV=None,
    # Te_eV=1e3 * np.linspace(0.1, 2.5, 25),
    Te_eV=None,
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    jp_fraction_re=None,
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

    if cases is None:
        cases = _CASES

    # --------------
    # load elements
    # --------------

    demiss = {}
    for ii in dmix.keys():
        demiss[ii], ddist = _load_spect_anis.main(
            dmix=dmix[ii],
            # d2cross
            d2cross_phi=d2cross_phi,
            # dist
            ne_m3=ne_m3,
            pnormW=pnormW,
            Ekin_max_eV=Ekin_max_eV,
            Te_eV=Te_eV,
            jp_fraction_re=jp_fraction_re,
        )

    # extract
    # E_ph = demiss[0]['common']['E_photon']['data']
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
            gs[ii, 2:],
            aspect='auto',
            sharex=ax0_map,
            sharey=ax0_map,
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
    # plot - cases
    # --------------

    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    jp_fracu = np.unique(ddist['plasma']['jp_fraction_re']['data'])
    Ekinu = np.unique(ddist['plasma']['Ekin_max_eV']['data'])

    # loop on cases
    for i0, (k0, v0) in enumerate(cases['case'].items()):

        # slice
        Te = Teu[np.argmin(np.abs(Teu - v0['Te']))]
        jpf = jp_fracu[np.argmin(np.abs(jp_fracu - v0['jp_frac']))]
        Ekin = Ekinu[np.argmin(np.abs(Ekinu - v0['Ekin_max_eV']))]
        ic = (
            (ddist['plasma']['jp_fraction_re']['data'] == jpf)
            & (ddist['plasma']['Te_eV']['data'] == Te)
            & (ddist['plasma']['Ekin_max_eV']['data'] == Ekin)
        )
        assert ic.sum() == 1
        ic = tuple([cc[0] for cc in ic.nonzero()])

        # --------
        # spectra

        # loop on dmix
        for ii in dmix.keys():

            kax = f'spect_{ii}'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                sli = ic + (slice(None), slice(None))
                for kdist in demiss[ii]['emiss'].keys():
                    emiss_E = demiss[ii]['emiss'][kdist]['ff']['data'][sli]

                    # plot
                    ax.fill_between(
                        demiss[ii]['E_ph']['data']*1e-3,
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
