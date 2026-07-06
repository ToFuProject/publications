import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import astropy.units as asunits
import datastock as ds


from . import _load_spect_anis
from ._savefig import main as savefig


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


_TE_PLOT = None


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    d2cross_phi=None,
    ne_m3=None,
    Te_plot=None,
    # integration method
    integration=None,
    # error
    error=None,
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
    # inputs
    # --------------

    if Te_plot is None:
        Te_plot = _TE_PLOT

    # error
    error = ds._generic_check._check_var(
        error, 'error',
        types=str,
        default='iso',
        allowed=['min', 'max', 'iso', 'anis']
    )

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
        jp_Am2=0.,
        jp_fraction_re=0.,
        # integration method
        integration=integration,
    )

    if Te_plot is None:
        Te_plot = np.unique(demiss['Te']['data'])

    units = demiss['emiss']['RE']['ff']['units']
    ne_m3 = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp_Am2 = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    E_ph_eV = demiss['E_ph']['data']

    # sanity checks
    emiss_iso = demiss['emiss']['maxwell']['ff_iso']['data']
    emiss_anis = demiss['emiss']['maxwell']['ff']['data']

    # sanity check - Ekin
    if not np.allclose(emiss_anis[0, ...], emiss_anis[1:, ...]):
        msg = "Maxwell ff seems to depend on Ekin_max_eV..."
        raise Exception(msg)
    emiss_anis = emiss_anis[0, ...]

    # sanity check - theta
    mean_theta = np.mean(emiss_anis, axis=-1)
    diff_theta = np.abs(emiss_anis - mean_theta[..., None])
    diff_rel = np.zeros(diff_theta.shape)
    iok = mean_theta > 0.

    diff_rel[iok] = diff_theta[(iok, slice(None))] / mean_theta[(iok, None)]
    if np.max(diff_rel) > 0.01:
        msg = (
            "Maxwell ff seems to depend on theta_ph_vsB...\n"
            f"\t- max deviation: {np.max(diff_rel)*100:3.2f} %"
        )
        raise Exception(msg)

    # squeeze
    emiss_iso = emiss_iso.squeeze()
    emiss_anis = mean_theta.squeeze()
    assert emiss_iso.shape == emiss_anis.shape

    # diff
    diff = np.full(emiss_anis.shape, np.nan)
    if error == 'min':
        emiss_ref = np.minimum(emiss_iso, emiss_anis)
    elif error == 'iso':
        emiss_ref = emiss_iso
    elif error == 'anis':
        emiss_ref = emiss_anis
    else:
        emiss_ref = np.maximum(emiss_iso, emiss_anis)

    iok = emiss_ref > 0.
    diff[iok] = 100 * np.abs(emiss_anis - emiss_iso)[iok] / emiss_ref[iok]

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
    ax.set_ylabel(
        r"$\epsilon^{Max}_{ff}$" + f"  ({asunits.Unit(units)})",
        size=fontsize,
        fontweight='bold',
    )
    tit = (
        "Validation of " + r"$\epsilon_{ff}^{Max}$"
        + " implemented from " + r"$\sigma_{EH}$" + "\n"
        "Maxwellian H (Z=1) distribution with  "
        + r"$j_{P}$" + f" = {jp_Am2:3.0f} A/m2,  "
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

    Teu = np.unique(demiss['Te']['data'])

    kax = 'abs'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(Te_plot):

            # ind
            ind = np.argmin(np.abs(Teu - te))
            te = Teu[ind]

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ind, slice(None))

            # SCRAM / FLYCHK
            l0, = ax.loglog(
                E_ph_eV * 1e-3,
                emiss_iso[sli],
                ls='-',
                lw=1,
                label=lab,
            )
            dcolor[ii] = l0.get_color()

            # ff
            l0, = ax.loglog(
                E_ph_eV * 1e-3,
                emiss_anis[sli],
                ls='--',
                lw=1,
                color=dcolor[ii],
            )

        vmax = np.nanmax(np.maximum(emiss_iso, emiss_anis))
        vmax_plot = 10**np.ceil(np.log10(vmax))

        leg = ax.legend(loc="center left")
        ax.add_artist(leg)
        ax.set_xlim(E_ph_eV.min()*1e-3, E_ph_eV.max()*1e-3)
        ax.set_ylim(vmax_plot/1e15, vmax_plot)

        # add linestyle legend
        lh = [
            Line2D([], [], c='k', ls='--', label="from " + r"$\sigma_{EH}$"),
            Line2D([], [], c='k', ls='-', label="CHIANTI"),
        ]
        ax.legend(handles=lh, loc='upper right')
        ax.set_xlim(right=1e2)
        ax.grid(True)

    # ---------------
    # plot diff
    # ---------------

    kax = 'diff'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(Te_plot):

            # ind
            ind = np.argmin(np.abs(Teu - te))
            te = Teu[ind]

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ind, slice(None))

            # SCRAM / FLYCHK
            l0, = ax.loglog(
                E_ph_eV * 1e-3,
                diff[sli],
                ls='-',
                lw=1,
                color=dcolor[ii],
                label=lab,
            )

        ax.set_ylim(1e-4, 1e4)
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

    return dax, demiss, ddist
