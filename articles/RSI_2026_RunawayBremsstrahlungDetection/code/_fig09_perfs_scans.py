import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from ._load_spect_anis import _JP_FRAC
from . import _perfs
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
    # plot
    figsize=(15, 4),
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

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    # --------------
    # compute
    # --------------

    (
        demiss_integ, dsignal, ddist,
        total_headon, diff_RE, diff_max,
        dang, theta,
        lresp, ldist,
    ) = _perfs.main(
        **locals(),
    )

    # --------------
    # extract
    # --------------

    nresp = len(lresp)

    # --------------
    # prepare
    # --------------

    # dynamic range
    dynamic = diff_RE / total_headon
    bits = None

    # RE vs Maxwell
    RE_vs_max = 100 * diff_RE / (diff_max + diff_RE)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.05, 'right': 0.98,
        'bottom': 0.12, 'top': 0.95,
        'wspace': 0.10, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=nresp, nrows=1, **dmargin)
    dax = {}

    # --------------
    # axes
    # --------------

    ax0 = None
    for iresp, kresp in enumerate(lresp):

        ax = fig.add_subplot(
            gs[0, iresp],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )

        ax.set_xlabel(
            r'$T_e$ (keV)',
            fontsize=fontsize,
            fontweight='bold',
        )

        ax.set_title(
            kresp,
            fontsize=fontsize,
            fontweight='bold',
        )

        ax.text(
            0.01,
            0.99,
            f"({string.ascii_lowercase[1 + iresp]})",
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        if iresp == 0:
            ax.set_ylabel(
                r"$F_{RE}$",
                fontsize=fontsize,
                fontweight='bold',
            )
        else:
            ax.tick_params(labelleft=False)

        dax[kresp] = ax

    # check
    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot vs theta
    # --------------

    for iresp, kresp in enumerate(lresp):

        kax = kresp
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # -------------
            # dynamic range

            # levels
            vmin_log10 = np.log10(np.nanmin(dynamic[iresp, ...]))
            vmax_log10 = np.log10(np.nanmax(dynamic[iresp, ...]))
            if vmax_log10 - vmin_log10 > 2:
                levels = np.logspace(
                    np.floor(vmin_log10),
                    np.ceil(vmax_log10),
                    6,
                )
            else:
                levels = 6

            # plot dynamic range
            sli = (0, slice(None), slice(None))
            cs = ax.contour(
                ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                ddist['plasma']['jp_fraction_re']['data'][sli],
                dynamic[iresp],
                colors='k',
                linestyles='-',
                levels=levels,
            )
            ax.clabel(cs, cs.levels, fmt=lambda vv: f"{vv:2.1e}", fontsize=12)

            # plot range
            cs = ax.contour(
                ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                ddist['plasma']['jp_fraction_re']['data'][sli],
                RE_vs_max[iresp],
                colors='b',
                linestyles='-',
                levels=np.r_[0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99],
            )
            ax.clabel(cs, cs.levels, fontsize=12)

            # ------------
            # decorate

            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 1)
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

    return dax, demiss_integ, dsignal, dynamic


# #######################################
# #######################################
#           Subroutine
# #######################################
