import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _perfs
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

    # Te
    if Te_eV is None:
        Te_eV = _TE

    # jp_fraction_re
    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    # --------------
    # compute
    # --------------

    (
        demiss_integ, dsignal,
        total_headon, diff_RE, diff_max,
        dang, theta,
        lresp, ldist,
    ) = _perfs.main(
        **locals(),
    )

    # --------------
    # prepare axes
    # --------------

    dmargin_theta = {
        'left': 0.08, 'right': 0.98,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }
    dmargin_perf = {
        'left': 0.08, 'right': 0.98,
        'bottom': 0.04, 'top': 0.65,
        'wspace': 0.25, 'hspace': 0.08,
    }

    fig = plt.figure(figsize=figsize)

    gs_theta = gridspec.GridSpec(ncols=1, nrows=3, **dmargin_theta)
    gs_perf = gridspec.GridSpec(ncols=1, nrows=2, **dmargin_perf)
    dax = {}

    # --------------
    # axes - vs theta
    # --------------

    ax = fig.add_subplot(gs_theta[0, 0], aspect='auto')
    ax.set_xlabel(
        r'$\theta_{ph,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon_{ff}^{RE,\eta} / "
        r"\max\left(\epsilon_{ff}^{RE,\eta}\right)$",
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(a)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['theta'] = ax

    # --------------
    # axes - Absolute
    # --------------

    ax = fig.add_subplot(gs_perf[0, 0], aspect='auto')
    ax.set_ylabel(
        r"$\epsilon^{RE,\eta} / \max\left(\epsilon^{RE,\eta}\right)$",
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(b)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )
    ax.tick_params(labelbottom=False)

    dax['abs'] = ax

    # --------------
    # axes - rel
    # --------------

    ax0 = ax
    ax = fig.add_subplot(
        gs_perf[1, 0],
        sharex=ax0,
        aspect='auto',
    )
    ax.set_ylabel(
        r"$\Delta M / M$",
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(c)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['rel'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot vs theta
    # --------------

    kax = 'theta'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ------------
        # loop on resp

        dcolor = {}
        for kresp in demiss_integ.keys():

            data = demiss_integ[kresp]['RE']['ff']['data']
            l0, = ax.plot(
                theta*180/np.pi,
                data / data.max(),
                ls='-',
                color=_perfs._DANGLES[dang[kresp]]['color'],
                lw=1,
                label=kresp,
            )
            dcolor[kresp] = l0.get_color()

        # ------------
        # loop on angles

        for kang, vang in _perfs._DANGLES.items():
            for kk in ['head-on', 'back']:
                ax.axvspan(
                    vang[kk][0]*180/np.pi,
                    vang[kk][1]*180/np.pi,
                    facecolor=vang['color'],
                    alpha=vang['alpha'],
                )

        # ------------
        # decorate

        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.legend(loc='upper right')
        ax.grid(True)

    # --------------
    # plot abs
    # --------------

    kax = 'abs'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ------------
        # loop on resp

        weight_counts = {'head-on': {}, 'back': {}}

        # loop
        for kdir in weight_counts.keys():

            # detail
            for kdist in ldist:
                for kemiss in sorted(dsignal[lresp[0]][kdist].keys()):
                    data = np.array([
                        dsignal[kresp][kdist][kemiss][kdir]['data'].squeeze()
                        for kresp in lresp
                    ])
                    weight_counts[kdir][f"{kdist} {kemiss}"] = data

            # normalize
            for kk, vv in weight_counts[kdir].items():
                weight_counts[kdir][kk] = vv / total_headon

        # --------
        # plot

        width = 0.2
        dcolor = {
            'maxwell bb': 'b',
            'maxwell fb': 'g',
            'maxwell ff': 'm',
            'RE ff': 'r',
        }
        lorder = ['maxwell bb', 'maxwell fb', 'maxwell ff', 'RE ff']
        for idir, kdir in enumerate(weight_counts.keys()):
            bottom = 0
            for kk in lorder:
                ax.bar(
                    np.arange(len(lresp)) + 1 - 0.15 + 0.3*idir,
                    weight_counts[kdir][kk],
                    width,
                    label=kk,
                    bottom=bottom,
                    color=dcolor[kk],
                )
                bottom += weight_counts[kdir][kk]

        ax.set_ylim(0, 1)
        ax.grid(True)

    # --------------
    # plot rel
    # --------------

    kax = 'rel'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ----------
        # plot

        # RE
        ax.semilogy(
            np.arange(len(lresp)) + 1,
            diff_RE / total_headon,
            marker='o',
            ms=10,
            ls='None',
            c='k',
            label='RE',
        )

        # maxwell
        ax.semilogy(
            np.arange(len(lresp)) + 1,
            diff_max / total_headon,
            marker='x',
            ms=10,
            ls='None',
            c='k',
            label='maxwellian',
        )

        vmin_log10 = np.floor(np.log10(min(
            np.min(diff_RE / total_headon),
            np.min(diff_max[diff_max > 0] / total_headon[diff_max > 0]),
        )))

        # decorate
        ax.set_ylim(10**np.floor(vmin_log10), 1)
        ax.set_xticks(np.arange(len(lresp)) + 1)
        ax.set_xticklabels(lresp)
        ax.legend()
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

    return dax, demiss_integ, dsignal


# #######################################
# #######################################
#           Subroutine
# #######################################
