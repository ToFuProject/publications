import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import datastock as ds


from ._load_spect_anis import _JP_FRAC
from . import _perfs
from ._fig08_perfs_single import _DCASES
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_RE = ['dreicer', 'avalanche 100 keV', 'avalanche 10 MeV']


_DLEVELS = {
    'bolo': {
        'xi': np.r_[1e-6, 1e-5, 5e-5, 1e-4, 1e-3],
        'kappa': np.r_[10, 20, 30, 50, 70, 80],
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'cvd_bare': {
        'xi': np.r_[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'kappa': np.r_[10, 20, 30, 50, 70, 80, 90],
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'cvd_filter': {
        'xi': np.r_[1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        'kappa': np.r_[10, 20, 30, 50, 70, 80, 90],
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'spectro': {
        'xi': 10,
        'kappa': 10,
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'mesxr_11_keV': {
        'xi': np.r_[0.01, 0.1, 0.2, 0.3],
        'kappa': 10,
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'mehxr_60_keV': {
        'xi': 10,
        'kappa': 10,
        'total': np.r_[0.1, 0.3, 0.5, 0.7, 0.9],
    },
}


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
    figsize=(10, 14),
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

    if re is None:
        re = _RE

    # --------------
    # compute
    # --------------

    dout = {}
    for ii, rei in enumerate(re):
        (
            demiss_integ, dsignal, ddist, dmix,
            total_headon, diff_RE, diff_max,
            dang, theta,
            lresp, ldist,
        ) = _perfs.main(
            dmix=dmix,
            ne_m3=ne_m3,
            jp_Am2=jp_Am2,
            d2cross_phi=d2cross_phi,
            re=rei,
            jp_fraction_re=jp_fraction_re,
        )

        dout[rei] = {
            'ii': ii,
            'demiss_integ': demiss_integ,
            'dsignal': dsignal,
            'ddist': ddist,
            'total_headon': total_headon,
            'diff_RE': diff_RE,
            'diff_max': diff_max,
        }

    # --------------
    # extract
    # --------------

    nresp = len(lresp)
    nre = len(re)

    # --------------
    # prepare
    # --------------

    for rei, vout in dout.items():
        # xi range
        dout[rei]['xi'] = vout['diff_RE'] / vout['total_headon']

        # RE vs Maxwell
        difftot = vout['diff_max'] + vout['diff_RE']
        dout[rei]['kappa'] = vout['diff_RE'] / difftot

    # --------------
    # prepare Te, F cases
    # --------------

    lkcase = sorted(_DCASES.keys())
    Te_case = np.array([_DCASES[kk]['Te_eV'] for kk in lkcase])
    Fre_case = np.array([_DCASES[kk]['jp_fraction_re'] for kk in lkcase])

    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]

    lk = list(dmix.keys())
    lc = [np.unique(dmix[kk])[0] for kk in lk]
    inds = np.argsort(lc)[::-1]
    lstr = [f"{lk[ss]} {lc[ss]*100:3.1f} \\%" for ss in inds]
    tit_mix = ",  ".join(lstr)

    tit = (
        r"$n_e$" + f" = {ne:1.0e}" + r"$/m^3$,  "
        + r"$j_P$" + f" = {jp*1e-6:1.0f}" + r"$MA/m^2$" + "\n"
        + tit_mix
    )

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.08, 'right': 0.90,
        'bottom': 0.04, 'top': 0.98,
        'wspace': 0.10, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)
    fig.suptitle(tit, fontsize=fontsize, fontweight='bold')

    gs = gridspec.GridSpec(ncols=nre, nrows=nresp, **dmargin)
    dax = {}

    # --------------
    # axes
    # --------------

    ax0 = None
    for ie, rei in enumerate(re):
        for iresp, kresp in enumerate(lresp):

            ax = fig.add_subplot(
                gs[iresp, ie],
                aspect='auto',
                sharex=ax0,
                sharey=ax0,
            )
            if ie == iresp == 0:
                ax0 = ax

            # title
            if iresp == 0:
                ax.set_title(
                    rei,
                    fontsize=fontsize,
                    fontweight='bold',
                )

            # xlabel
            if iresp == len(lresp) - 1:
                ax.set_xlabel(
                    r'$T_e$ (keV)',
                    fontsize=fontsize,
                    fontweight='bold',
                )
            else:
                ax.tick_params(labelbottom=False)

            # ylabel
            if ie == 0:
                ax.set_ylabel(
                    f"{kresp}\n" + r"$F_{RE}$",
                    fontsize=fontsize,
                    fontweight='bold',
                )

            # char
            ax.text(
                0.01,
                0.99,
                f"({string.ascii_lowercase[iresp + ie * nre]})",
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=fontsize,
                fontweight='bold',
                transform=ax.transAxes,
            )

            dax[f"{rei}_{kresp}"] = ax

    # check
    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot vs theta
    # --------------

    dcolor = {
        'xi': {
            'color': 'k',
            'label': r'$\xi_{RE} = \frac{\Delta_{ff}^{RE}}{M_i}$',
        },
        'kappa': {
            'color': 'b',
            'label': r"$\kappa = \frac{\Delta_{ff}^{RE}}{\Delta_{ff}^{Max}}$",
        },
        'tot': {
            'color': 'r',
            'label': r"$M_i$",
        },
    }
    for ie, rei in enumerate(re):
        for iresp, kresp in enumerate(lresp):

            kax = f"{rei}_{kresp}"
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                # -------------
                # xi range

                # set levels xi
                if _DLEVELS.get(kresp, {}).get('xi') is not None:
                    levels = _DLEVELS[kresp]['xi']

                # plot xi range
                sli = (0, slice(None), slice(None))
                cs = ax.contour(
                    ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                    ddist['plasma']['jp_fraction_re']['data'][sli],
                    dout[rei]['xi'][iresp],
                    colors=dcolor['xi'],
                    linestyles='-',
                    levels=levels,
                    label='xi',
                )
                ax.clabel(
                    cs,
                    cs.levels,
                    fmt=lambda vv: f"{vv:2.1e}",
                    fontsize=12,
                )

                # -------------
                # kappa range

                # set levels kappa
                if _DLEVELS.get(kresp, {}).get('kappa') is not None:
                    levels = _DLEVELS[kresp]['kappa']

                # plot kappa
                cs = ax.contour(
                    ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                    ddist['plasma']['jp_fraction_re']['data'][sli],
                    dout[rei]['kappa'][iresp],
                    colors=dcolor['kappa'],
                    linestyles='-',
                    levels=levels,
                    label='kappa',
                )
                ax.clabel(cs, cs.levels, fontsize=12)

                # -------------
                # total range

                # set levels total
                if _DLEVELS.get(kresp, {}).get('total') is not None:
                    levels = _DLEVELS[kresp]['total']

                # plot total head-on
                total = dout[rei]['total_headon'][iresp]
                cs = ax.contour(
                    ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                    ddist['plasma']['jp_fraction_re']['data'][sli],
                    total / np.nanmax(total),
                    colors=dcolor['total'],
                    linestyles='-',
                    levels=levels,
                    label='total_headon_norm',
                )
                ax.clabel(cs, cs.levels, fontsize=12)

                # add case
                ax.plot(
                    Te_case*1e-3,
                    Fre_case,
                    ls='None',
                    marker='x',
                    ms=6,
                    color='k',
                )

                # ------------
                # decorate

                if ie == iresp == 0:
                    ax.set_xlim(0, 2.5)
                    ax.set_ylim(0, 1)
                    ax.grid(True)

                if ie == len(re) - 1:
                    lh = [
                        mlines.Line2D(
                            [], [],
                            ls='-',
                            c=cc,
                            label=dcolor[kk]['label'],
                        )
                        for kk, cc in dcolor.items()
                    ]
                    ax.legend(
                        handles=lh,
                        loc='upper right',
                        bbox_to_anchor=(1.2, 1.),
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

    return dax, dout


# #######################################
# #######################################
#           Subroutine
# #######################################
