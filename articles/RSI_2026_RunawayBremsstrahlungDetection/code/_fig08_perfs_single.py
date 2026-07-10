import string


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


_DCASES = {
    0: {
        'Te_eV': 1e3,
        'jp_fraction_re': 0.1,
        're': 'avalanche 100 keV',
    },
    1: {
        'Te_eV': 0.1e3,
        'jp_fraction_re': 0.5,
        're': 'avalanche 10 MeV',
    },
    2: {
        'Te_eV': 2e3,
        'jp_fraction_re': 0.9,
        're': 'dreicer',
    },
}


_LCOMP = ['maxwell bb', 'maxwell fb', 'maxwell ff', 'RE ff']
_DCOLOR = {
    'maxwell bb': {
        'label': r'$\epsilon_{bb}^{Max}$',
        'color': 'b',
    },
    'maxwell fb': {
        'label': r'$\epsilon_{fb}^{Max}$',
        'color': 'g',
    },
    'maxwell ff': {
        'label': r'$\epsilon_{ff}^{Max}$',
        'color': 'm',
    },
    'RE ff': {
        'label': r'$\epsilon_{ff}^{RE}$',
        'color': 'r',
    },
}


# #######################################
# #######################################
#           Main
# #######################################


def main(
    dmix=None,
    # cases
    dcases=None,
    # d2cross
    d2cross_phi=None,
    # dist
    ne_m3=None,
    jp_Am2=None,
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    # plot
    figsize=(14, 10),
    # figsize=(8, 10),
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

    # dcases
    if dcases is None:
        dcases = _DCASES

    # --------------
    # compute
    # --------------

    dout = {}
    for kcase, vcase in dcases.items():
        (
            demiss_integ, dsignal, ddist,
            total_headon, diff_RE, diff_max,
            dang, theta,
            lresp, ldist,
        ) = _perfs.main(
            d2cross_pphi=d2cross_phi,
            Te_eV=vcase['Te_eV'],
            jp_fraction_re=vcase['jp_fraction_re'],
            re=vcase['re'],
        )

        dout[kcase] = {
            'Te_eV': vcase['Te_eV'],
            're': vcase['re'],
            'ddist': ddist,
            'demiss_integ': demiss_integ,
            'dsignal': dsignal,
            'total_headon': total_headon,
            'diff_RE': diff_RE,
            'diff_max': diff_max,
        }

    # --------------
    # prepare plot params
    # --------------

    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]

    width = 0.2

    # --------------
    # prepare axes
    # --------------

    dmargin_theta = {
        'left': 0.05, 'right': 0.99,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.18, 'hspace': 0.20,
    }
    dmargin_perf = {
        'left': 0.05, 'right': 0.99,
        'bottom': 0.08, 'top': 0.60,
        'wspace': 0.18, 'hspace': 0.08,
    }

    fig = plt.figure(figsize=figsize)

    gs_theta = gridspec.GridSpec(ncols=len(dout), nrows=3, **dmargin_theta)
    gs_perf = gridspec.GridSpec(ncols=len(dout), nrows=2, **dmargin_perf)
    dax = {}

    # ----------------------
    # loop on cases for axes
    # ----------------------

    ax0_theta = None
    ax0_abs = None
    ax0_xi = None
    for kcase, vcase in dout.items():

        Teu = np.unique(vcase['ddist']['plasma']['Te_eV']['data'])
        indTe = np.argmin(np.abs(Teu - vcase['Te_eV']))
        Te_eV = vcase['ddist']['plasma']['Te_eV']['data'][(0, indTe, 0)]
        jp_frac = np.unique(vcase['ddist']['plasma']['jp_fraction_re']['data'])[0]

        # tit
        tit = (
            r"$n_e$" + f" = {ne:1.0e}" + r"$/m^3$,  "
            + r"$j_P$" + f" = {jp*1e-6:1.0f}" + r"$MA/m^2$" + "\n"
            + r"$T_e$" + f" = {Te_eV*1e-3:2.1f} keV,  "
            + r"$F_{RE}$" + f" = {jp_frac:2.1f}\n"
            + vcase['re']
        )

        # --------------
        # axes - vs theta

        ax = fig.add_subplot(
            gs_theta[0, kcase],
            aspect='auto',
            sharex=ax0_theta,
            sharey=ax0_theta,
        )
        ax.set_title(
            tit,
            fontsize=fontsize,
            fontweight='bold',
        )
        ax.set_xlabel(
            r'$\theta_{ph,B}$ (deg)',
            fontsize=fontsize,
            fontweight='bold',
        )
        if kcase == 0:
            ax.set_ylabel(
                r"$\epsilon_{ff}^{RE,\eta} / "
                r"\max\left(\epsilon_{ff}^{RE,\eta}\right)$",
                fontsize=fontsize,
                fontweight='bold',
            )
            ax0_theta = ax

        ax.text(
            0.01,
            0.99,
            f'({string.ascii_lowercase[kcase]})',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        dax[f'theta_{kcase}'] = ax

        # --------------
        # axes - Absolute
        # --------------

        ax = fig.add_subplot(
            gs_perf[0, kcase],
            aspect='auto',
            sharex=ax0_abs,
            sharey=ax0_abs,
        )
        if kcase == 0:
            ax.set_ylabel(
                r"$\epsilon^{\eta} / \max\left(\epsilon^{\eta}\right)$",
                fontsize=fontsize,
                fontweight='bold',
            )
            ax.set_xlim(0, len(lresp) + 2)
            ax0_abs = ax

        ax.text(
            0.01,
            0.99,
            f'({string.ascii_lowercase[kcase + len(dout)]})',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )
        ax.tick_params(labelbottom=False)

        dax[f'abs_{kcase}'] = ax

        # --------------
        # axes - rel
        # --------------

        ax = fig.add_subplot(
            gs_perf[1, kcase],
            sharex=ax0_abs,
            sharey=ax0_xi,
            aspect='auto',
        )
        if kcase == 0:
            ax.set_ylabel(
                r"$\xi = \Delta_{i,ff}^{RE} / M_{i}^{head-on}$",
                fontsize=fontsize,
                fontweight='bold',
            )
            ax0_xi = ax

        ax.text(
            0.01,
            0.99,
            f'({string.ascii_lowercase[kcase + 2*len(dout)]})',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        dax[f'xi_{kcase}'] = ax

    dax = ds._generic_check._check_dax(dax)

    # ticklabels size
    for kax, vax in dax.items():
        dax[kax]['handle'].tick_params(
            axis='both',
            which='major',
            labelsize=fontsize - 1,
        )

    # --------------
    # plot vs theta
    # --------------

    for kcase, vcase in dout.items():

        kax = f'theta_{kcase}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # ------------
            # loop on resp

            for kresp in vcase['demiss_integ'].keys():

                data = vcase['demiss_integ'][kresp]['RE']['ff']['data']
                l0, = ax.plot(
                    theta*180/np.pi,
                    data / data.max(),
                    ls='-',
                    color=_perfs._DANGLES[dang[kresp]]['color'],
                    lw=1,
                    label=kresp,
                )

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

            if kcase == 0:
                ax.set_xlim(0, 180)
                ax.set_ylim(0, 1)
                ax.set_xticks([0, 45, 90, 135, 180])
                ax.grid(True)
            elif '10 MeV' in vcase['re']:
                ax.legend(loc='upper right', fontsize=fontsize - 2)

        # --------------
        # plot abs
        # --------------

        kax = f'abs_{kcase}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # ------------
            # loop on resp

            weight_counts = {'head-on': {}, 'back': {}}

            # loop
            for kdir in weight_counts.keys():

                # detail
                for kdist in ldist:
                    lk = sorted(vcase['dsignal'][lresp[0]][kdist].keys())
                    for kemiss in lk:
                        data = np.array([
                            vcase['dsignal'][kresp][kdist][kemiss][kdir]['data'].squeeze()
                            for kresp in lresp
                        ])
                        weight_counts[kdir][f"{kdist} {kemiss}"] = data

                # normalize
                for kk, vv in weight_counts[kdir].items():
                    weight_counts[kdir][kk] = vv / vcase['total_headon']

            # --------
            # plot

            for idir, kdir in enumerate(weight_counts.keys()):
                bottom = 0
                for kk in _LCOMP:
                    ax.bar(
                        np.arange(len(lresp)) + 1 - 0.15 + 0.3*idir,
                        weight_counts[kdir][kk],
                        width,
                        label=_DCOLOR[kk]['label'] if idir == 0 else None,
                        bottom=bottom,
                        color=_DCOLOR[kk]['color'],
                    )
                    bottom += weight_counts[kdir][kk]

            # decorate
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend(loc='upper right', fontsize=fontsize - 2)

        # --------------
        # plot rel
        # --------------

        kax = f'xi_{kcase}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # ----------
            # plot

            # RE
            ax.semilogy(
                np.arange(len(lresp)) + 1,
                vcase['diff_RE'] / vcase['total_headon'],
                marker='o',
                ms=10,
                ls='None',
                c='k',
                label='RE',
            )

            # maxwell
            ax.semilogy(
                np.arange(len(lresp)) + 1,
                vcase['diff_max'] / vcase['total_headon'],
                marker='x',
                ms=10,
                ls='None',
                c='k',
                label='maxwellian',
            )

            iok = vcase['diff_max'] > 0
            vmin_log10 = np.floor(np.log10(min(
                np.min(vcase['diff_RE'] / vcase['total_headon']),
                np.min(vcase['diff_max'][iok] / vcase['total_headon'][iok]),
            )))

            # decorate
            if kcase == 0:
                ax.set_ylim(10**np.floor(vmin_log10), 1)
                ax.set_xticks(np.arange(len(lresp)) + 1)
            ax.set_xticklabels(
                lresp,
                rotation=30,
                ha="right",
            )
            ax.legend(loc='lower right')
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
