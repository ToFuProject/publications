import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import datastock as ds


from ._load_spect_anis import _JP_FRAC
from ._fig02_dist_type import _DDIST_PLOT
from ._fig05_emiss import _DDMIX
from . import _perfs
from ._fig08_perfs_single import _DCASES
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_RE = ['dreicer', 'avalanche 100 keV', 'avalanche 10 MeV']


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
    figsize=(5, 7),
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

    if dmix is None:
        dmix = _DDMIX[1]

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

    # ind theta headon vs back
    ind_headon = theta <= np.pi/2.
    ind_back = theta >= np.pi/2.
    assert ind_headon.sum() == ind_back.sum()
    ind_headon = np.nonzero(ind_headon)[0][::-1]
    ind_back = np.nonzero(ind_back)[0]
    delta_theta = theta[ind_back] - np.pi/2.

    xticks = np.r_[0, 15, 30, 45, 60, 75, 90]
    xlab = [
        "0" if ix == 0
        else f"{90 - xx}\n- {90 + xx}"
        for ix, xx in enumerate(xticks)
    ]

    # delta_emiss
    delta_emiss = {kresp: {} for kresp in lresp}
    for ii, rei in enumerate(re):
        for kresp in lresp:
            data = dout[rei]['demiss_integ'][kresp]['RE']['ff']['data']
            delta = data[:, :, ind_headon] - data[:, :, ind_back]
            delta_emiss[kresp][rei] = {
                'data': delta,
                'units': dout[rei]['demiss_integ'][kresp]['RE']['ff']['units']
            }

    # -------------
    # prepare full
    # -------------

    shape_plasma = delta_emiss[kresp][rei]['data'].shape[:-1]
    nan = np.full(shape_plasma + (1,), np.nan)
    delta_theta_full = np.full(shape_plasma + (delta_theta.size + 1), np.nan)
    delta_theta_full[:, :, :-1] = delta_theta[None, None, :].ravel()

    # --------------
    # prepare Te, F cases
    # --------------

    nresp = len(lresp)
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
        'bottom': 0.04, 'top': 0.95,
        'wspace': 0.10, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)
    fig.suptitle(tit, fontsize=fontsize, fontweight='bold')

    gs = gridspec.GridSpec(ncols=1, nrows=nresp, **dmargin)
    dax = {}

    # --------------
    # axes
    # --------------

    ax0 = None
    for iresp, kresp in enumerate(lresp):

        ax = fig.add_subplot(
            gs[iresp, 0],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        if iresp == 0:
            ax0 = ax

        # xlabel
        if iresp == len(lresp) - 1:
            ax.set_xlabel(
                r'$\delta \theta_{ph,B}$ (deg)',
                fontsize=fontsize,
                fontweight='bold',
            )

        # ylabel
        ax.set_ylabel(
            f"{kresp}\n" + r"$\delta M_i$",
            fontsize=fontsize,
            fontweight='bold',
        )

        # char
        ax.text(
            0.01,
            0.99,
            f"({string.ascii_lowercase[iresp]})",
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        dax[kresp] = ax

    # check
    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot vs delta_theta
    # --------------

    for iresp, kresp in enumerate(lresp):

        kax = kresp
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # ------------------------
            # loop on xi, kappa, total

            for ie, rei in enumerate(re):

                # data (Te, jp_frac, dtheta)
                data = delta_emiss[kresp][rei]['data']
                data_norm = data / np.nanmax(data, axis=-1)[:, :, None]
                data_norm = np.concatenate((data_norm, nan), axis=-1).ravel()

                # concatenate
                ax.plot(
                    delta_theta_full * 180/np.pi,
                    data_norm,
                    color=_DDIST_PLOT[rei]['color'],
                    linestyle='-',
                    label=rei,
                )

                # ------------
                # decorate

                if iresp == 0:
                    ax.set_xlim(0, 90)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xlab)
                    ax.grid(True)

                elif iresp == len(lresp) - 1:
                    ax.legend(
                        handles=None,
                        loc='upper right',
                        bbox_to_anchor=(1.5, 1.),
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

    return dax, delta_theta, delta_emiss
