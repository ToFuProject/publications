import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _singlept_sensor
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_RE = ['avalanche 10 MeV', 'dreicer']
_JP_FRAC = 0.9
_TE = 0.5e3
_T = 5    # 5


_CASES = {
    0: {
        'helicity': False,
        'pitch': True,
        'ne': False,
    },
    1: {
        'helicity': True,
        'pitch': True,
        'ne': False,
    },
    2: {
        'helicity': True,
        'pitch': True,
        'ne': True,
    },
}


_FONTSIZE = 14


# #######################################
# #######################################
#           Main
# #######################################


def main(
    # coll
    coll=None,
    key_cam=None,
    config=None,
    res=None,
    # responsivity
    key_resp=None,
    # equilibrium
    t=None,
    # dmix
    dmix=None,
    # d2cross
    d2cross_phi=None,
    # dist
    ne_m3=None,
    jp_Am2=None,
    Te_eV=None,
    jp_fraction_re=None,
    # RE
    re=None,
    # assumptions
    cases=None,
    # ptcam
    angle0=None,
    angle1=None,
    # plot params
    dvmax=None,
    # plot
    dmargin=None,
    figsize=None,
    fontsize=None,
    # saving
    pfe_save=None,
    path_save=None,
    # unused
    **kwdargs,
):

    # -----------
    # inputs
    # -----------

    if t is None:
        t = _T

    if Te_eV is None:
        Te_eV = _TE

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    if re is None:
        re = _RE
    if isinstance(re, str):
        re = [re]

    if cases is None:
        cases = _CASES

    if key_resp is None:
        key_resp = 'cvd_filter'

    # --------------
    # compute
    # --------------

    (
        coll, config,
        dangles, dsig_los, dmetrics,
        krays_max, krays_min,
        angle0, angle1,
        ne, jp, jp_frac, Te_eV,
    ) = _singlept_sensor.main(
        # coll
        coll=coll,
        key_cam=key_cam,
        config=config,
        res=res,
        # responsivity
        key_resp=key_resp,
        # equilibrium
        t=t,
        # dmix
        dmix=dmix,
        # d2cross
        d2cross_phi=d2cross_phi,
        # dist
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        Te_eV=Te_eV,
        jp_fraction_re=jp_fraction_re,
        # RE
        re=re,
        # assumptions
        cases=cases,
        # ptcam
        angle0=angle0,
        angle1=angle1,
    )

    # -------------
    # inputs
    # -------------

    if fontsize is None:
        fontsize = _FONTSIZE

    if dvmax is None:
        dvmax = {}

    for rei in re:
        if dvmax.get(rei) is None:
            dvmax[rei] = np.nanmax([
                dsig_los[kcase][krays_max][rei]['RE']['ff']['data']
                for kcase in cases.keys()
            ])

    # -------------
    # prepare
    # -------------

    dang0 = angle0[1] - angle0[0]
    dang1 = angle1[1] - angle1[0]
    extent = (
        (angle0[0] - 0.5*dang0) * 180/np.pi,
        (angle0[-1] + 0.5*dang0) * 180/np.pi,
        (angle1[0] - 0.5*dang1) * 180/np.pi,
        (angle1[-1] + 0.5*dang1) * 180/np.pi,
    )

    kcase = sorted(cases.keys())[0]
    units = dsig_los[kcase][krays_max][re[0]]['RE']['ff']['units']

    # -------------
    # prepare fig
    # -------------

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.90,
            'bottom': 0.06, 'top': 0.93,
            'wspace': 0.18, 'hspace': 0.20,
        }
        dmargin_cbar = {
            'left': 0.93, 'right': 0.97,
            'bottom': 0.06, 'top': 0.93,
            'wspace': 0.18, 'hspace': 0.20,
        }

    fig = plt.figure(figsize=figsize)

    nc = np.max([kk for kk in cases.keys()]) + 2
    gs = gridspec.GridSpec(ncols=nc, nrows=len(re), **dmargin)
    gs_cbar = gridspec.GridSpec(ncols=1, nrows=len(re), **dmargin_cbar)
    dax = {}

    # ----------------------
    # fig title
    # ----------------------

    # tit
    tit = (
        r"$n_e$" + f" = {ne:1.0e}" + r"$/m^3$,  "
        + r"$j_P$" + f" = {jp*1e-6:1.0f}" + r"$MA/m^2$" + "\n"
        + r"$T_e$" + f" = {Te_eV*1e-3:2.1f} keV,  "
        + r"$F_{RE}$" + f" = {jp_frac:2.1f}\n"
        + key_resp
    )
    fig.suptitle(
        tit,
        fontsize=fontsize,
        fontweight='bold',
    )

    # ----------------------
    # loop on cases for axes
    # ----------------------

    ax0 = None
    for ire, rei in enumerate(re):
        for icase, (kcase, vcase) in enumerate(cases.items()):

            # tit
            hel = 'helicity' if vcase['helicity'] else 'no helicity'
            pitch = 'pitch angle distrib.' if vcase['pitch'] else 'no pitch'
            nep = 'ne profile' if vcase['ne'] else 'flat profile'
            tit = (
                f"{hel}\n"
                f"{pitch}\n"
                f"{nep}\n"
            )

            # --------------
            # axes - image

            ax = fig.add_subplot(
                gs[ire, kcase],
                aspect='equal',
                sharex=ax0,
                sharey=ax0,
            )
            if ire == 0:
                ax.set_title(
                    tit,
                    fontsize=fontsize,
                    fontweight='bold',
                )
            ax.set_xlabel(
                r'$\theta_0$ (deg)',
                fontsize=fontsize,
                fontweight='bold',
            )
            if icase == 0:
                ax.set_ylabel(
                    f"{rei}\n" + r"$\theta_1$ (deg)",
                    fontsize=fontsize,
                    fontweight='bold',
                )
                ax0 = ax

            ax.text(
                0.01,
                0.99,
                f'({string.ascii_lowercase[icase + nc*ire]})',
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=fontsize,
                fontweight='bold',
                transform=ax.transAxes,
            )

            dax[f'{rei}_{kcase}'] = ax

        # ---------
        # diff_RE

        ax = fig.add_subplot(
            gs[ire, -1],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
        )
        if ire == 0:
            ax.set_title(
                tit,
                fontsize=fontsize,
                fontweight='bold',
            )
        ax.set_xlabel(
            r'$\theta_0$ (deg)',
            fontsize=fontsize,
            fontweight='bold',
        )

        ax.text(
            0.01,
            0.99,
            f'({string.ascii_lowercase[nc - 1 + nc*ire]})',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        dax[f'{rei}_diff'] = ax

        # ---------
        # colorbar

        ax = fig.add_subplot(
            gs_cbar[ire, 0],
            aspect='auto',
        )
        ax.set_title(
            str(units),
            fontsize=fontsize,
            fontweight='bold',
        )

        dax[f'{rei}_cbar'] = ax

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

    for ire, rei in enumerate(re):
        for icase, (kcase, vcase) in enumerate(cases.items()):

            kax = f"{rei}_{kcase}"
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                data = dsig_los[kcase][krays_max][rei]['RE']['ff']['data']

                im = ax.imshow(
                    data.T,
                    extent=extent,
                    origin='lower',
                    cmap=plt.cm.viridis,
                    interpolation='nearest',
                    vmin=0,
                    vmax=dvmax[rei],
                )

                if icase == len(cases) - 1:
                    ax = dax[f'{rei}_cbar']['handle']
                    plt.colorbar(im, cax=ax)

        # ---------
        # diff

        kax = f"{rei}_diff"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            kcase = [
                k0 for k0, v0 in cases.items()
                if all([v1 for v1 in v0.values()])
            ][0]
            data = dsig_los[kcase][krays_max][rei]['RE']['ff']['data']
            data_back = dsig_los[kcase][krays_min][rei]['RE']['ff']['data']
            diff = data - data_back[::-1, :]

            im = ax.imshow(
                diff.T,
                extent=extent,
                origin='lower',
                cmap=plt.cm.viridis,
                interpolation='nearest',
                vmin=0,
                vmax=dvmax[rei],
            )

    # ----------
    # save
    # ----------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return dax
