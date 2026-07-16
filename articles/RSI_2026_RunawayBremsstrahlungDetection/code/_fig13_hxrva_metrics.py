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
    # ptcam
    angle0=None,
    angle1=None,
    # plot params
    dvmax=None,
    dvmin=None,
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

    if key_resp is None:
        key_resp = 'cvd_filter'

    # --------------
    # compute
    # --------------

    cases = {0: {'helicity': True, 'pitch': True, 'ne': True}},
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

    # ----------------------
    # loop on cases for axes
    # ----------------------

    dmetrics_plot = {
        0: {
            'key': 'xi',
            'tit': 'xi',
        },
        1: {
            'key': 'kappa',
            'tit': 'kappa',
        },
        2: {
            'key': 'meas_RE_diff',
            'tit': r'$\delta \epsilon_{ff}^{RE}$',
        },
        3: {
            'key': 'meas_RE_headon',
            'tit': r"$\epsilon_{ff}^{RE}$",
        },
    }

    # -----------
    # vmin, vmax

    if dvmax is None:
        dvmax = {}
    if dvmin is None:
        dvmin = {}

    lm = sorted(dmetrics_plot.keys())
    for km in lm:
        if dvmax.get(km) is None:
            dvmax[km] = np.nanmax([dmetrics[rei][km]['data'] for rei in re])
        if dvmin.get(km) is None:
            dvmin[km] = dvmax[km] / 1000

    # -------------
    # prepare fig
    # -------------

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.90,
            'bottom': 0.20, 'top': 0.93,
            'wspace': 0.18, 'hspace': 0.20,
        }
        dmargin_cbar = {
            'left': 0.93, 'right': 0.97,
            'bottom': 0.06, 'top': 0.20,
            'wspace': 0.18, 'hspace': 0.20,
        }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=4, nrows=len(re), **dmargin)
    gs_cbar = gridspec.GridSpec(ncols=4, nrows=1, **dmargin_cbar)
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
        for im, km in enumerate(lm):

            # --------------
            # axes - image

            ax = fig.add_subplot(
                gs[ire, im],
                aspect='equal',
                sharex=ax0,
                sharey=ax0,
            )
            if ire == 0:
                ax.set_title(
                    dmetrics_plot[km]['tit'],
                    fontsize=fontsize,
                    fontweight='bold',
                )
            if ire == len(re) - 1:
                ax.set_xlabel(
                    r'$\theta_0$ (deg)',
                    fontsize=fontsize,
                    fontweight='bold',
                )
            if im == 0:
                ax.set_ylabel(
                    f"{rei}\n" + r"$\theta_1$ (deg)",
                    fontsize=fontsize,
                    fontweight='bold',
                )
                ax0 = ax

            ax.text(
                0.01,
                0.99,
                f'({string.ascii_lowercase[im + 3*ire]})',
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=fontsize,
                fontweight='bold',
                transform=ax.transAxes,
            )

            dax[f'{rei}_{km}'] = ax

            # ---------
            # colorbar

            if ire == 0:
                ax = fig.add_subplot(
                    gs_cbar[0, im],
                    aspect='auto',
                )
                ax.set_title(
                    str(units),
                    fontsize=fontsize,
                    fontweight='bold',
                )

                dax[f'{km}_cbar'] = ax

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
        for im, km in enumerate(lm):

            kax = f"{rei}_{km}"
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                data = dmetrics[rei][km]['data']

                import pdb; pdb.set_trace()     # DB
                im = ax.imshow(
                    data.T,
                    extent=extent,
                    origin='lower',
                    cmap=plt.cm.viridis,
                    interpolation='nearest',
                    vmin=dvmin[km],
                    vmax=dvmax[km],
                )

                # colorbar
                if ire == 0:
                    ax = dax[f'{km}_cbar']['handle']
                    plt.colorbar(im, cax=ax)

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
