import os

import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds
import tofu as tf


from ._savefig import main as savefig


tfphysdist = tf.physics_tools.electrons.distribution


# #####################################################
# #####################################################
#       DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)


_DDIST = {
    # maxwell
    'Te_eV': np.r_[0.1e3, 2e3, 0.1e3, 2e3],
    'ne_m3': 1e19,
    'jp_Am2': 1e6,
    # RE
    'jp_fraction_re': np.r_[0.1, 0.1, 0.9, 0.9],
    'dominant': 'bump',
    'pnormW': np.r_[0.1, 4, 0.1, 4],
    'Ekin_max_eV': np.r_[100e3, 10e6, 100e3, 10e6],
    'Ekin_min_eV': 100,
    'step': 0.1,
    'theta_width': 20*np.pi/180,
    # coords
    'E_eV': np.logspace(0, 8, 80),
    'theta': np.linspace(0, 180, 181) * np.pi / 180,
}


# #####################################################
# #####################################################
#       main
# #####################################################


def main(
    # coords
    E_eV=None,
    theta=None,
    # Maxwell
    ne_m3=None,
    jp_Am2=None,
    # RE
    dominant=None,
    jp_fraction_re=None,
    Ekin_max_eV=None,
    Ekin_min_eV=None,
    step=None,
    pnormW=None,
    # plot
    figsize=(5, 7),
    fontsize=12,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Plot a Maxwellian + RE distribution

    Includes:
        - a 2d (E, pitch) contour plot
        - a 1d (E,) plot

    """

    # ------------
    # inputs
    # ------------

    din = locals()
    din = {
        kk: vv if din.get(kk) is None else din[kk]
        for kk, vv in _DDIST.items()
    }

    # ------------
    # compute
    # ------------

    # ddist = {'dist': dict, 'plasma': dist, 'coords': dist}
    ddist = tfphysdist.get_distribution(**din)

    # units
    units2d = asunits.Unit(ddist['dist']['RE']['dist']['units'])
    units1d = units2d * asunits.Unit(ddist['coords']['x1']['units'])

    # ------------
    # Derive 1d data
    # ------------

    dataRE = scpinteg.trapezoid(
        ddist['dist']['RE']['dist']['data'],
        x=ddist['coords']['x1']['data'],
        axis=-1,
    )
    dataMax = scpinteg.trapezoid(
        ddist['dist']['maxwell']['dist']['data'],
        x=ddist['coords']['x1']['data'],
        axis=-1,
    )

    # ------------
    # Derive levels, vmin, vmax
    # ------------

    Ekin_max = ddist['plasma']['Ekin_max_eV']['data']
    vminRE_2d = np.inf
    vminRE_1d = np.inf
    for ind in np.ndindex(dataRE.shape[:-1]):
        indE = np.argmin(np.abs(ddist['coords']['x0']['data'] - Ekin_max[ind]))
        sli = ind + (indE, slice(None))
        vmaxRE_2d = np.nanmax(ddist['dist']['RE']['dist']['data'][sli])
        vminRE_2d = min(vminRE_2d, vmaxRE_2d)
        sli = ind + (indE,)
        vmaxRE_1d = dataRE[sli]
        vminRE_1d = min(vminRE_1d, vmaxRE_1d)
    vmaxRE_2d = np.nanmax(ddist['dist']['RE']['dist']['data'])
    vmaxRE_1d = np.nanmax(dataRE)
    vmaxMax_2d = np.nanmax(ddist['dist']['maxwell']['dist']['data'])
    vmaxMax_1d = np.nanmax(dataMax)

    # 1d
    vmaxlog10_1d = np.log10(max(vmaxRE_1d, vmaxMax_1d))
    dlog10_1d = vmaxlog10_1d - np.log10(vminRE_1d)
    vmaxlog10_1d = np.ceil(vmaxlog10_1d)
    vminlog10_1d = np.floor(vmaxlog10_1d - 1 - 1.2*dlog10_1d)
    vmax_1d = 10**vmaxlog10_1d
    vmin_1d = 10**vminlog10_1d

    # 2d
    vmaxlog10_2d = np.log10(max(vmaxRE_2d, vmaxMax_2d))
    dlog10_2d = vmaxlog10_2d - np.log10(vminRE_2d)
    vmaxlog10_2d = np.ceil(vmaxlog10_2d)
    vminlog10_2d = np.floor(vmaxlog10_2d - 1 - 1.2*dlog10_2d)
    levels_2d = np.logspace(vminlog10_2d, vmaxlog10_2d - 1, 6)

    # --------------
    # labels
    # --------------

    dlabel = {}
    for ind in np.ndindex(ddist['dist']['RE']['dist']['data'].shape[:-2]):
        Te = ddist['plasma']['Te_eV']['data'][ind] * 1e-3
        jpf = ddist['plasma']['jp_fraction_re']['data'][ind]
        Ek = ddist['plasma']['Ekin_max_eV']['data'][ind]
        Ek = f"{Ek*1e-3:3.0f} keV" if np.log10(Ek) <= 6 else f"{Ek*1e-6:2.0f} Mev"

        dlabel[ind] = f"{jpf:2.1f}  ,    {Te:2.1f} keV,  {Ek}"

    # title
    ne = np.unique(ddist['plasma']['ne_m3']['data'])
    assert ne.size == 1
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])
    assert jp.size == 1
    tit = (
        r"$n_e$"
        + f" = {ne[0]:2.1e}, "
        + r"$j_{P,tot}$"
        + f" = {jp[0]*1e-6:2.1f} MA/m2"
    )

    # --------------
    # print
    # --------------

    _print(ddist)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.12, 'right': 0.98,
        'bottom': 0.06, 'top': 0.97,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # axes - 2d
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto', xscale='log')
    ax.set_ylabel(
        r'$\theta_{e_0,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_title(tit, fontsize=fontsize, fontweight='bold')
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

    dax['2d'] = ax

    # --------------
    # axes - 1d
    # --------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto', sharex=ax, yscale='log')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        f" ({units1d})",
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

    dax['1d'] = ax

    dax = ds._generic_check._check_dax(dax)

    # ------------
    # plot 1d
    # ------------

    kax = '1d'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # for legend
        ax.plot(
            [], [],
            c='w',
            ls='-',
            lw=1,
            label=r"$j_{frac}$, $T_e$,   $E_{e_0,max}$",
        )

        # loop plot
        for ind in np.ndindex(dataRE.shape[:-1]):
            sli = ind + (slice(None),)

            # Max
            l0, = ax.plot(
                ddist['coords']['x0']['data']*1e-3,
                dataMax[sli],
                ls='-',
                lw=1,
            )
            dcolor[ind] = l0.get_color()

            # RE
            ax.plot(
                ddist['coords']['x0']['data']*1e-3,
                dataRE[sli],
                ls='--',
                lw=1,
                color=dcolor[ind],
            )

            # Total
            ax.plot(
                ddist['coords']['x0']['data']*1e-3,
                dataMax[sli] + dataRE[sli],
                ls='-',
                lw=2,
                color=dcolor[ind],
                label=dlabel[ind],
            )

        # Add critical energy
        Ec = tf.physics_tools.electrons.convert_momentum_velocity_energy(
            momentum_normalized=ddist['dist']['RE']['p_crit']['data'],
        )['energy_kinetic_eV']['data']
        for ec in np.unique(Ec):
            ax.axvline(ec*1e-3, c='k', lw=1, ls='--')

        # decorate
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_ylim(vmin_1d, vmax_1d)

    # ------------
    # plot 2d
    # ------------

    kax = '2d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # loop plot
        for ind in np.ndindex(dataRE.shape[:-1]):
            sli = ind + (slice(None), slice(None))

            # data
            data = (
                ddist['dist']['maxwell']['dist']['data'][sli]
                + ddist['dist']['RE']['dist']['data'][sli]
            )

            # contour
            ax.contour(
                ddist['coords']['x0']['data']*1e-3,
                ddist['coords']['x1']['data']*180/np.pi,
                data.T,
                levels_2d,
                colors=dcolor[ind],
            )

        # decorate
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

    return dax, ddist


# #####################################################
# #####################################################
#       _print
# #####################################################


def _print(ddist, sep='  '):

    # -----------
    # header

    head = [
        'ind',
        'Te (keV)',
        'ne (1e20/m3)', 'Max / RE',
        'jp (MA/m2)', 'Max / RE',
    ]
    lmax = np.max([len(ss) for ss in head])

    # -----------
    # header

    lc = []
    for ind in np.ndindex(ddist['dist']['RE']['dist']['data'].shape[:-2]):
        Te = ddist['plasma']['Te_eV']['data'][ind]*1e-3
        ne = ddist['plasma']['ne_m3']['data'][ind]*1e-20
        jp = ddist['plasma']['jp_Am2']['data'][ind]*1e-6
        ne_max = ddist['dist']['maxwell']['integ_ne']['data'][ind]*1e-20
        ne_RE = ddist['dist']['RE']['integ_ne']['data'][ind]*1e-20
        jp_max = ddist['dist']['maxwell']['integ_jp']['data'][ind]*1e-6
        jp_RE = ddist['dist']['RE']['integ_jp']['data'][ind]*1e-6

        cc = [
            str(ind),
            f'{Te:2.1f}',
            f'{ne:2.2f}', f"{ne_max:2.2f} / {ne_RE:2.2f}",
            f'{jp:2.2f}', f"{jp_max:2.2f} / {jp_RE:2.2f}",
        ]
        lc.append(cc)
        lmax = max(lmax, np.max([len(ss) for ss in cc]))

    # ----------------
    # concatenate

    line = sep.join(['-'*lmax for ss in head])
    head = sep.join([ss.ljust(lmax) for ss in head])
    lc = [
        sep.join([ss.ljust(lmax) for ss in cc])
        for cc in lc
    ]

    msg = '\n'.join([head, line] + lc)
    print(msg)

    return
