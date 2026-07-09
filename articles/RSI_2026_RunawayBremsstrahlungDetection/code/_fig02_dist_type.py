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
    "maxwell": {
        'Te_eV': 1e3,
        'ne_m3': 1e19,
        'jp_Am2': 1e6,
    },
    'RE': {
        'dreicer': {
            'dominant': 'dreicer',
            'jp_fraction_re': 0.9,
            'Efield_par_Vm': 0.5,
            'Ekin_min_eV': 100,
            # used for legacy
            'pnormW': 0,
            'Ekin_max_eV': 0,
        },
        'avalanche 100 keV': {
            'dominant': 'avalanche',
            'jp_fraction_re': 0.9,
            'Efield_par_Vm': 0.5,
            'Ekin_max_eV': 100e3,
            'Ekin_min_eV': 100,
            'sigmap': 0.1,
            # used for legacy
            'pnormW': 0,
        },
        'avalanche 10 MeV': {
            'dominant': 'avalanche',
            'jp_fraction_re': 0.9,
            'Efield_par_Vm': 0.5,
            'Ekin_max_eV': 10e6,
            'Ekin_min_eV': 100,
            'sigmap': 10,
            # used for legacy
            'pnormW': 0,
        },
    },
    # coords
    'coords': {
        'E_eV': np.logspace(0, 8, 80),
        'theta': np.linspace(0, 180, 181) * np.pi / 180,
    }
}


_DDIST_PLOT = {
    'maxwell': {'color': 'k', 'hatch': 'None'},
    'dreicer': {'color': 'tab:blue', 'hatch': '||'},
    'avalanche 100 keV': {'color': 'tab:orange', 'hatch': '--'},
    'avalanche 10 MeV': {'color': 'tab:green', 'hatch': 'o'},
}


# #####################################################
# #####################################################
#       main
# #####################################################


def main(
    # plot
    figsize=(5, 7),
    fontsize=14,
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

    # ------------
    # compute
    # ------------

    ddist = {}
    for kdist, vdist in _DDIST['RE'].items():

        kwd = _DDIST['maxwell']
        kwd.update(**vdist)
        kwd.update(**_DDIST['coords'])

        ddist[kdist] = tfphysdist.get_distribution(**kwd)

    # sanity check - jp_RE
    lip = [v0['dist']['RE']['integ_jp']['data'] for v0 in ddist.values()]
    assert np.allclose(lip, lip[0])

    # sanity check - Maxwell
    lMax = np.array([
        v0['dist']['maxwell']['dist']['data'] for v0 in ddist.values()
    ])
    err_rel = (lMax - lMax[0:1, ...])
    iok = lMax[0] > 0
    sli = (slice(None), iok)
    err_rel[sli] = 100 * err_rel[sli] / lMax[(slice(0, 1), iok)]
    assert np.all(err_rel < 0.5), np.max(err_rel)

    # units
    kdomref = list(_DDIST['RE'].keys())[0]
    units2d = asunits.Unit(ddist[kdomref]['dist']['RE']['dist']['units'])
    units1d = units2d * asunits.Unit(ddist[kdomref]['coords']['x1']['units'])

    # ------------
    # Derive 1d data
    # ------------

    d1d = {}
    for dom in ddist.keys():
        d1d[dom] = scpinteg.trapezoid(
            ddist[dom]['dist']['RE']['dist']['data'],
            x=ddist[dom]['coords']['x1']['data'],
            axis=-1,
        )

    dataMax = scpinteg.trapezoid(
        ddist[kdomref]['dist']['maxwell']['dist']['data'],
        x=ddist[kdomref]['coords']['x1']['data'],
        axis=-1,
    )

    # ------------
    # Derive levels, vmin, vmax
    # ------------

    # 1d
    vmaxlog10_1d = np.ceil(np.log10(np.nanmax(dataMax)))
    vmax_1d = 10**(vmaxlog10_1d)
    vmin_1d = 10**(vmaxlog10_1d - 8)

    # 2d
    dvminmaxlog10_2d = {}
    for dom in ddist.keys():

        Ekin_max = ddist[dom]['plasma']['Ekin_max_eV']['data']
        vminRE_2d = np.inf
        for ind in np.ndindex(d1d[dom].shape[:-1]):
            indE = np.argmin(np.abs(
                ddist[dom]['coords']['x0']['data'] - Ekin_max[ind]
            ))
            sli = ind + (indE, slice(None))
            vmaxRE_2d = np.nanmax(
                ddist[dom]['dist']['RE']['dist']['data'][sli]
            )
            vminRE_2d = min(vminRE_2d, vmaxRE_2d)
            sli = ind + (indE,)
        vmaxRE_2d = np.nanmax(ddist[dom]['dist']['RE']['dist']['data'])
        vmaxMax_2d = np.nanmax(
            ddist[dom]['dist']['maxwell']['dist']['data']
        )

        # 2d
        vmaxlog10_2d = np.log10(max(vmaxRE_2d, vmaxMax_2d))
        dlog10_2d = vmaxlog10_2d - np.log10(vminRE_2d)
        dvminmaxlog10_2d[dom] = {
            'max': np.ceil(vmaxlog10_2d),
            'min': np.floor(vmaxlog10_2d - 1 - 1.2*dlog10_2d),
        }

    levels_2d = np.logspace(
        np.min([vv['min'] for vv in dvminmaxlog10_2d.values()]),
        np.max([vv['max'] for vv in dvminmaxlog10_2d.values()]) - 1,
        6,
    )

    # --------------
    # labels
    # --------------

    dlabel = {}
    for dom in ddist.keys():
        dlabel[dom] = f"{dom}"

    # title
    Te = np.unique(ddist[kdomref]['plasma']['Te_eV']['data'])
    assert Te.size == 1
    ne = np.unique(ddist[kdomref]['plasma']['ne_m3']['data'])
    assert ne.size == 1
    jp = np.unique(ddist[kdomref]['plasma']['jp_Am2']['data'])
    assert jp.size == 1
    jp_fraction_re = np.unique(ddist[kdomref]['plasma']['jp_fraction_re']['data'])
    assert jp_fraction_re.size == 1
    Ekin_max_eV = np.unique(ddist[kdomref]['plasma']['Ekin_max_eV']['data'])
    assert Ekin_max_eV.size == 1
    sigmap = np.unique(ddist[kdomref]['plasma']['sigmap']['data'])
    assert sigmap.size == 1
    vloop = np.unique(ddist[kdomref]['plasma']['Efield_par_Vm']['data'])
    assert vloop.size == 1
    tit = (
        r"$n_e$" + f" = {ne[0]:2.1e} " + r"$/m^3$" "+,  "
        + r"$T_e$" + f" = {Te[0]*1e-3:2.1f} keV,  "
        + r"$V_{loop}$" + f" = {vloop[0]:2.1f} V/m\n"
        + r"$j_{P,tot}$" + f" = {jp[0]*1e-6:2.1f} MA/m2,  "
        + r"$F_{RE}$" + f" = {jp_fraction_re[0]:2.1f}"
    )

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.12, 'right': 0.98,
        'bottom': 0.06, 'top': 0.93,
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

    # ticklabels size
    for kax, vax in dax.items():
        dax[kax]['handle'].tick_params(
            axis='both',
            which='major',
            labelsize=fontsize - 1,
        )

    # ------------
    # plot 1d
    # ------------

    kax = '1d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # Maxwell
        ax.plot(
            ddist[kdomref]['coords']['x0']['data']*1e-3,
            dataMax[0, ...],
            ls='-',
            lw=1,
            c='k',
            label='Maxwellian',
        )

        # loop plot
        for dom in ddist.keys():

            # RE
            l0, = ax.plot(
                ddist[dom]['coords']['x0']['data']*1e-3,
                d1d[dom][0, ...],
                ls='-',
                lw=1,
                color=_DDIST_PLOT[dom]['color'],
                label=dlabel[dom],
            )

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

        # Maxwell
        ax.contour(
            ddist[kdomref]['coords']['x0']['data']*1e-3,
            ddist[kdomref]['coords']['x1']['data']*180/np.pi,
            ddist[kdomref]['dist']['maxwell']['dist']['data'][0, ...].T,
            levels_2d,
            colors='k',
        )

        # loop plot
        for dom in ddist.keys():

            # data
            data = ddist[dom]['dist']['RE']['dist']['data'][0, ...]

            # contour
            ax.contour(
                ddist[dom]['coords']['x0']['data']*1e-3,
                ddist[dom]['coords']['x1']['data']*180/np.pi,
                data.T,
                levels_2d,
                colors=_DDIST_PLOT[dom]['color'],
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
