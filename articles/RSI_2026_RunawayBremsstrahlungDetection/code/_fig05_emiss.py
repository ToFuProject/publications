import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _load_spect_anis
from ._fig02_dist_type import _DDIST
from ._fig04_bremsstrahlung import _CASES
from ._savefig import main as savefig


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)


_RE = 'avalanche 100 keV'


_DDMIX = {
    # 0: 'H',
    1: 'O',
    2: {'O': 0.90, 'Fe': 0.10},
}


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    dmix=None,
    # cases
    cases=None,
    # d2cross
    d2cross_phi=None,
    # dist
    ne_m3=None,
    jp_Am2=None,
    # RE
    re=None,
    dominant=None,
    jp_fraction_re=np.linspace(0.025, 0.975, 11),
    Efield_par_Vm=None,
    Ekin_min_eV=None,
    Ekin_max_eV=None,
    sigmap=None,
    pnormW=0,
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    # plot
    figsize=(15, 8),
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

    if dmix is None:
        dmix = _DDMIX
    nmix = len(dmix)

    # Maxwell
    kwd_max = {'ne_m3': ne_m3, 'jp_Am2': jp_Am2}

    # RE
    if re is None:
        re = _RE
    assert re in _DDIST['RE'].keys()
    lRE = [
        'dominant', 'jp_fraction_re', 'Efield_par_Vm',
        'Efield_par_Vm', 'Ekin_max_eV', 'Ekin_min_eV',
        'sigmap', 'pnormW'
    ]
    kwd_RE = {
        kk: _DDIST['RE'][re].get(kk) if vv is None else vv
        for kk, vv in locals().items()
        if kk in lRE
    }

    if cases is None:
        cases = _CASES

    # --------------
    # load elements
    # --------------

    demiss = {}
    kwd = dict(kwd_max)
    kwd.update(**kwd_RE)
    for ii in dmix.keys():
        demiss[ii], ddist, dmix[ii] = _load_spect_anis.main(
            dmix=dmix[ii],
            # d2cross
            d2cross_phi=d2cross_phi,
            # dist
            **kwd,
        )

    # extract
    # E_ph = demiss[0]['common']['E_photon']['data']
    Te = ddist['plasma']['Te_eV']['data']
    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    units = demiss[0]['emiss']['maxwell']['ff']['units']

    # --------------
    # Elim
    # --------------

    shape = ddist['plasma']['Te_eV']['data'].shape
    Elim = {kk: np.full(shape, np.nan) for kk in dmix.keys()}
    for kk in dmix.keys():
        for ind in np.ndindex(shape):

            sli_anis = ind + (slice(None), 0)
            sli_iso = ind[:-1] + (0, slice(None), 0)

            emiss_max = (
                demiss[kk]['emiss']['maxwell']['ff']['data'][sli_anis]
                + demiss[kk]['emiss']['maxwell']['fb']['data'][sli_iso]
                + demiss[kk]['emiss']['maxwell']['bb']['data'][sli_iso]
            )

            emiss_RE = demiss[kk]['emiss']['RE']['ff']['data'][sli_anis]

            ilim = (emiss_RE > emiss_max)
            if np.any(ilim):
                Elim[kk][ind] = np.min(demiss[kk]['E_ph']['data'][ilim])

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.05, 'right': 0.99,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.20, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=3, nrows=nmix, **dmargin)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    tit0 = (
        r"$n_e$" + f" = {ne:1.1e} " + r"$/m^3$,  "
        + r"$j_P$" + f" = {jp:1.1e} " + r"$A/m^2$" + "\n"
        + re
    )

    ax0_spect = None
    ax0_map = None
    for im, mix in enumerate(sorted(dmix.keys())):

        lk = list(dmix[mix].keys())
        lc = [np.unique(dmix[mix][kk])[0] for kk in lk]
        inds = np.argsort(lc)[::-1]
        lstr = [f"{lk[ss]} {lc[ss]*100:3.1f} \%" for ss in inds]
        tit = ",  ".join(lstr)
        if im == 0:
            tit = tit0 + tit

        ax = fig.add_subplot(
            gs[im, :2],
            sharex=ax0_spect,
            sharey=ax0_spect,
            aspect='auto',
        )
        ax.set_title(
            tit,
            fontsize=fontsize,
            fontweight='bold',
        )
        ax.set_ylabel(
            r"$\epsilon$" + f' ({units})',
            fontsize=fontsize,
            fontweight='bold',
        )
        if im == 0:
            ax0_spect = ax
        elif im == nmix - 1:
            ax.set_xlabel(
                r"$E_{ph}$" + ' (keV)',
                fontsize=fontsize,
                fontweight='bold',
            )
        ax.text(
            0.,
            1.05,
            ['(a)', '(b)'][im],
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
        )

        dax[f'spect_{im}'] = ax

        # ----------------
        # ax - Elim
        # ----------------

        ax = fig.add_subplot(
            gs[im, 2:],
            aspect='auto',
            sharex=ax0_map,
            sharey=ax0_map,
        )
        if im == 0:
            ax0_map = ax
        elif im == nmix - 1:
            ax.set_xlabel(
                r"$T_e$" + ' (keV)',
                fontsize=fontsize,
                fontweight='bold',
            )
        ax.set_ylabel(r"$F_{RE}$", fontsize=fontsize, fontweight='bold')

        ax.text(
            0.,
            1.05,
            ['(c)', '(d)'][im],
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
        )
        dax[f'Elim_{im}'] = ax

    # ----------------
    # check dax format

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - cases
    # --------------

    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    jp_fracu = np.unique(ddist['plasma']['jp_fraction_re']['data'])
    Ekinu = np.unique(ddist['plasma']['Ekin_max_eV']['data'])

    # loop on cases
    for i0, (k0, v0) in enumerate(cases['case'].items()):

        # slice
        Te = Teu[np.argmin(np.abs(Teu - v0['Te']))]
        jpf = jp_fracu[np.argmin(np.abs(jp_fracu - v0['jp_frac']))]
        Ekin = Ekinu[np.argmin(np.abs(Ekinu - v0['Ekin_max_eV']))]
        ic = (
            (ddist['plasma']['jp_fraction_re']['data'] == jpf)
            & (ddist['plasma']['Te_eV']['data'] == Te)
            & (ddist['plasma']['Ekin_max_eV']['data'] == Ekin)
        )
        assert ic.sum() == 1
        ic = tuple([cc[0] for cc in ic.nonzero()])

        # --------
        # spectra

        # loop on dmix
        for ii in dmix.keys():

            kax = f'spect_{ii}'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                sli_anis = ic + (slice(None), slice(None))
                sli_iso = ic[:-1] + (0, slice(None), slice(None))

                # -----------
                # total -maxwell

                emiss_max = (
                    demiss[ii]['emiss']['maxwell']['ff']['data'][sli_anis]
                    + demiss[ii]['emiss']['maxwell']['fb']['data'][sli_iso]
                    + demiss[ii]['emiss']['maxwell']['bb']['data'][sli_iso]
                )

                # plot
                ax.fill_between(
                    demiss[ii]['E_ph']['data']*1e-3,
                    np.nanmin(emiss_max, axis=-1),
                    np.nanmax(emiss_max, axis=-1),
                    hatch=v0['hatch'],
                    facecolor='None',
                    alpha=0.5,
                    edgecolor=v0['color'],
                    ls='-',
                )

                # -----------
                # ff -maxwell

                ff_max = demiss[ii]['emiss']['maxwell']['ff']['data'][sli_anis]

                # plot
                ax.plot(
                    demiss[ii]['E_ph']['data']*1e-3,
                    np.nanmean(ff_max, axis=-1),
                    color=v0['color'],
                    ls='--',
                )

                # -----------
                # ff - RE

                emiss_RE = demiss[ii]['emiss']['RE']['ff']['data'][sli_anis]

                # plot
                ax.fill_between(
                    demiss[ii]['E_ph']['data']*1e-3,
                    np.nanmin(emiss_RE, axis=-1),
                    np.nanmax(emiss_RE, axis=-1),
                    hatch=v0['hatch'],
                    facecolor='None',
                    alpha=0.5,
                    edgecolor=v0['color'],
                    ls='--',
                    label=r'$\epsilon_{ff}^{RE}$' if ic == 0 else None,
                )

                # decorate
                vmax_log10 = np.ceil(np.log10(np.nanmax(emiss_max)))

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(10**(vmax_log10 - 18), 10**vmax_log10)
                ax.set_xlim(left=demiss[ii]['E_ph']['data'][0]*1e-3)
                ax.grid(True)

                # legend
                lab = r"$\epsilon_{ff}^{Max} + \epsilon_{fb}^{Max} + \epsilon_{bb}^{Max}$"
                ax.plot([], [], c='k', ls='-', label=lab)
                lab = r"$\epsilon_{ff}^{Max}$"
                ax.plot([], [], c='k', ls='--', label=lab)
                ax.legend(loc='upper right')

    # --------------
    # plot - Elim
    # --------------

    for kk in dmix.keys():
        kax = f'Elim_{kk}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            sli = (0, slice(None), slice(None))
            cs = ax.contour(
                ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                ddist['plasma']['jp_fraction_re']['data'][sli],
                Elim[kk][sli] * 1e-3,
                cmap=plt.cm.viridis,
                levels=np.r_[1, 2, 5, 7.5, 10, 15, 20],
                vmin=0.1,
                vmax=20,
            )

            # cases
            for i0, (k0, v0) in enumerate(cases['case'].items()):
                ax.plot(
                    v0['Te']*1e-3,
                    v0['jp_frac'],
                    marker='*',
                    markersize=8,
                    markerfacecolor=v0['color'],
                    color=v0['color'],
                )

            ax.clabel(cs, cs.levels, fontsize=12)

            ax.set_xlim(0, ddist['plasma']['Te_eV']['data'].max()*1e-3)
            ax.set_ylim(0, 1)

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return dax, demiss
