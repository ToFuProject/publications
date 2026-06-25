import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import datastock as ds
import tofu as tf


from ._fig02_dist import _DDIST


tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#       DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUTS = os.path.join(os.path.dirname(_PATH_HERE), 'inputs')
_PATH_SAVE = os.path.join(os.path.dirname(_PATH_HERE), 'figures')


_PFE_D2CROSS_PHI = os.path.join(
    _PATH_INPUTS,
    'd2cross_phi_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_nthetaph61_nthetae060_EH.npz',
    # 'd2cross_phi_Ee01eV-100MeV-240log_Eph1eV-100MeV-241log_nthetaph61_nthetae060_EH.npz',
)


_CASES = {
    'case': {
        '0': {
            'Te': 0.1e3,
            'jp_frac': 0.9,
            'Ekin_max_eV': 100e3,
            'color': 'r',
            'hatch': '//',
        },
        '1': {
            'Te': 2.0e3,
            'jp_frac': 0.1,
            'Ekin_max_eV': 10e6,
            'color': 'b',
            'hatch': "\\",
        },
    },
    'theta_ph_vsB': {
        'val': np.r_[0, 0.5, 1]*np.pi,
        'ls': ['-', '--', ':'],
    },
    'E_ph_eV': {
        'val': np.r_[0.1, 2, 20]*1e3,
        'ls': ['-', '-', '-'],
    },
}


# #####################################################
# #####################################################
#       main
# #####################################################


def main(
    d2cross_phi=None,
    # dist
    ne_m3=None,
    pnormW=np.r_[0.1, 5],
    Ekin_max_eV=np.r_[100e3, 10e6],
    # Te_eV=1e3 * np.linspace(0.1, 2.5, 25),
    Te_eV=1e3 * np.linspace(0.1, 2.5, 11),
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    jp_fraction_re=np.linspace(0.1, 0.9, 9),
    # emiss
    E_ph_eV=None,
    # cases
    cases=None,
    # plot
    figsize=(15, 8),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Plot a selection of Maxwellian and RE free-free spectra

    Includes selected angular amissivity
    Includes a 2d contour plot of photon energy threshold for RE dominance

    """

    # ------------
    # d2cross_phi
    # ------------

    if d2cross_phi is None:
        d2cross_phi = _PFE_D2CROSS_PHI

    # ------------
    # ddist
    # ------------

    ddist = locals()
    ddist = {
        kk: vv if ddist.get(kk) is None else ddist[kk]
        for kk, vv in _DDIST.items()
        if kk not in ['E_eV', 'theta']
    }
    if ne_m3 is not None:
        ddist['ne_m3'] = ne_m3

    if pnormW is not None:
        ddist['pnormW'] = pnormW

    if Ekin_max_eV is not None:
        ddist['Ekin_max_eV'] = Ekin_max_eV

    if pnormW is not None:
        ddist['pnormW'] = pnormW

    # shape
    ddist['Ekin_max_eV'] = np.atleast_1d(ddist['Ekin_max_eV'])[:, None, None]
    ddist['pnormW'] = np.atleast_1d(ddist['pnormW'])[:, None, None]
    if ddist['pnormW'].size != ddist['Ekin_max_eV'].size:
        raise Exception()
    ddist['Te_eV'] = Te_eV[None, :, None]
    ddist['jp_fraction_re'] = jp_fraction_re[None, None, :]

    nEkin = ddist['Ekin_max_eV'].shape[0]

    # --------------
    # integrated cross-section
    # --------------

    demiss, ddist, d2cross_phi = tfphysemis.get_xray_thin_integ_dist(
        # ----------------
        # cross-section
        # tabulated d2cross_phi
        d2cross_phi=d2cross_phi,
        # d2cross_phi computation
        E_ph_eV=E_ph_eV,
        E_e0_eV=None,
        E_e0_eV_npts=None,
        theta_e0_vsB_npts=None,
        phi_e0_vsB_npts=None,
        theta_ph_vsB=None,
        # inputs
        Z=None,
        # hypergeometric parameter
        ninf=None,
        source=None,
        # integration parameters
        nthetae=None,
        ndphi=None,
        # output customization
        version_cross=None,
        # save / load
        save_d2cross_phi=False,
        # ---------------------
        # optional responsivity
        dresponsivity=None,
        plot_responsivity_integration=None,
        # -----------
        # verb
        debug=False,
        verb=True,
        # ----------------
        # electron distribution
        **ddist,
    )

    units = demiss['emiss']['RE']['emiss']['units']
    ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]

    tit = (
        r"$n_e$" + f" = {ne:1.0e} /m3\n"
        + r"$j_{P,RE}$" + f" = {jp*1e-6:2.1f} MA/m2"
    )

    # --------------
    # cases
    # --------------

    if cases is None:
        cases = _CASES

    # --------------
    # Elim
    # --------------

    shape = ddist['plasma']['Te_eV']['data'].shape
    Elim = np.full(shape, np.nan)
    theta_Elim = 0
    for ii, ind in enumerate(np.ndindex(shape)):

        sli_emiss = ind + (slice(None), 0)
        emiss_RE = demiss['emiss']['RE']['emiss']['data'][sli_emiss]
        emiss_max = demiss['emiss']['maxwell']['emiss']['data'][sli_emiss]

        ilim = (emiss_RE > emiss_max)
        if np.any(ilim):
            Elim[ind] = np.min(demiss['E_ph_eV']['data'][ilim])

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.06, 'right': 0.98,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.25, 'hspace': 0.30,
    }
    dmargin_theta = {
        'left': 0.06, 'right': 0.98,
        'bottom': 0.06, 'top': 0.60,
        'wspace': 0.25, 'hspace': 0.10,
    }
    dmargin_map = dict(dmargin)
    dmargin_map['hspace'] = 0.10

    fig = plt.figure(figsize=figsize)

    nE = len(cases['E_ph_eV']['val'])
    gs = gridspec.GridSpec(ncols=nE + 2, nrows=3, **dmargin)
    gs_theta = gridspec.GridSpec(ncols=nE + 2, nrows=2, **dmargin_theta)
    gs_map = gridspec.GridSpec(ncols=nE + 2, nrows=nEkin, **dmargin_map)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    ax = fig.add_subplot(gs[0, :nE], aspect='auto')
    ax.set_xlabel(
        r"$E_{ph}$" + ' (keV)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon$" + f' ({units})',
        fontsize=fontsize,
        fontweight='bold',
    )
    dax['spectra'] = ax

    # ----------------
    # ax - theta - lin
    # ----------------

    ax0 = None
    for ii in range(nE):
        ax = fig.add_subplot(
            gs_theta[0, ii],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        if ii == 0:
            ax.set_ylabel(
                'emiss (norm.)',
                fontsize=fontsize,
                fontweight='bold',
            )
            ax.set_xlim(0, 180)
            ax.set_ylim(0, 1)
            ax0 = ax

        dax[f'theta_{ii}_lin'] = ax

    # ----------------
    # ax - theta - log
    # ----------------

    ax0 = None
    for ii in range(nE):
        ax = fig.add_subplot(
            gs_theta[1, ii],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        ax.set_xlabel(
            r'$\theta_{ph,B}$' + ' (deg)',
            fontsize=fontsize,
            fontweight='bold',
        )
        if ii == 0:
            ax.set_ylabel(
                f'{units}',
                fontsize=fontsize,
                fontweight='bold',
            )
            ax.set_xlim(0, 180)
            # ax.set_ylim(0, 1)
            ax0 = ax

        dax[f'theta_{ii}_log'] = ax

    # ----------------
    # ax - Elim
    # ----------------

    ax0 = None
    for ii in range(nEkin):
        ax = fig.add_subplot(
            gs_map[ii, nE:],
            aspect='auto',
            sharex=ax0,
            sharey=ax0,
        )
        if ii == 0:
            ax0 = ax
            ax.set_title(
                tit,
                fontsize=fontsize,
                fontweight='bold',
            )
        elif ii == nEkin - 1:
            ax.set_xlabel('Te (keV)', fontsize=fontsize, fontweight='bold')
        ax.set_ylabel('jp_frac', fontsize=fontsize, fontweight='bold')

        dax[f'Elim_{ii}'] = ax

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

        kax = 'spectra'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            sli = ic + (slice(None), slice(None))
            for kdist in demiss['emiss'].keys():
                emiss_E = demiss['emiss'][kdist]['emiss']['data'][sli]

                # plot
                ax.fill_between(
                    demiss['E_ph_eV']['data']*1e-3,
                    np.nanmin(emiss_E, axis=-1),
                    np.nanmax(emiss_E, axis=-1),
                    hatch=v0['hatch'],
                    facecolor='None',
                    alpha=0.5,
                    edgecolor=v0['color'],
                    ls='--' if kdist == 'RE' else '-',
                    label=f'{kdist}_{ic}',
                )

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

            # vlines
            for i1, cc in enumerate(cases['E_ph_eV']['val']):
                ax.axvline(
                    cc*1e-3,
                    c='k',
                    ls='--',
                    lw=1,
                    label=f"E_ph = {cc*1e-3:3.1f} keV",
                )

            # Elim
            iE = np.argmin(np.abs(demiss['E_ph_eV']['data'] - Elim[ic]))
            ax.plot(
                np.r_[Elim[ic], Elim[ic]] * 1e-3,
                [emiss_E[iE, 0], 1e16],
                color=v0['color'],
                ls='--',
                lw=1,
                label=f"E_lim = {Elim[ic]*1e-3:3.1f} keV",
            )

            # text
            trans = transforms.blended_transform_factory(
                ax.transData,
                ax.transAxes,
            )
            ax.text(
                Elim[ic] * 1e-3,
                1,
                r"$E_{ph,lim}$" + f"\n = {Elim[ic] * 1e-3:2.1f} keV",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=fontsize,
                fontweight='bold',
                color=v0['color'],
                transform=trans,
            )

            ax.set_ylim(1e0, 1e16)
            ax.set_xlim(1e-2, 2e4)

        # --------
        # theta

        for i1, cc in enumerate(cases['E_ph_eV']['val']):

            kax = f'theta_{i1}_lin'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                iE = np.argmin(np.abs(demiss['E_ph_eV']['data'] - cc))
                sli = ic + (iE, slice(None))

                for kdist in demiss['emiss'].keys():
                    emiss_theta = demiss['emiss'][kdist]['emiss']['data'][sli]

                    # plot
                    ax.plot(
                        demiss['theta_ph_vsB']['data']*180/np.pi,
                        emiss_theta / emiss_theta.max(),
                        # ls=cases['E_ph_eV']['ls'][i1],
                        ls='--' if kdist == 'RE' else '-',
                        lw=1,
                        marker='None',
                        color=v0['color'],
                        label=f'{kdist}_{ic}_{cc*1e-3:3.1f}keV',
                    )

                # deco
                ax.set_title(
                    r"$E_{ph,B}$" + f" = {cc*1e-3:3.1f} keV",
                    fontsize=fontsize,
                    fontweight='bold',
                )

            # ---------
            # theta - log

            kax = f'theta_{i1}_log'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                iE = np.argmin(np.abs(demiss['E_ph_eV']['data'] - cc))
                sli = ic + (iE, slice(None))

                for kdist in demiss['emiss'].keys():
                    emiss_theta = demiss['emiss'][kdist]['emiss']['data'][sli]

                    # plot
                    ax.semilogy(
                        demiss['theta_ph_vsB']['data']*180/np.pi,
                        emiss_theta,
                        # ls=cases['E_ph_eV']['ls'][i1],
                        ls='--' if kdist == 'RE' else '-',
                        lw=1,
                        marker='None',
                        color=v0['color'],
                        label=f'{kdist}_{ic}_{cc*1e-3:3.1f}keV',
                    )

            ax.set_ylim(2e4, 6e14)

    # --------------
    # plot - Elim
    # --------------

    for ii in range(nEkin):
        kax = f'Elim_{ii}'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            sli = (ii, slice(None), slice(None))
            cs = ax.contour(
                ddist['plasma']['Te_eV']['data'][sli] * 1e-3,
                ddist['plasma']['jp_fraction_re']['data'][sli],
                Elim[sli] * 1e-3,
                cmap=plt.cm.viridis,
                levels=np.r_[1, 2, 5, 7.5, 10, 15],
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

    # pfe_save
    pfe_save = ds._generic_check._check_var(
        pfe_save, 'pfe_save',
        types=(bool, str),
        default=False,
    )

    # saving
    if pfe_save is not False:
        if pfe_save is [None, True]:
            name = 'fig04_bremsstrahlung.png'
            if path_save is None:
                path_save = _PATH_SAVE
            pfe_save = os.path.join(_PATH_SAVE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax, demiss, ddist, d2cross_phi
