

import os


import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as asunits
import datastock as ds


import tofu as tf


tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')
_PATH_SAVE = os.path.join(_PATH_PAPER, 'figures')


# SPECTRAL MODELLING FILES
_LPFE_SPECT = [
    ff for ff in os.listdir(_PATH_INPUTS)
    if ff.endswith('_data.npz')
    and any([ss in ff for ss in ['_SCRAM86_', '_FLYCHK_']])
]
_DPFE_SPECT = {
    ff.split('_')[-2]: os.path.join(_PATH_INPUTS, ff)
    for ff in _LPFE_SPECT
}


# CROSS-SECTION FILES
_DPFE_DCROSS = {
    'EH0': os.path.join(
        _PATH_INPUTS,
        'd2cross_phi_Ee01eV-100MeV-240log_Eph1eV-100MeV-241log_nthetaph61_nthetae060_EH.npz',
    ),
    'EH1': os.path.join(
        _PATH_INPUTS,
        'd2cross_phi_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_nthetaph61_nthetae060_EH.npz'
    ),
}


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    d2cross_phi='EH1',
    elements='H',
    ne_m3=1e19,
    Te_plot=None,
    # plot
    figsize=(6, 8),
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

    # elements
    if isinstance(elements, str):
        elements = [elements]
    elements = tuple(ds._generic_check._check_var_iter(
        elements, 'elements',
        types=(list, tuple),
        types_iter=str,
        allowed=list(_DPFE_SPECT.keys()),
    ))

    # d2cross_phi
    d2cross_phi = ds._generic_check._check_var(
        d2cross_phi, 'd2cross_phi',
        types=str,
        allowed=list(_DPFE_DCROSS.keys()),
        default='EH0',
    )
    d2cross_phi = _DPFE_DCROSS[d2cross_phi]

    # --------------
    # load elements
    # --------------

    dspect = {
        kk: {
            'file': np.load(
                _DPFE_SPECT[kk],
                allow_pickle=True,
            )['arr_0'].tolist()
        }
        for kk in elements
    }

    ni_m3 = ne_m3

    # Extract Te
    lc = ['Te', 'E_photon']
    dcommon = {cc: {} for cc in lc}
    for cc in lc:
        ref = dspect[elements[0]]['file'][cc]['data']
        for kk, vv in dspect.items():
            assert np.allclose(vv['file'][cc]['data'], ref)
        dcommon[cc] = dspect[elements[0]]['file'][cc]

    # Compute emissivity
    units0 = 'J*cm^3/s/eV/atom/electron'
    units = "1 / (m3.s.eV.sr)"
    E_ph = dcommon['E_photon']['data']
    for kk, vv in dspect.items():
        assert str(vv['file']['emis_ff']['units']).replace('$', '') == units0

        # ph / m3 / s / eV / sr
        emiss_ff = (
            vv['file']['emis_ff']['data']
            * 1e-6    # cm3 => m3
            / (E_ph[None, :] * scpct.e)  # J => ph
            * ne_m3  # /electron => /m3
            * ni_m3  # /atom => /m3
            / (4*np.pi)  # => /sr
        )

        dspect[kk]['ff'] = {
            'data': emiss_ff,
            'units': units,
        }

    # --------------
    # load cross
    # --------------

    demiss, ddist, d2cross_phi = tfphysemis.get_xray_thin_integ_dist(
        # dist
        Te_eV=dcommon['Te']['data'],
        ne_m3=ne_m3,
        jp_Am2=0,
        Zeff=1,
        jp_fraction_re=0,
        # ----------------
        # cross-section
        # tabulated d2cross_phi
        d2cross_phi=d2cross_phi,
        # d2cross_phi computation
        E_ph_eV=dcommon['E_photon']['data'],
        # -----------
        # verb
        debug=False,
        verb=True,
    )

    # check units
    units2 = demiss['emiss']['maxwell']['emiss']['units']
    assert asunits.Unit(units) == asunits.Unit(units2)

    # check isotropy
    emiss = demiss['emiss']['maxwell']['emiss']['data']
    mean = np.nanmean(emiss, axis=-1)
    diff = emiss - mean[:, :, None]
    iok = emiss > 0.
    c0 = np.abs(diff[iok] / emiss[iok])
    assert np.all(c0 < 1e-2)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.13, 'right': 0.99,
        'bottom': 0.10, 'top': 0.88,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - abs

    ax = fig.add_subplot(
        gs[0, 0],
        xscale='log',
        yscale='log',
        aspect='auto',
    )
    ax.set_xlabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon^{Max}_{ff}$" + f"  ({asunits.Unit(units)})",
        size=fontsize,
        fontweight='bold',
    )
    tit = (
        "Validation of Bremstrahlung implementation from EH cross-section\n"
        "Maxwellian distribution with  "
        + r"$j_{P}$" + " = 0 A/m2,  "
        + r"$n_e$" + f" = {ne_m3:1.0e}"
    )
    ax.set_title(
        tit,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['abs'] = {'handle': ax}

    # --------------
    # ax - diff

    ax0 = ax
    ax = fig.add_subplot(
        gs[1, 0],
        sharex=ax0,
        yscale='log',
        aspect='auto',
    )
    ax.set_xlabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "error  (%)",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['diff'] = {'handle': ax}

    # standardize
    dax = ds._generic_check._check_dax(dax)

    # ---------------------
    # Absolute values
    # ---------------------

    kax = 'abs'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(dcommon['Te']['data']):

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ii, slice(None))

            # loop on elements
            for kk, vv in dspect.items():

                # SCRAM / FLYCHK
                l0, = ax.loglog(
                    dcommon['E_photon']['data']*1e-3,
                    vv['ff']['data'][sli],
                    ls='-',
                    lw=1,
                    label=lab,
                )
                dcolor[ii] = l0.get_color()

                # ff
                l0, = ax.loglog(
                    demiss['E_ph_eV']['data']*1e-3,
                    mean[sli],
                    ls='--',
                    lw=1,
                    color=dcolor[ii],
                    # label=lab,
                )

        ax.legend()

    # ---------------
    # plot diff
    # ---------------

    kax = 'diff'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # loop on Te

        for ii, te in enumerate(dcommon['Te']['data']):

            # label
            lab = r"$T_e$" + f" = {te*1e-3:3.2f} keV"

            # slice
            sli = (ii, slice(None))

            # interp
            interp = 10**(np.interp(
                np.log10(dcommon['E_photon']['data']),
                np.log10(demiss['E_ph_eV']['data']),
                np.log10(mean[sli]),
            ))

            # min
            emiss_min = np.minimum(interp, vv['ff']['data'][sli])
            iok = emiss_min > 0.
            diff = np.full(interp.shape, np.nan)
            diff[iok] = 100 * (
                np.abs(interp - vv['ff']['data'][sli]) / emiss_min
            )[iok]

            # loop on elements
            for kk, vv in dspect.items():

                # SCRAM / FLYCHK
                l0, = ax.loglog(
                    dcommon['E_photon']['data']*1e-3,
                    diff,
                    ls='-',
                    lw=1,
                    color=dcolor[ii],
                    label=lab,
                )

        ax.legend()

    # --------------
    # add a, b, c, d, e
    # --------------

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = f"{os.path.split(__file__)[-1][1:].replace('.py', '')}.png"
            if path_save is None:
                path_save = _PATH_SAVE
            pfe_save = os.path.join(_PATH_SAVE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax, demiss, ddist, dspect
