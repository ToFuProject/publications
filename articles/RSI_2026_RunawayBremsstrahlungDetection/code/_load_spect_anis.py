import os


import numpy as np
import scipy.interpolate as scpinterp
import astropy.units as asunits
import tofu as tf


from ._load_spect import main as load_spect
from ._fig02_dist import _DDIST


tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')


def _D2CROSS_PHI_FN(nE, ntheta):
    return (
        f"d2cross_phi_Ee01eV-100MeV-{nE}log_Eph1eV-100MeV-{nE+1}log"
        f"_nthetaph{ntheta+1}_nthetae0{ntheta}_EH.npz"
    )


_DPFE_D2CROSS_PHI = {
    'EH0': os.path.join(_PATH_INPUTS, _D2CROSS_PHI_FN(80, 60)),
    'EH1': os.path.join(_PATH_INPUTS, _D2CROSS_PHI_FN(240, 60)),
}
_D2CROSS_PHI = 'EH0'


_TE = 1e3 * np.linspace(0.1, 2.5, 25)
_JP_FRAC = np.linspace(0.1, 0.9, 9)
_EKIN_MAX_EV = np.r_[100e3, 10e6]
_PNORMW = np.r_[0.1, 5]


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    dmix=None,
    ne_m3=None,
    # ddist
    jp_Am2=None,
    jp_fraction_re=None,
    Ekin_max_eV=None,
    pnormW=None,
    # intergation method
    integration=None,
    # d2cross_phi
    d2cross_phi=None,
):
    """ For a given impurity mix and ne, Te, return a dict of emissivities

    Emissivities are broken down into 2 distributions
    - Maxwell:
        - ff: anisotro'upper
        - fb: isotropic (from SCRAM / FLYCHK / CHIANTI)
        - bb: isotropic
    - RE:
        - ff: anisotropic

    Returns dplasma:
    {
        'concentration': {'H': float / array, 'Ni': float / array, ...},
        'emiss_tot': {
            'maxwell': {
                'ff': {'data': array, 'units': str}},
                'fb': {'data': array, 'units': str}},
                'bb': {'data': array, 'units': str}},
            },
            'RE': {
                'ff': {'data': array, 'units': str}},
            },
        },
        'common': {
            'Te': {'data': array, 'units': str},
            'ne': {'data': array, 'units': str},
            'E_ph': {'data': array, 'units': str},
            'theta_ph': {'data': array, 'units': str},
        },
    }

    """

    # --------------
    # isotropic
    # --------------

    # load
    dplasma = load_spect(
        dmix=dmix,
        ne_m3=ne_m3,
    )

    # extract

    # ---------------------
    # extract Te and ddist
    # ---------------------

    Te_eV = dplasma['common']['Te']['data']
    Z_eff = dplasma['common']['Zeff']['data']

    # ------------
    # inputs
    # ------------

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC
    jp_fraction_re = np.atleast_1d(jp_fraction_re)

    if Ekin_max_eV is None:
        Ekin_max_eV = _EKIN_MAX_EV
    Ekin_max_eV = np.atleast_1d(Ekin_max_eV)

    if pnormW is None:
        pnormW = _PNORMW
    pnormW = np.atleast_1d(pnormW)

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

    if Ekin_max_eV is not None:
        ddist['Ekin_max_eV'] = Ekin_max_eV

    if pnormW is not None:
        ddist['pnormW'] = pnormW

    # shape
    ddist['Ekin_max_eV'] = ddist['Ekin_max_eV'][:, None, None]
    ddist['pnormW'] = ddist['pnormW'][:, None, None]
    if ddist['pnormW'].size != ddist['Ekin_max_eV'].size:
        raise Exception()
    ddist['Te_eV'] = Te_eV[None, :, None]
    if jp_Am2 is not None:
        ddist['jp_Am2'] = jp_Am2
    ddist['jp_fraction_re'] = jp_fraction_re[None, None, :]

    nEkin = ddist['Ekin_max_eV'].shape[0]

    # ------------
    # d2cross_phi
    # ------------

    if d2cross_phi is None:
        d2cross_phi = _D2CROSS_PHI

    if not d2cross_phi.endswith('.npz'):
        d2cross_phi = _DPFE_D2CROSS_PHI[d2cross_phi]

    # --------------
    # anisotropic
    # --------------

    # ------------------------
    # integrated cross-section

    demiss, ddist, d2cross_phi = tfphysemis.get_xray_thin_integ_dist(
        # ----------------
        # cross-section
        # tabulated d2cross_phi
        d2cross_phi=d2cross_phi,
        # d2cross_phi computation
        E_ph_eV=None,
        E_e0_eV=None,
        E_e0_eV_npts=None,
        theta_e0_vsB_npts=None,
        phi_e0_vsB_npts=None,
        theta_ph_vsB=None,
        # integration
        integration=integration,
        # -----------
        # verb
        debug=False,
        verb=True,
        # ----------------
        # electron distribution
        **ddist,
    )

    # add check units
    units0 = dplasma['emiss_tot']['ff']['units']
    units1 = demiss['emiss']['maxwell']['emiss']['units']
    assert asunits.Unit(units0) == asunits.Unit(units1)
    units = units0

    # --------------
    # shapes
    # --------------

    shape_Te = dplasma['common']['Te']['data'].shape
    shape_plasma = ddist['plasma']['Te_eV']['data'].shape
    assert shape_Te[0] == shape_plasma[1]

    shape_emiss_anis = demiss['emiss']['maxwell']['emiss']['data'].shape
    nEph1 = demiss['E_ph_eV']['data'].size
    ntheta = demiss['theta_ph_vsB']['data'].size
    shape_check = (nEkin, shape_Te[0], jp_fraction_re.size, nEph1, ntheta)
    assert shape_emiss_anis == shape_check

    # ----------------------
    # add Zeff to emiss_anis
    # ----------------------

    # reshape Zeff
    Zeff = dplasma['common']['Zeff']['data'][None, :, None, None, None]

    # add
    for k0, v0 in demiss['emiss'].items():
        demiss['emiss'][k0]['emiss']['data'] = v0['emiss']['data'] * Zeff

    # --------------
    # uniformize
    # --------------

    E_ph0 = dplasma['common']['E_photon']['data']
    E_ph1 = demiss['E_ph_eV']['data']
    ilow = E_ph1 < E_ph0.min()
    iup = E_ph1 > E_ph0.max()
    E_ph = np.r_[E_ph1[ilow], E_ph0, E_ph1[iup]]

    shape_new_anis = shape_emiss_anis[:-2] + (E_ph.size, ntheta)
    shape_new_iso = (1, shape_Te[0], 1, E_ph.size, 1)

    iin = np.r_[
        np.zeros(ilow.sum(), dtype=bool),
        np.ones(E_ph0.size, dtype=bool),
        np.zeros(iup.sum(), dtype=bool),
    ]
    iout = ~iin

    # ---------------------
    # isotropic - 0-padding

    diso = {}
    sli_in = (slice(None),) * 3 + (iin, slice(None))
    sli_broad = (None, slice(None), None, slice(None), None)
    for ff in ['ff', 'fb', 'bb']:
        diso[ff] = np.zeros(shape_new_iso, dtype=float)
        diso[ff][sli_in] = dplasma['emiss_tot'][ff]['data'][sli_broad]

    # ---------------------
    # anisotropic: interpolate

    danis = {}
    sli_out = (slice(None),) * 3 + (~iin, slice(None))
    iedges = (ilow | iup)
    sli_edges = (slice(None),) * 3 + (iedges, slice(None))
    for k0, v0 in demiss['emiss'].items():

        # tabulated
        danis[k0] = np.zeros(shape_new_anis, dtype=float)
        danis[k0][sli_out] = v0['emiss']['data'][sli_edges]

        # interpolated
        iok = v0['emiss']['data'] > 0.
        for ind in np.ndindex(v0['emiss']['data'].shape[:-2]):

            sli = ind + (slice(None), slice(None))
            iok = np.all(v0['emiss']['data'][sli] > 0, axis=-1)
            if not np.any(iok):
                continue
            sli = ind + (iok, slice(None))

            sli_inii = ind + (iin, slice(None))

            danis[k0][sli_inii] = np.power(
                10,
                scpinterp.make_interp_spline(
                    np.log10(E_ph1[iok]),
                    np.log10(v0['emiss']['data'][sli]),
                    k=1,
                    axis=-2,
                    check_finite=True,
                )(np.log10(E_ph0)),
            )

    # --------------
    # output
    # --------------

    demiss = {
        'emiss': {
            'maxwell': {
                'ff': {
                    'data': danis['maxwell'],
                    'units': units,
                },
                'ff_iso': {
                    'data': diso['ff'],
                    'units': units,
                },
                'fb': {
                    'data': diso['fb'],
                    'units': units,
                },
                'bb': {
                    'data': diso['bb'],
                    'units': units,
                },
            },
            'RE': {
                'ff': {
                    'data': danis['RE'],
                    'units': units,
                },
            },
        },
        'E_ph': {
            'data': E_ph,
            'units': 'eV',
        },
        'theta_ph_vsB': demiss['theta_ph_vsB'],
        'Te': ddist['plasma']['Te_eV'],
        'ne': ddist['plasma']['ne_m3'],
        'jp_fraction_re': ddist['plasma']['jp_fraction_re'],
        'Ekin_max_eV': ddist['plasma']['Ekin_max_eV'],
    }

    return demiss, ddist
