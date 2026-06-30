import os


import numpy as np
import astropy.units as asunits
import tofu as tf


tfphysemis = tf.physics_tools.electrons.emission


from ._load_spect import main as load_spect
from ._fig02_dist import _DDIST
from ._fig04_bremsstrahlung import _TE, _JP_FRAC, _EKIN_MAX_EV, _PNORMW


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    dmix=None,
    ne_m3=None,
    # ddist
    Te_eV=None,
    jp_fraction_re=None,
    Ekin_max_eV=None,
    pnormW=None,
    # d2cross_phi
    d2cross_phi=None,
):
    """ For a given impurity mix and ne, Te, return a dict of emissivities

    Emissivities are broken down into 2 distributions
    - Maxwell:
        - ff: anisotropic
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

    Te = dplasma['common']['Te']['data']

    # ---------------
    # get ion charges

    # coll = sp.Collection()
    # coll.add_ion(k0)
    Z_eff = None

    # ------------
    # inputs
    # ------------

    if Te_eV is None:
        Te_eV = _TE

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    if Ekin_max_eV is None:
        Ekin_max_eV = _EKIN_MAX_EV

    if pnormW is None:
        pnormW = _PNORMW

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
    ddist['Ekin_max_eV'] = np.atleast_1d(ddist['Ekin_max_eV'])[:, None, None]
    ddist['pnormW'] = np.atleast_1d(ddist['pnormW'])[:, None, None]
    if ddist['pnormW'].size != ddist['Ekin_max_eV'].size:
        raise Exception()
    ddist['Te_eV'] = Te_eV[None, :, None]
    ddist['jp_fraction_re'] = jp_fraction_re[None, None, :]

    nEkin = ddist['Ekin_max_eV'].shape[0]

    # ------------
    # safety check

    assert np.allclose(Te, ddist['Te_eV'].ravel())

    # --------------
    # integrated cross-section
    # --------------

    import pdb; pdb.set_trace()

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
        E_ph_eV=E_ph_eV,
        E_e0_eV=None,
        E_e0_eV_npts=None,
        theta_e0_vsB_npts=None,
        phi_e0_vsB_npts=None,
        theta_ph_vsB=None,
        # -----------
        # verb
        debug=False,
        verb=True,
        # ----------------
        # electron distribution
        **ddist,
    )

    # --------------
    # output
    # --------------

    demiss = {
        'emiss': {
            'maxwell': {
                'ff': {},
                'fb': dplasma['emiss_tot']['fb'],
                'bb': dplasma['emiss_tot']['bb'],
            },
            'RE': {
                'ff': {},
            },
        },
    }

    return demiss
