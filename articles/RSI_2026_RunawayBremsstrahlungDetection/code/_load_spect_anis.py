import os


import numpy as np
import astropy.units as asunits


from ._load_spect import main as load_spect


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

    dplasma = load_spect(
        dmix=dmix,
        ne_m3=ne_m3,
    )

    # --------------
    # anisotropic
    # --------------



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
