import os


import numpy as np
import scipy.constants as scpct


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')


# SPECTRAL MODELLING FILES
_LTYPES = ['_SCRAM86_']   # '_CHIANTI_', '_FLYCHK_'
_LPFE_SPECT = [
    ff for ff in os.listdir(_PATH_INPUTS)
    if ff.endswith('_data.npz')
    and any([ss.lower() in ff.lower() for ss in _LTYPES])
]
_DPFE_SPECT = {
    ff.split('_')[-2]: os.path.join(_PATH_INPUTS, ff)
    for ff in _LPFE_SPECT
}


# PLASMA
_NE = 1e19  # /m3


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    dmix=None,
    ne_m3=None,
):
    """ Extract ff, fb, bb emissivities from SCRAM / FLYCHK / CHIANTY data

    Uses user-provided ne_m3 and concentrations (via dmix)
    Uses the Te values foudn in the source files

    Returns dplasma:
    {
        'concentration': {'H': float / array, 'Ni': float / array, ...},
        'emiss': {
            'H': {
                'ff': {'data': array, 'units': str}},
                'fb': {'data': array, 'units': str}},
                'bb': {'data': array, 'units': str}},
            },
            ...
            'Ni': {
                'ff': {'data': array, 'units': str}},
                'fb': {'data': array, 'units': str}},
                'bb': {'data': array, 'units': str}},
            },
        },
        'emiss_tot': {
            'ff': {'data': array, 'units': str}},
            'fb': {'data': array, 'units': str}},
            'bb': {'data': array, 'units': str}},
        },
        'common': {
            'Te': {'data': array, 'units': str},
            'ne': {'data': array, 'units': str},
            'E_photon': {'data': array, 'units': str},
        },
    }

    """

    # --------------
    # inputs
    # --------------

    dmix = _check_mix(dmix)

    # --------------
    # load files
    # --------------

    dfiles = {
        k0: np.load(_DPFE_SPECT[k0], allow_pickle=True)['arr_0'].tolist()
        for k0 in dmix.keys()
    }

    # ---------------
    # Extract common
    # ---------------

    # Extract Te
    lk = list(dmix.keys())
    lc = ['Te', 'E_photon']
    dcommon = {cc: None for cc in lc}
    for cc in lc:
        ref = dfiles[lk[0]][cc]['data']
        rshape = ref.shape
        for kk, vv in dfiles.items():
            cc_shape = vv[cc]['data'].shape
            if vv[cc]['data'].shape != ref.shape:
                msg = (
                    "Different shapes!\n"
                    f"\t- dfiles['{kk}']['{cc}']['data'].shape: {cc_shape}\n"
                    f"\t- dfiles['{lk[0]}']['{cc}']['data'].shape: {rshape}\n"
                )
                raise Exception(msg)
            if not np.allclose(vv[cc]['data'], ref):
                msg = (
                    f"Common field '{cc}' not constant accross files!\n"
                    f"\t- {ref}\n"
                    f"\t- {vv[cc]['data']}\n"
                )
                raise Exception(msg)
        dcommon[cc] = dfiles[lk[0]][cc]

    # ne_m3 vs Te
    ne_m3, dcommon['Te']['data'] = _check_neTe(ne_m3, dcommon['Te']['data'])
    dcommon['ne'] = {
        'data': ne_m3,
        'units': '1/m3',
    }

    # --------------
    # shapes
    # --------------

    shape_Te = ne_m3.shape
    shape_conc = dmix[lk[0]].shape
    try:
        shape_plasma = np.broadcast_shapes(shape_Te, shape_conc)
        ne_m3 = np.broadcast_to(ne_m3, shape_plasma)
        for k0, v0 in dmix.items():
            dmix[k0] = np.broadcast_to(v0, shape_plasma)
    except Exception:
        msg = "Te and concentrations should be broadcastcable!"
        raise Exception(msg)

    shape_spect = dcommon['E_photon']['data'].shape
    shape_emiss = shape_plasma + shape_spect

    sli_E = (None,) * len(shape_plasma) + (slice(None),)
    sli_ne = (slice(None),) * len(shape_plasma) + (None,)

    # --------------
    # Zeff
    # --------------

    # Zeff
    Zeff = np.zeros(ne_m3.shape)
    sli_Z = (None,) * len(shape_plasma) + (slice(None),)
    sli_cc = (slice(None),) * len(shape_plasma) + (None,)
    for k0, v0 in dfiles.items():
        zz = np.arange(0, v0['Xz']['data'].shape[-1])
        fz = v0['Xz']['data']
        Zeff += np.sum(dmix[k0][sli_cc] * fz * zz[sli_Z]**2, axis=-1)

    dcommon['Zeff'] = {'data': Zeff, 'units': None}

    # --------------
    # dplasma
    # --------------

    lemiss = ['ff', 'fb', 'bb']
    dplasma = {
        'concentration': dmix,
        'emiss': {k0: {cc: None for cc in lemiss} for k0 in dmix.keys()},
        'common': dcommon,
        'Zmean': {k0: v0['<Z>'] for k0, v0 in dfiles.items()},
    }

    # --------------
    # load elements
    # --------------

    units0 = 'J*cm^3/s/eV/atom/electron'
    units = "1 / (m3.s.eV.sr)"
    E_ph = dcommon['E_photon']['data']
    for k0 in dmix.keys():

        # find emissivity key
        for ke in lemiss:
            lk = [kk for kk in dfiles[k0].keys() if f"emis_{ke}" in kk]
            if len(lk) == 1:
                kemiss = lk[0]
            elif len(lk) > 1:
                kchianti = [kk for kk in lk if 'chianti' in kk.lower()]
                if len(kchianti) > 0:
                    kemiss = kchianti[0]
                else:
                    msg = "Not sure how to choose between {lk}"
                    raise Exception(msg)
            else:
                msg = f"key 'emis_{ke}' not found in {k0}"
                raise Exception(msg)

            # check units
            uu = str(dfiles[k0][kemiss]['units']).replace('$', '')
            assert uu == units0

            # concentration
            cc = np.broadcast_to(dplasma['concentration'][k0], shape_plasma)

            # emiss
            emiss = dfiles[k0][kemiss]['data'].squeeze()

            # ph / m3 / s / eV / sr
            emiss = (
                np.broadcast_to(emiss, shape_emiss)
                * 1e-6    # cm3 => m3
                / (E_ph[sli_E] * scpct.e)  # J => ph
                * ne_m3[sli_ne]**2  # /electron => /m3
                * cc[sli_ne]  # /atom => /m3
                / (4*np.pi)  # => /sr
            )

            dplasma['emiss'][k0][ke] = {
                'data': emiss,
                'units': units,
                'ref': 'plasma.shape + (nE,)',
            }

    # -----------
    # total
    # -----------

    dplasma['emiss_tot'] = {
        ke: {
            'data': np.sum(
                [dplasma['emiss'][k0][ke]['data'] for k0 in dmix.keys()],
                axis=0,
            ),
            'units': units,
        }
        for ke in lemiss
    }

    return dplasma, dmix


# #####################################################
# #####################################################
#       Check
# #####################################################


def _check_mix(
    dmix=None,
):

    # -----------
    # dmix
    # -----------

    # None
    if dmix is None:
        dmix = 'H'

    # str
    if isinstance(dmix, str):
        dmix = {dmix: 1}

    # ---------
    # each dict

    dfail = {}
    lok = sorted(_DPFE_SPECT.keys())
    if isinstance(dmix, dict):

        # check each element
        for k0, v0 in dmix.items():

            # key
            if not (isinstance(k0, str) and k0 in lok):
                dfail[k0] = f"key not allowed {lok}"
                continue

            # value
            v0 = np.atleast_1d(v0)
            iok = np.isfinite(v0)
            iok[iok] = (v0[iok] >= 0) & (v0[iok] <= 1)
            if not np.all(iok):
                dfail[k0] = "non-finite or values not in [0, 1]"
                continue
            dmix[k0] = v0

    else:
        dfail['dmix'] = "not a dict!"

    # -------------------
    # overall consistency

    if len(dfail) == 0 and len(dmix) > 1:

        # broadcastable
        try:
            shape = np.broadcast_shapes(*[v0.shape for v0 in dmix.values()])

            # broadcast
            for k0, v0 in dmix.items():
                dmix[k0] = np.broadcast_to(v0, shape)

        except Exception:
            dfail['dmisc'] = "values not broadcastable!"

    # -------------
    # sum = 1

    total = np.sum([v0 for v0 in dmix.values()], axis=0)
    for k0, v0 in dmix.items():
        dmix[k0] = v0 / total

    # -----------
    # any error

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "\nArg 'dmix' must be a dict with of the form:\n"
            "\t- {'H': cH, 'Ni': cNi, ...}\n"
            "where:\n"
            "\t- each key is an element\n"
            "\t- each value is a concentration float / array in [0, 1]\n"
            "All values must be broadcastable together\n"
            "All values will be normalized such that their sum = 1\n"
            "\nIdentified issues:\n"
            + "\n".join(lstr)
            + f"\nProvided:\n{dmix}\n"
        )
        raise Exception(msg)

    return dmix


def _check_neTe(
    ne_m3=None,
    Te=None,
):

    # -----------
    # ne, Te
    # -----------

    # ne
    if ne_m3 is None:
        ne_m3 = _NE
    ne_m3 = np.atleast_1d(ne_m3)

    # Te
    if Te is None:
        Te = _TE
    Te = np.atleast_1d(Te)

    # -------------
    # broadcastable

    return np.broadcast_arrays(ne_m3, Te)
