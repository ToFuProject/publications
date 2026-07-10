import os
import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds
import tofu as tf


from ._load_spect import _PATH_INPUTS
from ._load_spect_anis import _JP_FRAC
from . import _perfs
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_PFE_CONFIG = os.path.join(
    _PATH_INPUTS,
    'TFG_Config_ExpSPARC_SPARC-V2_sh00000_Vers1.8.18.npz',
)
_PFE_COLL = os.path.join(
    _PATH_INPUTS,
    'Inversion_HXRVA_ptcam_dvezinet_20260709-204104.npz',
)



_RE = 'avalanche 10 MeV'
_JP_FRACT = 0.9
_TE = 1e3


_CASES = {
    0: {
        'helicity': False,
        'pitch': False,
        'ne': False,
    },
    1: {
        'helicity': True,
        'pitch': False,
        'ne': False,
    },
    2: {
        'helicity': True,
        'pitch': True,
        'ne': False,
    },
    3: {
        'helicity': True,
        'pitch': True,
        'ne': True,
    },
}


# #######################################
# #######################################
#           Main
# #######################################


def main(
    # coll
    coll=None,
    config=None,
    res=None,
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
    # assumptions
    cases=None,
    # unused
    **kwdargs,
):

    # -----------
    # inputs
    # -----------

    if Te_eV is None:
        Te_eV = _TE

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    if re is None:
        re = _RE

    if cases is None:
        cases = _CASES

    # --------------
    # compute
    # --------------

    (
        demiss_integ, dsignal, ddist,
        total_headon, diff_RE, diff_max,
        dang, theta,
        lresp, ldist,
    ) = _perfs.main(
        dmix=dmix,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        d2cross_phi=d2cross_phi,
        re=re,
        jp_fraction_re=jp_fraction_re,
    )

    # --------------
    # extract
    # --------------

    Teu = np.unique(ddist['plasma']['Te']['data'])
    neu = np.unique(ddist['plasma']['Te']['data'])
    jpu = np.unique(ddist['plasma']['Te']['data'])
    jp_fracu = np.unique(ddist['plasma']['Te']['data'])
    assert neu.size == jpu.size == jp_fracu.size == 1

    indTe = np.argmin(np.abs(Teu - Te_eV))
    sli_Te = (0, indTe, 0)
    Te_eV = Teu[indTe]
    ne = neu[0]
    jp = jpu[0]
    jp_frac = jp_fracu[0]

    # --------------
    # load coll
    # --------------

    # config
    if config is None:
        config = _PFE_CONFIG
    if isinstance(config, str):
        config = tf.data.load(config)

    # coll
    if coll is None:
        coll = _PFE_COLL
    if isinstance(coll, str):
        coll = tf.data.load(coll)

    # --------------
    # get rays
    # --------------

    wrays = coll._which_rays
    krays = [
        kk for kk, vv in coll.dobj[wrays].items()
        if vv['angles'] is not None
    ]
    assert len(krays) == 1
    krays = krays[0]

    # --------------
    # get angle vs B
    # --------------

    msg = "\nCompute angles vs B..."
    print(msg)

    # B-field keys
    kBR, kBZ, kBphi = [
        [kk for kk in coll.ddata.keys() if kk.endswith(f'_2dB{k0}')][0]
        for k0 in ['R', 'Z', 'phi']
    ]

    for kcase, vcase in cases.items():

        if vcase['helicity'] is True:
            # compute angles along rays
            dangle = coll.get_rays_angle_vs_vect(
                # rays
                key_rays=krays,
                res=res,
                segment=-1,
                # vector components
                key_XR=kBR,
                key_YZ=kBZ,
                key_Zphi=kBphi,
                geometry='toroidal',
                # separatrix
                key_sepR=True,
                key_sepZ=True,
                # verb
                verb=None,
            )

        else:
            dangle = None

    # ------------
    # safety check

    lcam = ['CCW_ptcam', 'CW_ptcam']
    n0 = dangle[lcam[0]]['angle']['data'].shape[1]
    n1 = dangle[lcam[1]]['angle']['data'].shape[1]

    if np.abs(n0 - n1) == 1:
        nmin = min(n0, n1)
        sli = (slice(None), slice(nmin), slice(None), slice(None))
        for kcam in lcam:
            for k0, v0 in dangle[kcam].items():
                if v0['data'].ndim == 4:
                    slii = sli
                else:
                    slii = sli[1:]
                dangle[kcam][k0]['data'] = v0['data'][slii]

    elif np.abs(n0 - n1) > 1:
        msg = "More than 1 pt of LOS sampling difference !"
        raise Exception(msg)

    # store
    for kcam, vcam in dangle.items():
        kref = f'{kcam}_samp_npts'
        coll.add_ref(kref, size=vcam['R']['data'].shape[0])
        for kk in vcam.keys():
            ref = list(dangle[kcam][kk]['ref'])
            ref[ref.index(None)] = kref
            dangle[kcam][kk]['ref'] = ref
            dangle[kcam][kk]['key'] = f"{kcam}_samp_{kk}"
            coll.add_data(
                **dangle[kcam][kk],
            )






    # ################
    # piggyback
    # ################

    # --------------
    # get keys of plasma quantities
    # --------------

    dplasma = _get_plasma_quantities(
        coll=coll,
        dangle=dangle,
    )

    # store
    for kcam, vcam in dplasma.items():
        for k0, v0 in vcam.items():
            coll.add_data(**v0)

    # ------------------
    # add jp_fraction_re
    # ------------------

    _add_jp_fraction_re(
        coll=coll,
        dplasma=dplasma,
        jp_fraction_re=jp_fraction_re,
    )

    # ------------------
    # add ne_m3_re for Maxwell RE
    # ------------------

    _add_neTe_re(
        coll=coll,
        dplasma=dplasma,
        ne_m3_re_fraction=ne_m3_re_fraction,
        Te_eV_re=Te_eV_re,
    )

    # ------------------
    # add nZ_m3
    # ------------------

    _add_nZ_m3(
        coll=coll,
        dplasma=dplasma,
    )

    # --------------
    # get demiss
    # --------------

    ldist, dbins, dindu = _get_demiss(
        dplasma=dplasma,
        coll=coll,
        dangle=dangle,
        extrapolate=extrapolate,
        plot_responsivity_integration=plot_responsivity_integration,
        pfe_d2cross_phi=pfe_d2cross_phi,
        # RE-specific
        Zeff=Zeff,
        Ekin_max_eV=Ekin_max_eV,
        Efield_par_Vm=Efield_par_Vm,
        lnG=lnG,
        sigmap=sigmap,
        Te_eV_re=Te_eV_re,
        dominant=dominant,
        # debug
        debug=debug_emissivity,
    )

    # --------------
    # get diff, total
    # --------------

    _add_diff_total(
        coll=coll,
        ldist=ldist,
        lcam=lcam,
    )

    # ---------------
    # get region max
    # ---------------

    _get_max_area(
        coll=coll,
        lcam=lcam,
        ldist=ldist,
    )

    # ---------------
    # get reverse_ray-tracing
    # ---------------

    _reverse_ray_tracing(
        coll=coll,
        lcam=lcam,
        pitch=pitch,
    )

    # -------------------------
    # plot total signal at t
    # -------------------------

    for tt in plot_t:
        _plot._plot_tot_indt(
            coll=coll,
            run=run,
            res=res,
            jp_fraction_re=jp_fraction_re,
            thick=thick,
            lcam=lcam,
            plot_t=tt,
            dbins=dbins,
            dindu=dindu,
        )

    # ---------------
    # plot time traces
    # ---------------

    dax_time_traces = _plot._plot_time_traces(
        coll=coll,
        lcam=lcam,
        run=run,
        thick=thick,
        jp_fraction_re=jp_fraction_re,
        dax=dax_time_traces,
        color=color_time_traces,
        plot_t=plot_t,
    )































    # --------------
    # store
    # --------------

    dimage = {}

    return dimage
