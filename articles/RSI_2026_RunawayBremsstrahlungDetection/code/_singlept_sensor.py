import copy


import numpy as np
import scipy.integrate as scpinteg
import matplotlib.path as mpath
import astropy.units as asunits
import tofu as tf


from ._fig11_hxrva_cad import _PFE_CONFIG, _PFE_COLL, _add_ptcam
from . import _perfs


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_RE = ['avalanche 10 MeV', 'dreicer']
_JP_FRAC = 0.9
_TE = 0.5e3
_RE = 0.01


_FONTSIZE = 14


# #######################################
# #######################################
#           Main
# #######################################


def main(
    # coll
    coll=None,
    key_cam=None,
    config=None,
    res=None,
    # responsivity
    key_resp=None,
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
    # ptcam
    angle0=None,
    angle1=None,
    # unused
    **kwdargs,
):

    # -----------
    # inputs
    # -----------

    if res is None:
        res = _RES

    if Te_eV is None:
        Te_eV = _TE

    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    if re is None:
        re = _RE
    if isinstance(re, str):
        re = [re]

    if key_resp is None:
        key_resp = 'cvd_filter'

    # --------------
    # compute
    # --------------

    demiss_integ = {}
    for rei in re:
        (
            demiss_integ[rei], dsignal, ddist, dmix,
            total_headon, diff_RE, diff_max,
            dang, theta,
            lresp, ldist,
        ) = _perfs.main(
            dmix=dmix,
            ne_m3=ne_m3,
            jp_Am2=jp_Am2,
            d2cross_phi=d2cross_phi,
            re=rei,
            jp_fraction_re=jp_fraction_re,
        )

    # --------------
    # extract
    # --------------

    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    neu = np.unique(ddist['plasma']['ne_m3']['data'])
    jpu = np.unique(ddist['plasma']['jp_Am2']['data'])
    jp_fracu = np.unique(ddist['plasma']['jp_fraction_re']['data'])
    assert neu.size == jpu.size == jp_fracu.size == 1

    indTe = np.argmin(np.abs(Teu - Te_eV))
    sli_Te = (indTe, 0, slice(None))
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
        config = tf.load(config)

    # --------------
    # load cameras
    # --------------

    # coll
    if coll is None:
        coll = _PFE_COLL
    if isinstance(coll, str):
        coll = tf.data.load(coll)

    # --------------
    # add pt_cam rays
    # --------------

    dkrays, angle0, angle1 = _add_ptcam(
        coll=coll,
        key_cam=key_cam,
        angle0=angle0,
        angle1=angle1,
        config=config,
    )
    assert len(dkrays) == 2

    # --------------
    # get angle vs B
    # --------------

    msg = "\nCompute angles vs B..."
    print(msg)

    dangles, axis_samp = _get_angles_vs_B(
        coll=coll,
        cases=cases,
        dkrays=dkrays,
        res=res,
        t=t,
    )

    # ################
    # interpolate ne
    # ################

    dne_coef = {}
    for kcam, krays in dkrays.items():

        R = dangles[krays]['helicity']['R']['data']
        Z = dangles[krays]['helicity']['Z']['data']
        kmR = [kk for kk in coll.ddata.keys() if kk.endswith('_magaxR')][0]
        kmZ = [kk for kk in coll.ddata.keys() if kk.endswith('_magaxZ')][0]
        magR = coll.ddata[kmR]['data']
        magZ = coll.ddata[kmZ]['data']

        ksepR = [kk for kk in coll.ddata.keys() if kk.endswith('_sepR')][0]
        ksepZ = [kk for kk in coll.ddata.keys() if kk.endswith('_sepZ')][0]
        sepR = coll.ddata[ksepR]['data']
        sepZ = coll.ddata[ksepZ]['data']

        sli = (slice(None),) + (None,)*R.ndim
        ne_coef = np.exp(
            - (R[None, ...] - magR[sli])**2/0.5**2
            - (Z[None, ...] - magZ[sli])**2/0.7**2
        )

        if t is not None:
            kt = f"{kmR.split('_')[0]}_t"
            indt = np.argmin(np.abs(coll.ddata[kt]['data'] - t))
            ne_coef = ne_coef[indt, ...]
            sepR = sepR[indt, ...]
            sepZ = sepZ[indt, ...]
            sep = mpath.Path(np.array([sepR, sepZ]).T)
        dne_coef[krays] = ne_coef

    # ################
    # interpolate emissivity
    # ################

    msg = "Interpolate emissivity..."
    print(msg)

    dsig_los = {
        kcase: {
            krays: {
                rei: {kdist: {} for kdist in ['maxwell', 'RE']}
                for rei in re
            }
            for krays in dkrays.values()
        }
        for kcase in cases.keys()
    }
    for kcase, vcase in cases.items():
        for kcam, krays in dkrays.items():

            # ------------------
            # find closest angle

            khel = 'helicity' if vcase['helicity'] is True else 'nohelicity'
            angles = dangles[krays][khel]['angle']['data']
            ref_ang = dangles[krays][khel]['angle']['ref']
            ref = tuple([
                rr for ii, rr in enumerate(ref_ang) if ii != axis_samp
            ])

            # ------------------
            # find closest angle

            sli = (None,) * angles.ndim + (slice(None),)
            delta = angles[..., None] - theta[sli]
            iang = np.argmin(np.abs(delta), axis=-1)
            shape = tuple([
                ss for ii, ss in enumerate(iang.shape)
                if ii != axis_samp
            ])

            # ----------
            # out of sep

            R = dangles[krays][khel]['R']['data']
            Z = dangles[krays][khel]['Z']['data']
            iokRZ = np.isfinite(R)
            pts = np.array([R[iokRZ], Z[iokRZ]]).T
            iin = np.zeros(R.shape, dtype=bool)
            iin[iokRZ] = sep.contains_points(pts)
            R[~iin] = np.nan

            iok = np.isfinite(angles) & np.isfinite(R)
            length = dangles[krays][khel]['length']['data']
            length[~iok] = np.nan
            ndimd = iang.ndim - length.ndim
            sli_ndimd = (slice(None),)*ndimd

            # -----------
            # units, ref

            for rei in re:
                for kdist, vdist in demiss_integ[rei][key_resp].items():
                    for kemiss, vemiss in vdist.items():
                        emiss = vemiss['data'][sli_Te]

                        # ne profile
                        if vcase['ne'] is True:
                            lengthi = length * dne_coef[krays]
                        else:
                            lengthi = length

                        # units + initialize
                        units_emiss = asunits.Unit(vemiss['units'])
                        emiss_los = np.zeros(shape, dtype=float)
                        if kemiss in ['fb', 'bb']:
                            emiss_los[...] = (
                                emiss * np.nansum(lengthi, axis=0)
                            )[None, ...]
                        else:
                            for ind in np.ndindex(lengthi.shape[1:]):
                                ioki = iok[(slice(None),) + ind]
                                if not np.any(ioki):
                                    continue
                                ll = lengthi[(ioki,) + ind]
                                sli_emiss = sli_ndimd + (ioki,) + ind
                                sli_los = sli_ndimd + ind
                                emiss_los[sli_los] = scpinteg.trapezoid(
                                    emiss[iang[sli_emiss]],
                                    x=ll,
                                    axis=axis_samp,
                                )

                        dsig_los[kcase][krays][rei][kdist][kemiss] = {
                            'data': emiss_los,
                            'units': units_emiss * asunits.Unit('m'),
                            'ref': ref,
                        }

                        # store
                        coll.add_data(
                            key=f"{kcase}_{krays}_{rei}_{kdist}_{kemiss}",
                            **dsig_los[kcase][krays][rei][kdist][kemiss],
                        )

    # ################
    # merits
    # ################

    kcase = [
        kcase for kcase, vcase in cases.items()
        if all([vv for vv in vcase.values()])
    ][0]

    lkrays = sorted(dsig_los[kcase].keys())
    krays_mean = [
        np.nanmean(dsig_los[kcase][kk][re[0]]['RE']['ff']['data'])
        for kk in lkrays
    ]
    krays_max = lkrays[np.argmax(krays_mean)]
    krays_min = lkrays[np.argmin(krays_mean)]

    dmetrics = {}
    for rei in re:

        # RE
        RE_headon = dsig_los[kcase][krays_max][rei]['RE']['ff']['data']
        RE_back = dsig_los[kcase][krays_min][rei]['RE']['ff']['data']
        diffRE = RE_headon - RE_back[::-1, :]

        # Max
        kd = 'maxwell'
        Max_headon_ff = dsig_los[kcase][krays_max][rei][kd]['ff']['data']
        Max_back_ff = dsig_los[kcase][krays_min][rei][kd]['ff']['data']
        diffMax = Max_headon_ff - Max_back_ff[::-1, :]

        if np.any(diffMax < 0) or np.any(diffRE < 0):
            import pdb; pdb.set_trace() # DB
            diffRE[diffRE < 0] = 0
            diffMax[diffMax < 0] = 0

        Max_headon_tot = (
            dsig_los[kcase][krays_max][rei][kd]['ff']['data']
            + dsig_los[kcase][krays_max][rei][kd]['fb']['data']
            + dsig_los[kcase][krays_max][rei][kd]['bb']['data']
        )
        total_headon = Max_headon_tot + RE_headon

        # metrics
        units = dsig_los[kcase][krays_max][rei]['RE']['ff']['units']
        ref = dsig_los[kcase][krays_max][rei]['RE']['ff']['ref']
        dmetrics[rei] = {
            'xi': {
                'data': diffRE / total_headon,
                'units': '',
                'ref': ref,
            },
            'kappa': {
                'data': diffRE / (diffRE + diffMax),
                'units': '',
                'ref': ref,
            },
            'meas_RE_diff': {
                'data': diffRE,
                'units': units,
                'ref': ref,
            },
            'meas_RE_headon': {
                'data': RE_headon,
                'units': units,
                'ref': ref,
            },
            'meas_headon': {
                'data': total_headon,
                'units': units,
                'ref': ref,
            },
        }

    return (
        coll, config,
        dangles, dsig_los, dmetrics,
        krays_max, krays_min,
        angle0, angle1,
        ne, jp, jp_frac, Te_eV,
    )


# ############################################
# ############################################
#       angles vs B
# ############################################


def _get_angles_vs_B(
    coll=None,
    cases=None,
    dkrays=None,
    res=None,
    t=None,
):

    # ----------------
    # keys of interest
    # ----------------

    # B-field keys
    kBR, kBZ, kBphi = [
        [kk for kk in coll.ddata.keys() if kk.endswith(f'_B{k0}')][0]
        for k0 in ['R', 'Z', 'phi']
    ]

    # Sep
    ksepR, ksepZ = [
        [kk for kk in coll.ddata.keys() if kk.endswith(f'_sep{k0}')][0]
        for k0 in ["R", "Z"]
    ]

    # time
    kt = f"{kBR.split('_')[0]}_t"

    # ----------------
    # loop on cases / rays
    # ----------------

    dangles = {
        krays: {'helicity': {}, 'nohelicity': {}}
        for krays in dkrays.values()
    }

    for kcam, krays in dkrays.items():

        # ---------------------------------------
        # compute angles along rays - helicity

        dout = coll.get_rays_angle_vs_vect(
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
        )[krays]

        # select time
        if t is not None:
            indt = np.argmin(np.abs(coll.ddata[kt]['data'] - t))
            t = coll.ddata[kt]['data'][indt]

            dout['angle']['data'] = dout['angle']['data'][indt, ...]
            dout['angle']['ref'] = dout['angle']['ref'][1:]

        dangles[krays]['helicity'] = dout

        # ---------------------------------------
        # compute angles along rays - helicity

        phi = dangles[krays]['helicity']['phi']['data']
        ephix = -np.sin(phi)
        ephiy = np.cos(phi)
        if np.nanmean(coll.ddata[kBphi]['data']) < 0:
            ephix, ephix = -ephix, -ephix
        vx, vy, vz = coll.get_rays_vect(krays)
        cos = ephix * (-vx) + ephiy * (-vy)
        ang = np.arccos(cos)
        if ang.ndim < dout['angle']['data'].ndim:
            nt = dout['angle']['data'].shape[0]
            ang = np.array([ang for ii in range(nt)])
        ang[~np.isfinite(dout['angle']['data'])] = np.nan

        dangles[krays]['nohelicity'] = copy.deepcopy(dout)
        dangles[krays]['nohelicity']['angle']['data'] = ang

        # ------------
        # add missing ref

        axis_samp = dout['angle']['ref'].index(None)

        kref = f'{krays}_nsamp0'
        coll.add_ref(
            key=kref,
            size=ang.shape[axis_samp],
        )

        # ------------
        # store

        ref = dout['angle']['ref']
        ref[ref.index(None)] = kref

        key = f"{krays}_helicity"
        coll.add_data(
            key=key,
            data=dangles[krays]['helicity']['angle']['data'],
            ref=tuple(ref),
            units=dangles[krays]['helicity']['angle']['units'],
            dim=dangles[krays]['helicity']['angle']['dim'],
        )

        key = f"{krays}_nohelicity"
        coll.add_data(
            key=key,
            data=dangles[krays]['nohelicity']['angle']['data'],
            ref=tuple(ref),
            units=dangles[krays]['nohelicity']['angle']['units'],
            dim=dangles[krays]['nohelicity']['angle']['dim'],
        )

    return dangles, axis_samp
