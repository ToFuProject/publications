import os
import copy
import string


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.integrate as scpinteg
import astropy.units as asunits
import datastock as ds
import tofu as tf


from ._load_spect import _PATH_INPUTS
from . import _perfs
# from ._savefig import main as savefig


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
    'Inversion_HXRVA_dvezinet_20260713-154229.npz',
)


_RE = ['avalanche 10 MeV', 'dreicer']
_JP_FRAC = 0.9
_TE = 0.5e3
_T = 5    # 5


_CASES = {
    0: {
        'helicity': False,
        'pitch': True,
        'ne': False,
    },
    # 1: {
        # 'helicity': True,
        # 'pitch': False,
        # 'ne': False,
    # },
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
    # plot
    dmargin_RE=None,
    figsize_RE=None,
    fontsize=None,
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
    if isinstance(re, str):
        re = [re]

    if cases is None:
        cases = _CASES

    if key_resp is None:
        key_resp = 'cvd_filter'

    # --------------
    # compute
    # --------------

    demiss_integ = {}
    for rei in re:
        (
            demiss_integ[rei], dsignal, ddist,
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

        sli = (slice(None),) + (None,)*R.ndim
        ne_coef = np.exp(
            - (R[None, ...] - magR[sli])**2/0.5**2
            - (Z[None, ...] - magZ[sli])**2/0.7**2
        )

        if t is not None:
            kt = f"{kmR.split('_')[0]}_t"
            indt = np.argmin(np.abs(coll.ddata[kt]['data'] - t))
            ne_coef = ne_coef[indt, ...]
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
            R = dangles[krays][khel]['angle']['data']
            Z = dangles[krays][khel]['angle']['data']
            iok = np.isfinite(angles) & np.isfinite(R)
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

            length = dangles[krays][khel]['length']['data']
            length[~iok] = np.nan
            ndimd = iang.ndim - length.ndim

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
                                sli_emiss = (slice(None),)*ndimd + (ioki,) + ind
                                sli_los = (slice(None),)*ndimd + ind
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
        diffRE = RE_headon - RE_back

        # Max
        Max_headon_ff = dsig_los[kcase][krays_max][rei]['maxwell']['ff']['data']
        Max_back_ff = dsig_los[kcase][krays_min][rei]['maxwell']['ff']['data']
        diffMax = Max_headon_ff - Max_back_ff

        Max_headon_tot = (
            dsig_los[kcase][krays_max][rei]['maxwell']['ff']['data']
            + dsig_los[kcase][krays_max][rei]['maxwell']['fb']['data']
            + dsig_los[kcase][krays_max][rei]['maxwell']['bb']['data']
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
        }

    # ################
    # plot - RE
    # ################

    dax_RE = None
    if t is not None:
        dax_RE = _plot_RE(
            coll=coll,
            cases=cases,
            re=re,
            key_resp=key_resp,
            dsig_los=dsig_los,
            # specifics
            kresp=key_resp,
            krays_max=krays_max,
            krays_min=krays_min,
            # angles
            angle0=angle0,
            angle1=angle1,
            # params
            Te_eV=Te_eV,
            ne=ne,
            jp=jp,
            jp_frac=jp_frac,
            # plot
            dmargin=dmargin_RE,
            figsize=figsize_RE,
            fontsize=fontsize,
        )

    # --------------
    # store
    # --------------

    # ################
    # plot - merits
    # ################

    dax_merits = None
    if t is not None and False:
        dax_merits = _plot_merits(
            coll=coll,
            cases=cases,
            re=re,
            key_resp=key_resp,
            dsig_los=dsig_los,
            # specifics
            kresp=key_resp,
            krays_max=krays_max,
            krays_min=krays_min,
            # angles
            angle0=angle0,
            angle1=angle1,
            # params
            Te_eV=Te_eV,
            ne=ne,
            jp=jp,
            jp_frac=jp_frac,
            # plot
            dmargin=dmargin_RE,
            figsize=figsize_RE,
            fontsize=fontsize,
        )

    # --------------
    # store
    # --------------


    return dax_RE, coll, dangles, dsig_los, dmetrics


# ############################################
# ############################################
#       add ptcam
# ############################################


def _add_ptcam(
    coll=None,
    key_cam=None,
    angle0=None,
    angle1=None,
    config=None,
):

    # -------
    # inputs
    # -------

    # angle0
    if angle0 is None:
        angle0 = (30*np.pi/180) * np.linspace(-1, 1, 180)

    angle0 = ds._generic_check._check_flat1darray(
        angle0, "angle0",
        dtype=float,
        unique=True,
    )

    # angle1
    if angle1 is None:
        angle1 = (40*np.pi/180) * np.linspace(-1, 1, 240)

    angle1 = ds._generic_check._check_flat1darray(
        angle1, "angle1",
        dtype=float,
        unique=True,
    )
    # -------
    # angles ref
    # -------

    ref_rays = ('nangle0', 'nangle1')
    nrays = (angle0.size, angle1.size)
    coll.add_ref(ref_rays[0], size=nrays[0])
    coll.add_ref(ref_rays[1], size=nrays[1])

    # ---------
    # angles
    # ---------

    coll.add_data(
        'angle0',
        data=angle0*180/np.pi,
        units='deg',
        ref=ref_rays[0],
    )

    coll.add_data(
        'angle1',
        data=angle1*180/np.pi,
        units='deg',
        ref=ref_rays[1],
    )

    # ---------------
    # add single points
    # ---------------

    if key_cam is None:
        key_cam = sorted(coll.dobj['camera'].keys())

    dkrays = {}
    for kcam in key_cam:

        # -------------
        # cent, vect

        cent = np.mean(coll.get_camera_cents_xyz(kcam), axis=1)
        phi0 = np.arctan2(cent[1], cent[0])
        ephi0 = np.r_[-np.sin(phi0), np.cos(phi0), 0]

        vect = {}
        dvect = coll.get_camera_unit_vectors(kcam)
        ls = ['x', 'y', 'z']
        for kv in ['nin', 'e0', 'e1']:
            vect[kv] = np.array([dvect[f'{kv}_{ss}'] for ss in ls])
            if vect[kv].ndim > 1:
                laxis = range(1, vect[kv].ndim)
                vect[kv] = np.mean(vect[kv], axis=tuple(laxis))

        # adjust e0
        e1 = np.r_[0, 0, 1.]
        e0 = np.cross(vect['nin'], e1)
        e0 = e0 / np.linalg.norm(e0)
        if np.sum(e0 * ephi0) < 0.:
            e0 = -e0
        vect['e0'] = e0
        e1 = np.cross(vect['nin'], e0)
        e1 = e1 / np.linalg.norm(e1)
        if e1[2] < 0:
            e1 = -e1
        vect['e1'] = e1

        # --------------------
        # using tofu built-in

        kray = f"{kcam}_pt"
        coll.add_single_point_camera2d(
            key=kray,
            cent=cent,
            angle0='angle0',
            angle1='angle1',
            config=config,
            **vect,
        )

        dkrays[kcam] = kray

    return dkrays, angle0, angle1


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


# ############################################
# ############################################
#       plot RE
# ############################################


def _plot_RE(
    coll=None,
    cases=None,
    key_resp=None,
    re=None,
    dsig_los=None,
    # specifics
    kresp=None,
    krays_max=None,
    krays_min=None,
    # angles
    angle0=None,
    angle1=None,
    # params
    Te_eV=None,
    ne=None,
    jp=None,
    jp_frac=None,
    # plot
    dmargin=None,
    figsize=(14, 10),
    fontsize=None,
    vmax=None,
    # unused
    *kwdargs,
):
    # -------------
    # inputs
    # -------------

    if fontsize is None:
        fontsize = _FONTSIZE

    dvmax = {}
    if vmax is None:
        dvmax = {
            rei: np.nanmax([
                dsig_los[kcase][krays_max][rei]['RE']['ff']['data']
                for kcase in cases.keys()
            ])
            for rei in re
        }

    # -------------
    # prepare
    # -------------

    dang0 = angle0[1] - angle0[0]
    dang1 = angle1[1] - angle1[0]
    extent = (
        (angle0[0] - 0.5*dang0) * 180/np.pi,
        (angle0[-1] + 0.5*dang0) * 180/np.pi,
        (angle1[0] - 0.5*dang1) * 180/np.pi,
        (angle1[-1] + 0.5*dang1) * 180/np.pi,
    )

    kcase = sorted(cases.keys())[0]
    units = dsig_los[kcase][krays_max][re[0]]['RE']['ff']['units']

    # -------------
    # prepare fig
    # -------------

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.90,
            'bottom': 0.06, 'top': 0.93,
            'wspace': 0.18, 'hspace': 0.20,
        }
        dmargin_cbar = {
            'left': 0.93, 'right': 0.97,
            'bottom': 0.06, 'top': 0.93,
            'wspace': 0.18, 'hspace': 0.20,
        }

    fig = plt.figure(figsize=figsize)

    nc = np.max([kk for kk in cases.keys()]) + 2
    gs = gridspec.GridSpec(ncols=nc, nrows=len(re), **dmargin)
    gs_cbar = gridspec.GridSpec(ncols=1, nrows=len(re), **dmargin_cbar)
    dax = {}

    # ----------------------
    # fig title
    # ----------------------

    # tit
    tit = (
        r"$n_e$" + f" = {ne:1.0e}" + r"$/m^3$,  "
        + r"$j_P$" + f" = {jp*1e-6:1.0f}" + r"$MA/m^2$" + "\n"
        + r"$T_e$" + f" = {Te_eV*1e-3:2.1f} keV,  "
        + r"$F_{RE}$" + f" = {jp_frac:2.1f}\n"
        + key_resp
    )
    fig.suptitle(
        tit,
        fontsize=fontsize,
        fontweight='bold',
    )

    # ----------------------
    # loop on cases for axes
    # ----------------------

    ax0 = None
    for ire, rei in enumerate(re):
        for icase, (kcase, vcase) in enumerate(cases.items()):

            # tit
            hel = 'helicity' if vcase['helicity'] else 'no helicity'
            pitch = 'pitch angle distrib.' if vcase['pitch'] else 'no pitch'
            nep = 'ne profile' if vcase['ne'] else 'flat profile'
            tit = (
                f"{hel}\n"
                f"{pitch}\n"
                f"{nep}\n"
            )

            # --------------
            # axes - image

            ax = fig.add_subplot(
                gs[ire, kcase],
                aspect='equal',
                sharex=ax0,
                sharey=ax0,
            )
            if ire == 0:
                ax.set_title(
                    tit,
                    fontsize=fontsize,
                    fontweight='bold',
                )
            ax.set_xlabel(
                r'$\theta_0$ (deg)',
                fontsize=fontsize,
                fontweight='bold',
            )
            if icase == 0:
                ax.set_ylabel(
                    f"{rei}\n" + r"$\theta_1$ (deg)",
                    fontsize=fontsize,
                    fontweight='bold',
                )
                ax0 = ax

            ax.text(
                0.01,
                0.99,
                f'({string.ascii_lowercase[icase + nc*ire]})',
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=fontsize,
                fontweight='bold',
                transform=ax.transAxes,
            )

            dax[f'{rei}_{kcase}'] = ax

        # ---------
        # diff_RE

        ax = fig.add_subplot(
            gs[ire, -1],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
        )
        if ire == 0:
            ax.set_title(
                tit,
                fontsize=fontsize,
                fontweight='bold',
            )
        ax.set_xlabel(
            r'$\theta_0$ (deg)',
            fontsize=fontsize,
            fontweight='bold',
        )

        ax.text(
            0.01,
            0.99,
            f'({string.ascii_lowercase[nc - 1 + nc*ire]})',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transAxes,
        )

        dax[f'{rei}_diff'] = ax

        # ---------
        # colorbar

        ax = fig.add_subplot(
            gs_cbar[ire, 0],
            aspect='auto',
        )
        ax.set_title(
            str(units),
            fontsize=fontsize,
            fontweight='bold',
        )

        dax[f'{rei}_cbar'] = ax

    dax = ds._generic_check._check_dax(dax)

    # ticklabels size
    for kax, vax in dax.items():
        dax[kax]['handle'].tick_params(
            axis='both',
            which='major',
            labelsize=fontsize - 1,
        )

    # --------------
    # plot vs theta
    # --------------

    for ire, rei in enumerate(re):
        for icase, (kcase, vcase) in enumerate(cases.items()):

            kax = f"{rei}_{kcase}"
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                data = dsig_los[kcase][krays_max][rei]['RE']['ff']['data']

                im = ax.imshow(
                    data.T,
                    extent=extent,
                    origin='lower',
                    cmap=plt.cm.viridis,
                    interpolation='nearest',
                    vmin=0,
                    vmax=dvmax[rei],
                )

                if icase == len(cases) - 1:
                    ax = dax[f'{rei}_cbar']['handle']
                    plt.colorbar(im, cax=ax)

        # ---------
        # diff

        kax = f"{rei}_diff"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            kcase = [
                k0 for k0, v0 in cases.items()
                if all([v1 for v1 in v0.values()])
            ][0]
            data = dsig_los[kcase][krays_max][rei]['RE']['ff']['data']
            data_back = dsig_los[kcase][krays_min][rei]['RE']['ff']['data']
            diff = data - data_back[::-1, :]

            im = ax.imshow(
                diff.T,
                extent=extent,
                origin='lower',
                cmap=plt.cm.viridis,
                interpolation='nearest',
                vmin=0,
                vmax=dvmax[rei],
            )

    return dax
