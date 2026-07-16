import os
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds
import tofu as tf


from ._load_spect import _PATH_INPUTS
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_PFE_CAD0 = os.path.join(
    _PATH_INPUTS,
    'OMP000U.png',
)
_PFE_CAD1 = os.path.join(
    _PATH_INPUTS,
    'OMP000U_Zoom.png',
)
_PFE_CAD2 = os.path.join(
    _PATH_INPUTS,
    'OMP000U_Zoom_inside.png',
)
_PFE_CONFIG = os.path.join(
    _PATH_INPUTS,
    'TFG_Config_ExpSPARC_SPARC-V2_sh00000_Vers1.8.18.npz',
)
_PFE_COLL = os.path.join(
    _PATH_INPUTS,
    'Inversion_BOLO_DMS_ptcam_dvezinet_20260709-205535.npz',
)


_DSAMPLING = {
    'dedge': {'res': 'min'},
    'dsurface': {'nb': 7},
}


# #######################################
# #######################################
#           Main
# #######################################


def main(
    coll=None,
    key_cam=None,
    config=None,
    # ptcam
    angle0=None,
    angle1=None,
    # fov
    dsampling=None,
    # plot
    figsize=(5, 7),
    # figsize=(8, 10),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):

    # -----------
    # inputs
    # -----------

    if key_cam is None:
        key_cam = 'IDIV_000U'

    if dsampling is None:
        dsampling = _DSAMPLING

    # load CAD image
    image0 = Image.open(_PFE_CAD0)
    image1 = Image.open(_PFE_CAD1)
    image2 = Image.open(_PFE_CAD2)

    # config
    if config is None:
        config = _PFE_CONFIG
    if isinstance(config, str):
        config = tf.load(config)

    # load coll
    if coll is None:
        coll = _PFE_COLL
    if isinstance(coll, str):
        coll = tf.data.load(coll)

    dkrays, angle0, angle1 = _add_ptcam(
        coll=coll,
        key_cam=key_cam,
        angle0=angle0,
        angle1=angle1,
        config=config,
    )

    wdiag = coll._which_diagnostic
    kdiag = [kd for kd in coll.dobj[wdiag].keys() if 'bolo' in kd.lower()][0]

    # -------------
    # add FOV
    # -------------

    # make rays
    kcam = list(dkrays.keys())[0]
    coll.add_rays_from_diagnostic(
        key=kdiag,
        dsampling_pixel=dsampling,
        dsampling_optics=dsampling,
        optics=-1,
        config=config,
        store=True,
        strict=False,
        key_rays=None,
        overwrite=None,
    )
    key_rays = f"{kcam.replace('_los', '').replace('BOLO_', '')}_rays"

    # FOV
    key = list(dkrays.values())[0]
    wrays = coll._which_rays
    shape = coll.dobj[wrays][key]['shape'][1:]
    ind_fov = np.zeros(shape, dtype=bool)
    dout_rays = coll.get_rays_angles_from_single_point_camera2d(
        key_single_pt_cam=key,
        key_rays=key_rays,
        return_indices=True,
        convex_axis=(-1, -2),
    )
    for k1, v1 in dout_rays['hull'].items():
        ind_fov = ind_fov | v1

    # --------------
    # compute
    # --------------

    # data
    assert len(dkrays) == 1
    kray = list(dkrays.values())[0]
    data, _, ref = coll.get_rays_quantity(
        key=kray,
        quantity='alpha',
    )

    # colormap
    colormap = np.zeros(data.shape[1:] + (4,), dtype=float)
    colormap[..., -1] = data[0, ...] / (np.pi/2)
    colormap[(ind_fov, 0)] = 1.

    # extent
    dang0 = angle0[1] - angle0[0]
    dang1 = angle1[1] - angle1[0]
    extent = (
        (angle0[0] - 0.5*dang0) * 180/np.pi,
        (angle0[-1] + 0.5*dang0) * 180/np.pi,
        (angle1[0] - 0.5*dang1) * 180/np.pi,
        (angle1[-1] + 0.5*dang1) * 180/np.pi,
    )

    # --------------
    # prepare figure
    # --------------

    dmargin = {
        'left': 0.04, 'right': 0.90,
        'bottom': 0.07, 'top': 0.99,
        'wspace': 0.18, 'hspace': 0.20,
    }

    n0 = 5
    n1 = 2*n0 - 2
    nc = 8
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols=2*nc, nrows=(2*n0 + 1 + n1), **dmargin)
    dax = {}

    # ----------------------
    # axes
    # ----------------------

    # --------------
    # axes - CAD0

    ax = fig.add_subplot(gs[:2*n0, :nc], aspect='auto')

    ax.text(
        0.,
        1.,
        '(a)',
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['CAD0'] = ax

    # --------------
    # axes - CAD1

    ax = fig.add_subplot(gs[:n0, nc:], aspect='auto')

    ax.text(
        0.,
        1.,
        '(b)',
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['CAD1'] = ax

    # --------------
    # axes - CAD2

    ax = fig.add_subplot(gs[n0:2*n0, nc:], aspect='auto')

    ax.text(
        0.,
        1.,
        '(c)',
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['CAD2'] = ax

    # --------------
    # axes - View

    ax = fig.add_subplot(gs[2*n0+1:, :-2], aspect='equal')

    ax.set_title(
        "view from a point-sphere camera model",
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_xlabel(
        r"$\theta_0$",
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\theta_1$",
        fontsize=fontsize,
        fontweight='bold',
    )

    ax.text(
        -0.15,
        1.0,
        '(d)',
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['view'] = ax

    # --------------
    # axes - colorbar

    ax = fig.add_subplot(gs[2*n0+1:, -2], aspect='auto')
    dax['cbar'] = ax

    dax = ds._generic_check._check_dax(dax)

    # ticklabels size
    for kax, vax in dax.items():
        dax[kax]['handle'].tick_params(
            axis='both',
            which='major',
            labelsize=fontsize - 1,
        )

    # --------------
    # plot CAD
    # --------------

    kax = "CAD0"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(image0)
        ax.set_axis_off()

    kax = "CAD1"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(image1)
        ax.set_axis_off()

    kax = "CAD2"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(image2)
        ax.set_axis_off()

    # --------------
    # plot view
    # --------------

    kax = "view"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            np.swapaxes(colormap, 0, 1),
            extent=extent,
            origin='lower',
            aspect='equal',
            cmap=plt.cm.gray_r,
            # interpolation='nearest',
            # vmin=0,
            # vmax=1,
        )

        # cax
        cax = dax['cbar']['handle']
        plt.colorbar(
            im,
            cax=cax,
            orientation='vertical',
        )


        deg_ticks = np.r_[0, 30, 60, 90]
        cax.set_yticks(deg_ticks/90.)
        cax.set_yticklabels(deg_ticks)
        cax.set_ylabel(
            'Incidence angle on PFC (deg)',
            fontsize=fontsize,
            fontweight='bold',
        )
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()

    # ----------
    # save
    # ----------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return dax, coll


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
        angle0 = (np.pi/180) * np.linspace(-50, 30, 240)

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

    # isBOLO
    wdiag = coll._which_diagnostic
    isbolo = any(['bolo' in kd.lower() for kd in coll.dobj[wdiag].keys()])

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

    wcam = coll._which_cam
    if key_cam is None:
        key_cam = 'IDIV_000U'
    if isinstance(key_cam, str):
        key_cam = [key_cam]


    dkrays = {}
    for kcam in key_cam:

        # --------------------
        # HXRVA => cent, vect

        if isbolo is True:

            wrays = coll._which_rays
            if kcam not in coll.dobj[wrays].keys():
                if kcam in coll.dobj[wcam].keys():
                    kdiag = [
                        kd for kd, vd in coll.dobj[wdiag].items()
                        if kcam in vd[wcam]
                    ][0]
                    kcam = coll.dobj[wdiag][kdiag]['doptics'][kcam]['los']

            kray = f"{kcam}_pt"
            coll.add_single_point_camera2d(
                key=kray,
                key_rays=kcam,
                angle0='angle0',
                angle1='angle1',
                e1=np.r_[0, 0, 1],
                config=config,
            )

        else:

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
