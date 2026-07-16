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


_PFE_CAD = os.path.join(
    _PATH_INPUTS,
    'HXR_MPP200_zoom.png',
)
_PFE_CONFIG = os.path.join(
    _PATH_INPUTS,
    'TFG_Config_ExpSPARC_SPARC-V2_sh00000_Vers1.8.18.npz',
)
_PFE_COLL = os.path.join(
    _PATH_INPUTS,
    'Inversion_HXRVA_dvezinet_20260713-154229.npz',
)


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
    # plot
    figsize=(14, 10),
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

    # load CAD image
    image = Image.open(_PFE_CAD)

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

    # --------------
    # compute
    # --------------

    # data
    data, _, ref = coll.get_rays_quantity(
        key='LEFT_pt',
        quantity='alpha',
    )

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
        'left': 0.05, 'right': 0.90,
        'bottom': 0.06, 'top': 0.93,
        'wspace': 0.18, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols=2, nrows=15, **dmargin)
    dax = {}

    # ----------------------
    # axes
    # ----------------------

    # --------------
    # axes - CAD

    ax = fig.add_subplot(gs[:, 0], aspect='auto')

    ax.text(
        0.01,
        0.99,
        '(a)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['CAD'] = ax

    # --------------
    # axes - View

    ax = fig.add_subplot(gs[:-2, 1], aspect='equal')

    ax.set_title(
        "iview from a point-sphere camera model",
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
        0.01,
        1.01,
        '(b)',
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['view'] = ax

    # --------------
    # axes - colorbar

    ax = fig.add_subplot(gs[-1, 1], aspect='auto')
    ax.set_title(
        'Incidence angle on PFC (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )

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

    kax = "CAD"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(
            image,
        )

        ax.set_axis_off()

    # --------------
    # plot view
    # --------------

    kax = "view"
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            data.T*180/np.pi,
            extent=extent,
            origin='lower',
            aspect='equal',
            cmap=plt.cm.gray,
            interpolation='nearest',
            vmin=0,
            vmax=90,
        )

        plt.colorbar(
            im,
            cax=dax['cbar']['handle'],
            orientation='horizontal',
        )
        dax['cbar']['handle'].set_xticks([0, 30, 60, 90])

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
