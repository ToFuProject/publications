import os
import copy


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import datastock as ds
import tofu as tf


from ._savefig import main as savefig


# #####################################################
# #####################################################
#       DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)
_PATH_SAVE = os.path.join(os.path.dirname(_PATH_HERE), 'figures')


_DR = {
    'R0': 1.8,
    'rplasma': 0.60,
    'RVes': [1.2, 2.66],
    'Rcryo': 4.6,
    ''
    'PP_R': np.r_[2.50, 4.7],  # 4.2
    'PP_width': 0.47,
    'PP_phi': np.r_[-180, -20, 0, 20] * np.pi/180,
}


_DSENSORS = {
    'in': {
        'pp': 0,
        'R': 2.55,
        'cw': False,
        'rplasma_ratio': 0.7,
        'color': 'b',
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
        'shielding': True,
        'dogleg': {
            'frac_h': 0.5,
            'frac_v': 0.75,
        },
        'text': {
            'str': "(0)",
        },
    },
    'ex1': {
        'pp': 1,
        'R': 6,
        'cw': False,
        'color': 'g',
        'width': 0.10,
        'dist': 4,
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
        'shielding': True,
        'opening': True,
        'wall': True,
        'beamdump': True,
        'neutrons_length': 1,
        'neutrons_width': 0.2,
        'reflector': {
            'angle': 15*np.pi/180,
            'R': _DR['PP_R'][1] + 0.1,
            'width': 0.25,
            'color': 'k',
            'lw': 2,
        },
        'text': {
            'str': "(2)",
        },
    },
    'ex2': {
        'pp': 2,
        'R': 6,
        'cw': False,
        'color': 'g',
        'width': 0.10,
        'dist': 4,
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
        'shielding': True,
        'opening': True,
        'wall': True,
        'beamdump': True,
        'neutrons_length': 1,
        'neutrons_width': 0.2,
        'text': {
            'str': "(1)",
        },
    },
    'ex3': {
        'pp': 3,
        'R': 6,
        'cw': False,
        'color': (0.1, 0.9, 0.1),
        'width': 0.10,
        'dist': 4,
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
        'shielding': True,
        'opening': True,
        'wall': True,
        'beamdump': True,
        'neutrons_length': 1,
        'neutrons_width': 0.2,
        'reflector': {
            'angle': -15*np.pi/180,
            'R': _DR['PP_R'][0],
            'width': 0.25,
            'color': 'k',
            'lw': 2,
        },
        'text': {
            'str': "(3)",
        },
    },
}


# #####################################################
# #####################################################
#       main
# #####################################################


def main(
    # tokamak
    R0=None,
    rplasma=None,
    RVes=None,
    Rcryo=None,
    # port plug
    PP_R=None,
    PP_width=None,
    PP_phi=None,
    # sensors
    res=None,
    dsensors=None,
    # plot
    figsize=(5, 7),
    fontsize=12,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Plot a top view of a SPARC-like tokamak with diagnostics

    Includes:
        - a in-vessel view
        - 3 ex-cryostat view with or without reflectors

    Also plots the associated observation angles vs the local magnetic field

    """

    # --------------
    # Load SPARC
    # --------------

    config, dinput = _fig02_check(**locals())

    phi = np.pi * np.linspace(-1, 1, 181)
    cos = np.cos(phi)
    sin = np.sin(phi)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.94,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # axes - hor
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='equal')
    ax.set_xlabel('X (m)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=fontsize, fontweight='bold')
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

    dax['hor'] = ax

    # --------------
    # axes - ang vs rplasma
    # --------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto')
    ax.set_xlabel('r / a', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        r'$\theta_{ph,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(b)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['theta_vs_B'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot hor
    # --------------

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # plot R

        lk = [kk for kk in dinput.keys() if kk[0] == 'R']
        for k0 in lk:

            if 'plasma' in k0:
                inner = dinput[k0]['data'][0] * np.array([cos, sin]).T
                outer = dinput[k0]['data'][1] * np.array([cos, sin]).T
                vertices = np.concatenate((inner, outer[::-1]), axis=0)

                codes = np.ones(
                    len(inner),
                    dtype=mpath.Path.code_type,
                ) * mpath.Path.LINETO
                codes[0] = mpath.Path.MOVETO
                all_codes = np.concatenate((codes, codes))

                path = mpath.Path(vertices, all_codes)
                patch = mpatches.PathPatch(
                    path,
                    facecolor='r',
                    alpha=0.1,
                    edgecolor='r',
                )
                ax.add_patch(patch)

            else:
                for v1 in dinput[k0]['data']:
                    ax.plot(
                        v1*cos,
                        v1*sin,
                        **dinput[k0]['prop'],
                    )

        # --------------
        # add arrows

        R = dinput['R0']['data'][0] + 0.5 * dinput['rplasma']['data'][0]
        phi = np.r_[130, 170] * np.pi / 180
        # dist = R * np.hypot(
        # np.cos(phi[0]) - np.cos(phi[1]),
        # np.sin(phi[0]) - np.sin(phi[1]),
        # )
        # rad = (R * (1 - np.cos(np.abs(np.diff(phi)/2))) / dist)[0]
        rad = -0.3
        ax.annotate(
            "",
            xy=(R*np.cos(phi[0]), R*np.sin(phi[0])),
            xycoords='data',
            xytext=(R*np.cos(phi[1]), R*np.sin(phi[1])),
            textcoords='data',
            color='r',
            fontweight='bold',
            fontsize=fontsize,
            horizontalalignment='center',
            verticalalignment='center',
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color='r',
                shrinkA=5, shrinkB=5,
                patchA=None, patchB=None,
                connectionstyle=f'arc3,rad={rad}',
            ),
        )
        ax.text(
            R*np.cos(np.mean(phi)),
            R*np.sin(np.mean(phi)),
            "RE",
            color='r',
            horizontalalignment='left',
            verticalalignment='top',
            fontweight='bold',
            fontsize=fontsize,
        )

        # --------------
        # plot port plug

        for ii, phi in enumerate(dinput['PP_phi']['data']):

            # edges
            width = dinput['PP_width']['data'][0]
            ppR0 = dinput['PP_R']['data'][0]
            ppR1 = dinput['PP_R']['data'][1]
            length = ppR1 - ppR0
            cent = 0.5 * (ppR0+ppR1) * np.r_[np.cos(phi), np.sin(phi)]
            xy = (
                cent[0] - 0.5 * length,
                cent[1] - 0.5 * width,
            )

            # patch white
            patch = mpatches.Rectangle(
                xy,
                length,
                width,
                angle=phi*180/np.pi,
                rotation_point='center',
                facecolor='w',
                alpha=1.,
                edgecolor='None',
                zorder=10,
            )
            ax.add_patch(patch)

            # central line
            ppR0 = dinput['PP_R']['data'][0]
            ppR1 = dinput['PP_R']['data'][1]

            centx = np.r_[ppR0, ppR1] * np.cos(phi)
            centy = np.r_[ppR0, ppR1] * np.sin(phi)

            ax.plot(
                centx,
                centy,
                c='k',
                lw=1,
                ls='--',
                alpha=0.3,
                zorder=15,
            )

            # patch dark
            patch = mpatches.Rectangle(
                (xy[0], cent[1] - 0.4*width),
                length,
                width*0.80,
                angle=phi*180/np.pi,
                rotation_point='center',
                facecolor='k',
                alpha=0.5,
                edgecolor='None',
                zorder=20,
            )
            ax.add_patch(patch)

            # edges
            ephi = np.r_[-np.sin(phi), np.cos(phi)]
            edgex = (
                centx[None, :]
                + 0.5 * width * ephi[0] * np.r_[1, np.nan, -1][:, None]
            ).ravel()
            edgey = (
                centy[None, :]
                + 0.5 * width * ephi[1] * np.r_[1, np.nan, -1][:, None]
            ).ravel()

            ax.plot(
                edgex,
                edgey,
                c='k',
                lw=1,
                ls='-',
                zorder=20,
            )

        # --------------
        # add sensors

        dsensors = _sensors(
            res=res,
            dsensors=dsensors,
            dinput=dinput,
        )

        for k0, v0 in dsensors.items():

            # FOV
            patch = mpatches.PathPatch(
                v0['path'],
                facecolor=v0['color'],
                alpha=v0['alpha'],
                zorder=50,
                edgecolor=v0['color'],
            )
            ax.add_patch(patch)

            # sensor
            ax.plot(
                [v0['cent'][0]],
                [v0['cent'][1]],
                c=v0['color'],
                ls='None',
                lw=2,
                marker=v0['marker'],
                zorder=30,
                label=v0.get('label', k0),
            )

            # FOV sampling
            ax.plot(
                v0['ptsx'],
                v0['ptsy'],
                c=v0['color'],
                marker=v0.get('marker', '.'),
                ls='None',
                zorder=60,
                ms=v0.get('ms', 4),
            )

            # patch
            if v0.get('patch') is not None:
                ax.add_patch(
                    v0['patch'],
                )

            # reflectors
            if v0.get('reflector') is not None:
                ax.plot(
                    v0['reflector']['x'],
                    v0['reflector']['y'],
                    c=v0['reflector']['color'],
                    lw=v0['reflector']['lw'],
                    label='reflector',
                    zorder=100,
                )

            # dogleg
            if v0.get('dogleg') is not None:
                for ii, (cc, lw) in enumerate([('w', 3), (v0['color'], 1)]):
                    ax.plot(
                        v0['dogleg']['x'],
                        v0['dogleg']['y'],
                        ls='-',
                        lw=lw,
                        color=cc,
                        zorder=100 + 10*ii,
                    )

            # neutrons
            if v0.get('neutrons') is not None:
                ax.fill(
                    v0['neutrons']['x'],
                    v0['neutrons']['y'],
                    v0['neutrons']['color'],
                    alpha=0.5,
                )

            # text
            if v0.get('text') is not None:
                ax.text(
                    v0['text']['x'],
                    v0['text']['y'],
                    v0['text']['str'],
                    horizontalalignment=v0['text']['horizontalalignment'],
                    verticalalignment=v0['text']['verticalalignment'],
                    fontsize=fontsize,
                    fontweight='bold',
                    color=v0['color'],
                    transform=ax.transData,
                )

        # overall text in-vessel
        lkin = [k0 for k0 in dsensors.keys() if k0.startswith('in')]
        xx = np.mean([dsensors[k0]['text']['x'] for k0 in lkin])
        yy = np.max([dsensors[k0]['text']['y'] for k0 in lkin])
        ax.text(
            xx,
            yy + 0.5,
            'In-vessel\nsensors',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=fontsize,
            fontweight='bold',
            color=dsensors[lkin[0]]['color'],
            transform=ax.transData,
        )

        # overall text ex-cryostat
        lkex = [k0 for k0 in dsensors.keys() if k0.startswith('ex')]
        xx = np.mean([dsensors[k0]['text']['x'] for k0 in lkex])
        yy = np.max([dsensors[k0]['text']['y'] for k0 in lkex])
        ax.text(
            xx,
            yy + 0.5,
            'Ex-cryostat\nsensors',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=fontsize,
            fontweight='bold',
            color=dsensors['ex1']['color'],
            transform=ax.transData,
        )

        # overall text neutrons
        ind = np.argmin([
            np.min(dsensors[k0]['neutrons']['y']) for k0 in lkex
        ])
        xx = np.max(dsensors[lkex[ind]]['neutrons']['x'])
        yy = np.min(dsensors[lkex[ind]]['neutrons']['y'])
        ax.text(
            xx,
            yy - 0.5,
            'neutrons',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=fontsize,
            fontweight='bold',
            color=dsensors[lkex[ind]]['neutrons']['color'],
            transform=ax.transData,
        )

    # ----------------
    # plot theta_vs_B
    # ----------------

    kax = 'theta_vs_B'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for k0, v0 in dsensors.items():

            ax.plot(
                v0['rplasma_norm'],
                v0['theta_vs_B'] * 180 / np.pi,
                marker=v0.get('marker', '.'),
                ms=v0.get('ms', 6),
                color=v0['color'],
                ls=v0.get('ls', 'None'),
                label=v0.get('label', k0),
            )

            # text
            if v0.get('text') is not None:
                trans = transforms.blended_transform_factory(
                    ax.transAxes, ax.transData,
                )
                ind = v0['rplasma_norm'] > 0.9
                yy = v0['theta_vs_B'][ind]
                if k0 == 'ex3':
                    yy = np.mean(yy[yy < np.pi/4])
                else:
                    yy = np.mean(yy)
                if k0 == 'ex1':
                    txt = '(1,2)'
                else:
                    txt = v0['text']['str']

                if k0 != 'ex2':
                    ax.text(
                        1.,
                        yy * 180 / np.pi,
                        txt,
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=fontsize,
                        fontweight='bold',
                        color=v0['color'],
                        transform=trans,
                    )

        ax.axhline(90, c='k', ls='--', lw=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 180)
        ax.grid(True)

        # comments
        ax.text(
            -0.97,
            55,
            'forward',
            horizontalalignment='left',
            verticalalignment='top',
            rotation=90,
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transData,
        )
        ax.text(
            -0.97,
            110,
            'backward',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=90,
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transData,
        )

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
    )

    return dax


def _fig02_check(
    config=None,
    # tokamak
    R0=None,
    rplasma=None,
    RVes=None,
    Rcryo=None,
    PP_R=None,
    PP_width=None,
    PP_phi=None,
    # unused
    **kwdargs,
):

    # ------------------
    # config
    # ------------------

    if config is None:
        config = 'SPARC'

    if isinstance(config, str):
        config = tf.load_config(config)

    # ------------------
    # Geometry - R
    # ------------------

    lk = ['R0', 'rplasma', 'RVes', 'Rcryo', 'PP_R', 'PP_width', 'PP_phi']
    dinput = {
        kk: {
            'data': None,
            'prop': {'color': 'k', 'ls': '-', 'lw': 1, 'label': None}
        }
        for kk in lk
    }

    # R0
    dinput['R0']['data'] = np.r_[float(ds._generic_check._check_var(
        R0, 'R0',
        types=(float, int),
        sign='>0',
        default=_DR['R0'],
    ))]
    dinput['R0']['prop']['ls'] = '--'

    # rplasma
    dinput['rplasma']['data'] = np.r_[float(ds._generic_check._check_var(
        rplasma, 'rplasma',
        types=(float, int),
        sign='>0',
        default=_DR['rplasma'],
    ))]
    assert dinput['rplasma']['data'] < dinput['R0']['data']

    dinput['Rplasma'] = {
        'data': dinput['R0']['data'] + dinput['rplasma']['data']*np.r_[-1, 1],
        'prop': {
            'color': 'r',
            'lw': 1,
            'ls': '-',
            'label': 'plasma',
        }
    }

    # RVes
    if RVes is None:
        RVes = _DR['RVes']
    dinput['RVes']['data'] = ds._generic_check._check_flat1darray(
        RVes, 'RVes',
        dtype=float,
        sign='>0',
        unique=True,
        size=2,
    )
    Rlim = dinput['R0']['data'] - dinput['rplasma']['data']
    assert dinput['RVes']['data'][0] < Rlim
    Rlim = dinput['R0']['data'] + dinput['rplasma']['data']
    assert dinput['RVes']['data'][1] > Rlim
    dinput['RVes']['prop']['lw'] = 2

    # rplasma
    dinput['Rcryo']['data'] = np.r_[float(ds._generic_check._check_var(
        Rcryo, 'Rcryo',
        types=(float, int),
        sign='>0',
        default=_DR['Rcryo'],
    ))]
    assert dinput['Rcryo']['data'] > dinput['RVes']['data'][1]
    dinput['Rcryo']['prop']['lw'] = 2

    # ------------------
    # Geometry - Port plug
    # ------------------

    # PP_R
    if PP_R is None:
        PP_R = _DR['PP_R']
    dinput['PP_R']['data'] = ds._generic_check._check_flat1darray(
        PP_R, 'PP_R',
        dtype=float,
        sign='>0',
        unique=True,
        size=2,
    )
    Rin = dinput['R0']['data'] + dinput['rplasma']['data']
    assert dinput['PP_R']['data'][0] > Rin
    assert dinput['PP_R']['data'][1] > dinput['Rcryo']['data']

    # PP_width
    dinput['PP_width']['data'] = np.r_[float(ds._generic_check._check_var(
        PP_width, 'PP_width',
        types=(float, int),
        sign='>0',
        default=_DR['PP_width'],
    ))]

    # PP_phi
    if PP_phi is None:
        PP_phi = _DR['PP_phi']
    PP_phi = ds._generic_check._check_flat1darray(
        PP_phi, 'PP_phi',
        dtype=float,
        unique=True,
    )
    dinput['PP_phi']['data'] = np.arctan2(np.sin(PP_phi), np.cos(PP_phi))

    return config, dinput


def _sensors(
    res=None,
    dsensors=None,
    # dinput
    dinput=None,
    # unused
    **kwdargs,
):

    # --------------
    # inputs
    # --------------

    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        sign='>0',
        default=0.01,
    )

    # --------------
    # initialize
    # --------------

    if dsensors is None:
        dsensors = copy.deepcopy(_DSENSORS)

    # --------------
    # check
    # --------------

    for k0, v0 in dsensors.items():
        for k1, v1 in v0.items():
            if v1 is None:
                dsensors[k0][k1] = _DSENSORS[k0].get(k1)
        for k1, v1 in _DSENSORS[k0].items():
            if dsensors[k0].get(k1) is None:
                dsensors[k0][k1] = v1

    # --------------
    # Derive
    # --------------

    for k0, v0 in dsensors.items():

        phi = dinput['PP_phi']['data'][v0['pp']]
        eR = np.r_[np.cos(phi), np.sin(phi)]
        ephi = np.r_[-np.sin(phi), np.cos(phi)]
        sign = v0["cw"] * 2 - 1
        width = dinput['PP_width']['data']
        length = dinput['PP_R']['data'][1] - dinput['PP_R']['data'][0]
        ppc = np.mean(dinput['PP_R']['data']) * eR

        # ----------
        # cent

        if k0 == 'in':
            cent = v0['R'] * eR + sign * 0.5 * width * ephi
        else:
            dphi = np.arctan2(width - v0['width'], length)
            eRs = eR * np.cos(dphi) + sign * ephi * np.sin(dphi)
            cent = ppc + v0["dist"] * eRs

            # store
            dsensors[k0]["dphi"] = dphi
            dsensors[k0]["ppc"] = ppc
            dsensors[k0]["eRs"] = eRs

        # ----------
        # FOV

        if k0 == 'in':

            R0 = dinput['R0']['data'][0]
            rplasma = dinput['rplasma']['data'][0]

            # out
            R = R0 + rplasma * v0["rplasma_ratio"]
            vect_out = _tangent(cent, R, sign)

            # in
            R = R0 - rplasma * v0["rplasma_ratio"]
            vect_in = _tangent(cent, R, sign)

        else:
            ephis = np.r_[-eRs[1], eRs[0]]
            vect_out = (
                (length + v0["dist"]) * (-eRs) + 0.5 * v0['width'] * ephis
            )
            vect_in = (
                (length + v0["dist"]) * (-eRs) - 0.5 * v0['width'] * ephis
            )
            vect_out = vect_out / np.linalg.norm(vect_out)
            vect_in = vect_in / np.linalg.norm(vect_in)

        # ------------
        # reflector in

        if dsensors[k0].get('reflector') is not None:

            ref_angle = dsensors[k0]['reflector']['angle']
            ref_R = dsensors[k0]['reflector']['R']
            ref_width = dsensors[k0]['reflector']['width']

            ref_kk, ref_isout = _intersect(cent, -eRs, ref_R)
            ref_kk = ref_kk[~ref_isout]

            ref_cent = cent + ref_kk * (-eRs)
            ref_epar = eRs * np.cos(ref_angle) + ephis * np.sin(ref_angle)
            ref_nin = np.r_[-ref_epar[1], ref_epar[0]]

            refx = ref_cent[0] + 0.5 * ref_width * np.r_[-1, 1] * ref_epar[0]
            refy = ref_cent[1] + 0.5 * ref_width * np.r_[-1, 1] * ref_epar[1]

            dsensors[k0]["reflector"]['x'] = refx
            dsensors[k0]["reflector"]['y'] = refy

            cent_out, vect_out2 = _intersect_line(
                cent, vect_out, ref_cent, ref_nin,
            )
            cent_in, vect_in2 = _intersect_line(
                cent, vect_in, ref_cent, ref_nin,
            )

            # if ref_R > 3 => update cent
            if ref_R > 3:

                cent_new = (
                    ref_cent
                    + np.sum((cent - ref_cent) * ref_epar) * ref_epar
                    - np.sum((cent - ref_cent) * ref_nin) * ref_nin
                )
                vect_out2 = vect_out
                vect_in2 = vect_in
            else:
                cent_new = cent

            if False:
                msg = (
                    f"\nReflector {k0}:\n"
                    f"\t- ref_R: {ref_R} vs {np.linalg.norm(ref_cent)}\n"
                    f"\t- ref_angle: {ref_angle*180/np.pi:3.1f} deg\n"
                    f"\t- ref_width: {ref_width}\n"
                    f"\t- ref_cent: {ref_cent}\n"
                    f"\t- ref_nin: {ref_nin}\n"
                    f"\t- ref_epar: {ref_epar}\n"
                    f"\t- ref_isout: {ref_isout}\n"
                    f"\t- ref_kk: {ref_kk}\n"
                    f"\t- cent: {cent}\n"
                    f"\t- cent_new: {cent_new}\n"
                    f"\t- cent_out: {cent_out}\n"
                    f"\t- cent_in: {cent_in}\n"
                    f"\t- eRs: {eRs}\n"
                    f"\t- vect_out: {vect_out}\n"
                    f"\t- vect_out2: {vect_out2}\n"
                    f"\t- vect_in: {vect_in}\n"
                    f"\t- vect_in2: {vect_in2}\n"
                )
                print(msg)
        else:
            cent_out = cent
            cent_in = cent
            cent_new = None
            vect_out2 = vect_out
            vect_in2 = vect_in

        dsensors[k0]['cent'] = cent if cent_new is None else cent_new

        # FOV
        xx, yy = _FOV(
            cent_out, vect_out2,
            cent_in, vect_in2,
            R0,
            rplasma,
            cent_new,
        )

        path = mpath.Path(np.array([xx, yy]).T)

        # Sample FOV
        DX = np.max(xx) - np.min(xx)
        DY = np.max(yy) - np.min(yy)
        nptsx = int(DX / res)
        nptsy = int(DY / res)
        ptsx = np.linspace(np.min(xx), np.max(xx), nptsx)
        ptsy = np.linspace(np.min(yy), np.max(yy), nptsy)
        ptsx = np.repeat(ptsx[:, None], nptsy, axis=1).ravel()
        ptsy = np.repeat(ptsy[None, :], nptsx, axis=0).ravel()
        iok = (
            path.contains_points(np.array([ptsx, ptsy]).T)
            & (np.hypot(ptsx, ptsy) >= R0 - rplasma)
            & (np.hypot(ptsx, ptsy) <= R0 + rplasma)
        )
        ptsx = ptsx[iok]
        ptsy = ptsy[iok]

        # ------------
        # Angle

        pts_phi = np.arctan2(ptsy, ptsx)
        pts_ephi0 = -np.sin(pts_phi)
        pts_ephi1 = np.cos(pts_phi)
        vect0 = ptsx - cent[0]
        vect1 = ptsy - cent[1]
        vectn = np.sqrt(vect0**2 + vect1**2)
        vect0 = vect0 / vectn
        vect1 = vect1 / vectn
        theta_vs_B = np.arccos(vect0 * pts_ephi0 + vect1 * pts_ephi1)

        # ------------
        # store

        dsensors[k0]["ptsx"] = ptsx
        dsensors[k0]["ptsy"] = ptsy
        dsensors[k0]["rplasma_norm"] = (
            (np.hypot(ptsx, ptsy) - R0) / dinput['rplasma']['data']
        )
        dsensors[k0]["theta_vs_B"] = theta_vs_B
        dsensors[k0]["path"] = path

        # patch
        if k0.startswith('ex'):
            xpath = (
                ppc[0]
                + 0.5 * v0['width'] * np.r_[-1, -1, 1, 1] * ephis[0]
                + 0.55 * length * np.r_[-1, 1, 1, -1] * eRs[0]
            )
            ypath = (
                ppc[1]
                + 0.5 * v0['width'] * np.r_[-1, -1, 1, 1] * ephis[1]
                + 0.55 * length * np.r_[-1, 1, 1, -1] * eRs[1]
            )

            path_patch = mpath.Path(np.array([xpath, ypath]).T)
            dsensors[k0]['patch'] = mpatches.PathPatch(
                path_patch,
                facecolor='w',
                alpha=1.,
                zorder=30,
                edgecolor='w',
            )

        # ----------
        # dogleg

        if v0.get('dogleg') is not None:
            # frac_h = v0['dogleg']['frac_h']
            frac_v = v0['dogleg']['frac_v']
            xx = np.r_[
                cent[0],
                ppc[0] - ephi[0] * frac_v * width/2 - eR[0] * length/2,
                ppc[0] - ephi[0] * frac_v * width/2,
                ppc[0] + ephi[0] * frac_v * width/2,
                ppc[0] + ephi[0] * frac_v * width/2 + eR[0] * 2*length/3,
            ]
            yy = np.r_[
                cent[1],
                ppc[1] - ephi[1] * frac_v * width/2 - eR[1] * length/2,
                ppc[1] - ephi[1] * frac_v * width/2,
                ppc[1] + ephi[1] * frac_v * width/2,
                ppc[1] + ephi[1] * frac_v * width/2 + eR[1] * 2*length/3,
            ]

            dsensors[k0]['dogleg']['x'] = xx
            dsensors[k0]['dogleg']['y'] = yy

        # ----------
        # neutrons

        if dsensors[k0].get('neutrons_length') is not None:

            poutpp = ppc + 0.5 * np.diff(dinput['PP_R']['data']) * eRs
            neutrons_length = dsensors[k0]['neutrons_length'] * eRs
            rad = dsensors[k0]['neutrons_width']
            theta = np.linspace(-1, 1, 101) * np.pi / 2
            cos = np.cos(theta)
            sin = np.sin(theta)
            xx = poutpp[0] + np.r_[
                0,
                neutrons_length[0] + rad * (cos * eRs[0] + sin * ephis[0]),
                0,
            ]
            yy = poutpp[1] + + np.r_[
                0,
                neutrons_length[1] + rad * (cos * eRs[1] + sin * ephis[1]),
                0,
            ]

            dsensors[k0]['neutrons'] = {
                'x': xx,
                'y': yy,
                'color': 'r',
            }

        # ----------
        # text

        if v0.get('text') is not None:
            if k0.startswith('in'):
                dsensors[k0]['text']['x'] = ppc[0] + length * 0.7 * eR[0]
                dsensors[k0]['text']['y'] = ppc[1] + length * 0.7 * eR[1]
                dsensors[k0]['text']['horizontalalignment'] = 'right'
                dsensors[k0]['text']['verticalalignment'] = 'center'
            else:
                if cent_new is None:
                    cent_new = cent
                dsensors[k0]['text']['x'] = cent_new[0] + length/10
                dsensors[k0]['text']['y'] = cent_new[1]
                dsensors[k0]['text']['horizontalalignment'] = 'left'
                dsensors[k0]['text']['verticalalignment'] = 'center'

    return dsensors


def _tangent(
    cent=None,
    R=None,
    sign=None,
):

    # ---------
    #

    phi = np.arctan2(cent[1], cent[0])
    eR = np.r_[np.cos(phi), np.sin(phi)]
    ephi = np.r_[-np.sin(phi), np.cos(phi)]

    ang = np.arcsin(R / np.hypot(*cent))
    vect = (-eR) * np.cos(ang) - sign * ephi * np.sin(ang)
    vect = vect / np.linalg.norm(vect)

    return vect


def _FOV(
    cent_out, vect_out,
    cent_in, vect_in,
    R0,
    rplasma,
    cent=None,
):

    # ------------------
    # intersect vect_out
    # ------------------

    kk_out_out, isout_out_out = _intersect(cent_out, vect_out, R0 + rplasma)
    kk_out_in, isout_out_in = _intersect(cent_out, vect_out, R0 - rplasma)

    kk_out = np.r_[kk_out_out, kk_out_in]
    iok_out = np.r_[isout_out_out, ~isout_out_in]
    iok = np.isfinite(kk_out) & iok_out
    assert iok.sum() >= 1
    kout = np.min(kk_out[iok])
    pt_out = cent_out + kout * vect_out

    # ------------------
    # intersect vect_in
    # ------------------

    kk_in_out, isout_in_out = _intersect(cent_in, vect_in, R0 + rplasma)
    kk_in_in, isout_in_in = _intersect(cent_in, vect_in, R0 - rplasma)

    kk_in = np.r_[kk_in_out, kk_in_in]
    iok_in = np.r_[isout_in_out, ~isout_in_in]
    iok = np.isfinite(kk_in) & iok_in
    assert iok.sum() >= 1
    kin = np.min(kk_in[iok])
    pt_in = cent_in + kin * vect_in

    assert np.allclose(np.linalg.norm(pt_out), np.linalg.norm(pt_in))
    Rpts = np.linalg.norm(pt_out)

    # ------------------
    # polyx, polyy
    # ------------------

    # ang
    ang_out = np.arctan2(pt_out[1], pt_out[0])
    ang_in = np.arctan2(pt_in[1], pt_in[0])
    ang_min = min(ang_out, ang_in)
    ang_max = max(ang_out, ang_in)
    if np.abs(ang_min - ang_max) > np.pi:
        ang_min, ang_max = ang_max, ang_min + 2*np.pi
    ang = np.linspace(ang_min, ang_max, 31)[::-1]

    polyx = np.r_[cent_out[0], Rpts * np.cos(ang), cent_in[0]]
    polyy = np.r_[cent_out[1], Rpts * np.sin(ang), cent_in[1]]

    # cent
    if cent is not None:
        polyx = np.r_[polyx, cent[0], cent_out[0]]
        polyy = np.r_[polyy, cent[1], cent_out[1]]

    return polyx, polyy


def _intersect(cent, vect, R):

    # ----------
    # kk

    # AM = ku
    # R^2 = (OA + AM)^2 = RA^2 + k^2 + 2 ku OA
    a = 1
    b = 2 * np.sum(vect * cent)
    c = np.sum(cent**2) - R**2
    delta = b**2 - 4 * a * c

    kk = np.full((2,), np.nan)
    if delta == 0:
        kk[0] = -b / (2*a)
    elif delta > 0:
        kk = (-b + np.r_[1, -1] * np.sqrt(delta)) / (2 * a)

    # ----------
    # isout

    xx = cent[0] + kk * vect[0]
    yy = cent[1] + kk * vect[1]
    phi = np.arctan2(yy, xx)
    isout = (vect[0] * np.cos(phi) + vect[1] * np.sin(phi)) > 0.

    assert isout.sum() <= 1

    return kk, isout


def _intersect_line(cent, vect, ref, nin):

    # ----------
    # kk

    # cent_M = kk * vect
    # (ref_cent + cent_M).nin = 0
    # (ref_cent + kk * vect).nin = 0
    kk = - np.sum((cent - ref) * nin) / np.sum(vect * nin)

    # ----------
    # reflected vector

    vect2 = vect - 2 * np.sum(vect * nin) * nin
    vect2 = vect2 / np.linalg.norm(vect2)

    return cent + kk * vect, vect2
