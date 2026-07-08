import copy


import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpinteg
import scipy.constants as scpct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as asunits
import datastock as ds


from . import _load_spect_anis
from ._fig02_dist_type import _DDIST
from ._fig05_emiss import _RE, _DDMIX
from ._fig07_responsivities import _PFE_RESPONSIVITIES
from ._savefig import main as savefig


# #######################################
# #######################################
#           DEFAULTS
# #######################################


_TE = 1e3
_JP_FRAC = 0.5


_DANGLES = {
    'in-vessel': {
        'head-on': np.r_[0, 40]*np.pi/180,
        'back': np.r_[140, 180]*np.pi/180,
        'resp': ['cvd_bare', 'cvd_filter', 'bolo'],
        'color': 'tab:orange',
        'alpha': 0.4,
    },
    'ex-cryostat': {
        'head-on': np.r_[60, 70]*np.pi/180,
        'back': np.r_[110, 120]*np.pi/180,
        'resp': [
            'mesxr_11_keV', 'mesxr_18_keV',
            'mehxr_20_keV', 'mehxr_60_keV',
            'spectro',
        ],
        'color': 'tab:green',
        'alpha': 0.4,
    },
}


_LRESP = [
    'bolo',
    'cvd_bare', 'cvd_filter',
    'mesxr_11_keV', 'mesxr_18_keV',
    'mehxr_20_keV', 'mehxr_60_keV',
]


# #######################################
# #######################################
#           Main
# #######################################


def main(
    dmix=None,
    # cases
    Te_eV=None,
    # d2cross
    d2cross_phi=None,
    # dist
    ne_m3=None,
    jp_Am2=None,
    # RE
    re=None,
    dominant=None,
    jp_fraction_re=None,
    Efield_par_Vm=None,
    Ekin_min_eV=None,
    Ekin_max_eV=None,
    sigmap=None,
    pnormW=0,
    # jp_fraction_re=np.linspace(0.025, 0.975, 39),
    # plot
    figsize=(8, 10),
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

    if dmix is None:
        dmix = _DDMIX[1]

    # Te
    if Te_eV is None:
        Te_eV = _TE

    # jp_fraction_re
    if jp_fraction_re is None:
        jp_fraction_re = _JP_FRAC

    # Maxwell
    kwd_max = {'ne_m3': ne_m3, 'jp_Am2': jp_Am2}

    # RE
    re = ds._generic_check._check_var(
        re, 're',
        types=str,
        default=_RE,
        allowed=sorted(_DDIST['RE'].keys()),
    )

    lRE = [
        'dominant', 'jp_fraction_re', 'Efield_par_Vm',
        'Efield_par_Vm', 'Ekin_max_eV', 'Ekin_min_eV',
        'sigmap', 'pnormW'
    ]
    kwd_RE = {
        kk: _DDIST['RE'][re].get(kk) if vv is None else vv
        for kk, vv in locals().items()
        if kk in lRE
    }

    # --------------
    # load elements
    # --------------

    demiss = {}
    kwd = dict(kwd_max)
    kwd.update(**kwd_RE)
    demiss, ddist, dmix = _load_spect_anis.main(
        dmix=dmix,
        # d2cross
        d2cross_phi=d2cross_phi,
        # dist
        **kwd,
    )

    # extract
    # E_ph = demiss[0]['common']['E_photon']['data']
    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    # ne = np.unique(ddist['plasma']['ne_m3']['data'])[0]
    # jp = np.unique(ddist['plasma']['jp_Am2']['data'])[0]
    # units = demiss['emiss']['maxwell']['ff']['units']

    indTe = np.argmin(np.abs(Teu - Te_eV))
    Te_eV = Teu[indTe]
    sli_emiss = (0, indTe, 0, slice(None), slice(None))

    # -------------------
    # load responsivities
    # -------------------

    # load
    dresp = {
        k0: v0.tolist()
        for k0, v0 in np.load(_PFE_RESPONSIVITIES, allow_pickle=True).items()
    }

    # -----------------------------
    # Integrate over responsivity
    # -----------------------------

    demiss_integ = {kk: {'maxwell': {}, 'RE': {}} for kk in dresp.keys()}
    for kresp, vresp in dresp.items():

        # Eph_resp
        iok_resp = np.isfinite(vresp['responsivity']['data'])
        iok_resp[iok_resp] = vresp['responsivity']['data'][iok_resp] > 0.
        Eph_resp = vresp['E_eV']['data'][iok_resp]

        # loop on dist
        for kdist, vdist in demiss['emiss'].items():

            # loop on emiss type
            for kemiss, vemiss in vdist.items():

                # skip
                if kemiss == 'ff_iso':
                    continue

                # Eph_emiss
                data = demiss['emiss'][kdist][kemiss]['data'][sli_emiss]
                iok_emiss = (
                    (demiss['E_ph']['data'] > Eph_resp.min())
                    & (demiss['E_ph']['data'] < Eph_resp.max())
                    & np.all(np.isfinite(data), axis=-1)
                    & np.all(data > 0., axis=-1)
                )
                Eph_emiss = demiss['E_ph']['data'][iok_emiss]

                # Eph
                Eph = np.unique(np.r_[Eph_resp, Eph_emiss])

                # interpolate responsivity
                resp = scpinterp.make_interp_spline(
                    Eph_resp,
                    vresp['responsivity']['data'][iok_resp],
                    k=1,
                    axis=0,
                    check_finite=True,
                )(Eph)

                # units
                unitsE = asunits.Unit(demiss['E_ph']['units'])
                unitsR = asunits.Unit(vresp['responsivity']['units'])
                unitsD = asunits.Unit(demiss['emiss'][kdist][kemiss]['units'])
                units = unitsE * unitsR * unitsD

                # ph vs energy
                if 'W' in str(unitsR):
                    data = data * demiss['E_ph']['data'][:, None] * scpct.e
                    units = units * asunits.Unit('W.s')
                else:
                    units = units * asunits.Unit('ph')

                # interpolate  emissivity
                if np.any(iok_emiss):
                    sli = (iok_emiss, slice(None))
                    emiss = np.power(
                        10,
                        scpinterp.make_interp_spline(
                            np.log10(Eph_emiss),
                            np.log10(data[sli]),
                            k=1,
                            axis=0,
                            bc_type=None,
                            check_finite=True,
                        )(np.log10(Eph)),
                    )

                    # integrate
                    demiss_integ[kresp][kdist][kemiss] = {
                        'data': scpinteg.trapezoid(
                            resp[:, None] * emiss,
                            x=Eph,
                            axis=-2,
                        ),
                        'units': units,
                    }
                else:
                    zeros = np.zeros(data.shape[1:], dtype=float)
                    demiss_integ[kresp][kdist][kemiss] = {
                        'data': zeros,
                        'units': units,
                    }

    # -----------------------------
    # Integrate over angles
    # -----------------------------

    dang = {}
    dsignal = copy.deepcopy(demiss_integ)
    theta = demiss['theta_ph_vsB']['data']
    for kresp, vresp in dresp.items():

        # get relevant angle
        kang = [
            kk for kk, vv in _DANGLES.items()
            if kresp in vv['resp']
        ][0]
        dang[kresp] = kang

        # integrate each direction
        for idir, kdir in enumerate(['head-on', 'back']):

            # iang
            iang = (
                (theta >= _DANGLES[kang][kdir][0])
                & (theta <= _DANGLES[kang][kdir][1])
            )

            # solid angle assuming cone
            sang = scpinteg.trapezoid(
                np.sin(theta[iang]),
                x=theta[iang],
                axis=-1,
            ) * 2*np.pi

            # loop on dist
            for kdist, vdist in demiss_integ[kresp].items():

                # loop on emiss type
                for kemiss, vemiss in vdist.items():

                    # integrate
                    if vemiss['data'].shape[-1] == 1:
                        data = vemiss['data'] * sang
                    else:
                        data = scpinteg.trapezoid(
                            vemiss['data'][iang] * np.sin(theta[iang]),
                            x=theta[iang],
                            axis=-1,
                        ) * 2*np.pi

                    # store
                    if idir == 0:
                        dsignal[kresp][kdist][kemiss] = {
                            kdir: {
                                'data': data,
                                'units': vemiss['units'] * asunits.Unit('sr'),
                            }
                        }
                    else:
                        dsignal[kresp][kdist][kemiss][kdir] = {
                            'data': data,
                            'units': vemiss['units'] * asunits.Unit('sr'),
                        }

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.08, 'right': 0.98,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=3, **dmargin)
    dax = {}

    # --------------
    # axes - vs theta
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel(
        r'$\theta_{ph,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon_{ff}^{RE,\eta} / \max\left(\epsilon_{ff}^{RE,\eta}\right)$",
        fontsize=fontsize,
        fontweight='bold',
    )
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

    dax['theta'] = ax

    # --------------
    # axes - Absolute
    # --------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto')
    ax.set_ylabel(
        r"$\epsilon^{RE,\eta} / \max\left(\epsilon^{RE,\eta}\right)$",
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

    dax['abs'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot vs theta
    # --------------

    kax = 'theta'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ------------
        # loop on resp

        dcolor = {}
        for kresp, vresp in dresp.items():

            data = demiss_integ[kresp]['RE']['ff']['data']
            l0, = ax.plot(
                theta*180/np.pi,
                data / data.max(),
                ls='-',
                color=_DANGLES[dang[kresp]]['color'],
                lw=1,
                label=kresp,
            )
            dcolor[kresp] = l0.get_color()

        # ------------
        # loop on angles

        for kang, vang in _DANGLES.items():
            for kk in ['head-on', 'back']:
                ax.axvspan(
                    vang[kk][0]*180/np.pi,
                    vang[kk][1]*180/np.pi,
                    facecolor=vang['color'],
                    alpha=vang['alpha'],
                )

        # ------------
        # decorate

        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.legend(loc='upper right')
        ax.grid(True)

    # --------------
    # plot abs
    # --------------

    kax = 'abs'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ------------
        # loop on resp

        weight_counts = {'head-on': {}, 'back': {}}
        lresp = _LRESP
        ldist = sorted(dsignal[lresp[0]].keys())

        # loop
        total_headon = 0
        for kdir in weight_counts.keys():

            # detail
            for kdist in ldist:
                for kemiss in sorted(dsignal[lresp[0]][kdist].keys()):
                    data = np.array([
                        dsignal[kresp][kdist][kemiss][kdir]['data'].squeeze()
                        for kresp in lresp
                    ])
                    weight_counts[kdir][f"{kdist} {kemiss}"] = data

                    if kdir == 'head-on':
                        total_headon += data

            # normalize
            for kk, vv in weight_counts[kdir].items():
                weight_counts[kdir][kk] = vv / total_headon

        # --------
        # plot

        width = 0.2
        dcolor = {
            'maxwell bb': 'b',
            'maxwell fb': 'g',
            'maxwell ff': 'm',
            'RE ff': 'r',
        }
        lorder = ['maxwell bb', 'maxwell fb', 'maxwell ff', 'RE ff']
        for idir, kdir in enumerate(weight_counts.keys()):
            bottom = 0
            for kk in lorder:
                ax.bar(
                    np.arange(len(lresp)) + 1 - 0.15 + 0.3*idir,
                    weight_counts[kdir][kk],
                    width,
                    label=kk,
                    bottom=bottom,
                    color=dcolor[kk],
                )
                bottom += weight_counts[kdir][kk]

        ax.set_ylim(0, 1)
        ax.grid(True)

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return demiss, dresp, demiss_integ, dsignal


# #######################################
# #######################################
#           Subroutine
# #######################################

