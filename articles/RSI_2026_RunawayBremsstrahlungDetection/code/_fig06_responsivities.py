import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from ._savefig import main as savefig


# #####################################################
# #####################################################
#       DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_SAVE = os.path.join(_PATH_PAPER, 'figures')
_PFE_RESPONSIVITIES = os.path.join(
    os.path.join(_PATH_PAPER, 'inputs'),
    'responsivities.npz',
)


# #####################################################
# #####################################################
#       main
# #####################################################


def main(
    pfe=None,
    lw=2,
    figsize=(7, 4),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Plot the spectral responsivities of a series of sensors

    """

    # --------------
    # load
    # --------------

    if pfe is None:
        pfe = _PFE_RESPONSIVITIES

    dresp = {
        k0: v0.tolist()
        for k0, v0 in np.load(pfe, allow_pickle=True).items()
    }

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.97,
        'bottom': 0.12, 'top': 0.98,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
    dax = {}

    # --------------
    # axes - resp
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('responsivity', fontsize=fontsize, fontweight='bold')

    dax['resp'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - resp
    # --------------

    kax = 'resp'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ---------------
        # loop on sensors

        dme = {'mesxr': False, 'mehxr': False, 'cvd': False}
        for k0, v0 in dresp.items():

            # resp, color, lab
            lk = [kk for kk in dme.keys() if kk in k0]
            if len(lk) == 1:
                kk = lk[0]
                if dme[kk] is False:
                    dme[kk] = v0.get('color', 'k')
                    lab = f"{kk}  -  {v0['responsivity']['units']}"
                else:
                    v0['color'] = dme[kk]
                    v0['ls'] = '--'
                    lab = None
            else:
                lab = f"{k0}  -  {v0['responsivity']['units']}"

            # lw
            if lw is None:
                lwi = v0.get('lw', 1)
            else:
                lwi = lw

            # plot
            ax.loglog(
                v0['E_eV']['data']*1e-3,
                v0['responsivity']['data'],
                ls=v0.get('ls', '-'),
                lw=lwi,
                c=v0.get('color', 'k'),
                marker=v0.get('marker', 'None'),
                label=lab,
            )

        ax.set_ylim(1e-4, 2)
        ax.grid(True)
        ax.legend()

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
    )

    return dax, dresp
