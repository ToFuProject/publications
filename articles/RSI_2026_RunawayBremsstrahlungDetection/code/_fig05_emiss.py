


import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tofu as tf


from ._savefig import main as savefig


tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_INPUTS = os.path.join(_PATH_PAPER, 'inputs')


# SPECTRAL MODELLING FILES
_LPFE_SPECT = [
    ff for ff in os.listdir(_PATH_INPUTS)
    if ff.endswith('_data.npz')
    and any([ss in ff for ss in ['_SCRAM86_', '_FLYCHK_']])
]
_DPFE_SPECT = {
    ff.split('_')[-2]: os.path.join(_PATH_INPUTS, ff)
    for ff in _LPFE_SPECT
}


# CROSS-SECTION FILES
_DPFE_DCROSS = {
    'EH0': os.path.join(
        _PATH_INPUTS,
        'd2cross_phi_Ee01eV-100MeV-240log_Eph1eV-100MeV-241log_nthetaph61_nthetae060_EH.npz',
    ),
    'EH1': os.path.join(
        _PATH_INPUTS,
        'd2cross_phi_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_nthetaph61_nthetae060_EH.npz'
    ),
}


# #####################################################
# #####################################################
#       Main
# #####################################################


def main(
    d2cross_phi='EH1',
    elements='H',
    ne_m3=1e19,
    Te_plot=None,
    # plot
    figsize=(6, 8),
    fontsize=14,
    # save
    path_save=None,
    pfe_save=None,
    # unused
    **kwdargs,
):
    """ Validate Bremsstrahlung vs SCRAM and FLYCHK

    """

    # --------------
    # inputs
    # --------------

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.13, 'right': 0.99,
        'bottom': 0.10, 'top': 0.92,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - abs

    ax = fig.add_subplot(
        gs[0, 0],
        xscale='log',
        yscale='log',
        aspect='auto',
    )
    ax.set_xlabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\epsilon^{Max}_{ff}$" + f"  ({asunits.Unit(units)})",
        size=fontsize,
        fontweight='bold',
    )
    tit = (
        "Validation of Bremstrahlung implemented from EH cross-section\n"
        "Maxwellian distribution with  "
        + r"$j_{P}$" + " = 0 A/m2,  "
        + r"$n_e$" + f" = {ne_m3:1.0e}"
    )
    ax.set_title(
        tit,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['abs'] = {'handle': ax}

    # --------------
    # save
    # --------------

    savefig(
        fig=fig,
        pfe_save=pfe_save,
        path_save=path_save,
        file=__file__,
    )

    return
