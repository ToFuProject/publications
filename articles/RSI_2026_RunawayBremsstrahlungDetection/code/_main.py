import os
import sys


_PATH_HERE = os.path.dirname(__file__)
_PATH_PUBLI = os.path.dirname(os.path.dirname(os.path.dirname(_PATH_HERE)))
_PATH_PROJECTS = os.path.dirname(_PATH_PUBLI)
_PATH_TF = os.path.join(_PATH_PROJECTS, 'tofu')
sys.path.insert(0, _PATH_TF)
import tofu as tf
sys.path.pop(0)


# #####################################################
# #####################################################
#           Local imports
# #####################################################


from ._fig01_cross import main as fig01
from ._fig02_dist  import main as fig02
from ._fig04_bremsstrahlung import main as fig04
from ._fig05_tokamak import main as fig05
from ._fig06_responsivities import main as fig06


# #####################################################
# #####################################################
#               main
# #####################################################


def main(
    ne_m3=None,
):

    # ---------------
    # plot all
    # ---------------

    for fig in [fig01, fig02, fig04, fig05, fig06]:
        fig(
            ne_m3=ne_m3,
        )

    return
