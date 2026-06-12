import os
import sys


_PATH_HERE = os.path.dirname(__file__)
_PATH_PUBLI = os.path.dirname(os.path.dirname(_PATH_HERE))
_PATH_PROJECTS = os.path.dirname(_PATH_PUBLI)
_PATH_TF = os.path.join(_PATH_PROJECTS, 'tofu')
sys.path.insert(0, _PATH_TF)
import tofu as tf
sys.path.pop(0)
_PATH_TFS = os.path.join(_PATH_PROJECTS, 'tofu_sparc')
sys.path.insert(0, _PATH_TFS)
import tofu_sparc as tfs
sys.path.pop(0)


# #####################################################
# #####################################################
#           Local imports
# #####################################################


from ._fig01 import main as fig01
from ._fig02 import main as fig02
from ._fig03 import main as fig03
from ._fig04 import main as fig04
from ._fig05 import main as fig05


# #####################################################
# #####################################################
#               main
# #####################################################


def main():

    # ---------------
    # plot all
    # ---------------

    for fig in [fig01, fig02, fig03, fig04, fig05]:
        fig()

    return
