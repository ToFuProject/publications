

import os
import datastock as ds


# #######################################
# #######################################
#           Default
# #######################################


# PATHS
_PATH_HERE = os.path.dirname(__file__)
_PATH_PAPER = os.path.dirname(_PATH_HERE)
_PATH_SAVE = os.path.join(_PATH_PAPER, 'figures')


# #######################################
# #######################################
#           Main
# #######################################


def main(
    fig=None,
    pfe_save=None,
    path_save=None,
    file=None,
):

    # -------------
    # inputs
    # -------------

    # pfe_save
    pfe_save = ds._generic_check._check_var(
        pfe_save, 'pfe_save',
        types=(bool, str, type(None)),
        default=False,
    )

    # -------------
    # saving
    # -------------

    if pfe_save is not False:
        if pfe_save in [None, True]:
            name = f"{os.path.split(file)[-1][1:].replace('.py', '')}.png"
            if path_save is None:
                path_save = _PATH_SAVE
            pfe_save = os.path.join(_PATH_SAVE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return
