'''

Look at SCRAM runs for Ca

cjperks
June 16, 2026

'''


# Modules
import numpy as np
import sys, os
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import scipy.constants as cnt
from scipy.interpolate import interp1d
import ast

from atomic_world.run_SCRAM.main import read_v86 as r86

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Species
elem = 'O_'
Znuc = 8

# Plasma
ne_cm3 = 1e14
Te_eV = 1e3

Te_long = np.logspace(np.log10(1000), np.log10(10000), 51) # [eV]
ne_long = 1e14*np.ones_like(Te_long)

charges = np.arange(0,Znuc+1)[None,:]

###################################################
#
#           SCRAM data
#
###################################################

# Common file path
fol = os.path.join(
    '/home/cjperks',
    'sharing/for_dVEZINET',
    '260616_SXR_emis'
    )
# Data file
fil = '260616_SCRAM86_%s_data.npz'%(elem.split('_')[0])

# Loads pickle object
dscram = np.load(
    os.path.join(fol, fil),
    allow_pickle=True
    )['arr_0'][()]


#########################################
#
#           Plotting
#
#########################################

Znuc = r86._get_nele(sym=elem.split('_')[0])

### --- Plots ion/rec rates for a charge state --- ###

fig1, ax1 = plt.subplots(
    1,2,
    figsize = (10,6)
    )
nele = 2

# Rates
ax1[0].plot(
    dscram['Te']['data']/1e3,
    dscram['scd']['data'][:,Znuc-nele],
    'b-',
    #label = 'ci+ea; SCRAM'
    )
ax1[0].plot(
    dscram['Te']['data']/1e3,
    dscram['acd']['data'][:,Znuc-nele],
    'b--',
    #label = 'rr+dr; SCRAM'
    )

ax1[0].plot(
    np.nan, np.nan,
    'k-',
    label = 'ioniz.'
    )
ax1[0].plot(
    np.nan, np.nan,
    'k--',
    label = 'recomb.'
    )

ax1[0].grid('on')
ax1[0].set_xscale('log')
ax1[0].set_xlabel(r'$T_e$ [keV]')
ax1[0].set_yscale('log')
ax1[0].set_ylabel(r'rate [$cm^3/s/electron$]')
ax1[0].set_xlim(0.1, 2.5)
ax1[0].set_ylim(1e-14,1e-9)

leg = ax1[0].legend(
    loc = 'lower right',
    labelcolor='linecolor',
    borderaxespad=0,
    )

# Abundance
ax1[1].plot(
    dscram['Te']['data']/1e3,
    dscram['Xz']['data'][:,Znuc-nele],
    'b-',
    label = 'SCRAM'
    )

ax1[1].grid('on')
ax1[1].set_xscale('log')
ax1[1].set_xlabel(r'$T_e$ [keV]')
ax1[1].set_yscale('log')
ax1[1].set_ylabel(r'abundance [frac]')
ax1[1].set_xlim(0.1, 2.5)
ax1[1].set_ylim(1e-3, 1e0)

leg = ax1[1].legend(
    loc = 'lower left',
    labelcolor='linecolor',
    borderaxespad=0,
    )


fig1.suptitle(
    r'%s; nele=%i'%(
        elem, nele
        )
    )

plt.tight_layout()



### --- Plots <Z> and cooling curve --- ###

fig2, ax2 = plt.subplots(
    1,2,
    #figsize = (12,6)
    )

# <Z>
ax2[0].plot(
    dscram['Te']['data']/1e3,
    dscram['<Z>']['data'],
    'b-',
    label = 'SCRAM'
    )


ax2[0].plot(
    [0.1, 30],
    [Znuc-2, Znuc-2],
    'k--'
    )
ax2[0].plot(
    [0.1, 30],
    [Znuc-10, Znuc-10],
    'k--'
    )
if Znuc > 28:
    ax2[0].plot(
        [0.1, 30],
        [Znuc-28, Znuc-28],
        'k--'
        )

ax2[0].grid('on')
ax2[0].set_xscale('log')
ax2[0].set_xlabel(r'$T_e$ [keV]')
ax2[0].set_ylabel('<Z>')
ax2[0].set_xlim(0.1, 2.5)
ax2[0].set_ylim(max(Znuc-10,0),Znuc)

leg = ax2[0].legend(
    loc = 'lower right',
    labelcolor='linecolor',
    borderaxespad=0,
    )



# Prad
ax2[1].plot(
    dscram['Te']['data']/1e3,
    dscram['Prad']['data'],
    'b-',
    label = 'SCRAM'
    )


ax2[1].grid('on')
ax2[1].set_xscale('log')
ax2[1].set_xlabel(r'$T_e$ [keV]')
ax2[1].set_yscale('log')
ax2[1].set_ylabel(r'$P_{rad}$ [$W*m^3/atom/electron$]')
ax2[1].set_xlim(0.1, 2.5)
ax2[1].set_ylim(1e-35,1e-32)


fig2.suptitle(
    r'%s'%(
        elem
        )
    )

plt.tight_layout()



### --- Plots the spectrum at a single Te --- ###


fig3, ax3 = plt.subplots()

ind = np.argmin(abs(
    Te_eV - dscram['Te']['data']
    ))


ax3.plot(
    dscram['E_photon']['data']/1e3,
    dscram['emis_tot']['data'][ind,:],
    'b-',
    label='tot'
    )

if True:
    ax3.plot(
        dscram['E_photon']['data']/1e3,
        dscram['emis_ff']['data'][ind,:],
        'b--',
        label='ff'
        )

    ax3.plot(
        dscram['E_photon']['data']/1e3,
        dscram['emis_fb']['data'][ind,:],
        'b:',
        label='fb'
        )

    ax3.plot(
        dscram['E_photon']['data']/1e3,
        dscram['emis_bb']['data'][ind,:],
        'b-.',
        label='bb'
        )

if False:

    # List of ions of interest
    ss = dscram['emis_ion']['name_long']
    neles = ast.literal_eval(ss[ss.index('['):])
    for nn, nele in enumerate(neles):
        ax3.plot(
            dscram['E_photon']['data']/1e3,
            (
                dscram['emis_ion']['data'][ind,:,nn]
                *dscram['Xz']['data'][ind,Znuc-nele]    # [1/ion] --> [1/atom], applies coronal charge balance
                ),
            '-.',
            color = colors[nn],
            label='nele=%i'%(nele)
            )


if False:
    ax3.plot(
        dscram['E_photon']['data']/1e3,
        dscram['emis_tot']['data'][ind,:]*ff_fil_fine,
        'r-',
        label='filtered'
        )


ax3.grid('on')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylim(1e-34, 1e-27)

ax3.set_xlabel('Photon energy [keV]')
ax3.set_ylabel(r'Emissivity [$J*cm^3/s/eV/atom/electron$]')

leg = ax3.legend(
    loc = 'upper right',
    labelcolor='linecolor',
    borderaxespad=0,
    )


ax3.set_title(
    r'%s; $T_e$=%0.1f keV'%(
        elem, Te_eV/1e3
        )
    )

plt.tight_layout()




### --- Plots the spectrum at v. Te --- ###


fig4, ax4 = plt.subplots(
    #figsize = (10,8)
    )


cmap = get_cmap('rainbow')
norm = Normalize(
    vmin=dscram['Te']['data'].min()/1e3,
    vmax=dscram['Te']['data'].max()/1e3,
    )

for ind_te in np.arange(len(dscram['Te']['data'])):

    color = cmap(norm(
        dscram['Te']['data'][ind_te]/1e3
        ))

    ax4.plot(
        dscram['E_photon']['data']/1e3,
        dscram['emis_tot']['data'][ind_te,:],
        color = color
        )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for Matplotlib < 3.7
cbar = plt.colorbar(sm, ax=ax4)
cbar.set_label(r'$T_e$ [keV]')

ax4.grid('on')
ax4.set_yscale('log')
ax4.set_xscale('log')

ax4.set_ylim(1e-34, 1e-27)

ax4.set_xlabel('Photon energy [keV]')
ax4.set_ylabel(r'Emissivity [$J*cm^3/s/eV/atom/electron$]')


ax4.set_title(
    r'%s; SCRAM v8.6'%(
        elem
        )
    )

plt.tight_layout()


