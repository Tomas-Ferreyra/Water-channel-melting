#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:02:25 2025

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import glob 
from time import time

#%%

# path = '/Volumes/Ice blocks/Scan water channel/25-10-24/'
path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'
# path = '/Volumes/Ice blocks/Scan water channel/25-12-12/'

# blink = np.load(path+'led_blink.npz')
# u_mes,d_mes = blink['u_mes'], blink['d_mes']

ice_x = np.load(path+'ice_x_0.npy')
ice_y = np.load(path+'ice_y_0.npy')
ice_z = np.load(path+'ice_z_0.npy')

#%%

plt.rcParams.update({'font.size':18})

i = 58
plt.figure( figsize=(5,10) )
ss=  plt.scatter( ice_z[i], ice_y[i], c=ice_x[i], s=1 )
plt.axis('equal')
cbar = plt.colorbar(ss, location='top')
# plt.xlim(-50,150)
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.title(r'$h$ (mm)',fontsize=12, pad=70)
plt.savefig('./Documents/t29_m30(2).png',dpi=200, bbox_inches='tight')
plt.show()


#%%
i = 1

points = np.array([ice_z[i], ice_y[i], ice_x[i]]).T
nans = np.where(np.sum(np.isnan(points),axis=1)>0)[0]
points_f = np.delete( points, nans, axis=0 )
# tri = Delaunay( points_f )

# plt.figure()
# plt.tricontour( ice_z[i], ice_y[i], ice_x[i] )
# plt.contour( ice_z[i], ice_y[i], ice_x[i], level=10 )
# plt.show()

# zz,yy = np.meshgrid(  )

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf( points_f[:,0], points_f[:,1], points_f[:,2], linewidth=0.2, antialiased=True)
plt.show()


#%%
# =============================================================================
# Water channel data
# =============================================================================

t1 = time()
path = '/Volumes/Ice blocks/Scan water channel/25-09-29/'
file = path + '2*.txt'
filename = glob.glob(file)[-1]

# file = path + '251212_171218.txt'
data = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )

t2 = time()
t2-t1
#%%
tiempo = data[:,0] / 60
Q_tunnel_setp = data[:,1]
Q_tunnel = data[:,2]
S_pump_setp = data[:,3]
Pdiff_CVF = data[:,4]
T_amb = data[:,5]
T_top = data[:,6]
T_bot = data[:,7]

plt.rcParams.update({'font.size':12})

# plt.figure()
# # plt.plot( data[:,d] )
# plt.plot(tiempo, Q_tunnel_setp )
# plt.plot(tiempo, Q_tunnel )
# plt.plot(tiempo, S_pump_setp )
# plt.grid()
# plt.show()

plt.figure()
plt.plot(tiempo, T_amb, label='amb' )
plt.plot(tiempo, T_top, label='top' )
plt.plot(tiempo, T_bot, label='bot' )
plt.grid()
plt.legend()
plt.show()





#%%
