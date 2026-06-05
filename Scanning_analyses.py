#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:02:25 2025

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as snd

import glob 
import h5py

from tqdm import tqdm
from time import time
from scipy.spatial import Delaunay
from scipy.ndimage import minimum_filter

from skimage.feature import peak_local_max
from skimage.filters import gaussian, roberts, sobel
from skimage.segmentation import watershed, mark_boundaries
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import local_minima, disk, remove_small_holes, binary_erosion, binary_dilation, binary_closing, binary_opening 


plt.rcParams.update({ 'font.size':12 })

def nangauss(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        V = altus.copy()
        V[np.isnan(altus)] = 0
        W = 0 * altus.copy() + 1
        W[np.isnan(altus)] = 0
        VV,WW = snd.gaussian_filter(V, sigma), snd.gaussian_filter(W, sigma)
        gdp = VV/WW
        gdp[np.isnan(altus)] = np.nan
        return gdp

def normalize(im):
    return (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))

def get_data(path, surface=True, channel=True, start=0, end=None):
    """
    Load reconstructed surface and/or water channel data

    Parameters
    ----------
    path : str
        path to folder of experiment.
    surface : bool, optional
        Load reconstructed surface data. The default is True.
    channel : bool, optional
        Load water channel data. The default is True.
    start: int
        Time of expeirment start of channel recording. The default is 0.
    end: int or None. end > start
        Time of expeirment end of channel recording. The default is None.

    Returns
    -------
    Reconstructed surface data:
        top and bottom camera reconstrucions. *_t: time (sec); *_x,*_y,*_z: x,y,z positions in mm
    Water channel data:
        tiempo: time, in sec
        Q_tunnel: flow rate, in m3/h
        T_amb: ambient temperature, in °C
        T_top: outlet temperature (after measurment section), in °C
        T_bot: inlet temperature (before measurment section), in °C
    """
    
    if surface and (not channel):
        with h5py.File(path + 'reconstructed_profile.hdf5', 'r') as f:
            
            top_t = f['Up/time'][:]
            top_x = f['Up/x'][:]
            top_y = f['Up/y'][:]
            top_z = f['Up/z'][:]
    
            dow_t = f['Down/time'][:]
            dow_x = f['Down/x'][:]
            dow_y = f['Down/y'][:]
            dow_z = f['Down/z'][:]
            
        bp = np.load(path+'back_position.npz')
        back_t, back_x = bp['time'], bp['position']
            
        return top_t, top_x, top_y, top_z, dow_t, dow_x, dow_y, dow_z, back_t, back_x, 0,0,0,0,0
    
    elif (not surface) and channel:
        filename = glob.glob(path + '2*.txt')[-1]
        expdata = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )
    
        tiempo = expdata[:,0]
        Q_tunnel_setp = expdata[:,1]
        Q_tunnel = expdata[:,2]
        S_pump_setp = expdata[:,3]
        Pdiff_CVF = expdata[:,4]
        T_amb = expdata[:,5]
        T_top = expdata[:,6]
        T_bot = expdata[:,7]
    
        return 0,0,0,0,0,0,0,0,0,0, tiempo[start:end], Q_tunnel[start:end], T_amb[start:end], T_top[start:end], T_bot[start:end]
    
    elif surface and channel:
        with h5py.File(path + 'reconstructed_profile.hdf5', 'r') as f:
            
            top_t = f['Up/time'][:]
            top_x = f['Up/x'][:]
            top_y = f['Up/y'][:]
            top_z = f['Up/z'][:]
    
            dow_t = f['Down/time'][:]
            dow_x = f['Down/x'][:]
            dow_y = f['Down/y'][:]
            dow_z = f['Down/z'][:]
            
        bp = np.load(path+'back_position.npz')
        back_t, back_x = bp['time'], bp['position']

        filename = glob.glob(path + '2*.txt')[-1]
        expdata = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )
    
        tiempo = expdata[:,0]
        Q_tunnel_setp = expdata[:,1]
        Q_tunnel = expdata[:,2]
        S_pump_setp = expdata[:,3]
        Pdiff_CVF = expdata[:,4]
        T_amb = expdata[:,5]
        T_top = expdata[:,6]
        T_bot = expdata[:,7]

        return top_t, top_x, top_y, top_z, dow_t, dow_x, dow_y, dow_z, back_t, back_x, \
                    tiempo[start:end], Q_tunnel[start:end], T_amb[start:end], T_top[start:end], T_bot[start:end]
    
    
    
def get_data_grid(path, start=0, end=None):

    with h5py.File(path + 'grid_reconstructed_profile.hdf5', 'r') as h5:
        
        back_t = h5['Back/time'][:]
        back_x = h5['Back/pos'][:]

        rec_t = h5['Reconstruction/time'][:]
        rec_x = h5['Reconstruction/x'][:]
        rec_y = h5['Reconstruction/y'][:]
        rec_z = h5['Reconstruction/z'][:]
        
    filename = glob.glob(path + '2*.txt')[-1]
    expdata = np.loadtxt( filename, skiprows=8, usecols=[0,2,3,4,6,19,20,21] )

    tiempo = expdata[:,0]
    # Q_tunnel_setp = expdata[:,1]
    Q_tunnel = expdata[:,2]
    # S_pump_setp = expdata[:,3]
    # Pdiff_CVF = expdata[:,4]
    T_amb = expdata[:,5]
    T_top = expdata[:,6]
    T_bot = expdata[:,7] 
        
    return rec_t, rec_x, rec_y, rec_z, back_t, back_x, \
            tiempo[start:end], Q_tunnel[start:end], T_amb[start:end], T_top[start:end], T_bot[start:end]


def aspect_ratio_fig(ax, xdata, ydata, zdata ):
    
    lxd, lxi = np.nanmax(xdata), np.nanmin(xdata)
    lyi, lyd = np.nanmin(ydata), np.nanmax(ydata)
    lzd, lzi = np.nanmin(zdata), np.nanmax(zdata)
    
    ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.05)
    ax.set_zlim(lzd-5,lzi+5)
    ax.set_xlim(lxd+5,lxi-5)
    ax.set_ylim(lyi-5,lyd+5)
    

def to_be_filled( img ):
    mask = ~np.isnan( img )

    kernel = np.ones( (2,4) )
    mask1 = binary_opening(mask, kernel)    
    mask2 = binary_closing(mask1, kernel)    
    lab3 = label(mask2)
    mask3 = lab3==1

    empty_points = (mask3*1. - mask)>0
    return empty_points
    

def fill_reconstruction_frame( img, sigma = 1):
    sig_x, sig_y = 1*sigma,5*sigma

    filled_img = np.copy(img)
    empty_points = to_be_filled( img )
    eys, exs = np.where(empty_points)
    idy,idx = np.indices(img.shape)


    for n in range(len(eys)):
        ey,ex = eys[n], exs[n]
        
        dist2 = (idy - ey)**2 / sig_y**2 + (idx - ex)**2 / sig_x**2
        wiegh = np.exp( -dist2/2 )
        filt = (wiegh>0.01) * (~np.isnan(img))
        
        ddd = np.average( img[filt], weights=wiegh[filt] )
        filled_img[ey,ex] = ddd 

    return filled_img
        
def fill_reconstruction(rec_x, sigma=1):
    nt = len(rec_x)
    filled_rec = np.zeros_like(rec_x)
    for n in range(nt):
        filled_rec[n] = fill_reconstruction_frame( rec_x[n], sigma = 1)
        
    return filled_rec
    
    
def kurvature(rec_x, rec_t, rec_y, rec_z, sigmas=[0,10,2] ):
    if rec_x.ndim == 3:    
        tsig, ysig, zsig = sigmas
        rec_xg = nangauss(rec_x, [tsig,ysig,zsig])
    
        gt, gy, gz = np.gradient( rec_xg, rec_t, rec_y[:,0], rec_z[0] )
        _,gyy,gyz = np.gradient( gy , rec_t, rec_y[:,0], rec_z[0] )
        _,gzy,gzz = np.gradient( gz , rec_t, rec_y[:,0], rec_z[0] )
        kurv = ( (1 + gyy**2)*gzz + (1+gzz**2)*gyy - 2*gz*gy*gzy) / (1+gz**2+gy**2)**(3/2)
        
    elif rec_x.ndim == 2:
        tsig, ysig, zsig = sigmas
        rec_xg = nangauss(rec_x, [ysig,zsig])
    
        gy, gz = np.gradient( rec_xg, rec_y[:,0], rec_z[0] )
        gyy,gyz = np.gradient( gy , rec_y[:,0], rec_z[0] )
        gzy,gzz = np.gradient( gz , rec_y[:,0], rec_z[0] )
        kurv = ( (1 + gyy**2)*gzz + (1+gzz**2)*gyy - 2*gz*gy*gzy) / (1+gz**2+gy**2)**(3/2)
        
    
    return kurv, rec_xg, gyy, gzz, gy,gz,gt

def get_markers(img, kurvi, min_distance=1, fp1=disk(3), fp2=disk(2), fp3=None):
    
    amg = np.copy(img)
    amg[ np.isnan(img) ] = 0.
    
    mask1 = ~np.isnan(kurvi)
    mask2 = binary_closing( mask1, footprint=fp1)
    mask3 = binary_erosion( mask2, footprint=fp2)
    mask = mask3 * mask1
    
    my,mx = peak_local_max( amg, min_distance=min_distance, labels=mask, footprint=fp3   ).T
    
    marks = np.zeros_like(amg)
    marks[my,mx] = 1
    # marks[ binary_dilation( np.isnan(img), disk(2) ) ] = 0

    return marks>0

def countour_mean( wtshed, img, lab ):
    sccaa = wtshed == lab
    ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )    
    conval = img[ccoo]
    return np.nanmean(conval)


def image_nanstdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanstd(intensities[region]) #, ddof=0)
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region]) #, ddof=0)
def image_minmax(region, intensities):
    return np.nanmax(intensities[region]) - np.nanmin(intensities[region])  
def image_min(region, intensities):
    return np.nanmin(intensities[region])  



def watershed_segmentation(rec_f, rec_t, rec_y, rec_z, sigmas=[15, 3], kernel_s=4, sig=1.4, min_distance=5, fp1=disk(3), fp2=disk(2) ):
    ysig, zsig = sigmas
    
    wats, lmins = [], []
    scaprop, scapropk = [], []
    kurv, rec_xg, _,_,_,_,_ = kurvature(rec_f, rec_t, rec_y, rec_z, sigmas=[0, ysig, zsig ] )

    kernel = np.ones((5*kernel_s,1*kernel_s))

    for i in tqdm(range(len(rec_f))):
     
        sigma = np.array([5,1]) * sig
        img = nangauss(normalize(rec_f[i]), sigma)

        lmin = get_markers( img, kurv[i], min_distance=5, fp1=fp1, fp2=fp2, fp3=kernel )
        
        mask = ~np.isnan(kurv[i])

        wts = watershed( kurv[i], mask = mask, markers=label(lmin), watershed_line=False )
            
        wats.append(wts)
        lmins.append(lmin)
        
        scaprop.append( regionprops(wts, intensity_image=rec_f[i], extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        scapropk.append( regionprops(wts, intensity_image=kurv[i], extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        
    return wats, scaprop, scapropk


def scallop_props(rec_f, rec_y, scaprop, scapropk, dz=2.5, dy=0.5, area_thres=100, std_thres=0.013, dist_top=50, dist_bot=120):
    
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme, ssi = [], [], [], []
        
    for i in tqdm(range(len(scaprop))):
    # for i in [60]:
        cen, cenw, scas, slab = [], [], [], []
        nsd, sd, mm, sdi = [], [], [], []
        me, nijs = [], []
        
        mask = ~np.isnan( rec_f[i]) 
        medmax = ( np.median(np.max( rec_y * mask, axis=0 )) - np.min(rec_y) ) / dy
        
        for j in range(len(scaprop[i])):
            ksd = scapropk[i][j].image_stdev
            sarea = scaprop[i][j].area * dz*dy
            cey,cex = scaprop[i][j].centroid
            
    
            if sarea > area_thres and ksd > std_thres and cey < medmax - dist_top and cey > dist_bot:
                cen.append( scaprop[i][j].centroid )
                cenw.append( scaprop[i][j].centroid_weighted )
                scas.append( sarea /dz/dy )
                slab.append( scaprop[i][j].label )
                
                nsd.append( scaprop[i][j].image_nanstdev )
                sd.append( scapropk[i][j].image_stdev )
                sdi.append( scaprop[i][j].image_stdev )
                mm.append( scaprop[i][j].image_minmax )

                me.append( countour_mean( wats[i], rec_f[i], j ) - scaprop[i][j].image_min )
                nijs.append( (scaprop[i][j].moments_normalized).T )
    
        cen, cenw, scas, slab = np.array(cen), np.array(cenw), np.array(scas), np.array(slab)
        nsd, sd, mm, sdi = np.array(nsd), np.array(sd), np.array(mm), np.array(sdi)
        me, nijs = np.array(me), np.array(nijs)
        
        if len(nijs) > 0:
            bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
            hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        else:
            bn,hn = np.nan, np.nan
        
        nsca.append( len(cen) )
    
        ssi.append( sdi )
        ssca.append(scas )
        cents.append(cen )
        centws.append(cenw )
        labe.append(slab )
        lx.append( bn )
        ly.append( hn )
        ssd.append(sd )
        smm.append(mm )
        sme.append(me)

    return lx,ly, ssca, cents, centws, nsca, nscaf, labe, ssd, smm, sme , ssi


def calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=False, integral=True ):
    tsig, ysig, zsig = sigmas
    rec_xg = nangauss(rec_x + rec_back[:,None,None] , [tsig,ysig,zsig])
    
    if f == 0:
        gt1, _,_ = np.gradient( rec_xg[:15], rec_t[:15], rec_y[:,0], rec_z[0], edge_order=2 )
        gt2, _,_ = np.gradient( rec_xg[17:], rec_t[17:], rec_y[:,0], rec_z[0], edge_order=2 )
        _, gy, gz = np.gradient( rec_xg, rec_t, rec_y[:,0], rec_z[0], edge_order=2 )
        
        gt = np.zeros_like( rec_xg ) * np.nan
        gt[:15], gt[17:] = gt1,gt2
        
    else:
        gt, gy, gz = np.gradient( rec_xg, rec_t, rec_y[:,0], rec_z[0] )
        
    mask = ~np.isnan(gt)
    areas = np.trapezoid( np.trapezoid( mask, dx=dz, axis=2 ), dx=dy, axis=1)

    gt0 = np.copy(gt)
    gt0[np.isnan(gt0)] = 0
    av_melt = np.trapezoid( np.trapezoid( gt0, dx=dz, axis=2 ), dx=dy, axis=1)
    av_melt = av_melt / areas /2
    
    st = np.nanstd( av_melt )
    
    if time_val:
        if integral: return gt, av_melt, 0
        else: return gt, np.nanmean( gt, axis=(1,2) ), 0
            
    else:
        if integral: 
            
            
            av_melt[np.isnan(av_melt)] = 0
            avg_melt = np.trapezoid( av_melt, x=rec_t  ) / rec_t[-1]
            return gt, avg_melt, st
        else: return gt, np.nanmean( gt ), np.nanstd( gt )



    
all_folders = ['25-08-07','25-09-29','25-10-24','25-11-17','25-12-12','26-01-27','26-02-17','26-03-05', '26-05-26' ]
grid_vels   = [ 0.5      , 0.3      , 0.9      , 0.3      , 0.0      , 0.3      , 0.9      , 0.3      ,  0.3       ]
nf = len(all_folders)

starts = [ 1900 , 650  , 750  , 1250 , 2060 , 1000 , 108  , 190640, 940 ]
ends   = [ 21700, 28000, 28600, 35250, 18980, 12580, 14980, 205690, 9433 ]

rho_ice = 916.8 # kg / m^3
latent = 334e3 # J / kg  or m^2/s^2 
thcon = 0.6 # m kg / s^3 °C
len_ice = 0.71 # m
nu = 1.004e-6 # m^2 / s

length0 = 0.7 #m

area_Q = 0.3 * 0.08

#%%

t1 = time()

f = 8

path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
surface = True
channel = True

rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )

if f==0: rec_x[16] = np.nan
elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]
    

rec_f = fill_reconstruction(rec_x, sigma=1)
rec_back = np.interp(rec_t, back_t, back_x )
    
lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
dz, dy = 2.5, 0.5

t2 = time()
print(t2-t1)

#%%

i = 20
ysig = 15
zsig = 3

lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]

# fig, ax = plt.subplots(1,2, figsize=(6,5))

# s1 = ax[0].imshow( rec_x[i], extent=lims, origin='lower' )
# s2 = ax[1].imshow( rec_f[i], extent=lims, origin='lower' )

# plt.colorbar(s1, ax=ax[0])
# plt.colorbar(s2, ax=ax[1])
# plt.show()

# kurv, _,_,_,_,_,_ = kurvature(rec_x, rec_t, rec_y, rec_z, sigmas=[0, ysig, zsig ] )
# plt.figure()
# plt.imshow( kurv[i], extent=lims, origin='lower' )
# plt.colorbar()
# plt.show()

# kurv, _,_,_,_,_,_ = kurvature(rec_f, rec_t, rec_y, rec_z, sigmas=[0, ysig, zsig ] )
# plt.figure()
# plt.imshow( kurv[i], extent=lims, origin='lower' )
# plt.colorbar()
# plt.show()

# plt.figure()
# # for i in range(len(rec_t)):
# for i in range(8,19):
    
#     plt.plot( rec_y[:,0], rec_x[i,:,30] + rec_back[i], '-', label=i )
# plt.legend()
# plt.show()

plt.figure()
# plt.plot( tiempo, T_amb, '.-' , label='Amb' )
plt.plot( tiempo, T_top, '.-' , label='Top')
plt.plot( tiempo, T_bot, '.-' , label='Bot' )

# plt.plot( tiempo, Q_tunnel, '.-', label='Q' )
plt.plot( tiempo, Q_tunnel / area_Q / 3600 , '.-', label='vel (m/s)'  )

# plt.plot( T_top, '.-' , label='Top')
# plt.plot( T_bot, '.-' , label='Bot' )
# plt.plot( Q_tunnel / area_Q / 3600 , '.-', label='vel (m/s)'  )

plt.legend()
plt.show()




#%%
# =============================================================================
# Nu vs Re
# =============================================================================
#approx Re (with spped from set ) and Nu

surface = True
channel = True


Nus, Res = [],[]
Nus_sd, Res_sd = [],[]

for f in tqdm(range(len(all_folders))):

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    
    if f==0: rec_x[16] = np.nan
    elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]
        
    
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
        
    lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
    dz, dy = 2.5, 0.5
    
    
    temp = 12.3 #°C
    temp =  (T_top[0] + T_bot[0])/2 
    
    gt, melt_rate, std_melt = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=False, integral=True )
    Nu = melt_rate/1000 * rho_ice * latent * length0 / (thcon * temp )
    nusd = std_melt/1000 * rho_ice * latent * length0 / (thcon * temp )
    
    area_Q = 0.3 * 0.08
    
    u_shear, std_shear = np.median(Q_tunnel) / area_Q / 3600, np.std(Q_tunnel) / area_Q / 3600
    Re = u_shear * length0 / nu
    resd = std_shear * length0 / nu
    
    Nus.append(Nu); Res.append(Re)
    Nus_sd.append(nusd); Res_sd.append(resd)
    
    
Nus,Res = np.array(Nus), np.array(Res)
Nus_sd,Res_sd = np.array(Nus_sd), np.array(Res_sd)

#%%

plt.figure()

cmap = plt.get_cmap('viridis',9)

# for i in range(len(Nus)):

plt.errorbar(Res, Nus, yerr=Nus_sd, markersize=0, fmt='.', capsize=5 , alpha=0.5)
gr = plt.scatter(Res, Nus,  c=grid_vels, cmap=cmap )


res = np.linspace(7e4,4.5e5, 100) 
# plt.plot( Res, Res**(3/4) * .1 )
plt.plot( res, res**(2/3) * .3, 'k--', label=r'Nu $\propto$ Re$^{2/3}$')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Re')
plt.ylabel('Nu')

plt.legend(loc='upper left' )

plt.colorbar(gr)

# plt.savefig('./Documents/Nu_Re.pdf',dpi=400, bbox_inches='tight')
plt.show()

# print(u_shear, temp)
# print(Nu, Re)

# plt.figure()
# plt.plot( tiempo, T_amb, '.-' , label='Amb' )
# plt.plot( tiempo, T_top, '.-' , label='Top')
# plt.plot( tiempo, T_bot, '.-' , label='Bot' )

# # plt.plot( tiempo, Q_tunnel, '.-', label='Q' )
# plt.plot( tiempo, Q_tunnel / area_Q / 3600 , '.-', label='vel (m/s)'  )

# plt.legend()
# plt.show()


#%%

avh = np.nanmean( rec_x, axis=(1,2) )

plt.figure()
plt.plot( back_t/60, back_x, '.-' )
plt.plot( rec_t/60, rec_back, '.' )

# plt.plot( rec_t/60, avh + rec_back, '.-'  )

# plt.show()


#%%
# =============================================================================
# Watershed
# =============================================================================

all_lx, all_ly, all_ssca, all_sme = [],[],[],[] 

rec_ts = []

# for f in range(len(all_folders)):
for f in [7]:

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    
    if f==0: rec_x[16] = np.nan
    elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]
        
    
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
        
    lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
    dz, dy = 2.5, 0.5
    
    s = 4
    sig = 1.4
    ysig, zsig  = 15, 3
    sigmas = [ysig, zsig]
    
    std_thres = 0.013
    if f == 7: std_thres = 0.015
    
    kurv, _,_,_,_,_,_ = kurvature(rec_f, rec_t, rec_y, rec_z, sigmas=[0, ysig, zsig ] )    
    wats, scaprop, scapropk = watershed_segmentation(rec_f, rec_t, rec_y, rec_z, sigmas=sigmas, kernel_s=s, sig=sig, min_distance=5, fp1=disk(3), fp2=disk(2) )
    lx,ly, ssca, cents, centws, nsca, nscaf, labe, ssd, smm, sme, ssi = scallop_props(rec_f, rec_y, scaprop, scapropk, \
                                                                                      dz=dz, dy=dy, area_thres=100, std_thres=std_thres, dist_top=50, dist_bot=120)
    
    all_lx.append(lx); all_ly.append(ly) 
    all_ssca.append(ssca); all_sme.append(sme)
    rec_ts.append(rec_t)
    
    # lx,ly, ssca, cents, centws, nsca, nscaf, labe, ssd, smm, sme = scallop_props(rec_f, rec_y, scaprop, scapropk, dz=dz, dy=dy, area_thres=0, std_thres=0.0, dist_top=-50)

        
#%%

n = 39

sobb = sobel(wats[n]) > 0
soy,sox = np.where(sobb)

colorb = (0,0,0)
cmap = plt.cm.RdBu
img_rgb = cmap(normalize(rec_f[n]))[..., :3]  # RGBA -> RGB
img_rgb[np.isnan(rec_f[n])] = np.nan

# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normalize)
# sm.set_array([])


fig, ax = plt.subplots(1,4, figsize=(8,5), layout='constrained', sharey=True )

# plt.imshow( rec_f[n], origin='lower', aspect=1/5 )
c1 = ax[0].imshow( rec_f[n], cmap=cmap, origin='lower', extent=lims)

ax[1].imshow( mark_boundaries( img_rgb, wats[n], color=colorb ), origin='lower', extent=lims) # aspect=1/5 )

ax[2].imshow( mark_boundaries( img_rgb, wats[n], color=colorb ), origin='lower', extent=lims) # aspect=1/5 )

mask = np.zeros_like(wats[n])
for j in range(len(labe[n])):
    mask += wats[n] == labe[n][j]
mask += np.isnan(rec_f[n])
amask = np.ma.masked_where(mask, mask)
ax[2].imshow( amask, alpha = 0.6, cmap='gray', origin='lower', extent=lims) # aspect=1/5)

c2 = ax[3].imshow( kurv[n], origin='lower', extent=lims )

ax[0].set_ylabel(r'$y$ (mm)', labelpad=-4.)
for j in range(4): ax[j].set_xlabel(r'$x$ (mm)')

fig.colorbar(c1, ax=ax[0], location='left' , aspect=50., label=r'$h$ (mm)' )
fig.colorbar(c2, ax=ax[3], location='right', aspect=50., label=r'$k$ (mm$^{-1}$)' ) 

# plt.savefig('./Documents/watershed_exp7.pdf',dpi=400, bbox_inches='tight')
plt.show()


#%%

n = 38

sobb = sobel(wats[n]) > 0
soy,sox = np.where(sobb)

plt.figure()
plt.imshow( rec_f[n], origin='lower', aspect=1/5 )
plt.plot( sox, soy,  'k.', markersize=1)
plt.plot( cents[n][:,1], cents[n][:,0], 'r.' )

for j in range(len(ssca[n])):
    plt.text(cents[n][j][1], cents[n][j][0] , str(round(ssd[n][j]/1. ,3)) )
    # plt.text(cents[n][j][1], cents[n][j][0] , str(round(ssi[n][j]/1. ,3)) )
    # plt.text(cents[n][j][1], cents[n][j][0] , str(round(ssca[n][j]/1. ,3)) )
    # plt.text(cents[n][j][1], cents[n][j][0] , str(round( ssd[n][j] / ssca[n][j] * 100 ,3)) )
    
plt.colorbar()
plt.show()

#%%

rec_back = np.interp(rec_t, back_t, back_x )

plt.figure()

for i in range(40):
    plt.plot( rec_y[:,0], -(rec_x[i,:,30] + rec_back[i]), '-', label=i )
plt.legend()
plt.show()



# plt.figure()
# plt.plot( back_t, back_x, '.-')
# plt.plot( rec_t, rec_back, '.')
# plt.show()

#%%
# =============================================================================
# Wavelength
# =============================================================================


fig, ax = plt.subplots()

# for f in [0,8]:
#     lx,ly = all_lx[f], all_ly[f]
#     rec_t = rec_ts[f]
    
#     mesx, sdsx = [],[]
#     mesy, sdsy = [],[]
    
#     for i in range( len(lx) ):
#         mesx.append( np.mean( lx[i] * dz ) )
#         sdsx.append( np.std( lx[i] * dz ) )
    
#         mesy.append( np.mean( ly[i] * dy ) )
#         sdsy.append( np.std( ly[i] * dy ) )
    
#     mesx, sdsx = np.array( mesx ), np.array( sdsx )
#     mesy, sdsy = np.array( mesy ), np.array( sdsy )
    
    
#     ax.plot( rec_t/60, mesx, '--', color='orange' ) #label=r'$\lambda_x$', 
#     ax.fill_between(rec_t/60, mesx-sdsx, mesx+sdsx, alpha=0.5, color='orange' )
    
#     ax.plot( rec_t/60, mesy, '--', color='g' ) #label=r'$\lambda_y$'
#     ax.fill_between(rec_t/60, mesy-sdsy, mesy+sdsy, alpha=0.5, color='g' )

# leg1 = [ax.plot([], [], '--', color=c)[0] for c in ['orange', 'g']]
# lgd = ax.legend(leg1, [r'$\lambda_x$', r'$\lambda_y$'])



for f in [4,5,6,7]:

    lx,ly = all_lx[f], all_ly[f]
    rec_t = rec_ts[f]
    
    mesx, sdsx = [],[]
    mesy, sdsy = [],[]
    
    for i in range( len(lx) ):
        mesx.append( np.mean( lx[i] * dz ) )
        sdsx.append( np.std( lx[i] * dz ) )
    
        mesy.append( np.mean( ly[i] * dy ) )
        sdsy.append( np.std( ly[i] * dy ) )
    
    mesx, sdsx = np.array( mesx ), np.array( sdsx )
    mesy, sdsy = np.array( mesy ), np.array( sdsy )

    # color = None
    # ax.plot( rec_t/60, mesx, '--', color=color ) #label=r'$\lambda_x$', 
    # ax.fill_between(rec_t/60, mesx-sdsx, mesx+sdsx, alpha=0.5, color=color )

    ax.plot( rec_t/60, mesy, '-'  ) #label=r'$\lambda_x$', 
    ax.fill_between(rec_t/60, mesy-sdsy, mesy+sdsy, alpha=0.5)
    
    color = 'b'
    # ax.plot( rec_t/60, mesy, '--', color=ç ) #label=r'$\lambda_y$'
    # ax.fill_between(rec_t/60, mesy-sdsy, mesy+sdsy, alpha=0.5, color=color )

# leg1 = [ax.plot([], [], '--', color=c)[0] for c in ['r', 'b']]
# lgd = ax.legend(leg1, [r'$\lambda_x$', r'$\lambda_y$'])


# plt.legend()
ax.set_xlabel(r'$t$ (min)')
ax.set_ylabel(r'$\lambda_y$ (mm)')

# plt.savefig('./Documents/wavelengthy_65.pdf',dpi=400, bbox_inches='tight')

plt.show()

#%%
# =============================================================================
# Area
# =============================================================================


fig, ax = plt.subplots()

# for f in [4,5,6,7]:
for f in [0,8]:

    ssca = all_ssca[f] 
    rec_t = rec_ts[f]
    
    mesa, sdsa = [],[]
    
    for i in range( len(ssca) ):
        mesa.append( np.mean( ssca[i] * dz*dy /100 ) )
        sdsa.append( np.std( ssca[i] * dz*dy /100 ) )
    
    mesa, sdsa = np.array( mesa ), np.array( sdsa )

    # for i in range( len(ssca) ):
    #     if ssca[i] is not np.nan:
    #         for l in range( len(ssca[i]) ):
    #             plt.plot( rec_t[i], ssca[i][l] * dz*dy /100 , '.' )
    
    
    plt.plot( rec_t/60, mesa )
    plt.fill_between(rec_t/60, mesa-sdsa, mesa+sdsa, alpha=0.5 )

# plt.legend()
plt.xlabel(r'$t$ (min)')
plt.ylabel(r'$A$ (cm$^2$)')

# plt.savefig('./Documents/area_65.pdf',dpi=400, bbox_inches='tight')

plt.show()


#%%
# =============================================================================
# Amplitude
# =============================================================================

fig, ax = plt.subplots()

# for f in [4,5,6,7]:
for f in [0,8]:

    sme = all_sme[f] 
    rec_t = rec_ts[f]

    
    mesa, sdsa = [],[]
    
    for i in range( len(sme) ):
        mesa.append( np.mean( sme[i] ) )
        sdsa.append( np.std( sme[i] ) )
    
    mesa, sdsa = np.array( mesa ), np.array( sdsa )
        
    # for i in range( len(sme) ):
    #     if sme[i] is not np.nan:
    #         for l in range( len(sme[i]) ):
    #             plt.plot( rec_t[i], sme[i][l] , '.' )
        
    plt.plot( rec_t/60, mesa )
    plt.fill_between(rec_t/60, mesa-sdsa, mesa+sdsa, alpha=0.5 )

# plt.legend()
plt.xlabel(r'$t$ (min)')
plt.ylabel(r'$H$ (mm)')

# plt.savefig('./Documents/amplitud_65.pdf',dpi=400, bbox_inches='tight')

plt.show()












#%%












#%%

i = 20
# edges = np.where( roberts(wts) >0 )
pmix, pmiy = rec_z[ np.where( lmins[i] ) ], rec_y[ np.where( lmins[i] ) ]


# plt.figure()
# plt.imshow( mark_boundaries( normalize(kurv[i]), wats[i]), extent=lims, origin='lower' )
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow( (kurv[i]), extent=lims, origin='lower' )
plt.plot( pmix, pmiy, 'r.' )
# plt.imshow( wats[i], extent=lims, origin='lower' )
plt.colorbar()
plt.show()

plt.figure()
plt.imshow( mark_boundaries( normalize(rec_x[i]), wats[i]), extent=lims, origin='lower' )
plt.plot( pmix, pmiy, 'r.' )
plt.colorbar()
plt.show()

plt.figure()
plt.imshow( normalize(rec_x[i]), extent=lims, origin='lower' )
plt.plot( pmix, pmiy, 'r.' )
plt.colorbar()
plt.show()


#%%

import matplotlib.tri as mtri
# from matplotlib.colors import LightSource


i = 14

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.view_init(80,-80, 0)

zg,yg,xg = rec_z.ravel(), rec_y.ravel(), rec_x[i].ravel()
me = np.nanmean(xg)

trigri = mtri.Triangulation( zg, yg)
aa = ax.plot_trisurf( zg, yg, -xg + me, triangles=trigri.triangles, cmap=plt.cm.jet, vmax=20, vmin=-20 )
# aa = ax.plot_trisurf( zg, yg, -xg + me, triangles=trigri.triangles, cmap=plt.cm.jet, vmax=0, vmin=-30 )
aspect_ratio_fig(ax, zg, yg, -xg+me)

# bb = ax.scatter( rec_z[edges], rec_y[edges], -rec_x[i][edges] + me, s=3, c='k'  )

# ax.set_xticks([140,100,60,20])
# ax.set_zticks([])
plt.colorbar(aa, ax=ax)
plt.gca().invert_xaxis()

# plt.savefig('./Documents/65_20.pdf',dpi=400, bbox_inches='tight')
plt.show()

#%%

# quiza deberia hacer un colosing e interpolar en valores con nan para hacer el watershed 
# sino cada vez que hay linea de nans el scallop queda dividido


i = 10
s = 4
ysig = 5 * s
zsig = 1 * s

# kernel = np.ones((25,5))
kernel = np.ones((5*s,1*s))

# kurv, rec_xg, gyy, gxx, gy,gz,gt = kurvature(rec_f, rec_t, rec_y, rec_z, sigmas=[0, ysig, zsig ] )


sigma = 0
img = nangauss(normalize(rec_f[i]), sigma)
marks0 = get_markers( img, kurv[i], min_distance=5, fp1=disk(0), fp2=disk(0), fp3=kernel  )  # fp1=disk(3), fp2=disk(2), fp3=kernel 

sigma = np.array([5,1]) * 1.4
img = nangauss(normalize(rec_f[i]), sigma)
marksg = get_markers( img, kurv[i], min_distance=5, fp1=disk(0), fp2=disk(0), fp3=kernel  )  # fp1=disk(3), fp2=disk(2), fp3=kernel 


plt.figure()
plt.imshow( normalize(rec_x[i]), extent=lims, origin='lower' )
plt.plot( rec_z[marks0], rec_y[marks0], 'r.', markersize=7 )
plt.plot( rec_z[marksg], rec_y[marksg], 'b.', markersize=5 )
plt.show()

# plt.figure()
# plt.imshow( kurv[i], extent=lims, origin='lower' )
# plt.plot( rec_z[marks0], rec_y[marks0], 'r.' )
# plt.plot( rec_z[marksg], rec_y[marksg], 'b.' )
# plt.show()





#%%

plt.figure()
# plt.imshow( ~np.isnan(kurv[i]) , extent=lims, origin='lower')
plt.imshow( binary_closing(~np.isnan(kurv[i]), disk(3)) , extent=lims, origin='lower')
plt.show()


#%%


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot sample data
X = rec_z
Y = rec_y
Z = rec_x[i]
ax.plot_surface(X, Y, Z, cmap=plt.cm.Spectral_r)
aspect_ratio_fig(ax, X,Y,Z)
# ax.view_init(50,-0, 90 )
ax.view_init(-50, 150, -70 )

def on_move(event):
    """Callback function to update angles in the title."""
    # Get current angles from the axis object
    elev, azim, roll = ax.elev, ax.azim, ax.roll
    # Update title with the new coordinates
    ax.set_title(f'Elevation: {elev:.1f}°, Azimuth: {azim:.1f}°, Roll: {roll:.1f}°')
    fig.canvas.draw_idle()

# Connect the mouse release event to the callback
# 'button_release_event' ensures it updates after you finish rotating
cid = fig.canvas.mpl_connect('button_release_event', on_move)

# For real-time updates while dragging, use 'motion_notify_event' instead
# cid = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()





#%%

for i in range(40,64):
# for i in []:

    mask = ~np.isnan(rec_x[i])
    
    fig,ax = plt.subplots(1,3, figsize=(6,5))
    ax[0].imshow( mask, extent=lims, origin='lower'  )
    ax[0].set_title( i )


    kernel = np.ones( (2,4) )
    mask1 = binary_opening(mask, kernel)    
    mask2 = binary_closing(mask1, kernel)    
    lab3 = label(mask2)
    mask3 = lab3==1

    ep = (mask3*1. - mask)>0

    ax[1].imshow( mask3, extent=lims, origin='lower'  )
    ax[1].set_title( i )

    ax[2].imshow( ep, extent=lims, origin='lower'  )
    ax[2].set_title( i )

    plt.show()





# i = 4
# mask = ~np.isnan(rec_x[i])

# plt.figure()
# # plt.imshow( mask, extent=lims, origin='lower'  )
# plt.imshow( rec_x[i], extent=lims, origin='lower'  )
# plt.show()

# problems 10, 5, 4,39, 37, 35, 34, 29, 23, 45



#%%
                
        
        
    
    
    


def interpolate_gaussian( xdata, ydata, zdata, zn, yn, dist_threshold=3, sigmas=[5,10], disable_bar=False ):
    sig_x, sig_y = sigmas
    sig_x2, sig_y2 = sig_x**2, sig_y**2
    values = xdata
    
    # zn = np.linspace( np.min(zdata), np.max(zdata), grid_size[0] )
    # yn = np.linspace( np.min(ydata), np.max(ydata), grid_size[1] )
    # zn,yn = np.meshgrid(zn,yn)

    points = np.column_stack((zdata, ydata))
    tree = cKDTree(points)
    grid_points = np.column_stack((zn.ravel(), yn.ravel()))
    dist, _ = tree.query(grid_points, k=1)
    gdist = dist.reshape(zn.shape)

    gg = np.full_like(zn, np.nan) * 1.

    ny,nx = np.shape(zn)
    for l in tqdm(range(ny), disable=disable_bar):
        for j in range(nx):
            zp,yp = zn[l,j], yn[l,j]
            dist2 = (zdata - zp)*(zdata - zp) / sig_x2 + (ydata - yp)*(ydata - yp) / sig_y2
            wiegh = np.exp( -dist2/2 )

            try: gg[l,j] = np.average(values, weights=wiegh)
            except ZeroDivisionError: gg[l,j] = 0.
            
    gg[ gdist > dist_threshold ] = np.nan    
    return gg




i = 45

# empty_points = to_be_filled(rec_x[i])
filled_img = fill_reconstruction( rec_x[i], sigma = 1)


fig,ax = plt.subplots(1,2, figsize=(6,5))
ax[0].imshow( rec_x[i], extent=lims, origin='lower'  )
ax[0].set_title( i )


ax[1].imshow( filled_img, extent=lims, origin='lower'  )
ax[1].set_title( i )

plt.show()




#%%

i = 4
img = rec_x[i]

si = 1
sig_x, sig_y = 5*si,1*si

filled_img = np.copy(img)
empty_points = to_be_filled( img )
eys, exs = np.where(empty_points)
idy,idx = np.indices(img.shape)

t1 = time()

for n in tqdm(range(len(eys))):
    ey,ex = eys[n], exs[n]
    
    dist2 = (idy - ey)**2 / sig_x**2 + (idx - ex)**2 / sig_y**2
    wiegh = np.exp( -dist2/2 )
    filt = (wiegh>0.01) * (~np.isnan(img))
    
    ddd = np.average( img[filt], weights=wiegh[filt] )
    filled_img[ey,ex] = ddd 

t2 = time()
print(ddd)
print(t2-t1)

# b = a[ey-s:ey+s+1, ex-s:ex+s+1]

plt.figure()
plt.imshow(img, extent=lims, origin='lower'  )
# plt.imshow(img )
# plt.plot( ex,ey,'r.' )
plt.show()

plt.figure()
plt.imshow(filled_img, extent=lims, origin='lower'  )
# plt.imshow( filled_img )
# plt.plot( ex,ey,'r.' )
plt.show()


# plt.figure()
# plt.imshow( np.sqrt(dist2) )
# plt.show()







#%%
t1 = time()

f = 7

path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
surface = True
channel = True

# top_t, top_x, top_y, top_z, bot_t, bot_x, bot_y, bot_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data(path, surface, channel )
top_t, top_x, top_y, top_z, bot_t, bot_x, bot_y, bot_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data(path, surface, channel, starts[f], ends[f])


t2 = time()
print(t2-t1)


mean_tx = np.nanmedian(top_x,axis=1)
mean_tx = mean_tx - mean_tx[0] 
mean_bx = np.nanmedian(bot_x,axis=1)
mean_bx = mean_bx - mean_bx[0] 

bpos_i = np.interp(top_t , back_t, back_x)
bpos_i2 = np.interp(bot_t , back_t, back_x)


plt.figure()

plt.plot( top_t , mean_tx, '.-' )
plt.plot( top_t , mean_tx + bpos_i, '.-' , label='top' )
plt.plot( bot_t , mean_bx, '.-' )
plt.plot( bot_t , mean_bx + bpos_i2, '.-' , label='bot' )

# plt.plot( mean_tx + bpos_i, '.-' , label='top' )
# plt.plot( mean_bx + bpos_i2, '.-' , label='bot' )

# plt.plot( back_t, back_x, '.-' )
# plt.plot( top_t , bpos_i, '.-' )

# plt.plot( tiempo, T_top, label='Outlet' )
# plt.plot( tiempo, T_bot, label='Inlet' )
# plt.plot( tiempo, T_amb, label='Ambient' )
# plt.plot( tiempo, Q_tunnel)

plt.grid()
plt.legend()
plt.show()

print( np.nanmedian(Q_tunnel) )


#%%

# for f in range(nf):
for f in [7]:

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    top_t, top_x, top_y, top_z, bot_t, bot_x, bot_y, bot_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data(path, surface, channel)
    
    
    t2 = time()
    print(t2-t1)
    
    
    i = 30
    plt.figure( figsize=(5,10), layout='constrained' )
    st =  plt.scatter( top_z[i], top_y[i], s=1, c=top_x[i] )
    sb =  plt.scatter( bot_z[i], bot_y[i], s=1, c=bot_x[i] )
    plt.axis('equal')
    cbar = plt.colorbar(st, location='top')
    cbar = plt.colorbar(sb, location='bottom')
    # plt.xlim(-50,150)
    plt.xlabel(r'$x$ (mm)')
    plt.ylabel(r'$y$ (mm)')
    # plt.title(r'$h$ (mm)',fontsize=12, pad=70)
    plt.title(all_folders[f] )
    # plt.savefig('./Documents/t29_m30(2).png',dpi=200, bbox_inches='tight')
    plt.show()


#%%

tops, bots, qs, temps = [], [], [], []
for f in tqdm(range(nf)):
    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    top_t, top_x, top_y, top_z, bot_t, bot_x, bot_y, bot_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data(path, surface, channel)

    mean_tx = np.nanmedian(top_x,axis=1)
    mean_tx = mean_tx - mean_tx[0] 
    mean_bx = np.nanmedian(bot_x,axis=1)
    mean_bx = mean_bx - mean_bx[0] 
    
    filt,filb = np.isnan(mean_tx), np.isnan(mean_bx) 
    
    bpos_i  = np.interp(top_t, back_t, back_x)
    bpos_i2 = np.interp(bot_t, back_t, back_x)
    
    tfit = np.polyfit( top_t[~filt], (mean_tx + bpos_i )[~filt], 1 )
    bfit = np.polyfit( bot_t[~filb], (mean_bx + bpos_i2)[~filb], 1 )
    tops.append( tfit[0] )
    bots.append( bfit[0] )
    if f == 4:
        qs.append( np.nanmedian(Q_tunnel[:20000]) )
    elif f == 7:
        qs.append( np.nanmedian(Q_tunnel[185000:]) )
    else:
        qs.append( np.nanmedian(Q_tunnel) )
    temps.append( T_top[0] )

tops, bots, qs, temps = np.array(tops), np.array(bots), np.array(qs), np.array(temps)

Nust = tops * rho_ice * latent * len_ice / (thcon * temps)
Nusb = bots * rho_ice * latent * len_ice / (thcon * temps)


plt.figure()
# plt.scatter( qs, tops, marker='^', c=grid_vels )
# plt.scatter( qs, bots, marker='v', c=grid_vels )
ss = plt.scatter( qs, Nust, marker='^', c=grid_vels, label='Top camera' )
plt.scatter( qs, Nusb, marker='v', c=grid_vels, label='Bottom camera' )

plt.colorbar(ss)

plt.xlabel('Q (m^3/h)')
plt.ylabel('Nu')
plt.legend()
plt.show()

#%%
i = 25

# filt = ~np.isnan(top_x[i])
# filb = ~np.isnan(bot_x[i])
filt = (top_y[i] < 35) * (top_y[i] > -25)
filb = (bot_y[i] < 35) * (bot_y[i] > -25)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(top_z[i][filt], top_x[i][filt], top_y[i][filt], alpha=0.5)# c=top_x[i][filt])
ax.scatter(bot_z[i][filb], bot_x[i][filb], bot_y[i][filb], alpha=0.5)# c=bot_x[i][filb])

# aspect_ratio_fig(ax, top_z[i][fil], top_x[i][fil], top_y[i][fil])


all_z, all_y, all_x = np.concatenate([top_z[i],bot_z[i]]), np.concatenate([top_y[i],bot_y[i]]), np.concatenate([top_x[i],bot_x[i]]) 
# all_fil = ~np.isnan(all_x)
all_fil = (all_y < 35) * (all_y > -25)

# aspect_ratio_fig(ax,  all_z[all_fil], all_x[all_fil], all_y[all_fil])

plt.show()

#%%
import matplotlib.tri as mtri

i = 25
# filt = ~np.isnan(top_x[i])
# filb = ~np.isnan(bot_x[i])
filt = (top_y[i] < 35) * (top_y[i] > -25)
filb = (bot_y[i] < 35) * (bot_y[i] > -25)

trit = mtri.Triangulation( top_z[i][filt], top_y[i][filt])
trib = mtri.Triangulation( top_z[i][filb], top_y[i][filb])

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot_trisurf( top_z[i][filt], top_x[i][filt], top_y[i][filt], triangles=tri.triangles, cmap=plt.cm.Spectral  )

ax.plot_trisurf( top_z[i][filt], top_y[i][filt], top_x[i][filt], triangles=trit.triangles, cmap=plt.cm.Blues  )
ax.plot_trisurf( bot_z[i][filb], bot_y[i][filb], bot_x[i][filb], triangles=trib.triangles, cmap=plt.cm.Reds  )


# aspect_ratio_fig(ax, top_z[i][filt], top_y[i][filt], -top_x[i][filt] )

all_z, all_y, all_x = np.concatenate([top_z[i],bot_z[i]]), np.concatenate([top_y[i],bot_y[i]]), np.concatenate([top_x[i],bot_x[i]]) 
# all_fil = ~np.isnan(all_x)
all_fil = (all_y < 35) * (all_y > -25)

# aspect_ratio_fig(ax,  all_z[all_fil], all_y[all_fil], all_x[all_fil])


plt.show()



#%%





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
