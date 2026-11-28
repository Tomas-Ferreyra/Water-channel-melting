#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:02:25 2025

@author: tomasferreyrahauchar
"""

import numpy as np
import warnings
import glob 
import h5py
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import cmcrameri

from tqdm import tqdm
from time import time

import scipy.ndimage as snd
from scipy.spatial import Delaunay
from scipy.ndimage import minimum_filter
from scipy.optimize import curve_fit, least_squares, linear_sum_assignment
from scipy.stats import linregress
from scipy.signal import correlate2d, correlate, find_peaks


from skimage.feature import peak_local_max
from skimage.filters import gaussian, roberts, sobel
from skimage.segmentation import watershed, mark_boundaries
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import local_minima, disk, remove_small_holes, binary_erosion, binary_dilation, binary_closing, binary_opening 

import os

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


def smooth_reconstruction(rec_x, rec_back, sigmas, slope_v='edge'):
    """
    slope = 'edge' use the slope at both edge (start and end)
    slope = 'all' fits an overall slope to use
    """ 
    tsig, ysig, zsig = sigmas
    rec = rec_x + rec_back[:, None, None]

    aaa = np.nanmean(rec, axis=(1, 2))
    missing = np.where(np.isnan(aaa))[0]

    # fill completely missing frames
    if len(missing):
        gaps = np.split( missing, np.where(np.diff(missing) > 1)[0] + 1 )

        for gap in gaps:
            start, end, n = gap[0], gap[-1], len(gap)
            before, after = rec[max(0, start-2):start], rec[end+1:end+3]
            if len(before) == 0 or len(after) == 0: continue

            img_left = ( 2 * before[-1] - before[-2] if len(before) >= 2 else before[-1] )
            img_right = ( 2 * after[0] - after[1] if len(after) >= 2 else after[0] )

            weights = np.linspace(0, 1, n+2)[1:-1]
            rec[gap] = ( (1 - weights[:, None, None]) * img_left[None, :, :] + weights[:, None, None] * img_right[None, :, :] )

    # Linear extrapolation at both ends
    aaa = np.nanmean(rec, axis=(1, 2))
    pad = int(4 * tsig)

    if slope_v == 'edge':
        left = aaa[0] + np.arange(-pad, 0) * (aaa[1] - aaa[0])
        right = aaa[-1] + np.arange(1, pad+1) * (aaa[-1] - aaa[-2])
        left_shift = left - aaa[0]
        right_shift = right - aaa[-1]
        
    elif slope_v == 'all':
        valid = np.isfinite(aaa)
        slope, intercept = np.polyfit(np.arange(len(aaa))[valid], aaa[valid], 1)        
        left = aaa[0] + np.arange(-pad, 0) * slope
        right = aaa[-1] + np.arange(1, pad+1) * slope
        left_shift = left - aaa[0]
        right_shift = right - aaa[-1]

    rec_pad = np.concatenate([ rec[0][None] + left_shift[:, None, None], rec, 
                              rec[-1][None] + right_shift[:, None, None] ], axis=0)

    # Gaussian smoothing
    rec_smooth = nangauss( rec_pad, sigmas )

    return rec_smooth[pad:-pad] if pad > 0 else rec_smooth 


def calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=False, integral=True, smooth=True, slope_v='edge' ):
    """
    Calculate the melt rate from a smoothed reconstruction.
    
    Computes spatial and temporal gradients of the Gaussian-smoothed
    reconstruction, then calculates spatially averaged melt rates and
    their temporal average or standard deviation.
    
    Parameters
    ----------
    f : int
        Flag controlling gradient calculation, with special handling for f == 0.
    rec_x : ndarray
        Reconstruction data.
    rec_back : ndarray
        Background correction added to the reconstruction.
    sigmas : list of float, optional
        Gaussian smoothing widths in time, y, and z.
    time_val : bool, optional
        If True, return time-dependent melt rates.
    integral : bool, optional
        If True, calculate spatially averaged and temporally integrated
        melt rates; otherwise, use simple means.
    
    Returns
    -------
    gt : ndarray
        Temporal gradient of the smoothed reconstruction.
    melt_rate : float or ndarray
        Average melt rate, or time-dependent melt rates if time_val=True.
    st : float
        Standard deviation of the time-dependent average melt rate,
        or 0 when time_val=True.
    """
    tsig, ysig, zsig = sigmas
    if smooth:
        rec_xg = smooth_reconstruction( rec_x, rec_back, sigmas, slope_v)
    else:
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
    # areas = np.trapezoid( np.trapezoid( mask, dx=dz, axis=2 ), dx=dy, axis=1)
    areas = np.trapz( np.trapz( mask, dx=dz, axis=2 ), dx=dy, axis=1)

    gt0 = np.copy(gt)
    gt0[np.isnan(gt0)] = 0
    # av_melt = np.trapezoid( np.trapezoid( gt0, dx=dz, axis=2 ), dx=dy, axis=1)
    av_melt = np.trapz( np.trapz( gt0, dx=dz, axis=2 ), dx=dy, axis=1)
    av_melt = av_melt / areas /2
    
    st = np.nanstd( av_melt )
    
    if time_val:
        if integral: return gt, av_melt, 0
        else: return gt, np.nanmean( gt, axis=(1,2) ), 0
            
    else:
        if integral: 
            av_melt[np.isnan(av_melt)] = 0
            # avg_melt = np.trapezoid( av_melt, x=rec_t  ) / rec_t[-1]
            avg_melt = np.trapz( av_melt, x=rec_t  ) / rec_t[-1]
            return gt, avg_melt, st
        else: return gt, np.nanmean( gt ), np.nanstd( gt )


def area_superficial_2d(f, x, y, mask):
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    
    fy, fx = np.gradient(f, dy, dx)
    
    # Control de indeterminaciones en bordes abruptos (como la esfera)
    fx = np.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0)
    fy = np.nan_to_num(fy, nan=0.0, posinf=0.0, neginf=0.0)
    
    func = np.sqrt(1 + fx**2 + fy**2)    
    x0 = x[0, :]
    y0 = y[:, 0]
    
    if mask is None:
        mask = np.ones_like(f, dtype=bool)
    func[~mask] = 0.0
    
    return np.trapz(np.trapz(func, x0, axis=1), y0, axis=0)

def area_superficial_multi_2d(f, x, y, mask=None):
    # Asegurar que f sea una matriz 3D (N, p, q). Si es 2D, se transforma a (1, p, q)
    if f.ndim == 2:
        f = f[np.newaxis, :, :]
    
    N, p, q = f.shape
    
    # Extraer el espaciado escalar a partir de las mallas de coordenadas 2D (p, q)
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    
    # Extraer vectores de integración unidimensionales
    x0 = x[0, :]
    y0 = y[:, 0]
    
    # Configurar la lógica de la máscara según su forma
    if mask is None:
        # Sin máscara: se acepta todo para las N superficies
        mascaras = np.ones((N, p, q), dtype=bool)
    elif mask.ndim == 2:
        # Máscara única de (p, q) se repite para las N superficies
        mascaras = np.repeat(mask[np.newaxis, :, :], N, axis=0)
    else:
        # Máscara ya viene con forma (N, p, q)
        mascaras = mask
        
    areas = np.zeros(N)
    
    # Procesar cada una de las N superficies
    for i in range(N):
        fi = f[i]
        mi = mascaras[i]
        
        fy, fx = np.gradient(fi, dy, dx)
        
        # Control de indeterminaciones en bordes
        fx = np.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0)
        fy = np.nan_to_num(fy, nan=0.0, posinf=0.0, neginf=0.0)
        
        func = np.sqrt(1 + fx**2 + fy**2)    
        func[~mi] = 0.0
        
        # Integración y almacenamiento del resultado
        areas[i] = np.trapz(np.trapz(func, x0, axis=1), y0, axis=0)
        
    # Si la entrada original era una sola superficie 2D, devolvemos un escalar en vez de un arreglo
    return areas[0] if N == 1 and areas.size == 1 else areas

    
def polyfitn(x, y, z, order=4):
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    xfit, yfit, zfit = x[mask], y[mask], z[mask]
    
    expos = list(dict.fromkeys( (e1, e2) for expo in itertools.permutations(list(range(order+1))*2, 2)
                                   for e1, e2 in [expo] if e1 + e2 <= order ))
    
    A = np.array([xfit**e1 * yfit**e2 for e1, e2 in expos]).T
    coeff, r, rank, s = np.linalg.lstsq(A, zfit, rcond=None)    
    return coeff, r, rank, s

def poln(coeff,xn,yn,order=4):
    expos = list(dict.fromkeys( [expo for expo in itertools.permutations(list(range(order+1))*2,2) if sum(expo) < order+1] ))
    terms = np.array( [xn**e1 * yn**e2 for (e1,e2) in expos ] )
    coso = 0 
    for k in range(len(coeff)):
        coso += terms[k] * coeff[k]
    return coso



def add_dimension_label(ax, p1, p2, label, fontsize=10, color='black', offset=0.02, side=1, ha='center', va='center'):

    # Project 3D endpoints to 2D
    x1, y1, _ = proj3d.proj_transform(*p1, ax.get_proj())
    x2, y2, _ = proj3d.proj_transform(*p2, ax.get_proj())

    # Convert to axes coordinates
    p1_2d = ax.transAxes.inverted().transform( ax.transData.transform((x1, y1)) )
    p2_2d = ax.transAxes.inverted().transform( ax.transData.transform((x2, y2)) )

    dx, dy = p2_2d - p1_2d
    angle = np.degrees(np.arctan2(dy, dx))

    # Keep text readable
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # Midpoint
    midpoint = (p1_2d + p2_2d) / 2

    # Perpendicular offset
    length = np.hypot(dx, dy)
    if length > 0: midpoint += side * offset * np.array([-dy, dx]) / length

    ax.text2D( midpoint[0], midpoint[1], label, transform=ax.transAxes, rotation=angle, 
              rotation_mode='anchor', ha=ha, va=va, fontsize=fontsize, color=color )
    
def add_bounding_box(ax, x, y, z, L=10, color='black', lw=0.8):

    # Bounding limits with padding
    xmin, xmax = np.nanmin(x)-L, np.nanmax(x)+L
    ymin, ymax = np.nanmin(y)-L, np.nanmax(y)+L
    zmin, zmax = np.nanmin(z)-L, np.nanmax(z)+L

    # 8 corners of the box
    corners = [ (xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
                (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax) ]

    # 12 edges
    edges = [ (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7) ]

    for n, (i, j) in enumerate(edges):
        if n in [0,1,4,5,8,9]:
            ax.plot( [corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]], [corners[i][2], corners[j][2]],
                    color=color, linewidth=lw, zorder=100  )
        else:
            ax.plot( [corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]], [corners[i][2], corners[j][2]],
                    color=color, linewidth=lw )

    return (xmin, xmax), (ymin, ymax), (zmin, zmax)


def preprocess(rec_x):
    data = rec_x.copy().astype(float)
    for t in range(data.shape[0]):
        frame = data[t]
        frame -= np.nanmean(frame)
        norm = np.sqrt(np.nanmean(frame**2))
        if norm > 0:
            frame /= norm
    return data

def spatiotemporal_correlation(rec_x, max_lag=None, mode='same'):

    nt, ny, nx = rec_x.shape
    if max_lag is None: max_lag = nt - 1

    if mode == 'full': shape = (2*ny - 1, 2*nx - 1)
    elif mode == 'same': shape = (ny, nx)
    else: print('Full or same for mode')
    valid_frames = ~np.all(np.isnan(rec_x), axis=(1, 2))

    correlations = []
    for dt in tqdm(range(1, max_lag + 1)):
        C = np.zeros(shape)
        n_pairs = 0
        for t in range(nt - dt):
            if not (valid_frames[t] and valid_frames[t + dt]): continue
            C += correlate( rec_x[t], rec_x[t + dt], mode=mode, method='fft' )
            n_pairs += 1
        
        if n_pairs > 0: C /= n_pairs
        else: C[:] = np.nan

        # C /= (nt - dt)
        correlations.append(C)

    return np.array(correlations)

def largest_valid_window(valid):
    ny, nx = valid.shape
    heights = np.zeros(nx, dtype=int)
    best_area = 0
    best_window = None
    for y in range(ny):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []
        for x in range(nx + 1):
            h = heights[x] if x < nx else 0
            start = x
            while stack and stack[-1][1] > h:
                start, height = stack.pop()
                area = height * (x - start)
                if area > best_area:
                    best_area = area
                    best_window = ( y - height + 1, y + 1, start, x )
            stack.append((start, h))
    return best_window

def get_displacements(Corr, ny, nx, mode='full'):
    if mode == 'full':
        y_center = ny - 1
        x_center = nx - 1

    elif mode == 'same':
        y_center = ny // 2
        x_center = nx // 2

    else: raise ValueError("mode must be 'full' or 'same'")

    displacements = []
    for C in Corr:
        if not np.any(np.isfinite(C)):
            displacements.append([np.nan, np.nan])
            continue

        iy, ix = np.unravel_index( np.nanargmax(C), C.shape )
        dx = ix - x_center
        dy = iy - y_center
        displacements.append([dx, dy])

    return np.array(displacements)


def get_minmax(rec_z, rec_y, rec_x, rec_back, sigmas, slope_v='all', tol = 1e-8 ):
    tsig, ysig, zsig = sigmas
    rec_xg = smooth_reconstruction( rec_x, rec_back, sigmas, slope_v)
    
    gt, gy, gz = np.gradient(rec_xg, rec_t, rec_y[:,0], rec_z[0], edge_order=2)
    _, gyy, gyz = np.gradient(gy, rec_t, rec_y[:,0], rec_z[0], edge_order=2)
    _, gzy, gzz = np.gradient(gz, rec_t, rec_y[:,0], rec_z[0], edge_order=2)
    
    grad_norm = np.sqrt(gy**2 + gz**2)
    thing = grad_norm < 0.05
    
    labels = np.zeros_like(thing, dtype=int)
    n_labels = []
    stationary_points = []    
    for i in tqdm(range(thing.shape[0])):
        lab, n = snd.label( thing[i], structure=np.ones((3,3)) )
        labels[i] = lab
        n_labels.append(n)
    
        points = []
        for j in range(1, n_labels[i] + 1):
            iy, iz = np.where(lab == j)
            k = np.argmin(grad_norm[i, iy, iz])
            points.append((iy[k], iz[k]))
        stationary_points.append(np.array(points))
        
    minima = []
    maxima = []
    saddles = []
    for i, points in enumerate(stationary_points):
        min_i = []
        max_i = []
        saddle_i = []    
        for iy, iz in points:
            H = np.array([ [gyy[i, iy, iz], gyz[i, iy, iz]], [gzy[i, iy, iz], gzz[i, iy, iz]] ])    
            H = 0.5 * (H + H.T)
            eig = np.linalg.eigvalsh(H)
    
            if np.all(eig > tol): min_i.append((iy, iz))
            elif np.all(eig < -tol): max_i.append((iy, iz))
            elif eig[0] < -tol and eig[1] > tol: saddle_i.append((iy, iz))
    
        minima.append(np.array(min_i, dtype=int).reshape(-1, 2))
        maxima.append(np.array(max_i, dtype=int).reshape(-1, 2))
        saddles.append(np.array(saddle_i, dtype=int).reshape(-1, 2))
    
    return minima, maxima


def track_extrema(points, min_length=10, max_gap=2, max_dist_y=5.0, max_dist_z=5.0):
    """
    Track minima or maxima over time.

    Parameters
    ----------
    points : list of arrays
        points[i] contains (iy, iz) coordinates at time i.
    min_length : int
        Minimum number of detections for a valid track.
    max_gap : int
        Maximum number of missing frames allowed between detections.
    max_distance : float
        Maximum allowed displacement per frame, in grid indices.

    Returns
    -------
    tracks : list of dict
        Each track contains:
          'frames': time indices
          'positions': (iy, iz) positions
    """

    active = []
    finished = []

    for t, detections in enumerate(points):
        detections = np.asarray(detections).reshape(-1, 2)

        # Remove tracks that have been missing too long
        still_active = []

        for tr in active:
            if t - tr['last_frame'] <= max_gap + 1: still_active.append(tr)
            else: finished.append(tr)

        active = still_active

        # No detections at this time
        if len(detections) == 0: continue

        if len(active) == 0:
            for p in detections:
                active.append({ 'frames': [t], 'positions': [p.astype(float)], 'last_frame': t })
            continue

        # Cost matrix between active tracks and new detections
        cost = np.full((len(active), len(detections)), 1e9)
        for j, tr in enumerate(active):

            # Estimate velocity from last two detections
            last_p = tr['positions'][-1]
            last_t = tr['frames'][-1]

            if len(tr['positions']) >= 2:
                prev_p = tr['positions'][-2]
                prev_t = tr['frames'][-2]
                velocity = (last_p - prev_p) / (last_t - prev_t)
            else:
                velocity = np.zeros(2)
            
            dt = t - last_t
            predicted = last_p + velocity * dt

            # Independent distance constraints
            dy = np.abs(detections[:, 0] - predicted[0])
            dz = np.abs(detections[:, 1] - predicted[1])
            valid = ( (dy <= max_dist_y * dt) & (dz <= max_dist_z * dt) )

            # Normalized distance for assignment
            distances = np.sqrt( (dy / max_dist_y)**2 + (dz / max_dist_z)**2 )
            cost[j, valid] = distances[valid]


        # Globally optimal one-to-one assignment
        rows, cols = linear_sum_assignment(cost)

        matched_detections = set()
        matched_tracks = set()

        for j, k in zip(rows, cols):
            if cost[j, k] >= 1e9: continue

            tr = active[j]
            tr['frames'].append(t)
            tr['positions'].append(detections[k].astype(float))
            tr['last_frame'] = t

            matched_tracks.add(j)
            matched_detections.add(k)

        # Unmatched detections start new tracks
        for k, p in enumerate(detections):
            if k not in matched_detections:
                active.append({ 'frames': [t], 'positions': [p.astype(float)], 'last_frame': t })

    # Finish remaining tracks
    finished.extend(active)

    # Filter tracks by minimum number of detections
    tracks = []
    for tr in finished:
        if len(tr['frames']) >= min_length:
            tracks.append({ 'frames': np.array(tr['frames']), 'positions': np.array(tr['positions']) })

    return tracks

def average_track_velocity(tracks, rec_y, rec_z, dt):
    for tr in tracks:
        pp = tr['positions'].astype(int)
        t = tr['frames'] * dt
        z, y = rec_z[0, pp[:, 1]], rec_y[pp[:, 0], 0]
        
        fit_y, fit_z = linregress(t, y), linregress(t, z)

        tr['vy_avg'], tr['vz_avg'] = fit_y.slope, fit_z.slope
        
        tr['vy_err'], tr['vz_err'] = fit_y.stderr, fit_z.stderr
    return tracks

def average_track_gt(gt, tracks, radius_y=2, radius_z=2):
    for tr in tracks:
        values = []

        for t, pos in zip(tr['frames'], tr['positions']):
            iy, iz = pos.astype(int)

            # Extract neighbourhood, handling boundaries
            y0 = max(0, iy - radius_y)
            y1 = min(gt.shape[1], iy + radius_y + 1)
            z0 = max(0, iz - radius_z)
            z1 = min(gt.shape[2], iz + radius_z + 1)
            neighbourhood = gt[t, y0:y1, z0:z1]

            # Mean ignoring NaNs
            if np.any(np.isfinite(neighbourhood)):
                values.append(np.nanmean(neighbourhood))

        tr['gt_avg'] = np.mean(values) if values else np.nan
        tr['gt_std'] = np.std(values) if values else np.nan
        tr['gt_n'] = len(values)

    return tracks


def plot_colored_line(ax, y_line, x_line, gt_line, norm,  cmap='coolwarm', linewidth=2, mmm=0):
    x, y = x_line - mmm, y_line
    c = gt_line - np.nanmean(gt_line)

    # Remove invalid points
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[valid], y[valid], c[valid]

    # Build line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create colored line
    # outline = LineCollection( segments, colors='black', linewidth=linewidth + 1.5, zorder=1 )
    # ax.add_collection(outline)
    
    # lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=1.0, antialiased=True)
    
    lc = LineCollection( segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=1, zorder=2 )
    lc.set_array((c[:-1] + c[1:]) / 2)
    lc.set_path_effects([ pe.Stroke(linewidth=linewidth + 1.5, foreground='black'), pe.Normal() ])

    ax.add_collection(lc)
    
    # lc.set_array((c[:-1] + c[1:]) / 2)
    # lc.set_linewidth(linewidth)
    # ax.add_collection(lc)
    return lc



def density_millero(t, s):
    """
    Computes density of seawater in kg/m^3.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    return rho_sw

def mu_visco(T):
    """ Vogel equation """
    return 2.414e-5 * 10**( 247.8 / (T + 133.15) )
def nu_visco(T):
    """ Vogel equation """
    return mu_visco(T) / density_millero(T, 0)

def thermal_cond(T):
    " Holzmann CFD: https://holzmann-cfd.com/community/blog-and-tools/cae-blog/thermophysical-properties-water"
    T = T + 273.15
    return -9.29827e-6 * T**2 + 0.0071857 * T - 0.710696

def heat_capacity(T):
    " Holzmann CFD: https://holzmann-cfd.com/community/blog-and-tools/cae-blog/thermophysical-properties-water"
    T = T + 273.15
    return -0.000127063 * T**3 + 0.13736 * T**2 - 48.6714 * T + 9850.69

def heat_diff(T):
    return thermal_cond(T) / (heat_capacity(T) * density_millero(T,0) )


def temp_t(T_top, T_bot, rec_t, tiempo):
    converted_t = rec_t + tiempo[-1] - rec_t[-1] - 10
    indices = np.searchsorted(tiempo,converted_t)
    return (T_top[indices] + T_bot[indices])/2 


os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plt.rcParams.update({
    # 'axes.edgecolor': 'black',
    "text.usetex": True,
    "font.family": "serif",
    'font.size':18,
})

    
all_folders = ['25-08-07','25-09-29','25-10-24','25-11-17','25-12-12','26-01-27','26-02-17','26-03-05','26-05-26','26-07-09' ]
grid_vels   = [ 0.5      , 0.3      , 0.9      , 0.3      , 0.0      , 0.3      , 0.9      , 0.3      , 0.3      , 0.3       ]
nf = len(all_folders)

starts = [ 1900 , 650  , 750  , 1250 , 2060 , 1000 , 108  , 190640, 940 , 920  ]
ends   = [ 21700, 28000, 28600, 35250, 18980, 12580, 14980, 205690, 9433, 9538 ]

rho_ice = 916.8 # kg / m^3
latent = 334e3 # J / kg  or m^2/s^2 
thcon = 0.6 # m kg / s^3 °C
len_ice = 0.71 # m
nu = 1.004e-6 # m^2 / s

length0 = 0.7 #m

area_Q = 0.3 * 0.08

#%%

t1 = time()

f = 4

path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
surface = True
channel = True

rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
# rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, 0, -1 )

if f==0: rec_x[16] = np.nan
elif f==3: rec_x[68:71] = np.nan
elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
elif f==2: 
    back_t, back_x = back_t[:-1], back_x[:-1]
    rec_x[:,618:1130,50:] = np.nan

t_med = time()    

rec_f = fill_reconstruction(rec_x, sigma=1)
rec_back = np.interp(rec_t, back_t, back_x )
    
lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
dz, dy = 2.5, 0.5

t2 = time()
print(t2-t1)

print( f'{f},\tvel: {np.median(Q_tunnel)/area_Q/3600:.2f} m/s,\t T:{np.median(T_top):.1f} °C,\t GSF: {grid_vels[f]}'  )
print(rec_x.shape)

#%%

i = 50
ysig = 15
zsig = 3

lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]

rec_xf = np.copy(rec_x)
filt_y = np.argmin(np.all(rec_y <= 0, axis=1))
rec_xf[:,:filt_y,:] = np.nan


# fig, ax = plt.subplots(1,2, figsize=(6,5))

# s1 = ax[0].imshow( rec_x[i], extent=lims, origin='lower'  )
# s2 = ax[1].imshow( rec_f[i], extent=lims, origin='lower' )

# # s1 = ax[0].imshow( rec_x[i], extent=lims , origin='lower' )
# # s2 = ax[1].imshow( rec_xf[i], extent=lims , origin='lower' )

# # s1 = ax[0].imshow( rec_z, extent=lims, origin='lower' )
# # s2 = ax[1].imshow( rec_y, extent=lims, origin='lower' )

# plt.colorbar(s1, ax=ax[0])
# plt.colorbar(s2, ax=ax[1])
# plt.show()

# creo que esto no sirve remover outliers
rec_xf = rec_x - np.nanmedian( rec_x, axis=(1,0) )
sd_rec = np.nanstd( rec_xf )
print(sd_rec)

rec_sf = rec_xf / sd_rec
fil = np.abs(rec_xf / sd_rec) > 4.3
rec_xf[fil] = np.nan 

fig, ax = plt.subplots(1,3, figsize=(9,5))

s1 = ax[0].imshow( rec_x[i], extent=lims, origin='lower' )
s2 = ax[1].imshow( rec_sf[i], extent=lims, origin='lower' )
s3 = ax[2].imshow( rec_xf[i], extent=lims, origin='lower' )

# s1 = ax[0].imshow( rec_x[i], extent=lims , origin='lower' )
# s2 = ax[1].imshow( rec_xf[i], extent=lims , origin='lower' )

# s1 = ax[0].imshow( rec_z, extent=lims, origin='lower' )
# s2 = ax[1].imshow( rec_y, extent=lims, origin='lower' )

plt.colorbar(s1, ax=ax[0])
plt.colorbar(s2, ax=ax[1])
plt.colorbar(s3, ax=ax[2])
plt.show()


fig, ax = plt.subplots(1,2, figsize=(6,5))

vmax = np.nanmax(np.abs( rec_x[i] - np.nanmean(rec_x[i]) ))
norm = Normalize(vmin=-vmax, vmax=vmax)

s1 = ax[0].imshow( rec_x[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='Spectral', norm=norm )
s2 = ax[1].imshow( rec_f[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='Spectral', norm=norm )

# s1 = ax[0].imshow( rec_x[i], extent=lims , origin='lower' )
# s2 = ax[1].imshow( rec_xf[i], extent=lims , origin='lower' )

# s1 = ax[0].imshow( rec_z, extent=lims, origin='lower' )
# s2 = ax[1].imshow( rec_y, extent=lims, origin='lower' )

plt.colorbar(s1, ax=ax[0])
plt.colorbar(s2, ax=ax[1])
plt.show()

#%%

i = 80

y_std = np.nanstd( rec_x, axis=2 )
gy_std = nangauss(y_std, (1,30))
nt, ny, nz = rec_x.shape


plt.figure()
# plt.imshow( rec_x[i], extent=lims, origin='lower' )
# plt.imshow( y_std.T )
# plt.colorbar()

for i in range(0,nt,2):
    # plt.plot( rec_y[:,0], y_std[i], '-', color=(i/nt,0.3,1-i/nt) )
    # plt.plot( rec_y[:,0], gy_std[i], '--', color=(i/nt,0.3,1-i/nt) )
    plt.plot( rec_y[:,0], np.gradient(gy_std[i]), '--', color=(i/nt,0.3,1-i/nt) )
    
# plt.plot( rec_y[:,0], np.gradient(gy_std[84]), '--', color=(i/nt,0.3,1-i/nt) )

# nnt = 10
# for i in range(0,nnt,1):
#     plt.plot( rec_y[:,0], y_std[i], '-', color=(i/nnt,0.3,1-i/nnt) )
plt.title(f'f = {f}')
plt.show()

if f == 3:
    iy_bot = np.nanargmax( np.gradient(gy_std[84]) )
else:
    iy_bot = np.nanargmax( np.gradient(gy_std[-1]) )
y_bot = rec_y[iy_bot,0]

print( y_bot )

#%%

# find here
x0 = -400

tempt = temp_t(T_top, T_bot, rec_t, tiempo)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    nt, ny, nz = rec_x.shape
    gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )

    mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
    med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
    ushearT = med_Q / mean_x * 25/27 
    
    gt_y = np.nanmean( gt, axis=2 )
    Nu_x = rho_ice * latent / (thermal_cond(tempt) * tempt )[:,None] * (rec_y[:,0]-x0)[None,:]/1000 * gt_y/1000    
    Re_x = ushearT[:,None] * (rec_y[:,0]-x0)[None,:]/1000 / nu_visco(tempt)[:,None]

print(gt_y.shape)

# fig, ax = plt.subplots(1,2, layout='constrained', figsize=(6,5))
# # s1 = ax[0].imshow( Nu_x , extent=lims, origin='lower' )
# # s2 = ax[1].imshow( Nu_xx , extent=lims, origin='lower' )

# s1 = ax[0].imshow( Nu_x, origin='lower' )
# s2 = ax[1].imshow( Re_x, origin='lower' )

# ax[0].set_aspect(30)
# ax[1].set_aspect(30)
# plt.colorbar(s1, ax=ax[0])
# plt.colorbar(s2, ax=ax[1])
# plt.show()

fig, ax = plt.subplots()

norm = plt.Normalize(vmin=0, vmax=(nt-1)/2)
cmap = plt.cm.coolwarm

for i in range(1, nt, 1):
# for i in range(50,70):
    if np.all(np.isnan(Re_x[i])) or np.all(np.isnan(Nu_x[i])):
        continue

    color = cmap(norm(i/2))
    ax.plot(Re_x[i], Nu_x[i], '-', color=color, label=i)
    
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Re(x)')
ax.set_ylabel('Nu(x)')
ax.set_title(f'f = {f}')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'$t$ (min)')


powe = 0.8
# rree = np.logspace(3.3,5,1000)
# ax.plot( rree, rree**powe * 0.2, 'k--' )
rree = np.logspace(4.3,5.5,1000)
ax.plot( rree, rree**powe * 0.3, 'k--' )

# xtext = 4e3
xtext = 4e4
ytext = 0.3 * xtext**powe
fig.canvas.draw()
# Calculate visual inclination
p1 = ax.transData.transform((xtext, ytext))
p2 = ax.transData.transform((xtext*1.01, ytext*1.01**powe))
angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
ax.text(xtext, ytext*1.15, rf'Re$^{{{powe}}}$', rotation=angle, rotation_mode='anchor', ha='center', va='bottom', fontsize=12)


plt.show()

#%%


#%%
from scipy.ndimage import binary_erosion as bin_erosion


from scipy.ndimage import maximum_filter

i = 52
lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]

# diy, dix = 25, 5

# t1 = time()
# y, x = np.ogrid[-diy:diy+1, -dix:dix+1]
# footprint = (y/diy)**2 + (x/dix)**2 <= 1
# img = np.nan_to_num(rec_f[i], nan=-np.inf)
# valid = np.isfinite(img)
# interior = bin_erosion(valid, iterations=5)

# filtered = maximum_filter(img, footprint=footprint)
# peaks = (img == filtered) & np.isfinite(img)
# ab = np.argwhere(peaks)
# ab = ab[interior[ab[:, 0], ab[:, 1]]]

# t2 = time()
# img = np.nan_to_num(rec_f[i], nan=-np.inf)
# aa = peak_local_max( img, min_distance=10, exclude_border=True, labels=~np.isnan(rec_f[i]), num_peaks=15 )
# t3 = time()

# print(t3-t2, t2-t1)




plt.figure()
plt.imshow( rec_f[i], extent=lims, origin='lower', cmap='Spectral' )
# plt.imshow( img, extent=lims, origin='lower', cmap='Spectral' )
# plt.imshow( ~np.isnan(rec_f[i]), extent=lims, origin='lower' )

if len(aa)>0: my, mx = aa.T
plt.plot(rec_z[0,mx], rec_y[my,0], 'ks')
if len(ab)>0: my, mx = ab.T
plt.plot(rec_z[0,mx], rec_y[my,0], 'b.')


plt.colorbar()
plt.show()

#%%

sigmas = [1,20,4]
slope_v='all'
low_y = -80

filt_y = np.argmin(np.all(rec_y <= low_y, axis=1)) 
rec_nx = np.copy(rec_x)
rec_nx[:,:filt_y,:] = np.nan



dt = 30
t1 = time()
gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )
minima, maxima = get_minmax(rec_z, rec_y, rec_nx, rec_back, sigmas, slope_v='all', tol = 1e-8 )
#get tracks
min_tracks = track_extrema( minima, min_length=10, max_gap=2, max_dist_y=15.0, max_dist_z=3. )
max_tracks = track_extrema( maxima, min_length=10, max_gap=2, max_dist_y=15.0, max_dist_z=3. )
#add vels
min_tracks = average_track_velocity(min_tracks, rec_y, rec_z, dt)
max_tracks = average_track_velocity(max_tracks, rec_y, rec_z, dt)
#add melt rates
min_tracks = average_track_gt(gt, min_tracks, radius_y=5, radius_z=1)
max_tracks = average_track_gt(gt, max_tracks, radius_y=5, radius_z=1)

min_mvy = np.array([tr['vy_avg'] for tr in min_tracks])
min_svy = np.array([tr['vy_err'] for tr in min_tracks])
min_mvz = np.array([tr['vz_avg'] for tr in min_tracks])
min_svz = np.array([tr['vz_err'] for tr in min_tracks])
min_mmr = np.array([tr['gt_avg'] for tr in min_tracks])
min_smr = np.array([tr['gt_std'] for tr in min_tracks])

max_mvy = np.array([tr['vy_avg'] for tr in max_tracks])
max_svy = np.array([tr['vy_err'] for tr in max_tracks])
max_mvz = np.array([tr['vz_avg'] for tr in max_tracks])
max_svz = np.array([tr['vz_err'] for tr in max_tracks])
max_mmr = np.array([tr['gt_avg'] for tr in max_tracks])
max_smr = np.array([tr['gt_std'] for tr in max_tracks])

t2 = time()
print(t2-t1)


print('         \t   Vy     \t    Vz     \t    Melt rate ' )

mean_vy = np.mean(min_mvy)
std_vy_total = np.sqrt( np.var(min_mvy, ddof=1) + np.mean(min_svy**2) )

mean_vz = np.mean(min_mvz)
std_vz_total = np.sqrt( np.var(min_mvz, ddof=1) + np.mean(min_svz**2) )

mean_mr = np.mean(min_mmr)
std_mr_total = np.sqrt( np.var(min_mmr, ddof=1) + np.mean(min_smr**2) )

print(f'Mean Min : \t {mean_vy:.3f} mm/s \t {mean_vz:.3f} mm/s \t {mean_mr:.3f} mm/s ' )
print(f'Error Min: \t {std_vy_total:.3f} mm/s \t {std_vz_total:.3f} mm/s \t {std_mr_total:.3f} mm/s ' )

mean_vy = np.mean(max_mvy)
std_vy_total = np.sqrt( np.var(max_mvy, ddof=1) + np.mean(max_svy**2) )

mean_vz = np.mean(max_mvz)
std_vz_total = np.sqrt( np.var(max_mvz, ddof=1) + np.mean(max_svz**2) )

mean_mr = np.mean(max_mmr)
std_mr_total = np.sqrt( np.var(max_mmr, ddof=1) + np.mean(max_smr**2) )

print(f'Mean Max : \t {mean_vy:.3f} mm/s \t {mean_vz:.3f} mm/s \t {mean_mr:.3f} mm/s ' )
print(f'Error Max: \t {std_vy_total:.3f} mm/s \t {std_vz_total:.3f} mm/s \t {std_mr_total:.3f} mm/s ' )


i = 20
# plt.figure()

# plt.imshow( rec_x[i], extent=lims, origin='lower', cmap='Spectral' )
# # plt.imshow( thing[i], extent=lims, origin='lower' )
# # plt.imshow( grad_norm[i], extent=lims, origin='lower' )


# # plt.imshow(gyy[-1], extent=lims, origin='lower', cmap='Spectral' )
# # plt.imshow(gzz[-1], extent=lims, origin='lower', cmap='Spectral' )
# # plt.imshow( np.abs(gz[-1]), extent=lims, origin='lower', cmap='Spectral' )
# # plt.imshow( (np.abs(gy[-1]) + np.abs(gz[-1])) < 0.01, extent=lims, origin='lower' )

# pp = minima[i].T
# plt.plot( rec_z[0,pp[1]], rec_y[pp[0],0], 'k.' )
# pp = maxima[i].T
# plt.plot( rec_z[0,pp[1]], rec_y[pp[0],0], 'b.' )

# plt.colorbar()
# plt.show()


plt.figure()
plt.imshow( rec_x[i], extent=lims, origin='lower', cmap='Spectral' )
nt = len(rec_x)
for j in range(nt):
    pp = minima[j].T
    plt.plot( rec_z[0,pp[1]], rec_y[pp[0],0], '.', color=(i/nt,0.2,1-i/nt) )
    # pp = maxima[i].T
    # plt.plot( rec_z[0,pp[1]], rec_y[pp[0],0], '.', color=(i/nt,0.2,1-i/nt) )
# plt.axis('equal')

plt.colorbar()
plt.show()


plt.figure()
plt.imshow(rec_x[i], extent=lims, origin='lower', cmap='Spectral')
# for tr in max_tracks:
for tr in min_tracks:
    pp = tr['positions'].astype(int)
    plt.plot(rec_z[0, pp[:, 1]], rec_y[pp[:, 0], 0], '-o', markersize=3)
plt.colorbar()
plt.show()


# plt.figure()
# plt.errorbar(min_mvz, min_mvy, xerr=min_svz, yerr=min_svy, fmt='s', label='Min')
# plt.errorbar(max_mvz, max_mvy, xerr=max_svz, yerr=max_svy, fmt='o', label='Max')
# plt.legend()
# plt.show()



#%%


# i = 0
# fig, ax = plt.subplots(1,2, figsize=(6,5))

# vmax = np.nanmax(np.abs( rec_x[i] - np.nanmean(rec_x[i]) ))
# norm = Normalize(vmin=-vmax, vmax=vmax)

# s1 = ax[0].imshow( rec_x[i], extent=lims, origin='lower', cmap='Spectral' )
# s2 = ax[1].imshow( rec_p[i], extent=lims, origin='lower', cmap='Spectral' )

# plt.colorbar(s1, ax=ax[0])
# plt.colorbar(s2, ax=ax[1])
# plt.show()


# C = spatiotemporal_correlation(rec_x, max_lag=5)

# plt.imshow(C[4], origin='lower', aspect='auto')
# plt.colorbar(label='Correlation')
# plt.xlabel('Displacement x')
# plt.ylabel('Displacement y')
# plt.show()
#%%



#%%

# plt.figure()
# plt.imshow( rec_xf[i] , extent=lims, origin='lower', cmap='Spectral' )
# plt.colorbar()
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




# plt.figure()
# # plt.plot( tiempo, T_amb, '.-' , label='Amb' )
# plt.plot( tiempo, T_top, '.-' , label='Top')
# plt.plot( tiempo, T_bot, '.-' , label='Bot' )

# # # # plt.plot( tiempo, Q_tunnel, '.-', label='Q' )
# # # plt.plot( tiempo, Q_tunnel / area_Q / 3600 , '.-', label='vel (m/s)'  )

# # # plt.plot( rec_t, np.nanmedian( rec_x, axis=(1,2)), '.-', label='rec_x' )

# # plt.plot( rec_t, np.median(Q_tunnel) / (np.nanmedian( rec_x, axis=(1,2)) /1000 * 0.3) / 3600 , '.-', label='rec_x' )
# # plt.hlines( np.median(Q_tunnel) / area_Q / 3600, 0, rec_t[-1], colors='k', linestyles='dashed', label='Area' )



# # # plt.plot( T_top, '.-' , label='Top')
# # # plt.plot( T_bot, '.-' , label='Bot' )
# # # plt.plot( Q_tunnel / area_Q / 3600 , '.-', label='vel (m/s)'  )

# plt.legend()
# plt.show()


#%%
# =============================================================================
# 3D figure
# =============================================================================

fsize = 16
plt.rcParams.update({ 'font.size':fsize })

# i = len(rec_x) - 1 # time
i = 84

L = 2  # padding in mm
cmap_name = 'Spectral'

# zg,yg,xg = rec_z.ravel(), rec_y.ravel(), rec_x[i].ravel()
zg,yg,xg = rec_z.ravel(), rec_y.ravel(), rec_f[i].ravel()

# xg[yg>250] = np.nan
xg, yg, zg = xg[yg<250], yg[yg<250], zg[yg<250]
me = np.nanmean(xg)

trigri = mtri.Triangulation( zg, yg)

fraction = 1.  
np.random.seed(42)
n = int(len(trigri.triangles) * fraction)
idx = np.random.choice(len(trigri.triangles), n, replace=False)

fig = plt.figure(figsize=(10,8))
# fig = plt.figure()
ax = plt.axes(projection='3d')

height = xg - me 
vmax = np.nanmax(np.abs(height))
norm = Normalize(vmin=-vmax, vmax=vmax)

aa = ax.plot_trisurf( zg, height, yg, triangles=trigri.triangles[idx], color='white', shade=False, linewidth=0, antialiased=False )
# aspect_ratio_fig(ax, zg, height, yg)
ax.invert_xaxis()
ax.set_axis_off()

bounds = add_bounding_box(ax, zg, height, yg, L=L)
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_zlim(bounds[2])

aa.set_zorder(10)

ax.view_init( 10, -60, 0)

(xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])
ax.set_zlim(bounds[2][0], bounds[2][1])

ax.set_box_aspect( [2, 2 * abs(bounds[1][1]-bounds[1][0]) / abs(bounds[0][1]-bounds[0][0]), 
                    2 * abs(bounds[2][1]-bounds[2][0]) / abs(bounds[0][1]-bounds[0][0])], zoom=0.85 )


# add_dimension_label( ax, (xmin, ymax, zmax), (xmax, ymax, zmax), f'{xmax-xmin:.0f} mm', side=-1, va='bottom' ) # X dimension (zg)
add_dimension_label( ax, (xmin, ymin, zmin), (xmax, ymin, zmin), f'{xmax-xmin:.0f} mm', side=-1, va='center', fontsize=fsize ) # X dimension (zg)
add_dimension_label( ax, (xmax, ymin, zmin), (xmax, ymax, zmin), f'{ymax-ymin:.0f} mm', side=-1, va='center', fontsize=fsize ) # Y dimension (height)
add_dimension_label( ax, (xmax, ymax, zmin), (xmax, ymax, zmax), f'{zmax-zmin:.0f} mm', side=-1, va='center', fontsize=fsize ) # Z dimension (yg)

if cmap_name in mpl.colormaps:
    cmap = mpl.colormaps[cmap_name]
elif hasattr(cmcrameri.cm, cmap_name):
    cmap = getattr(cmcrameri.cm, cmap_name)
else:
    raise ValueError(f"Unknown colormap: {cmap_name}")


tri_values = np.mean(height[trigri.triangles[idx]], axis=1)
facecolors = cmap(norm(tri_values))
aa.set_facecolors(facecolors)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label=r'$h-\langle h\rangle$', pad=-0.2, shrink=0.8)


filename = '/Volumes/Ice blocks/Scan water channel/Figures/f' + str(f) + '_3dlast.pdf'
# plt.savefig(filename, dpi=400, bbox_inches='tight', transparent=True)

plt.show()

#%%
# =============================================================================
# Time evolution
# =============================================================================
fsize = 14
plt.rcParams.update({ 'font.size':fsize })

centered = True
# ies = [0,20,40,60]
ies = [0,15,30,45]

if centered:
    vmax = np.nanmax(np.abs( rec_x[ies[-1]] - np.nanmean(rec_x[ies[-1]]) ))
    norm = Normalize(vmin=-vmax, vmax=vmax)
else:
    vmax = np.nanmax( rec_x[ies[-1]] - np.nanmean(rec_x[ies[-1]]) )
    vmin = np.nanmin( rec_x[ies[-1]] - np.nanmean(rec_x[ies[-1]]) )
    norm = Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1,4, figsize=(7,5), layout='constrained', sharex=True, sharey=True)

for j, i in enumerate(ies):
    # s1 = ax[j].imshow( rec_x[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='Spectral', norm=norm )
    s1 = ax[j].imshow( rec_f[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='Spectral', norm=norm )
    # ax[j].axis('off')

    ax[j].set_title(f't = {i*0.5} min', fontsize=fsize+1)
    ax[j].tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

ydif = rec_y.max() - rec_y.min()
zdif = rec_z.max() - rec_z.min()

ax[0].set_ylabel(f'{ydif:.0f} mm')
for j in range(4):
    ax[j].set_xlabel(f'{zdif:.0f} mm')

cbar = fig.colorbar(s1, ax=ax )
cbar.set_label(r'$h-\langle h\rangle$' )

filename = '/Volumes/Ice blocks/Scan water channel/Figures/f' + str(f) + '_2dtime.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%
# =============================================================================
# Comparison same experiment
# =============================================================================
fsize = 14
plt.rcParams.update({ 'font.size':fsize })

centered = True
fs = [4,5,6,7]

fig, ax = plt.subplots(1,4, figsize=(10,5), layout='constrained' )

for j, f in enumerate(fs):
    t1 = time()    
    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    # rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, 0, -1 )
    
    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]
    
    t_med = time()    
    
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
        
    lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
    dz, dy = 2.5, 0.5
    
    t2 = time()
    print(t2-t1)
    
    print( f'{f},\tvel: {np.median(Q_tunnel)/area_Q/3600:.2f} m/s,\t T:{np.median(T_top):.1f} °C,\t GSF: {grid_vels[f]}'  )
    print(rec_x.shape)
    print()
    
    # i = len(rec_x) - 2 # time
    i = 30 # time
    
    if centered:
        vmax = np.nanmax(np.abs( rec_x[i] - np.nanmean(rec_x[i]) ))
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        vmax = np.nanmax( rec_x[i] - np.nanmean(rec_x[i]) )
        vmin = np.nanmin( rec_x[i] - np.nanmean(rec_x[i]) )
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    
    s1 = ax[j].imshow( rec_f[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='Spectral', norm=norm )
    # s1 = ax[j].imshow( rec_f[i] - np.nanmean(rec_x[i]), extent=lims, origin='lower', cmap='RdYlBu', norm=norm )
    cbar = fig.colorbar(s1, ax=ax[j], label=r'$h-\langle h\rangle$' )

    ax[j].set_title(f't = {i*0.5} min', fontsize=fsize+1)
    ax[j].tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

    ydif = rec_y.max() - rec_y.min()
    zdif = rec_z.max() - rec_z.min()
    ax[j].set_ylabel(f'{ydif:.0f} mm')
    ax[j].set_xlabel(f'{zdif:.0f} mm')

filename = '/Volumes/Ice blocks/Scan water channel/Figures/comparison_same.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%
# =============================================================================
# Melt rates
# =============================================================================

x0 = -400
order = 4
surf_norm = True
av_kth = False
lwidth = 1/2 # 1 or 1/2 for full channel depth or half channel
low_y = -80

Nus, Res, Prs = [],[], []
Nus_sd, Res_sd = [],[]
h_rms, h_sd, area_diff = [],[],[]
bh_rms, bh_sd = [],[]

Nus_x, Res_x = [], []

# top
TNus, TRes, TPrs = [],[], []
TNus_sd, TRes_sd = [],[]
Th_rms, Th_sd, Tarea_diff = [],[],[]
Tbh_rms, Tbh_sd = [],[]


for f in tqdm(range(len(all_folders)-1)):

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    
    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
    elif f==2: 
        back_t, back_x = back_t[:-1], back_x[:-1]
        rec_x[:,618:1130,50:] = np.nan
    
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
        
    lims = [np.nanmin(rec_z), np.nanmax(rec_z),np.nanmin(rec_y),np.nanmax(rec_y)]
    dz, dy = 2.5, 0.5
    
    # filt_y = np.argmin(np.all(rec_y <= 0, axis=1)) 
    # rec_x[:,:filt_y,:] = np.nan
    # rec_f[:,:filt_y,:] = np.nan
    
    # if f == 0:
    #     fig,ax = plt.subplots(1,2)
    #     ax[0].imshow( rec_f[20], origin="lower" )
    #     ax[0].set_aspect(0.2)
    

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            
        gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )
        gt_y = np.nanmean( gt, axis=2 )
        
        mascara = ~np.isnan(rec_f)
        sup_ar = area_superficial_multi_2d(rec_f, rec_z, rec_y, mascara)
        flat_a = area_superficial_multi_2d( np.ones_like(rec_f), rec_z, rec_y, mascara)
        
        melt_rate_sd = 0.0025 # mm/s
        tempt = temp_t(T_top, T_bot, rec_t, tiempo)
        mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
        med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
        
        ushearT = med_Q / mean_x * 25/27 #np.median(Q_tunnel) / ( mean_x /1000 * 0.3) / 3600
        ushearT_s = 25/27 * np.sqrt( ( std_Q / mean_x )**2 + ( med_Q * std_x / mean_x**2 )**2 )
            
        if av_kth:
            kth_mean, diff_mean, nu_mean = np.nanmean(thermal_cond(tempt)), np.nanmean(heat_diff(tempt)), np.nanmean(nu_visco(tempt))
            PrT = np.array( [nu_mean / diff_mean] * len(tempt) )
            ReT = ushearT * mean_x/1000*lwidth / nu_mean
            ReT_s = 1/1000 / nu_mean * np.sqrt( (ushearT_s * mean_x*lwidth)**2 + (ushearT * std_x*lwidth)**2  )

            Nu_x = rho_ice * latent / (kth_mean * tempt )[:,None] * (rec_y[:,0]-x0)[None,:]/1000 * gt_y/1000
            Re_x = ushearT[:,None] * (rec_y[:,0]-x0)[None,:]/1000 / nu_mean

            if surf_norm: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt ) * flat_a/sup_ar
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt ) * flat_a/sup_ar
            else: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt )
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt )
                
        else:
            PrT = nu_visco(tempt) / heat_diff(tempt)
            ReT = ushearT * mean_x/1000*lwidth / nu_visco(tempt)
            ReT_s = 1/1000 / nu_visco(tempt) * np.sqrt( (ushearT_s * mean_x*lwidth)**2 + (ushearT * std_x*lwidth)**2  )

            Nu_x = rho_ice * latent / (thermal_cond(tempt) * tempt )[:,None] * (rec_y[:,0]-x0)[None,:]/1000 * gt_y/1000    
            Re_x = ushearT[:,None] * (rec_y[:,0]-x0)[None,:]/1000 / nu_visco(tempt)[:,None]

            if surf_norm: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
            else: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt )
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt )
    
        Res.append(ReT); Prs.append(PrT); Nus.append(NuT)
        Res_sd.append(ReT_s); Nus_sd.append(NuT_s)
        
        Res_x.append(Re_x); Nus_x.append(Nu_x)
    
        diff = np.sqrt( ( rec_x - np.nanmean(rec_x, axis=(1,2) )[:,None,None] )**2 )
        hhr = np.nanmean( diff, axis=(1,2) ) 
    
        area_diff.append( flat_a/sup_ar ); h_rms.append(hhr); h_sd.append(std_x)
        
        back_f = np.zeros_like(rec_x)
        for i in range(len(back_f)):
            coeff, r,rank,s = polyfitn(rec_z, rec_y, rec_x[i], order=order)
            back_f[i] = poln( coeff, rec_z, rec_y, order=order )
        back_f[np.isnan(rec_x)] = np.nan
        
        rec_fx = rec_x - back_f
        std_x = np.nanstd(rec_fx, axis=(1,2))
        diff = np.sqrt( ( rec_fx - np.nanmean(rec_fx, axis=(1,2) )[:,None,None] )**2 )
        hhr = np.nanmean( diff, axis=(1,2) ) 
        
        bh_rms.append(hhr); bh_sd.append(std_x)

    
    # Top
    filt_y = np.argmin(np.all(rec_y <= low_y, axis=1)) 
    rec_x[:,:filt_y,:] = np.nan
    rec_f[:,:filt_y,:] = np.nan
    
    # if f == 0:
    #     ax[1].imshow( rec_f[20], origin="lower" )
    #     ax[1].set_aspect(0.2)
    #     plt.show()
    

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            
        gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )
        
        mascara = ~np.isnan(rec_f)
        sup_ar = area_superficial_multi_2d(rec_f, rec_z, rec_y, mascara)
        flat_a = area_superficial_multi_2d( np.ones_like(rec_f), rec_z, rec_y, mascara)
        
        melt_rate_sd = 0.0025 # mm/s
        tempt = temp_t(T_top, T_bot, rec_t, tiempo)
        mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
        med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
        
        ushearT = med_Q / mean_x * 25/27 #np.median(Q_tunnel) / ( mean_x /1000 * 0.3) / 3600
        ushearT_s = 25/27 * np.sqrt( ( std_Q / mean_x )**2 + ( med_Q * std_x / mean_x**2 )**2 )
        
        if av_kth:
            kth_mean, diff_mean, nu_mean = np.nanmean(thermal_cond(tempt)), np.nanmean(heat_diff(tempt)), np.nanmean(nu_visco(tempt))
            PrT = np.array( [nu_mean / diff_mean] * len(tempt) )
            ReT = ushearT * mean_x/1000*lwidth / nu_mean
            ReT_s = 1/1000 / nu_mean * np.sqrt( (ushearT_s * mean_x*lwidth)**2 + (ushearT * std_x*lwidth)**2  )

            Nu_x = rho_ice * latent / (kth_mean * tempt )[:,None] * (rec_y[:,0]-x0)[None,:]/1000 * gt_y/1000
            Re_x = ushearT[:,None] * (rec_y[:,0]-x0)[None,:]/1000 / nu_mean

            if surf_norm: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt ) * flat_a/sup_ar
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt ) * flat_a/sup_ar
            else: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt )
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / ( kth_mean * tempt )
                
        else:
            PrT = nu_visco(tempt) / heat_diff(tempt)
            ReT = ushearT * mean_x/1000*lwidth / nu_visco(tempt)
            ReT_s = 1/1000 / nu_visco(tempt) * np.sqrt( (ushearT_s * mean_x*lwidth)**2 + (ushearT * std_x*lwidth)**2  )

            Nu_x = rho_ice * latent / (thermal_cond(tempt) * tempt )[:,None] * (rec_y[:,0]-x0)[None,:]/1000 * gt_y/1000    
            Re_x = ushearT[:,None] * (rec_y[:,0]-x0)[None,:]/1000 / nu_visco(tempt)[:,None]

            if surf_norm: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
            else: 
                NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt )
                NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000*lwidth / (thermal_cond(tempt) * tempt )
            
        # # if surf_norm: NuT = melt_rate/1000 * rho_ice * latent * length0 / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
        # # else: NuT = melt_rate/1000 * rho_ice * latent * length0 / (thermal_cond(tempt) * tempt )
        # if surf_norm: NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000 / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
        # else: NuT = melt_rate/1000 * rho_ice * latent * mean_x/1000 / (thermal_cond(tempt) * tempt )
        # ReT = ushearT * mean_x/1000 / nu_visco(tempt)
        # PrT = nu_visco(tempt) / heat_diff(tempt)
        
        # ReT_s = 1/1000 / nu_visco(tempt) * np.sqrt( (ushearT_s * mean_x)**2 + (ushearT * std_x)**2  )
        # # if surf_norm: NuT_s = melt_rate_sd/1000 * rho_ice * latent * length0 / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
        # # else: NuT_s = melt_rate_sd/1000 * rho_ice * latent * length0 / (thermal_cond(tempt) * tempt )
        # if surf_norm: NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000 / (thermal_cond(tempt) * tempt ) * flat_a/sup_ar
        # else: NuT_s = melt_rate_sd/1000 * rho_ice * latent * mean_x/1000 / (thermal_cond(tempt) * tempt )
    
        TRes.append(ReT); TPrs.append(PrT); TNus.append(NuT)
        TRes_sd.append(ReT_s); TNus_sd.append(NuT_s)
    
        diff = np.sqrt( ( rec_x - np.nanmean(rec_x, axis=(1,2) )[:,None,None] )**2 )
        hhr = np.nanmean( diff, axis=(1,2) ) 
    
        Tarea_diff.append( flat_a/sup_ar ); Th_rms.append(hhr); Th_sd.append(std_x)

        back_f = np.zeros_like(rec_x)
        for i in range(len(back_f)):
            coeff, r,rank,s = polyfitn(rec_z, rec_y, rec_x[i], order=order)
            back_f[i] = poln( coeff, rec_z, rec_y, order=order )
        back_f[np.isnan(rec_x)] = np.nan
        
        rec_fx = rec_x - back_f
        std_x = np.nanstd(rec_fx, axis=(1,2))
        diff = np.sqrt( ( rec_fx - np.nanmean(rec_fx, axis=(1,2) )[:,None,None] )**2 )
        hhr = np.nanmean( diff, axis=(1,2) ) 
        
        Tbh_rms.append(hhr); Tbh_sd.append(std_x)
    

#%%
rough = 'rms' # 'sd' or 'rms'

re_power = 0.0

def power_law(x, a,p):
    return a * x**p

m_nus, m_res = [], [] 
s_nus, s_res = [], [] 

fig, axs = plt.subplots(2,4, layout='constrained', figsize=(12,8), sharey=False, sharex=False)

ax = axs[0,:]
for i in range(len(Nus)):
    ax[0].plot( Res[i], Nus[i] / Res[i]**re_power, '.-' )
    ax[0].fill_between( Res[i], (Nus[i]+Nus_sd[i]) / Res[i]**re_power, 
                       (Nus[i]-Nus_sd[i]) / Res[i]**re_power, alpha=0.3  )
    
    ax[2].plot( Prs[i], Nus[i] / Res[i]**re_power, '.-' )
    ax[2].fill_between( Prs[i], (Nus[i]+Nus_sd[i]) / Res[i]**re_power, 
                       (Nus[i]-Nus_sd[i]) / Res[i]**re_power, alpha=0.3  )
    
    if rough == 'rms':
        ax[3].plot( h_rms[i], Nus[i] / Res[i]**re_power, '.-' )
        ax[3].fill_between( h_rms[i], (Nus[i]+Nus_sd[i]) / Res[i]**re_power, 
                          ( Nus[i]-Nus_sd[i]) / Res[i]**re_power, alpha=0.3  )
    elif rough == 'sd':
        ax[3].plot( h_sd[i], Nus[i] / Res[i]**re_power, '.-' )
        ax[3].fill_between( h_sd[i], (Nus[i]+Nus_sd[i]) / Res[i]**re_power, 
                           (Nus[i]-Nus_sd[i]) / Res[i]**re_power, alpha=0.3  )
    
    mean_Re = np.nanmean(Res[i])
    err_Re = np.sqrt(np.nansum(Res_sd[i]**2)) / np.sum(~np.isnan(Res_sd[i]))
    std_Re = np.nanstd(Res[i])
    
    mean_Nu = np.nanmean(Nus[i])
    err_Nu = np.sqrt(np.nansum(Nus_sd[i]**2)) / np.sum(~np.isnan(Nus_sd[i]))
    std_Nu = np.nanstd(Nus[i])
    
    # ax[0].errorbar( mean_Re, mean_Nu, yerr=err_Nu, fmt='s' )
    ax[1].errorbar( mean_Re, mean_Nu / mean_Re**re_power, xerr=err_Re, 
                   yerr=err_Nu / mean_Re**re_power, fmt='.' )
    ax[1].errorbar( mean_Re, mean_Nu / mean_Re**re_power, xerr=std_Re, 
                   yerr=std_Nu / mean_Re**re_power, fmt='.', alpha=0.3 )
    
    m_nus.append(mean_Nu), m_res.append(mean_Re)
    s_nus.append(std_Nu), s_res.append(std_Re)
    
# res_p = np.logspace( 3.7, 4.6, 1000 )
# ax[1].plot( res_p, power_law(res_p, 0.18, 0.92) / res_p**re_power, 'k--' )
# ax[0].plot( res_p, power_law(res_p, 0.18, 0.92) / res_p**re_power, 'k--' )

# plt.show()

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

(a,p), cov = curve_fit(power_law, m_res, m_nus, sigma=s_nus, absolute_sigma=True )
perr = np.sqrt(np.diag(cov))
    
print("All")
print(a, p)
print( perr[0], perr[1] )


m_nus, m_res = [], [] 
s_nus, s_res = [], [] 

# fig, ax = plt.subplots(1,4, layout='constrained', figsize=(12,5), sharey=True, sharex=False)

ax = axs[1,:]
for i in range(len(Nus)):
    ax[0].plot( TRes[i], TNus[i] / TRes[i]**re_power, '.-' )
    ax[0].fill_between( TRes[i], (TNus[i]+TNus_sd[i]) / TRes[i]**re_power, 
                       (TNus[i]-TNus_sd[i]) / TRes[i]**re_power, alpha=0.3  )
    
    ax[2].plot( TPrs[i], TNus[i] / TRes[i]**re_power, '.-' )
    ax[2].fill_between( TPrs[i], (TNus[i]+TNus_sd[i]) / TRes[i]**re_power, 
                       (TNus[i]-TNus_sd[i]) / TRes[i]**re_power, alpha=0.3  )
    
    if rough == 'rms':
        ax[3].plot( Th_rms[i], TNus[i] / TRes[i]**re_power, '.-' )
        ax[3].fill_between( Th_rms[i], (TNus[i]+TNus_sd[i]) / TRes[i]**re_power, 
                           (TNus[i]-TNus_sd[i]) / TRes[i]**re_power, alpha=0.3  )
    elif rough == 'sd':
        ax[3].plot( Th_sd[i], TNus[i], '.-' )
        ax[3].fill_between( Th_sd[i], (TNus[i]+TNus_sd[i]) / TRes[i]**re_power, 
                           (TNus[i]-TNus_sd[i]) / TRes[i]**re_power, alpha=0.3  )

    mean_Re = np.nanmean(TRes[i])
    err_Re = np.sqrt(np.nansum(TRes_sd[i]**2)) / np.sum(~np.isnan(Res_sd[i]))
    std_Re = np.nanstd(TRes[i])
    
    mean_Nu = np.nanmean(TNus[i])
    err_Nu = np.sqrt(np.nansum(TNus_sd[i]**2)) / np.sum(~np.isnan(TNus_sd[i]))
    std_Nu = np.nanstd(TNus[i])
    
    # ax[0].errorbar( mean_Re, mean_Nu, yerr=err_Nu, fmt='s' )
    ax[1].errorbar( mean_Re, mean_Nu / mean_Re**re_power, 
                   xerr=err_Re, yerr=err_Nu / mean_Re**re_power, fmt='.' )
    ax[1].errorbar( mean_Re, mean_Nu / mean_Re**re_power, 
                   xerr=std_Re, yerr=std_Nu / mean_Re**re_power, fmt='.', alpha=0.3 )
    
    m_nus.append(mean_Nu), m_res.append(mean_Re)
    s_nus.append(std_Nu), s_res.append(std_Re)
    
# res_p = np.logspace( 3.7, 4.6, 1000 )
# ax[1].plot( res_p, power_law(res_p, 0.18, 0.92) / res_p**re_power, 'k--' )
# ax[0].plot( res_p, power_law(res_p, 0.18, 0.92) / res_p**re_power, 'k--' )

ax[0].set_xlabel('Re')
ax[1].set_xlabel(r'$\langle Re \rangle$')
ax[2].set_xlabel('Pr')
if rough == 'rms': ax[3].set_xlabel(r'$h_{rms}$')
elif rough == 'sd': ax[3].set_xlabel(r'$\sigma_h$')

axs[0,0].set_ylabel('Nu (all)')
axs[1,0].set_ylabel('Nu (top)')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

ax[0].set_yticks([100,300,600], minor=True)
ax[0].set_yticklabels(['100','300','600'], minor=True)
ax[1].set_xticks([7000,40000], minor=True)
ax[1].set_xticklabels(['7000','40000'], minor=True)
ax[1].set_yticks([100,300,600], minor=True)
ax[1].set_yticklabels(['100','300','600'], minor=True)

plt.show()


(a,p), cov = curve_fit(power_law, m_res, m_nus, sigma=s_nus, absolute_sigma=True )
perr = np.sqrt(np.diag(cov))

print("Top")
print(a, p)
print( perr[0], perr[1] )

#%%

rough = 'rms' # 'sd'
part = 'top' # 'top'


def power_law_all(X, a, b, c, d):
    Re, Pr, hrms = X
    return a * Re**b * Pr**c * hrms**d

if part == 'all':
    Nu_all = np.concatenate(Nus)
    Re_all = np.concatenate(Res)
    Pr_all = np.concatenate(Prs)
    hrms_all = np.concatenate(h_rms)
    hsd_all = np.concatenate(h_sd)
elif part == 'top':
    Nu_all = np.concatenate(TNus)
    Re_all = np.concatenate(TRes)
    Pr_all = np.concatenate(TPrs)
    hrms_all = np.concatenate(Th_rms)
    hsd_all = np.concatenate(Th_sd)

mask = ( np.isfinite(Nu_all) & np.isfinite(Re_all) & np.isfinite(Pr_all) & np.isfinite(hrms_all) & 
        (Nu_all > 0) & (Re_all > 0) & (Pr_all > 0) & (hrms_all > 0) )

Nu_all = Nu_all[mask]
Re_all = Re_all[mask]
Pr_all = Pr_all[mask]
hrms_all = hrms_all[mask]
hsd_all = hsd_all[mask]

if rough == 'rms':
    popt, pcov = curve_fit( power_law_all, (Re_all, Pr_all, hrms_all), Nu_all, p0=[0.1, 0.9, 0.3, 0.1], maxfev=10000 )
    Nu_pred = power_law_all( (Re_all, Pr_all, hrms_all), *popt )
elif rough == 'sd':
    popt, pcov = curve_fit( power_law_all, (Re_all, Pr_all, hsd_all), Nu_all, p0=[0.1, 0.9, 0.3, 0.1], maxfev=10000 )
    Nu_pred = power_law_all( (Re_all, Pr_all, hsd_all), *popt )
perr = np.sqrt(np.diag(pcov))
a, b, c, d = popt

print(np.array2string(popt, precision=3, suppress_small=True))
print(np.array2string(perr, precision=3, suppress_small=True))
# print( popt )
# print( perr )


fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(Nu_all, Nu_pred)

lims = [ min(Nu_all.min(), Nu_pred.min()), max(Nu_all.max(), Nu_pred.max()) ]

ax.plot(lims, lims, 'k--')

ax.set_xlabel('Measured Nu')
ax.set_ylabel('Predicted Nu')
ax.set_title(part)
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.show()

#%%
# =============================================================================
# Comparison wall roughness heat transfer
# =============================================================================
import matplotlib.colors as mcolors 
from matplotlib.lines import Line2D

# norm = Normalize(vmin=0.5, vmax=10.0)
norm = Normalize(vmin=0.5, vmax=11.0)

cmap = plt.colormaps['magma_r'].resampled(256)
cmap = mcolors.LinearSegmentedColormap.from_list( 'magma_r_truncated', cmap(np.linspace(0.15, 0.75, 256)) )

def pr_color(Pr):
    return cmap(norm(Pr))

# Kevin paper numbers, roughness : k/h = 1/18
# possibly
Re_kev = np.array( [5.2e3, 7.6e3,9.3e3,1.3e4,2.6e4] * 3 )
Nu_kev = np.array( [16,23,28,37,66,27,39,47,62,111,44,64,76,100,173] )
Pr_kev = np.array( [0.5]*5 + [1.0]*5 + [2.0]*5 )


# Dipprey tesis table 
#tube e-3
Re_dipe3 = np.array( [2.44,4.18,6.94,12.1,3.22,5.49,9.14,15.9,4.86,8.30,13.8,23.9,10.7,18.2,30.2,52.2] ) * 1e4
Ch_dipe3 = np.array( [1.14,1.05,0.982,0.892,1.32,1.20,1.10,1.02,1.52,1.43,1.29,1.20,2.07,1.88,1.71,1.62] ) * 1e-3
Pr_dipe3 = np.array( [5.94]*4 + [4.38]*4 + [2.79]*4 + [1.20]*4 )
Nu_dipe3 = Ch_dipe3 * Re_dipe3  * Pr_dipe3 

Re_dipd3 = np.array( [1.40,2.41,4.10,6.82,11.9,1.85,3.16,5.40,8.98,15.7,2.76,4.77,8.15,13.6,23.5,6.05,10.6,17.8,29.8,51.4] ) * 1e4
# Ch_dipd3 = np.array( [1.28,1.21,1.17,1.12,1.14,1.46,1.39,1.32,1.30,1.31,1.77,1.62,1.57,1.51,1.56,2.40,2.28,2.18,2.10,2.07] ) * 1e-3 
Ch_dipd3 = np.array( [1.28,1.23,1.21,1.21,1.31,1.46,1.41,1.39,1.43,1.50,1.78,1.69,1.69,1.71,1.78,2.52,2.47,2.45,2.42,2.37] ) * 1e-3 #paper seems to use this one
Pr_dipd3 = np.array( [5.94]*5 + [4.38]*5 + [2.79]*5 + [1.20]*5 )
Nu_dipd3 = Ch_dipd3 * Re_dipd3  * Pr_dipd3 

Re_dipc9 = np.array( [2.37,4.01,6.68,11.6,3.10,5.23,8.80,15.3,4.68,7.92,13.3,22.9,10.2,17.3,29.2,50.1] ) * 1e4
Ch_dipc9 = np.array( [2.00,2.15,2.09,1.94,2.43,2.47,2.34,2.17,2.87,2.91,2.74,2.50,3.82,3.61,3.42,3.13] ) * 1e-3
Pr_dipc9 = np.array( [5.94]*4 + [4.38]*4 + [2.79]*4 + [1.20]*4 )
Nu_dipc9 = Ch_dipc9 * Re_dipc9  * Pr_dipc9 

Re_dipa4 = np.array( [1.35,2.32,3.94,6.56,11.5,1.78,3.05,5.19,8.64,15.1,2.68,4.59,7.85,13.0,22.6,5.85,10.1,17.2,28.6,49.3] ) * 1e4
Ch_dipa4 = np.array( [3.10,3.03,2.87,2.60,2.25,3.54,3.43,3.17,2.83,2.50,4.14,3.95,3.58,3.16,2.82,5.03,4.64,4.21,3.89,3.41] ) * 1e-3
Pr_dipa4 = np.array( [5.94]*5 + [4.38]*5 + [2.79]*5 + [1.20]*5 )
Nu_dipa4 = Ch_dipa4 * Re_dipa4  * Pr_dipa4 

# ravigururajan
Re_ravt1p10 = np.array( [2481,3460,5662,7473,9394,10867,12570,13947,15261] )
Nu_ravt1p10 = np.array( [23,37,56,74,95,110,128,146,168] )
Pr_ravt1p10 = np.array( [10]*len(Re_ravt1p10) )
Re_ravt2p10 = np.array( [2949,4002,5582,6873,7841,8946,10135,11246,12307,13374] )
Nu_ravt2p10 = np.array( [33,42,62,77,88,101,115,129,143,159] )
Pr_ravt2p10 = np.array( [10]*len(Re_ravt2p10) )
Re_ravt3p10 = np.array( [2970,4348,5466,6775,7623,8696,10271,13092,11318,14131] )
Nu_ravt3p10 = np.array( [33,51,68,89,104,121,139,181,156,195] )
Pr_ravt3p10 = np.array( [10]*len(Re_ravt3p10) )

Re_ravt1p6 = np.array( [5140,9868,13109,13953,19070,23312,26424,29732,33922,37637,41182,44752,48296] )
Nu_ravt1p6 = np.array( [45,79,118,128,158,211,204,227,254,292,334,374,413] )
Pr_ravt1p6 = np.array( [6]*len(Re_ravt1p6) )
Re_ravt2p6 = np.array( [6153,9658,16476,19601,21297,25681,29299,32968,37879,40321,42617,47290] )
Nu_ravt2p6 = np.array( [63,98,163,177,210,257,292,319,348,365,399,441] )
Pr_ravt2p6 = np.array( [6]*len(Re_ravt2p6) )
Re_ravt3p6 = np.array( [9789,13563,17168,20704,24453,27322,32048,35316,38646] )
Nu_ravt3p6 = np.array( [112,152,198,240,285,323,372,411,460] )
Pr_ravt3p6 = np.array( [6]*len(Re_ravt3p6) )

# Prandtl-number blocks shared by all three tables
p10 = [6.25e-3, 1.25e-2, 2.5e-2, 6.25e-2, 1.25e-1, 2.5e-1, 5e-1, 1, 4, 16]   # 10 values
p11 = p10[:8] + [2] + p10[8:]                                                  # same plus Pr = 2

# Table 3 – CHF forcing (43 cases), Pirozzoli Wei
Re_b_chf = np.array( [5.3e3]*10 + [1.7e4]*10 + [4.4e4]*11 + [8.25e4]*8 + [1.33e5]*3 + [2.85e5] )
Pr_chf = np.array( p10 + p10 + p11 + p10[:8] + [6.25e-3, 1.25e-2, 1] + [1] )
Nu_chf = np.array( [6.11, 6.147, 6.466, 7.607, 9.434, 12.26, 16.12, 21.19, 36.11, 60.85, 6.65, 7.1, 8.295, 12.34, 18.05, 
            26.96, 39.16, 55.37, 103.3, 182.1, 7.349, 8.677, 11.64, 20.16, 32.52, 51.44, 78.95, 116.6, 165.0, 229.7, 
            419.3, 8.395, 10.96, 16.01, 29.83, 49.6, 81.08, 127.2, 192.5, 9.836, 13.69, 283.9, 530.5] )

# Table 4 – CWT forcing (31 cases)
Re_b_cwt = np.array( [5.3e3]*10 + [1.7e4]*6 + [4.4e4]*11 + [8.25e4] + [1.33e5]*2 + [2.85e5] )
Pr_cwt = np.array( p10 + [6.25e-3, 1.25e-2, 2.5e-2, 1, 4, 16] + p11 + [1] + [6.25e-3, 1] + [1] )
Nu_cwt = np.array( [4.822, 4.92, 5.223, 6.469, 8.406, 11.37, 15.38, 20.59, 36.05, 60.86, 5.15, 5.593, 6.81, 54.28, 102.5, 
            181.9, 5.731, 7.012, 9.944, 18.14, 30.09, 48.89, 76.49, 114.8, 163.9, 228.3, 419.2, 190.0, 8.107, 281.8, 527.3] )

# Table 5 – UIH forcing (27 cases)
Re_b_uih = np.array( [5.3e3]*10 + [1.7e4]*3 + [4.4e4]*11 + [8.25e4, 1.33e5, 2.85e5] )
Pr_uih = np.array( p10 + [1, 4, 16] + p11 + [1, 1, 1] )
Nu_uih = np.array( [7.015, 7.101, 7.39, 8.604, 10.56, 13.56, 17.63, 22.9, 38.91, 63.71, 57.63, 105.8, 185.0, 8.023, 9.414, 
            12.57, 21.51, 34.27, 53.82, 81.77, 119.9, 168.0, 233.3, 421.2, 197.8, 290.4, 540.0] )


#smooth wall
def smooth(Pr, Ret=np.logspace(2, 6, 300), at='Kader'):
    kap, kth, A = 0.4, 0.46, 5.0
    if at=='Zhong': Ath = 11.7*Pr**(2/3) - np.log(11.7*Pr**(-1/3))/kth
    elif at=='Book': Ath = 13*Pr**(2/3) - 7
    elif at=='Kader': Ath = (3.85*Pr**(1/3) - 1.3)**2 + 2.12 * np.log(Pr)
    U = np.log(Ret)/kap + A
    T = np.log(Ret)/kth + Ath
    return Ret*U, Ret*U*Pr/(U*T) 

fsize = 16
plt.rcParams.update({ 'font.size':fsize })

expo = 0.44
trans = 0.35 #0.35
marks = 30
side = 'top' # 'all' or 'top'

m_nus, m_res, m_prs = [], [], []
s_nus, s_res = [], [] 

for i in range(len(Nus)):
    if side  == 'top':
        mean_Re = np.nanmean(TRes[i])
        err_Re = np.sqrt(np.nansum(TRes_sd[i]**2)) / np.sum(~np.isnan(TRes_sd[i]))
        std_Re = np.nanstd(TRes[i])
        mean_Nu = np.nanmean(TNus[i])
        err_Nu = np.sqrt(np.nansum(TNus_sd[i]**2)) / np.sum(~np.isnan(TNus_sd[i]))
        std_Nu = np.nanstd(TNus[i])
        mean_Pr = np.nanmean(Prs[i])    
        m_nus.append(mean_Nu), m_res.append(mean_Re)
        s_nus.append(std_Nu), s_res.append(std_Re)
        m_prs.append(mean_Pr)
    
    elif side  == 'all':
        mean_Re = np.nanmean(Res[i])
        err_Re = np.sqrt(np.nansum(Res_sd[i]**2)) / np.sum(~np.isnan(Res_sd[i]))
        std_Re = np.nanstd(Res[i])
        mean_Nu = np.nanmean(Nus[i])
        err_Nu = np.sqrt(np.nansum(Nus_sd[i]**2)) / np.sum(~np.isnan(Nus_sd[i]))
        std_Nu = np.nanstd(Nus[i])
        mean_Pr = np.nanmean(Prs[i])    
        m_nus.append(mean_Nu), m_res.append(mean_Re)
        s_nus.append(std_Nu), s_res.append(std_Re)
        m_prs.append(mean_Pr)

m_res, m_nus, m_prs = np.array(m_res), np.array(m_nus), np.array(m_prs)
s_res, s_nus = np.array(s_res), np.array(s_nus)

fig, ax = plt.subplots(1,2, layout='constrained', figsize=(11,5))

ax[0].scatter( Re_kev, Nu_kev, c=Pr_kev, s=marks, marker='*', cmap=cmap, norm=norm, alpha=trans )
ax[1].scatter( Re_kev, Nu_kev/Pr_kev**expo, c=Pr_kev, s=marks, marker='*', cmap=cmap, norm=norm, alpha=trans )

ax[1].errorbar( m_res, m_nus / m_prs**expo,  xerr=s_res, yerr=s_nus / m_prs**expo,  fmt='.', color=pr_color(10), 
               capsize=2, markersize=10, markerfacecolor='white', markeredgewidth=1.3 )

ax[1].scatter( Re_dipe3, Nu_dipe3 / Pr_dipe3**expo, c=Pr_dipe3, s=marks, marker='s', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_dipd3, Nu_dipd3 / Pr_dipd3**expo, c=Pr_dipd3, s=marks, marker='o', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_dipc9, Nu_dipc9 / Pr_dipc9**expo, c=Pr_dipc9, s=marks, marker='D', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_dipa4, Nu_dipa4 / Pr_dipa4**expo, c=Pr_dipa4, s=marks, marker='^', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )

ax[1].scatter( Re_ravt1p10, Nu_ravt1p10 / Pr_ravt1p10**expo, c=Pr_ravt1p10, s=marks, marker='<', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_ravt2p10, Nu_ravt2p10 / Pr_ravt2p10**expo, c=Pr_ravt2p10, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_ravt3p10, Nu_ravt3p10 / Pr_ravt3p10**expo, c=Pr_ravt3p10, s=marks, marker='v', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_ravt1p6, Nu_ravt1p6 / Pr_ravt1p6**expo, c=Pr_ravt1p6, s=marks, marker='<', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_ravt2p6, Nu_ravt2p6 / Pr_ravt2p6**expo, c=Pr_ravt2p6, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[1].scatter( Re_ravt3p6, Nu_ravt3p6 / Pr_ravt3p6**expo, c=Pr_ravt3p6, s=marks, marker='v', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )

# ax[1].scatter( Re_b_chf, Nu_chf / Pr_chf**expo, c=Pr_chf, s=marks, marker='<', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
# ax[1].scatter( Re_b_cwt, Nu_cwt / Pr_cwt**expo, c=Pr_cwt, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
# ax[1].scatter( Re_b_uih, Nu_uih / Pr_uih**expo, c=Pr_uih, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )


rree, nnuu = smooth(10, np.logspace(2, 4.5, 300))
ax[1].plot( rree, nnuu / 10**expo, '-', color=pr_color(10), zorder=0 )
rree, nnuu = smooth(5, np.logspace(2, 4.5, 300))
ax[1].plot( rree, nnuu / 5**expo, '-', color=pr_color(5), zorder=0 )
rree, nnuu = smooth(2, np.logspace(2, 4.5, 300))
ax[1].plot( rree, nnuu / 2**expo, '-', color=pr_color(2), zorder=0 )


# ax[0].errorbar( m_res, m_nus, xerr=s_res, yerr=s_nus, fmt='.', color=pr_color(10), 
#                capsize=2, markersize=10, markerfacecolor='white', markeredgewidth=1.3 )
ax[0].errorbar( m_res, m_nus, xerr=s_res, yerr=s_nus,  fmt='.', color=pr_color(10), 
               capsize=2, markersize=10, markerfacecolor='white', markeredgewidth=1.3 )


ax[0].scatter( Re_dipe3, Nu_dipe3, c=Pr_dipe3, s=marks, marker='s', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_dipd3, Nu_dipd3, c=Pr_dipd3, s=marks, marker='o', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_dipc9, Nu_dipc9, c=Pr_dipc9, s=marks, marker='D', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_dipa4, Nu_dipa4, c=Pr_dipa4, s=marks, marker='^', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )

ax[0].scatter( Re_ravt1p10, Nu_ravt1p10, c=Pr_ravt1p10, s=marks, marker='<', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_ravt2p10, Nu_ravt2p10, c=Pr_ravt2p10, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_ravt3p10, Nu_ravt3p10, c=Pr_ravt3p10, s=marks, marker='v', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_ravt1p6, Nu_ravt1p6, c=Pr_ravt1p6, s=marks, marker='<', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_ravt2p6, Nu_ravt2p6, c=Pr_ravt2p6, s=marks, marker='>', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )
ax[0].scatter( Re_ravt3p6, Nu_ravt3p6, c=Pr_ravt3p6, s=marks, marker='v', cmap=cmap, norm=norm, alpha=trans, edgecolors='none' )

rree, nnuu = smooth(10, np.logspace(2, 4.5, 300))
ax[0].plot( rree, nnuu, '-', color=pr_color(10), zorder=100 )
rree, nnuu = smooth(5, np.logspace(2, 4.5, 300))
ax[0].plot( rree, nnuu, '-', color=pr_color(5), zorder=100 )
rree, nnuu = smooth(2, np.logspace(2, 4.5, 300))
ax[0].plot( rree, nnuu, '-', color=pr_color(2), zorder=100 )


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'Pr', location='right')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')


ax[0].set_xlabel('Re')
ax[1].set_xlabel('Re')
ax[0].set_ylabel('Nu')
# ax[1].set_ylabel(r'Nu/Pr$^{1/2}$')
expstr=str(expo)
ax[1].set_ylabel(rf'Nu/Pr$^{{{expstr}}}$')


g = 'dimgray'   # neutral colour: Pr is read from the colorbar
def mk(marker, label, **kw):
    return Line2D([], [], linestyle='', marker=marker, color=g, markerfacecolor=g, markersize=6, label=label, **kw)

handles = [
    mk('*', 'Zhong et al. 2023 (DNS, k/h = 1/18)'),
    # Dipprey & Sabersky 1963 (water, tubes)
    mk('s', 'Dipprey \& Sabersky: E-3 (smooth)'),
    mk('o', r'Dipprey \& Sabersky: D-3 ($\epsilon_s/D$ = 0.0024)'),
    mk('D', r'Dipprey \& Sabersky: C-9 ($\epsilon_s/D$ = 0.0138)'),
    mk('^', r'Dipprey \& Sabersky: A-4 ($\epsilon_s/D$ = 0.0488)'),
    # Ravigururajan (Pr = 10 and 6)
    mk('<', 'Ravigururajan: tube 1'),
    mk('>', 'Ravigururajan: tube 2'),
    mk('v', 'Ravigururajan: tube 3'),
    # your measurements (errorbar points)
    # Line2D([], [], linestyle='', marker='.', markersize=10, color=g,
    #        label='Current work'),
    Line2D([], [], linestyle='', marker='o', markersize=8,
       markerfacecolor='white', markeredgecolor=g,
       markeredgewidth=1.5, color=g,
       label='Current work'),
    # smooth-wall model lines
    Line2D([], [], linestyle='-', color=g,
           label='Smooth-wall log-law model (Pr = 2, 5, 10)'), ]

# 'outside upper center' or 'outside lower center'
fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False)


filename = '/Volumes/Ice blocks/Scan water channel/Figures/Nu_Re.pdf'
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Roughness and Nu
# =============================================================================

rough_param = 'rms' # 'rms' or 'sd
side = 'top' # 'all' or 'top'
trans=0.7
marks = 5
compensate = 1
exp_c = 0.8

fig, ax = plt.subplots(1,3, layout='constrained', figsize=(11,3))

mres = [np.nanmean(TRes[i]) for i in range(len(TRes))] 

norm = plt.Normalize(vmin=np.nanmin(mres), vmax=np.nanmax(mres))
# cmap = plt.cm.viridis
cmap = plt.colormaps['viridis'].resampled(256)
cmap = mcolors.LinearSegmentedColormap.from_list( 'viridis', cmap(np.linspace(0.0, 0.85, 256)) )

for i in range(len(TNus)):
    t_i = np.arange(len(Tbh_rms[i])) * 0.5
    c = cmap(norm(mres[i]))

    if side == 'top':
        if rough_param == 'rms':
            ax[1].plot(t_i, Tbh_rms[i], '.-', color=c, alpha=trans, markersize=marks )
            if compensate: ax[2].plot(Tbh_rms[i], TNus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
            elif not compensate: ax[2].plot(Tbh_rms[i], TNus[i], '.-', color=c, alpha=trans, markersize=marks  )        
        elif rough_param == 'sd':
            ax[1].plot(t_i, Tbh_sd[i], '.-', color=c, alpha=trans, markersize=marks  )
            if compensate: ax[2].plot(Tbh_sd[i], TNus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
            elif not compensate: ax[2].plot(Tbh_sd[i], TNus[i], '.-', color=c, alpha=trans, markersize=marks  )
    
        if compensate: ax[0].plot(t_i, TNus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
        elif not compensate: ax[0].plot(t_i, TNus[i], '.-', color=c, alpha=trans, markersize=marks  )

    elif side == 'all':
        if rough_param == 'rms':
            ax[1].plot(t_i, bh_rms[i], '.-', color=c, alpha=trans, markersize=marks )
            if compensate: ax[2].plot(bh_rms[i], Nus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
            elif not compensate: ax[2].plot(bh_rms[i], Nus[i], '.-', color=c, alpha=trans, markersize=marks  )        
        elif rough_param == 'sd':
            ax[1].plot(t_i, bh_sd[i], '.-', color=c, alpha=trans, markersize=marks  )
            if compensate: ax[2].plot(bh_sd[i], Nus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
            elif not compensate: ax[2].plot(bh_sd[i], Nus[i], '.-', color=c, alpha=trans, markersize=marks  )
    
        if compensate: ax[0].plot(t_i, Nus[i] / mres[i]**exp_c, '.-', color=c, alpha=trans, markersize=marks  )
        elif not compensate: ax[0].plot(t_i, Nus[i], '.-', color=c, alpha=trans, markersize=marks  )



bh_all = []
Nu_all = []

for i in range(len(TNus)):
    if mres[i] > 10000:
        if side == 'top':
            if rough_param == 'rms': bh = Tbh_rms[i]
            elif rough_param == 'sd': bh = Tbh_sd[i]
            Nu = TNus[i] / mres[i]**exp_c

        elif side == 'all':
            if rough_param == 'rms': bh = bh_rms[i]
            elif rough_param == 'sd': bh = bh_sd[i]
            Nu = Nus[i] / mres[i]**exp_c

        valid = np.isfinite(bh) & np.isfinite(Nu)
        bh_all.extend(bh[valid])
        Nu_all.extend(Nu[valid])

bh_all = np.array(bh_all)
Nu_all = np.array(Nu_all)
slope, intercept, r_value, p_value, std_err = linregress(bh_all, Nu_all)

print(f"Slope = {slope:.4f}")
print(f"Intercept = {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")

bh_fit = np.linspace(np.min(bh_all), np.max(bh_all), 200)
Nu_fit = slope * bh_fit + intercept

ax[2].plot( bh_fit, Nu_fit, 'k--', linewidth=1.5, label=f'Nu = {slope:.3f} bh + {intercept:.3f}'  )
# ax[2].legend()


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'Re')

ax[0].set_xlabel(r'$t$ (min)')
ax[1].set_xlabel(r'$t$ (min)')

if compensate:
    ax[0].set_ylabel(rf'Nu/Re$^{{{exp_c}}}$')
    ax[2].set_ylabel(rf'Nu/Re$^{{{exp_c}}}$')
elif not compensate:
    ax[0].set_ylabel(r'Nu')
    ax[2].set_ylabel(r'Nu')

if rough_param == 'rms':  
    ax[1].set_ylabel(r'$\overline{h}_{rms}$')
    ax[2].set_xlabel(r'$\overline{h}_{rms}$')
elif rough_param == 'sd': 
    ax[1].set_ylabel(r'$\overline{h}_{std}$')
    ax[2].set_xlabel(r'$\overline{h}_{std}$')

filename = '/Volumes/Ice blocks/Scan water channel/Figures/corr_h_Nu.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()


#%%
# =============================================================================
# Nu_x vs Re_x
# =============================================================================

pr_pow = 0
powe = 0.8
fs = [3,2,0,4]

def plot_pow(fig, ax, re0, re1, powe, intercept=0.3, top=True):
    rree = np.logspace(re0, re1, 1000)
    ax.plot( rree, rree**powe * intercept, 'k--' )
    if top:
        xtext = rree[500]
        ytext = intercept * xtext**powe
    else:
        xtext = rree[-500]
        ytext = intercept * xtext**powe
    fig.canvas.draw()

    p1 = ax.transData.transform((xtext, ytext))
    p2 = ax.transData.transform((xtext*1.01, ytext*1.01**powe))
    angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
    if top: ax.text(xtext, ytext*1.15, rf'Re$^{{{powe}}}$', rotation=angle, rotation_mode='anchor', ha='center', va='bottom' )
    else: ax.text(xtext, ytext*0.85, rf'Re$^{{{powe}}}$', rotation=angle, rotation_mode='anchor', ha='center', va='top' )

# fig, ax = plt.subplots(1,4, layout='constrained', figsize=(15,4), sharey=True, sharex=False)
fig, ax = plt.subplot_mosaic([[0,0,1,1,2,2,3,3,4],
                              [5,5,5,5,5,6,6,6,6]], layout='tight', figsize=(12,8), sharey=True, gridspec_kw={'width_ratios': [1,1,1,1,1,1,1,1,0.7]})

for j in range(4):    
    ax[j].set_xscale('log')
    ax[j].set_yscale('log')
    
    ax[j].set_xlabel('Re(x)')
    
    re_app = np.nanmean(Res[fs[j]])
    re_app = round(re_app, -int(np.floor(np.log10(abs(re_app))))+1)
    ax[j].set_title(f'Re = {re_app:.0f}')

ax[0].set_ylabel('Nu(x)')    
ax[0].set_ylim(8,9e3)

# f = fs[0]
for j,f in enumerate(fs):
    Nu_x, Re_x, Pr = Nus_x[f], Res_x[f], Prs[f] 
    nt = len(Re_x)
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.coolwarm
    
    for i in range(1, nt, 1):
        if np.all(np.isnan(Re_x[i])) or np.all(np.isnan(Nu_x[i])): continue
        color = cmap(norm(i/nt))
        ax[j].plot(Re_x[i], Nu_x[i] / Pr[i]**(pr_pow) , '-', color=color, label=i)
    
    re0, re1 = np.log10(np.nanmin(Re_x)), np.log10(np.nanmax(Re_x))
    # re0, re1 = 3.3,5.5
    plot_pow(fig, ax[j], re0, re1, 0.8, intercept=0.3)
    iitt = [0.3,0.5,1.,1.2]
    plot_pow(fig, ax[j], re0, re1, 1/2, intercept=iitt[j], top=False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
pos = ax[4].get_position()
cax_top = fig.add_axes([ pos.x0 + pos.width*0., pos.y0 + pos.height*0., pos.width*0.25, pos.height*1.0 ])
fig.colorbar(sm, cax=cax_top, label=r'$t/t_{end}$', aspect=1/20)

ax[4].axis('off')

# fig.colorbar(sm, ax=ax[4], label=r'$t/t_{end}$', pad=-0.5)

# # plt.savefig('./Downloads/nux.png', dpi=400, bbox_inches='tight', transparent=False )
# plt.show()

# fig, ax = plt.subplots(1,1, layout='constrained', figsize=(9,4), sharey=True, sharex=False)
ax[6].axis('off')

ax[5].set_xscale('log')
ax[5].set_yscale('log')

ax[5].set_xlabel('Re(x)')

# re_app = np.nanmean(Res[fs[j]])
# re_app = round(re_app, -int(np.floor(np.log10(abs(re_app))))+1)
# ax[5]s.set_title(f'Re = {re_app:.0f}')

cmaps = [plt.cm.Greys, plt.cm.Purples, plt.cm.Greens, plt.cm.Oranges]

ax[5].set_ylabel('Nu(x)')    
ax[5].set_ylim(8,9e3)

sms, labels = [],[]
for j,f in enumerate(fs):
    Nu_x, Re_x, Pr = Nus_x[f], Res_x[f], Prs[f] 
    nt = len(Re_x)
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = cmaps[j]
    
    for i in range(1, nt, 1):
        if np.all(np.isnan(Re_x[i])) or np.all(np.isnan(Nu_x[i])): continue
        color = cmap(norm(i/nt))
        ax[5].plot(Re_x[i], Nu_x[i] / Pr[i]**(pr_pow) , '-', color=color, alpha=0.5)
        
    re_app = np.nanmean(Res[fs[j]])
    re_app = round(re_app, -int(np.floor(np.log10(abs(re_app))))+1)
    # ax[5]s.set_title(f'Re = {re_app:.0f}')
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    sms.append(sm)
    labels.append(f'$t/t_{{end}}$ (Re = {re_app:.0f})')
    # fig.colorbar(sm, ax=ax[6], label=r'$t/t_{end}$' + f' (Re = {re_app:.0f})', pad=0.9-j*0.2)
    
pos = ax[6].get_position()
n = 4
gap = 0.09
width = (pos.width - (n-1)*gap) / n /2
for j in range(n):
    cax = fig.add_axes([ pos.x0 + j*(width+gap), pos.y0 + pos.height*0.05, width, pos.height*0.99 ])
    fig.colorbar( sms[j], cax=cax, label=labels[j] )
    
re0, re1 = 3.3,5.5
plot_pow(fig, ax[5], re0, re1, 0.8, intercept=0.3)
iitt = [0.3,0.5,1.,1.2]
plot_pow(fig, ax[5], re0, re1, 1/2, intercept=iitt[0], top=False)


filename = '/Volumes/Ice blocks/Scan water channel/Figures/nux_rex.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()



#%%


#%%



#%%
# =============================================================================
# Watershed
# =============================================================================

lwidth = 1/2 # 1 or 1/2

all_lx, all_ly, all_ssca, all_sme = [],[],[],[] 
m_res, rec_ts = [], []

for f in range(len(all_folders) - 1):
# for f in [7]:

    print(f)
    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    
    # if f==0: rec_x[16] = np.nan
    # elif f==3: rec_x[68:71] = np.nan
    # elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]

    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
    elif f==2: 
        back_t, back_x = back_t[:-1], back_x[:-1]
        rec_x[:,618:1130,50:] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")

        tempt = temp_t(T_top, T_bot, rec_t, tiempo)
        mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
        med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
        
        ushearT = med_Q / mean_x * 25/27 #np.median(Q_tunnel) / ( mean_x /1000 * 0.3) / 3600
        ushearT_s = 25/27 * np.sqrt( ( std_Q / mean_x )**2 + ( med_Q * std_x / mean_x**2 )**2 )
        ReT = ushearT * mean_x/1000*lwidth / nu_visco(tempt)
        m_res.append( np.nanmean(ReT) )
        
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
# Wavelength, Area and amplitud
# =============================================================================

fig, ax = plt.subplots(1,3, layout='constrained', figsize=(11,4))

cmap = plt.cm.viridis
cmap = mpl.colors.LinearSegmentedColormap.from_list( 'truncated_viridis', plt.cm.viridis(np.linspace(0, 0.8, 256)) )
# norm = mpl.colors.Normalize(vmin=min(mres), vmax=max(mres))
norm = mpl.colors.Normalize(vmin=14000, vmax=20000)

for j,f in enumerate([0, 8, 4, 5, 6, 7]):

    lx, ly = all_lx[f], all_ly[f]
    rec_t = rec_ts[f]
    ssca = all_ssca[f] 
    sme = all_sme[f]

    mesy = np.array([np.mean(ly[i] * dy) for i in range(len(ly))])
    sdsy = np.array([np.std(ly[i] * dy) for i in range(len(ly))])

    mesa = np.array([np.mean( ssca[i] * dz*dy /100 ) for i in range(len(ly))])
    sdsa = np.array([np.std( ssca[i] * dz*dy /100 ) for i in range(len(ly))])
    
    mesh = np.array([np.mean( sme[i] ) for i in range(len(ly))])
    sdsh = np.array([np.std( sme[i] ) for i in range(len(ly))])
    
    mesy, sdsy = np.array( mesy ), np.array( sdsy )
    mesa, sdsa = np.array( mesa ), np.array( sdsa )
    mesh, sdsh = np.array( mesh ), np.array( sdsh )


    color = cmap(norm(mres[f]))

    ax[0].plot(rec_t/60, mesy, '-', color=color, lw=1.8)
    ax[0].errorbar( rec_t[2*j::10]/60, mesy[2*j::10], yerr=sdsy[2*j::10], fmt='none', ecolor=color,
                elinewidth=1.5, capsize=2, alpha=0.7 )

    ax[1].plot(rec_t/60, mesa, '-', color=color, lw=1.8)
    ax[1].errorbar( rec_t[2*j::10]/60, mesa[2*j::10], yerr=sdsa[2*j::10], fmt='none', ecolor=color,
                elinewidth=1.5, capsize=2, alpha=0.7 )

    ax[2].plot(rec_t/60, mesh, '-', color=color, lw=1.8)
    ax[2].errorbar( rec_t[2*j::10]/60, mesh[2*j::10], yerr=sdsh[2*j::10], fmt='none', ecolor=color,
                elinewidth=1.5, capsize=2, alpha=0.7 )

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

ax[0].set_xlabel(r'$t$ (min)')
ax[1].set_xlabel(r'$t$ (min)')
ax[2].set_xlabel(r'$t$ (min)')
ax[0].set_ylabel(r'$\lambda_y$ (mm)')
ax[1].set_ylabel(r'$A$ (cm$^2$)')
ax[2].set_ylabel(r'$H$ (mm)')

cbar = fig.colorbar(sm, ax=ax[2])
cbar.set_label(r'Re')

filename = '/Volumes/Ice blocks/Scan water channel/Figures/scallops.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()


#%%

# =============================================================================
# Trying other things
# =============================================================================

# Cross correlation

displacs = []

for f in range(len(all_folders) - 1):
# for f in [7]:

    print(f)
    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
        
    # if f==0: rec_x[16] = np.nan
    # elif f==3: rec_x[68:71] = np.nan
    # elif f==2: back_t, back_x = back_t[:-1], back_x[:-1]

    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
    elif f==2: 
        back_t, back_x = back_t[:-1], back_x[:-1]
        rec_x[:,618:1130,50:] = np.nan


    nt, ny, nz = rec_x.shape
    valid_frames = ~np.all(np.isnan(rec_f), axis=(1, 2))
    nan_f = np.isnan(rec_f[valid_frames])
    valid = np.all(~nan_f, axis=0)
    
    y0, y1, x0, x1 = largest_valid_window(valid)
    x1 = x1-2
    
    rec_p = preprocess(rec_f)
    rec_p_clean = rec_p[:, y0:y1, x0:x1]
    
    Corr = spatiotemporal_correlation(rec_p_clean, max_lag=5, mode='same')
    displacements = get_displacements(Corr, ny, nz, mode='same')
    
    dis_z = displacements[:, 0]
    dis_y = displacements[:, 1]
    
    vz_dis, vy_dis = np.mean(dis_z*dz) / 30, np.mean(dis_y*dy) / 30
    displacs.append( (vz_dis, vy_dis) )
    # print()
    # print(dis_z, dis_y)
    # print(dis_z*dz, dis_y*dy)
    
    # print( np.mean(dis_z*dz) / 30, np.mean(dis_y*dy) / 30)
    
#%%
# track max and mins
sigmas = [1,20,4]
slope_v='all'
low_y = -80
dt = 30

vys_mi, vzs_mi, mrs_mi = [],[],[]
vys_si, vzs_si, mrs_si = [],[],[]
vys_ma, vzs_ma, mrs_ma = [],[],[]
vys_sa, vzs_sa, mrs_sa = [],[],[]

m_res = []

for f in range(len(all_folders) - 1):

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )

    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
    elif f==2: 
        back_t, back_x = back_t[:-1], back_x[:-1]
        rec_x[:,618:1130,50:] = np.nan
        
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
        gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )
        
        tempt = temp_t(T_top, T_bot, rec_t, tiempo)
        mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
        med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
        
        ushearT = med_Q / mean_x * 25/27 #np.median(Q_tunnel) / ( mean_x /1000 * 0.3) / 3600
        ushearT_s = 25/27 * np.sqrt( ( std_Q / mean_x )**2 + ( med_Q * std_x / mean_x**2 )**2 )
        ReT = ushearT * mean_x/1000*lwidth / nu_visco(tempt)
        m_res.append( np.nanmean(ReT) )

        
    filt_y = np.argmin(np.all(rec_y <= low_y, axis=1)) 
    rec_nx = np.copy(rec_x)
    rec_nx[:,:filt_y,:] = np.nan

    minima, maxima = get_minmax(rec_z, rec_y, rec_nx, rec_back, sigmas, slope_v='all', tol = 1e-8 )
    #get tracks
    min_tracks = track_extrema( minima, min_length=10, max_gap=2, max_dist_y=15.0, max_dist_z=3. )
    max_tracks = track_extrema( maxima, min_length=10, max_gap=2, max_dist_y=15.0, max_dist_z=3. )
    #add vels
    min_tracks = average_track_velocity(min_tracks, rec_y, rec_z, dt)
    max_tracks = average_track_velocity(max_tracks, rec_y, rec_z, dt)
    #add melt rates
    min_tracks = average_track_gt(gt, min_tracks, radius_y=5, radius_z=1)
    max_tracks = average_track_gt(gt, max_tracks, radius_y=5, radius_z=1)
    
    min_mvy = np.array([tr['vy_avg'] for tr in min_tracks])
    min_svy = np.array([tr['vy_err'] for tr in min_tracks])
    min_mvz = np.array([tr['vz_avg'] for tr in min_tracks])
    min_svz = np.array([tr['vz_err'] for tr in min_tracks])
    min_mmr = np.array([tr['gt_avg'] for tr in min_tracks])
    min_smr = np.array([tr['gt_std'] for tr in min_tracks])
    
    max_mvy = np.array([tr['vy_avg'] for tr in max_tracks])
    max_svy = np.array([tr['vy_err'] for tr in max_tracks])
    max_mvz = np.array([tr['vz_avg'] for tr in max_tracks])
    max_svz = np.array([tr['vz_err'] for tr in max_tracks])
    max_mmr = np.array([tr['gt_avg'] for tr in max_tracks])
    max_smr = np.array([tr['gt_std'] for tr in max_tracks])
        
    print(f'f = {f}')
    # print('         \t   Vy     \t    Vz     \t    Melt rate ' )
    
    mean_vy = np.mean(min_mvy)
    std_vy = np.sqrt( np.var(min_mvy, ddof=1) + np.mean(min_svy**2) )
    
    mean_vz = np.mean(min_mvz)
    std_vz = np.sqrt( np.var(min_mvz, ddof=1) + np.mean(min_svz**2) )
    
    mean_mr = np.mean(min_mmr)
    std_mr = np.sqrt( np.var(min_mmr, ddof=1) + np.mean(min_smr**2) )
    
    vys_mi.append(mean_vy); vzs_mi.append(mean_vz); mrs_mi.append(mean_mr) 
    vys_si.append(std_vy); vzs_si.append(std_vz); mrs_si.append(std_mr) 
    
    # print(f'Mean Min : \t {mean_vy:.3f} mm/s \t {mean_vz:.3f} mm/s \t {mean_mr:.3f} mm/s ' )
    # print(f'Error Min: \t {std_vy:.3f} mm/s \t {std_vz:.3f} mm/s \t {std_mr:.3f} mm/s ' )
    
    mean_vy = np.mean(max_mvy)
    std_vy = np.sqrt( np.var(max_mvy, ddof=1) + np.mean(max_svy**2) )
    
    mean_vz = np.mean(max_mvz)
    std_vz = np.sqrt( np.var(max_mvz, ddof=1) + np.mean(max_svz**2) )
    
    mean_mr = np.mean(max_mmr)
    std_mr = np.sqrt( np.var(max_mmr, ddof=1) + np.mean(max_smr**2) )
    
    vys_ma.append(mean_vy); vzs_ma.append(mean_vz); mrs_ma.append(mean_mr) 
    vys_sa.append(std_vy); vzs_sa.append(std_vz); mrs_sa.append(std_mr) 
    
    # print(f'Mean Max : \t {mean_vy:.3f} mm/s \t {mean_vz:.3f} mm/s \t {mean_mr:.3f} mm/s ' )
    # print(f'Error Max: \t {std_vy:.3f} mm/s \t {std_vz:.3f} mm/s \t {std_mr:.3f} mm/s ' )
    

vys_mi, vzs_mi, mrs_mi = np.array(vys_mi), np.array(vzs_mi), np.array(mrs_mi)
vys_si, vzs_si, mrs_si = np.array(vys_si), np.array(vzs_si), np.array(mrs_si)
vys_ma, vzs_ma, mrs_ma = np.array(vys_ma), np.array(vzs_ma), np.array(mrs_ma)
vys_sa, vzs_sa, mrs_sa = np.array(vys_sa), np.array(vzs_sa), np.array(mrs_sa)
vys_mi[8], vys_ma[8] = np.nan, np.nan 

#%%


fig, ax = plt.subplots(1,1, layout='constrained')

# plt.errorbar(np.arange(9), vys_mi, yerr = vys_si, fmt='.', label='Minima vy' )
# plt.errorbar(np.arange(9), vys_ma, yerr = vys_sa, fmt='.', label='Maxima vy' )
# plt.errorbar(np.arange(9), vzs_mi, yerr = vzs_si, fmt='.', label='Minima vz' )
# plt.errorbar(np.arange(9), vzs_ma, yerr = vzs_sa, fmt='.', label='Maxima vz' )

angles_mi, angles_ma = np.arctan( mrs_mi / vys_mi )*180/np.pi, np.arctan( mrs_ma / vys_ma )*180/np.pi
angles_si = (180 / np.pi) * np.sqrt( vys_mi**2 * mrs_si**2 + mrs_mi**2 * vys_si**2 ) / (vys_mi**2 + mrs_mi**2)
angles_sa = (180 / np.pi) * np.sqrt( vys_ma**2 * mrs_sa**2 + mrs_ma**2 * vys_sa**2 ) / (vys_ma**2 + mrs_ma**2)

ax.errorbar(mres, angles_mi, yerr = angles_si, fmt='.', label='Minima', capsize=3 )
ax.errorbar(mres, angles_ma, yerr = angles_sa, fmt='.', label='Maxima', capsize=3 )

ax.set_xlabel('Re')
ax.set_ylabel(r'$\phi$ (°)')

ax.legend(loc='upper left')
plt.show()


fs = [0,4,5,6,7]
mmi_ag, mma_ag = np.mean(angles_mi[fs]), np.mean(angles_ma[fs])
smi_ag = np.sqrt( np.var(angles_mi[fs], ddof=1) + np.mean(angles_si[fs]**2) )
sma_ag = np.sqrt( np.var(angles_ma[fs], ddof=1) + np.mean(angles_sa[fs]**2) )

print( 'Minima \t Maxima')
print(f'{mmi_ag:.1f} \t {mma_ag:.1f} ')
print(f'{smi_ag:.1f} \t {sma_ag:.1f} ')

#%%
plt.rcParams.update({'font.size':16})
# Visualize local melt rates
fs = [3,1,0,4]
line_val = 70

fig, ax = plt.subplots(1,4, figsize=(12, 4), layout='constrained', sharey=True, sharex=False)

for j,f in enumerate(fs):

    path = '/Volumes/Ice blocks/Scan water channel/'+ all_folders[f] +'/'
    surface = True
    channel = True
    
    rec_t, rec_x, rec_y, rec_z, back_t, back_x, tiempo, Q_tunnel, T_amb, T_top, T_bot = get_data_grid(path, starts[f], ends[f] )
    
    ll = np.where(rec_z[0] == line_val)[0][0]
    print(rec_x.shape)
    print(rec_z[0,ll])
    print()

    if f==0: rec_x[16] = np.nan
    elif f==3: rec_x[68:71] = np.nan
    elif f==4: rec_x, rec_t = rec_x[:-2], rec_t[:-2] # removed mostly for melt rates
    elif f==2: 
        back_t, back_x = back_t[:-1], back_x[:-1]
        rec_x[:,618:1130,50:] = np.nan
        
    rec_f = fill_reconstruction(rec_x, sigma=1)
    rec_back = np.interp(rec_t, back_t, back_x )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
        gt, melt_rate, _ = calculate_melt_rate(f, rec_x, rec_back, sigmas=[2,10,2], time_val=True, integral=True, smooth=True, slope_v='all' )
        
        tempt = temp_t(T_top, T_bot, rec_t, tiempo)
        mean_x, std_x = np.nanmedian(rec_x, axis=(1,2)), np.nanstd(rec_x, axis=(1,2))
        med_Q, std_Q = np.median(Q_tunnel), np.std(Q_tunnel) 
        
        ushearT = med_Q / mean_x * 25/27 #np.median(Q_tunnel) / ( mean_x /1000 * 0.3) / 3600
        ushearT_s = 25/27 * np.sqrt( ( std_Q / mean_x )**2 + ( med_Q * std_x / mean_x**2 )**2 )
        ReT = ushearT * mean_x/1000*lwidth / nu_visco(tempt)
        res = np.nanmean(ReT)

    gt_dev = gt - np.nanmean(gt, axis=1, keepdims=True)
    vmax_gt = np.nanmax(np.abs(gt_dev))
    norm_gt = plt.Normalize(-vmax_gt, vmax_gt)
    
    cmap = 'seismic'

    nt, ny, nx = rec_x.shape   
    if f == 3: nt, ny, nx = rec_x[:85].shape   
    for i in range(0, nt, nt//10):
        y_line = rec_y[:, ll]
        x_line = rec_x[i, :, ll] + rec_back[i]
        gt_line = gt[i, :, ll]
        lc = plot_colored_line( ax[j], y_line, x_line, gt_line, norm=norm_gt, cmap=cmap, linewidth=2 )
        ax[j].autoscale()
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm_gt, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax[j], label=r'$\dot{h} - \langle h \rangle_{x}$')
    
    ax[j].set_xlabel(r'$z$ (mm)')
    
    re_app = np.nanmean(res)
    re_app = round(re_app, -int(np.floor(np.log10(abs(re_app))))+1)
    ax[j].set_title(f'Re = {re_app:.0f}')
    
ax[0].set_ylabel(r'$x$ (mm)')

filename = '/Volumes/Ice blocks/Scan water channel/Figures/local_mr.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()
    






#%%









#%%





#%%





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





