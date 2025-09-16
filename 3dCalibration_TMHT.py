#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:41:15 2025

@author: tomasferreyrahauchar
"""
import imageio.v2 as imageio
# import imageio.v3 as iio
from tqdm import tqdm
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import linregress
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks
from scipy.linalg import pinv
from scipy import ndimage

from skimage.color import rgb2gray
from skimage.filters import gaussian, try_all_threshold, rank, roberts, sato, frangi
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.util import img_as_ubyte
#%%

def get_line(x1,x2,y1,y2, x):
    m = (y2-y1)/(x2-x1)
    y = m * (x-x1) + y1
    return y, m

#%%
# =============================================================================
# Angles for calibration
# =============================================================================
t1 = time()
# im1 = imageio.imread('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0447.NEF')
# im2 = imageio.imread('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0450.NEF')
im1 = imageio.imread('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0433.NEF')
im2 = imageio.imread('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera back/DSC_0434.NEF')

t2 = time()
t2-t1
#%%

xc = np.linspace(3151.1,3176,10)
yc, mc = get_line(3151.5, 3172.5, 100, 4760, xc)

xd = np.linspace(4274,4275.5,10)
yd, md = get_line(4274, 4275.5, 100, 5500, xd)

plt.figure()
plt.imshow( im1[:,:,:] )
plt.plot(xc, yc, 'r-'  )
plt.plot(xd, yd, 'g-'  )
plt.plot(xc+xd[0]-xc[0], yc, 'r-'  )
# plt.gca().invert_yaxis()
plt.show()

print(np.arctan( np.abs((mc-md)/(1+mc*md)) ) * 180/np.pi, '°')
print(np.arctan( np.abs((mc-md)/(1+mc*md)) ), 'rad')
#%%
xc = np.linspace(3499,3549,10)
yc, mc = get_line(3499,3549, 100, 5500, xc)

xd = np.linspace(3466.3,3516.1,10)
yd, md = get_line(3466.3,3516.1, 50.6, 5447.5, xd)

plt.figure()
plt.imshow( im2[:,:,:] )
plt.plot(xd, yd, 'g-'  )
# plt.plot(xc, yc, 'r-'  )
plt.plot(xc+xd[0]-xc[0], yc, 'r-'  )
plt.show()

print(np.arctan( np.abs((mc-md)/(1+mc*md)) ) * 180/np.pi, '°')
print(np.arctan( np.abs((mc-md)/(1+mc*md)) ), 'rad')

#%%
# =============================================================================
# Calibration
# =============================================================================
def order_points( centsx, centsy, max_iter=100, dist_close=4, dist_long=10 ):
    puntos = np.hstack((np.array([centsx]).T, np.array([centsy]).T, np.array([[0]*len(centsy)]).T ))
    mima = [np.argmax(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
    # mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
    or_puntos = np.empty((0,3), int)

    xgr, ygr = [], []
    itere = 0
    while len(puntos) > 0 and itere < max_iter:
        mima = [np.argmax(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
        # mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
        dists = []
        p1, p2 = puntos[mima][0], puntos[mima][1] 
        for i in range(len(puntos)):
            p3 = puntos[i]
            dist = np.linalg.norm( np.cross(p2-p1, p1-p3) ) / np.linalg.norm(p2-p1)
            dists.append(dist)
        dists = np.array(dists)
        fil = dists < dist_close
        orde = np.argsort( puntos[fil][:,1] )[::-1]
        fil2 = (dists > dist_close) * (dists < dist_long) 
        
        or_puntos = np.vstack((or_puntos, puntos[fil][orde])) 
        
        xgr += list( np.arange(len(orde)) ) 
        ygr += list( np.zeros(len(orde))+itere ) 
        
        puntos = puntos[~(fil+fil2)]
        itere += 1

    if itere >= max_iter: print('Maximum iterations reached')
    return or_puntos, np.array(xgr), np.array(ygr)

def calibration_first(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] 
    ytop = v[3]*x + v[4]*y + v[5] 
    bot = v[6]*x + v[7]*y + 1

    X,Y = xtop / bot, ytop / bot
    return X,Y

def calibration_second(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] + v[3]*x**2 +  v[4]*y**2 +  v[5]*x*y
    ytop = v[6]*x + v[7]*y + v[8] + v[9]*x**2 + v[10]*y**2 + v[11]*x*y
    bot = v[12]*x + v[13]*y + 1 + v[14]*x**2 + v[15]*y**2 + v[16]*x*y

    X,Y = xtop / bot, ytop / bot
    return X,Y

def rotate_grid(theta, c1, c2):
    """
    Parameters
    ----------
    theta : float, angle in radians.
    c1 : array, 1st component to rotate.
    c2 : array, 2nd component to rotate

    Returns
    -------
    rotated arrays
    """
    c1r = np.cos(theta) * c1 + np.sin(theta) * c2
    c2r = -np.sin(theta) * c1 + np.cos(theta) * c2
    return c1r, c2r
    
#%%
t1 = time()

vid1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0456.MOV', 'ffmpeg') # 7548 frames
vid2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0457.MOV', 'ffmpeg') # 7464 frames
# vid3 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0458.MOV', 'ffmpeg') # 384 frames

ny,nx,_ = np.shape(vid1.get_data(0))

t2 = time()
print(t2-t1)
# vid1.count_frames(), vid2.count_frames(), vid3.count_frames()
#%%
xmed, ymed, ies = [],[], []

for i in tqdm(range(0,7548,20)):
    im = np.array(vid1.get_data(i)[500:1300,:])
    img = im[:,:,0] * 0.3 + im[:,:,1] * 0.7    #rgb2gray(im)
    bni = np.mean( np.where( img > 170 ), axis=1)
    xmed.append( bni[1] )
    ymed.append( bni[0] + 500 )
    ies.append(i)
    
for i in tqdm(range(0,7464,20)):
    im = np.array(vid2.get_data(i)[900:1500,:])
    img = im[:,:,0] * 0.3 + im[:,:,1] * 0.7 #rgb2gray(im)
    bni = np.mean( np.where( img > 195 ), axis=1)
    xmed.append( bni[1] )
    ymed.append( bni[0] + 900 )
    ies.append(i + 7548)
    
#%%
fram1 = [ 405, 1325, 3120, 3960, 5240, 6095, 7380] 
fram2 = [ 8235, 9495, 10685, 11895, 13290, 14060, 15000 ]

ims = []
for i in tqdm(fram1):
    imm = np.zeros((ny,nx))
    for j in range(i-20,i):
        im = np.array(vid1.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7)
    ims.append(imm / 20)
for i in tqdm(fram2):
    imm = np.zeros((ny,nx))
    for j in range(i-7548-20,i-7548):
        im = np.array(vid2.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7) 
    ims.append(imm / 20)
    
#%%
t1 = time()
masks = []
g1s = []
for n in tqdm(range(14)):
    
    g1 = gaussian(ims[n],1)
    g1s.append(g1)

    mask = np.zeros_like(ims[n])
    if n in [0,1,2,3,4,5,6,7,9]:
        imr = roberts(ims[n][450:1650])
        img = gaussian(imr, (20,20))
        imf = gaussian(imr, (1,1))
        
        rrr = remove_small_objects(imf/img > 2.6, min_size=5000, connectivity=1)
        lrl = binary_closing(label(~rrr)==2, disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)
    
    elif n in [8,10]:
        imr = roberts(ims[n][450:1650])
        img = gaussian(imr, (10,10))
        imf = gaussian(imr, (1,1))
        
        pol = remove_small_objects( (imf/ img > 1.9), min_size=1000 )
        rrr = label(~pol)
        lrl = binary_closing( rrr==2, disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)

    elif n == 11:
        imr = roberts(ims[n][450:1650])
        img = gaussian(imr, (10,10))
        imf = gaussian(imr, (1,1))
        
        pol = remove_small_objects( (imf/ img > 1.9), min_size=1000 )
        rrr = label(~pol)
        lrl = binary_closing( (rrr==2)+(rrr==4), disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)

    elif n == 12:
        imr = roberts(ims[n][450:1650])
        img = gaussian(imr, (10,10))
        imf = gaussian(imr, (1,1))
        
        pol = remove_small_objects( (imf/ img > 1.9), min_size=1000 )
        rrr = label(~pol)
        lrl = binary_closing( (rrr==2)+(rrr==3), disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)

    elif n == 13:
        imr = roberts(ims[n][450:1650])
        img = gaussian(imr, (10,10))
        imf = gaussian(imr, (1,1))
        
        pol = remove_small_objects( (imf/ img > 1.7), min_size=1000 )
        rrr = label(~pol)
        lrl = binary_closing( (rrr>1) * (rrr<15), disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)

masks = np.array(masks).astype(int)

t2 = time()
print(t2-t1)
#%%

cosos = []

spxs,spys = [],[]
xres, yres, zres = [],[],[]
xrrs, yrrs, zrrs = [],[],[] 
for i in tqdm(range(14)):

    # ig = (g1s[i]*masks[i]).astype(int) 
    ig = (g1s[i]*masks[i]).astype('uint8') 
    c2 = rank.autolevel( ig , disk(20), mask=masks[i]) * masks[i]
    coso = ( gaussian(c2,0.1) - gaussian(c2,5)  ) * masks[i]
    coso = (coso - np.min(coso)) / (np.max(coso)-np.min(coso)) 
    coso[~masks[i]] = 1
        
    if i < 10: bob = blob_log( (1-coso) , threshold=0.03, min_sigma=1, max_sigma=1.2, num_sigma=3  )
    else: bob = blob_log( (1-coso) , threshold=0.04, min_sigma=1, max_sigma=1.2, num_sigma=3, overlap=0.0  ) 
    pont = np.zeros_like(coso)
    edgem = masks[i]*1. - binary_erosion(masks[i], disk(1))*1.
    edy,edx = np.where( edgem )
    
    for blob in bob:
        y, x, r = blob
        dis = np.min( (edx - x)**2 + (edy - y)**2 )
        # if dis > 10:
        if dis > 6 and y < 1700:
            pont[round(y),round(x)] = 1
    pons = gaussian(pont,(1.,2.5))
    pons = pons / np.max(pons) > 0.1
    
    pol = label(pons)
    props = regionprops(pol, (1-coso) )
    cy = np.array( [ (props[i].centroid_weighted)[0] for i in range(len(props)) ] )
    cx = np.array( [ (props[i].centroid_weighted)[1] for i in range(len(props)) ] )
        
    orpu, xgr, ygr = order_points(cx, cy, max_iter=107, dist_close=4, dist_long=20)
    spx,spy = orpu[:,0], orpu[:,1]

    xre = xgr * 3 # mm
    yre = (ygr) * 6 # mm
    zre = np.ones_like(xre) * i*10 # mm
    
    xrr, yrr = rotate_grid(0.006869142801436136, xre, yre)
    zrr, yrr = rotate_grid(0.001585531211732638, zre, yrr)
    # xrr, zrr = rotate_grid(0, xrr, zrr)
    # xrr, zrr = xrr-39, zrr+np.median(zre)

    spxs += list( spx )
    spys += list( spy )
    xres += list( xre )
    yres += list( yre )
    zres += list( zre )
    xrrs += list( xrr )
    yrrs += list( yrr )
    zrrs += list( zrr )

spxs,spys = np.array(spxs), np.array(spys)
xres, yres, zres = np.array(xres), np.array(yres), np.array(zres)
xrrs, yrrs, zrrs = np.array(xrrs), np.array(yrrs), np.array(zrrs)
#%%
fin = 10
    
plt.figure()
plt.imshow( g1s[-1] )
plt.plot( spx, spy, '.' )
plt.plot( spx[:fin], spy[:fin], '.' )
# plt.axis('equal')
# plt.gca().invert_yaxis()
plt.show()


plt.figure()
plt.plot(xrr, yrr, '.')
plt.plot(xrr[:fin], yrr[:fin], '.')
# plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid()
plt.show()


# plt.figure()
# plt.plot( zre, yre, '.' )
# plt.plot( zrr, yrr, '.' )
# plt.plot( zre[:fin], yre[:fin], '.' )

# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# # plt.axis('equal')
# plt.show()

#%%
def cal_direct(v,x,y,z, order):
    if order == 1:
        xtop = v[0]*x + v[1]*y + v[2]*z  + v[3]
        ytop = v[4]*x + v[5]*y + v[6]*z  + v[7] 
        bot  = 1 
        X, Y = xtop/bot, ytop/bot 
        return X, Y
    if order == 2:
        xtop = v[0]*x  + v[1]*y  + v[2]*z  + v[3]  + v[4]*x**2  + v[5]*y**2  + v[6]*x*y
        ytop = v[7]*x  + v[8]*y  + v[9]*z  + v[10] + v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot  = 1 
        X, Y = xtop/bot, ytop/bot
        return X, Y
    if order == '2b':
        xtop = v[0]*x  + v[1]*y  + v[2]*z  + v[3]  + v[4]*x**2  + v[5]*y**2  + v[6]*x*y
        ytop = v[7]*x  + v[8]*y  + v[9]*z  + v[10] + v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot  = v[14]*x + v[15]*y + v[16]*z + 1     + v[17]*x**2 + v[18]*y**2 + v[19]*x*y
        X, Y = xtop/bot, ytop/bot 
        return X, Y


def calib_invzl(v,x,y,z, order, x0s=[[50,300]]):
    
    xr, yr = np.zeros_like(x), np.zeros_like(y)

    if order == 1:
        def func(val, x,y,z):
            xtop = v[0]*val[0] + v[1]*val[1] + v[2]*z  + v[3]
            ytop = v[4]*val[0] + v[5]*val[1] + v[6]*z  + v[7] 
            bot  = 1 # v[8]*val[0] + v[9]*val[1] + v[10]*z + 1  
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == 2:
        def func(val, x,y,z):
            xtop = v[0]*val[0]  + v[1]*val[1]  + v[2]*z  + v[3]  + v[4]*val[0]**2  + v[5]*val[1]**2  + v[6]*val[0]*val[1]
            ytop = v[7]*val[0]  + v[8]*val[1]  + v[9]*z  + v[10] + v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            bot  = 1 # v[14]*val[0] + v[15]*val[1] + v[16]*z + 1     + v[17]*val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == '2b':
        def func(val, x,y,z):
            xtop = v[0]*val[0]  + v[1]*val[1]  + v[2]*z  + v[3]  + v[4]*val[0]**2  + v[5]*val[1]**2  + v[6]*val[0]*val[1]
            ytop = v[7]*val[0]  + v[8]*val[1]  + v[9]*z  + v[10] + v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            bot  = v[14]*val[0] + v[15]*val[1] + v[16]*z + 1     + v[17]*val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    
    for i in tqdm(range(len(x))):
        xp, yp, zp = x[i], y[i], z[i]
        x0 = x0s[i%len(x0s)]
        funci = lambda sol: func(sol, xp, yp, zp)
        mini = minimize(funci, x0, method='Nelder-Mead')
        xr[i], yr[i] = mini.x[0], mini.x[1]

    return xr, yr

def residual_firstz(v):
    xtop = v[0]*xrrs + v[1]*yrrs + v[2]*zrrs  + v[3]
    ytop = v[4]*xrrs + v[5]*yrrs + v[6]*zrrs  + v[7]
    bot = 1 # v[8]*xrrs + v[9]*yrrs + v[10]*zrrs + 1

    X,Y = xtop / bot, ytop / bot
    return (X - spxs)**2 + (Y - spys)**2

def residual_secondz(v):
    xtop = v[0]*xrrs  + v[1]*yrrs  + v[2]*zrrs  + v[3]  + v[4]*xrrs**2  + v[5]*yrrs**2  + v[6]*xrrs*yrrs  
    ytop = v[7]*xrrs  + v[8]*yrrs  + v[9]*zrrs  + v[10] + v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs 
    bot = 1 # v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1     + v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs

    X,Y = xtop / bot, ytop / bot
    return (X - spxs)**2 + (Y - spys)**2

def residual_secondzb(v):
    xtop = v[0]*xrrs  + v[1]*yrrs  + v[2]*zrrs  + v[3]  + v[4]*xrrs**2  + v[5]*yrrs**2  + v[6]*xrrs*yrrs
    ytop = v[7]*xrrs  + v[8]*yrrs  + v[9]*zrrs  + v[10] + v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs
    bot =  v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1     + v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs

    X,Y = xtop / bot, ytop / bot
    return (X - spxs)**2 + (Y - spys)**2


lal = least_squares(residual_firstz, [0,1,1000,1,0,1000,0,0] , method='lm', xtol=1e-12, max_nfev=10000)

ls0 = np.zeros(14)
ls0[:4],ls0[7:11] = lal.x[:4], lal.x[4:8]
la2 = least_squares(residual_secondz, ls0 , method='lm', xtol=1e-12, max_nfev=10000)

ls0 = np.zeros(20)
ls0[:14] = la2.x[:14]
lab = least_squares(residual_secondzb, ls0 , method='lm', xtol=1e-12, max_nfev=10000)

# Xc, Yc = calib_invzl(lal.x, spxs, spys, zrrs, order=1)
# X2, Y2 = calib_invzl(la2.x, spxs, spys, zrrs, order=2)
Xb, Yb = calib_invzl(lab.x, spxs, spys, zrrs, order='2b')
#%%
cutoffs = [0, 2781, 5562, 8343, 11124, 13905, 16686, 19467, 22248, 25029, 27810, 30590, 33371, 36152, 38934 ]

# X, Y = cal_direct(lal.x, xrrs, yrrs, zrrs, 1)
# X, Y = cal_direct(la2.x, xrrs, yrrs, zrrs, 2)
X, Y = cal_direct(lab.x, xrrs, yrrs, zrrs, '2b')

# i = 7
for i in range(14):
    ini, fin = cutoffs[i], cutoffs[i+1]
    
    plt.figure()
    plt.plot( spxs[ini:fin], spys[ini:fin], '.' )
    plt.plot( X[ini:fin], Y[ini:fin], '.' )
    plt.gca().invert_yaxis()
    plt.show()

# i = 11
for i in range(14):
    ini, fin = cutoffs[i], cutoffs[i+1]
    
    plt.figure()
    plt.plot( xrrs[ini:fin], yrrs[ini:fin], '.' )
    # plt.plot( Xc[ini:fin], Yc[ini:fin], '.' )
    # plt.plot( X2[ini:fin], Y2[ini:fin], '.' )
    plt.plot( Xb[ini:fin], Yb[ini:fin], '.' )
    plt.show()


#%%
# =============================================================================
# Transformation from 2nd calib video to 1st
# =============================================================================

vidc1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0468.MOV', 'ffmpeg') # 7272 frames
vidc2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0469.MOV', 'ffmpeg') # 7272 frames

vidu1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0456.MOV', 'ffmpeg') # 7548 frames
vidu2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/Camera 1/DSC_0457.MOV', 'ffmpeg') # 7548 frames

#%%

fram1 = [ 405 ] 
fram2 = [15000]

imsu = []
for i in tqdm(fram1):
    imm = np.zeros((ny,nx))
    for j in range(i-20,i):
        im = np.array(vidu1.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7)
    imsu.append(imm / 20)
for i in tqdm(fram2):
    imm = np.zeros((ny,nx))
    for j in range(i-7548-20,i-7548):
        im = np.array(vid2.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7) 
    imsu.append(imm / 20)

masks = []
g1su = []
for n in tqdm(range(2)):
    
    g1 = gaussian(imsu[n],1)
    g1su.append(g1)
    mask = np.zeros_like(imsu[n])

    if n == 0:
        imr = roberts(imsu[n][450:1650])
        img = gaussian(imr, (20,20))
        imf = gaussian(imr, (1,1))
        
        rrr = remove_small_objects(imf/img > 2.6, min_size=5000, connectivity=1)
        lrl = binary_closing(label(~rrr)==2, disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)
        
    elif n == 1:
        imr = roberts(imsu[n][450:1650])
        img = gaussian(imr, (10,10))
        imf = gaussian(imr, (1,1))
        
        pol = remove_small_objects( (imf/ img > 1.7), min_size=1000 )
        rrr = label(~pol)
        lrl = binary_closing( (rrr>1) * (rrr<15), disk(5))
        rlr = binary_dilation(lrl, disk(3))
        mask[450:1650] = rlr
        masks.append(mask)

spxsu, spysu = [],[]
xresu, yresu = [],[]
for i in tqdm(range(2)):

    # ig = (g1s[i]*masks[i]).astype(int) 
    ig = (g1su[i]*masks[i]).astype('uint8') 
    c2 = rank.autolevel( ig , disk(20), mask=masks[i]) * masks[i]
    coso = ( gaussian(c2,0.1) - gaussian(c2,5)  ) * masks[i]
    coso = (coso - np.min(coso)) / (np.max(coso)-np.min(coso)) 
    coso[~(masks[i]>0)] = 1
        
    if i == 0: bob = blob_log( (1-coso) , threshold=0.03, min_sigma=1, max_sigma=1.2, num_sigma=3  )
    else: bob = blob_log( (1-coso) , threshold=0.04, min_sigma=1, max_sigma=1.2, num_sigma=3, overlap=0.0  ) 
    pont = np.zeros_like(coso)
    edgem = masks[i]*1. - binary_erosion(masks[i], disk(1))*1.
    edy,edx = np.where( edgem )
    
    for blob in bob:
        y, x, r = blob
        dis = np.min( (edx - x)**2 + (edy - y)**2 )
        # if dis > 10:
        if dis > 6 and y < 1700:
            pont[round(y),round(x)] = 1
    pons = gaussian(pont,(1.,2.5))
    pons = pons / np.max(pons) > 0.1
    
    pol = label(pons)
    props = regionprops(pol, (1-coso) )
    cy = np.array( [ (props[i].centroid_weighted)[0] for i in range(len(props)) ] )
    cx = np.array( [ (props[i].centroid_weighted)[1] for i in range(len(props)) ] )
        
    orpu, xgr, ygr = order_points(cx, cy, max_iter=107, dist_close=4, dist_long=20)
    spxu,spyu = orpu[:,0], orpu[:,1]
    
    xreu = xgr * 3 # mm
    yreu = (ygr) * 6 # mm
    
    spxsu += list( spxu )
    spysu += list( spyu )
    xresu += list( xreu )
    yresu += list( yreu )



fram1 = [ 390 ]
fram2 = [11865]

imsc = []
for i in tqdm(fram1):
    imm = np.zeros((ny,nx))
    for j in range(i-20,i):
        im = np.array(vidc1.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7) / 20
    imsc.append(imm)
for i in tqdm(fram2):
    imm = np.zeros((ny,nx))
    for j in range(i-7272-20,i-7272):
        im = np.array(vidc2.get_data(j))
        imm += (im[:,:,0] * 0.3 + im[:,:,1] * 0.7) / 20
    imsc.append(imm)
    
masks = []
g1sc = []
for i in tqdm(range(2)):
    
    g1 = gaussian(imsc[i],1)
    g1sc.append(g1)
    if i == 0:
        fals = felzenszwalb(imsc[i], scale=1e-11, sigma=0.1, min_size=300000)
        masks.append( fals == 3 )
    elif i==1:
        fals = felzenszwalb(imsc[i], scale=1e-6, sigma=0.5, min_size=5000)
        masks.append( (fals > 210) * (fals < 250) * (fals != 211) * (fals != 213) * (fals != 216) * (fals != 222) )

spxsc, spysc = [],[]
xresc, yresc = [],[]
for i in tqdm(range(2)):

    # ig = (g1s[i]*masks[i]).astype(int) 
    ig = (g1sc[i]*masks[i]).astype('uint8') 
    c2 = rank.autolevel( img_as_ubyte(ig), disk(20), mask=masks[i]) * masks[i]
    coso = ( gaussian(c2,0.1) - gaussian(c2,5)  ) * masks[i]
    coso = (coso - np.min(coso)) / (np.max(coso)-np.min(coso)) 
    coso[~(masks[i]>0)] = 1
    
    if i ==0: bob = blob_log( (1-coso) , threshold=0.05, min_sigma=1, max_sigma=1.2, num_sigma=3  )
    else: bob = blob_log( (1-coso) , threshold=0.034, min_sigma=1, max_sigma=1.2, num_sigma=3, overlap=0.0  ) 
    pont = np.zeros_like(coso)
    edgem = masks[i]*1. - binary_erosion(masks[i], disk(1))*1.
    edy,edx = np.where( edgem )
    
    for blob in bob:
        y, x, r = blob
        dis = np.min( (edx - x)**2 + (edy - y)**2 )
        # if dis > 10:
        if dis > 6:
            pont[round(y),round(x)] = 1
    pons = gaussian(pont,(1.,2.5))
    pons = pons / np.max(pons) > 0.1
    
    pol = label(pons)
    props = regionprops(pol, (1-coso) )
    cy = np.array( [ (props[i].centroid_weighted)[0] for i in range(len(props)) ] )
    cx = np.array( [ (props[i].centroid_weighted)[1] for i in range(len(props)) ] )
    
    if i ==0: orpu, xgr, ygr = order_points(cx, cy, max_iter=107, dist_close=4, dist_long=10)
    else: orpu, xgr, ygr = order_points(cx, cy, max_iter=107, dist_close=4, dist_long=30) 
    spxc,spyc = orpu[:,0], orpu[:,1]
    
    xrec = xgr * 3 # mm
    yrec = (ygr) * 6 # mm
    
    spxsc += list( spxc )
    spysc += list( spyc )
    xresc += list( xrec )
    yresc += list( yrec )

xresc, yresc = np.array(xresc), np.array(yresc) 
xresu, yresu = np.array(xresu), np.array(yresu) 
xresu = np.delete(xresu, 3456)
yresu = np.delete(yresu, 3456)

spxsc, spysc = np.array(spxsc), np.array(spysc) 
spxsu, spysu = np.array(spxsu), np.array(spysu) 
spxsu = np.delete(spxsu, 3456)
spysu = np.delete(spysu, 3456)

#%%
def ellipse(r1,r2,c1,c2):
    the = np.linspace(0,2*np.pi,1000)
    r = (r1*r2) / np.sqrt( (r1*np.cos(the))**2 + (r2*np.sin(the))**2 )
    return r*np.sin(the)+c1, r*np.cos(the)+c2

x1s = [345,906,1465,1744,2022,2575,3125,3670] #correct (imc)
y1s = [57,77,100,110,120,139,159,181]

x2s = [289,861,1429,1712,1993,2555,3116,3677] #incorrect (imu)
y2s = [55,70,86,93,100,114,129,143]

ex1 = np.array( [ellipse(55, 20, x1s[i], y1s[i]+180)[0] for i in range(8)] ).flatten()
ey1 = np.array( [ellipse(55, 20, x1s[i], y1s[i]+180)[1] for i in range(8)] ).flatten()
ex2 = np.array( [ellipse(57, 21, x2s[i], y2s[i]+180)[0] for i in range(8)] ).flatten()
ey2 = np.array( [ellipse(57, 21, x2s[i], y2s[i]+180)[1] for i in range(8)] ).flatten()


#%%

ax1, ay1 = spxsc, spysc
ax2, ay2 = spxsu, spysu

def recal(v,x,y, order=1):
    if order == 1:
        xt = v[0] * x + v[3] * y + v[6]
        yt = v[1] * x + v[4] * y + v[7]
        bt = v[2] * x + v[5] * y + 1
    if order == 2:
        xt = v[0] * x + v[3] * y + v[6] + v[8] * x*y + v[11] * x**2 + v[14] * y**2
        yt = v[1] * x + v[4] * y + v[7] + v[9] * x*y + v[12] * x**2 + v[15] * y**2
        bt = v[2] * x + v[5] * y +  1   + v[10]* x*y + v[13] * x**2 + v[16] * y**2
    X,Y = xt/bt, yt/bt
    return X,Y

def func1(v):
    xt = v[0] * ax1 + v[3] * ay1 + v[6]
    yt = v[1] * ax1 + v[4] * ay1 + v[7]
    bt = v[2] * ax1 + v[5] * ay1 + 1
    X,Y = xt/bt, yt/bt
    return (X-ax2)**2+(Y-ay2)**2
def func2(v):
    xt = v[0] * ax1 + v[3] * ay1 + v[6] + v[8] * ax1*ay1 + v[11] * ax1**2 + v[14] * ay1**2
    yt = v[1] * ax1 + v[4] * ay1 + v[7] + v[9] * ax1*ay1 + v[12] * ax1**2 + v[15] * ay1**2
    bt = v[2] * ax1 + v[5] * ay1 +  1   + v[10]* ax1*ay1 + v[13] * ax1**2 + v[16] * ay1**2
    X,Y = xt/bt, yt/bt
    return (X-ax2)**2+(Y-ay2)**2

al1 = least_squares(func1, [1.018,0,0,0,0.975,0,0,0], method='lm')

ls0 = np.zeros(17)
ls0[:8] = al1.x * 0.9
al2 = least_squares(func2, ls0) # method='lm' )

# X,Y = recal(al1.x, ax1,ay1, order=1)
X,Y = recal(al2.x, ax1,ay1, order=2)

plt.figure()
# plt.plot(ax1,ay1, 'b.')

plt.plot(X,Y, 'g.')

plt.plot(ax2,ay2, 'r.')

plt.gca().invert_yaxis()
# plt.axis('equal')
plt.show()

#%%
xx1, yy1 = np.concatenate( (ex1, spxsc) ), np.concatenate( (ey1, spysc) )
xx2, yy2 = np.concatenate( (ex2, spxsu) ), np.concatenate( (ey2, spysu) )

X,Y = recal(al2.x, xx1, yy1, order=2)

plt.figure()

# plt.plot( xx1, yy1, '.')
plt.plot(X,Y, 'g.')
plt.plot( xx2, yy2, 'r.')

plt.gca().invert_yaxis()
# plt.axis('equal')
plt.show()


#%%
# =============================================================================
# 
# =============================================================================

#%%

print( lab.x )

print( al2.x ) 
#%%








#%%
# =============================================================================
# Calibration with two cameras
# =============================================================================

import imageio.v2 as imageio
# import imageio.v3 as iio
from tqdm import tqdm
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import linregress
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks
from scipy.linalg import pinv
from scipy import ndimage

from skimage.color import rgb2gray
from skimage.filters import gaussian, try_all_threshold, rank, roberts, sato, frangi
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_closing, binary_dilation, binary_opening
from skimage.measure import label, regionprops
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.util import img_as_ubyte

from itertools import combinations

def position_grid( vid, back, cutoff, n_frames, interval=10, in_frame=0, lims=[None,None,None,None] ):
    xmed,ymed,ies = [],[],[]
    for i in tqdm(range(in_frame, n_frames, interval)):
        im = np.array(vid.get_data(i))
        im = 0.3 * im[lims[0]:lims[1],lims[2]:lims[3],0] + 0.7 * im[lims[0]:lims[1],lims[2]:lims[3],1]
        imb = im - back[lims[0]:lims[1],lims[2]:lims[3]]
        bni = np.mean( np.where( imb < cutoff ), axis=1)
        xmed.append(bni[1])
        ymed.append(bni[0])
        ies.append(i)
        
    return np.array(xmed),np.array(ymed),np.array(ies)

def get_frames(xms, ies, h1, h2):
    peksp,_ = find_peaks( np.gradient(-xms), distance=30, height=h1)
    peksm,_ = find_peaks( np.gradient(xms), distance=30, height=h2)

    peksp, peksm = peksp[:13] - 3, peksm[13]-3 
    peks = np.hstack( (peksp,peksm) )
    return ies[peks], peks

def image_position(frames, vids, lengths):
    ny,nx,_ = np.shape(vids[0].get_data(0))
    ll = np.cumsum( np.hstack((0,lengths)) )
    nv = np.array( [ np.sum(ll <= frames[i])-1 for i in range(len(frames)) ] )
    truframes = frames - ll[nv]
    
    ims = []
    for i in tqdm(range(len(frames))):
        ima = np.zeros((ny,nx))
        for j in range(-59,1):
            im = np.array( vids[ nv[i] * ((truframes[i] + j)>-1) ].get_data( truframes[i] + j + lengths[0] * ((truframes[i] + j)<0)) )
            imn = 0.3 * im[:,:,0] + 0.7 * im[:,:,1]
            ima += imn
        ims.append(ima/60) 
    return ims

def get_points(imb, tol_f=0.01, tol_ga=0.3):    
    g1 = normalize( gaussian(imb, 1) )
    gl = normalize( gaussian(imb, 10))
    imt = frangi(g1) > tol_f
    
    rim = remove_small_objects( imt , 45)
    pos = imt*1-rim
    ker = np.array([[0,1,0],[1,1,1],[0,1,0]])
    op = binary_opening(pos, ker)
    dots = remove_small_objects(op,10)
    dots[gl < tol_ga] = 0
    
    lab = label(dots)
    props = regionprops(lab, -g1 )
    cy = np.array( [ (props[i].centroid_weighted)[0] for i in range(len(props)) ] )
    cx = np.array( [ (props[i].centroid_weighted)[1] for i in range(len(props)) ] )
    
    tt = np.arange(len(cx)).astype(str)
    pp = np.vstack((cx,cy)).T
    return g1, pp, tt

def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def iscollinear(points):
    """
    Parameters
    ----------
    points : (N,M,2) array
        N sets of M points to check (in 2D).

    Returns
    -------
    array of booleans.
    """
    check = np.zeros(len(points),dtype=bool)
    for i in range(len(points)):
        k = points[i]
        vecdiv = (k-k[0]) * (k[1]-k[0])[::-1] 
        ks = np.diff(vecdiv,axis=1)[:,0]

        check[i] = np.all( ks==0 ) 
    
    return check

def find_basis(out, position, gridpos, eps, lim_ind):
    order = np.argsort( np.sum((out - position)**2,axis=1) )
    orout, orgridi = out[order], gridpos[order]
    
    for i in range(3, len(orout)+1):
        subs = list(combinations(np.arange(i), 3))
        subs_pts, subs_grid = orout[subs], orgridi[subs]
        is_col = iscollinear(subs_grid)
        subs_pts, subs_grid = subs_pts[~is_col], subs_grid[~is_col]
        
        if len(subs_pts)>0:
            ibase = np.argmin( np.sum((subs_pts - position)**2,axis=(1,2)) )
            basep, baseg = subs_pts[ibase], subs_grid[ibase]
            break

    if len(subs_pts)>0:
        vp1, vp2 = basep[1] - basep[0], basep[2] - basep[0]
        vg1, vg2 = baseg[1] - baseg[0], baseg[2] - baseg[0]
        vpoi = position - basep[0]
        
        a = np.linalg.det( np.vstack((vp1,vp2)) )
        b = np.linalg.det( np.vstack((vp1,vpoi)) )
        c = np.linalg.det( np.vstack((vpoi,vp2)) )

        pti = c/a * vg1 + b/a * vg2 + baseg[0]
        newind = np.round(pti) 
        err =  np.sqrt( np.sum( (newind - pti)**2 ) )

        belongs_grid = (err < eps) and (newind[0]>=lim_ind[0]) and (newind[0]<=lim_ind[1]) and (newind[1]>=lim_ind[2]) and (newind[1]<=lim_ind[3])
    else:
        newind = np.zeros(2)*np.nan
        belongs_grid = False
    
    return newind, belongs_grid

def find_indices(points, known, eps=0.25, lim_ind=[-1e5,1e5,-1e5,1e5]):
    """
    Parameters
    ----------
    points : (N,2)-array
        Points to match in  a grid.
    known : list of 3 integers 
        Indeces in points for (0,0), (1,0) and (0,1).
    eps : TYPE, optional
        DESCRIPTION. The default is 0.25.
    lim_ind : list length 4, optional
        min and max indices for [left,right,bottom,top]
    Returns
    -------
    (M,2)-array
        position of points in grid.
    (M,2)-array
        grid position of points.
    """
    N,col = np.shape(points)
    if col != 2: 
        return 'Points need to be 2D'
    
    pts = np.copy(points)
    out, gridpos = np.zeros((3,2)), np.zeros((3,2))
    out[0],out[1],out[2] = pts[known[0]], pts[known[1]], pts[known[2]]
    gridpos[1,0], gridpos[2,1] = 1, 1    
    
    pts = np.delete(pts, known, axis=0)
    
    itere = 0
    while len(pts) > 0 and itere <= N:
        itere += 1
        med = np.mean(out,axis=0)
        iclosest = np.argmin( np.sum((pts - med)**2,axis=1) )
        position = pts[iclosest]
        pts = np.delete(pts, iclosest, axis=0)
        
        newind, belongs_grid = find_basis(out, position, gridpos, eps, lim_ind)
        if belongs_grid:
            out = np.vstack((out,position))
            gridpos = np.vstack((gridpos,newind))
    
    return out, gridpos.astype('int64')

def cal_direct(v, points, order):
    """
    Parameters
    ----------
    v : list
        list of coefficients from fit.
    points : (N,3) array
        (x,y,z) real world coordinates.
    order : {'1','2','2b'}
        which function to use.

    Returns
    -------
    X : array
        pixel x-position of points.
    Y : array
        pixel y-position of points.
    """
    x,y,z = points.T
    if order == '1':
        xtop = v[0]*x + v[1]*y + v[2]*z  + v[3]
        ytop = v[4]*x + v[5]*y + v[6]*z  + v[7] 
        bot  = 1 
        X, Y = xtop/bot, ytop/bot 
        return X, Y
    if order == '2':
        xtop = v[0]*x  + v[1]*y  + v[2]*z  + v[3]  + v[4]*x**2  + v[5]*y**2  + v[6]*x*y
        ytop = v[7]*x  + v[8]*y  + v[9]*z  + v[10] + v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot  = 1 
        X, Y = xtop/bot, ytop/bot
        return X, Y
    if order == '2b':
        xtop = v[0]*x  + v[1]*y  + v[2]*z  + v[3]  + v[4]*x**2  + v[5]*y**2  + v[6]*x*y
        ytop = v[7]*x  + v[8]*y  + v[9]*z  + v[10] + v[11]*x**2 + v[12]*y**2 + v[13]*x*y
        bot  = v[14]*x + v[15]*y + v[16]*z + 1     + v[17]*x**2 + v[18]*y**2 + v[19]*x*y
        X, Y = xtop/bot, ytop/bot 
        return X, Y

def calib_invzl(v, points, order, x0s=[[50,300]]):
    """
    Parameters
    ----------
    v : list
        list of coefficients from fit.
    points : (N,3) array
        (x,y,z) coordinates. (x,y) should be pixel position of point and (z) the corresponding real world position of that point
    order : {'1','2','2b'}
        which function to use.
    x0s : (M,2) array, optional
        Initial guess for minimize. M=1 or M=N. If N=1 then it would be use as initial values for the points. M=N is for provinding initial values for all points
        . The default is [[50,300]].

    Returns
    -------
    X : array
        real world x-position of points.
    Y : array
        real world y-position of points.
    """
    x,y,z = points.T  
    xr, yr = np.zeros_like(x), np.zeros_like(y)

    if order == '1':
        def func(val, x,y,z):
            xtop = v[0]*val[0] + v[1]*val[1] + v[2]*z  + v[3]
            ytop = v[4]*val[0] + v[5]*val[1] + v[6]*z  + v[7] 
            bot  = 1 # v[8]*val[0] + v[9]*val[1] + v[10]*z + 1  
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == '2':
        def func(val, x,y,z):
            xtop = v[0]*val[0]  + v[1]*val[1]  + v[2]*z  + v[3]  + v[4]*val[0]**2  + v[5]*val[1]**2  + v[6]*val[0]*val[1]
            ytop = v[7]*val[0]  + v[8]*val[1]  + v[9]*z  + v[10] + v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            bot  = 1 # v[14]*val[0] + v[15]*val[1] + v[16]*z + 1     + v[17]*val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    if order == '2b':
        def func(val, x,y,z):
            xtop = v[0]*val[0]  + v[1]*val[1]  + v[2]*z  + v[3]  + v[4]*val[0]**2  + v[5]*val[1]**2  + v[6]*val[0]*val[1]
            ytop = v[7]*val[0]  + v[8]*val[1]  + v[9]*z  + v[10] + v[11]*val[0]**2 + v[12]*val[1]**2 + v[13]*val[0]*val[1]
            bot  = v[14]*val[0] + v[15]*val[1] + v[16]*z + 1     + v[17]*val[0]**2 + v[18]*val[1]**2 + v[19]*val[0]*val[1]
            eq1, eq2 = xtop/bot - x, ytop/bot - y
            return eq1**2+eq2**2
    
    for i in tqdm(range(len(x))):
        xp, yp, zp = x[i], y[i], z[i]
        x0 = x0s[i%len(x0s)]
        funci = lambda sol: func(sol, xp, yp, zp)
        mini = minimize(funci, x0, method='Nelder-Mead')
        xr[i], yr[i] = mini.x[0], mini.x[1]

    return xr, yr

def calibration_fit(real_pos, pix_pos, cal_check=False):
    """
    Parameters
    ----------
    real_pos : (N,3) array
        point positions in real world.
    pix_pos : (N,2) array
        pixel position in image.
    cal_check: bool
        if True, also returns (x,y) real world coordinates of image points, by inverting the calibration.
        Usefu; for checking if calibration worked.
    Returns
    -------
    dict
        results from least squares.
    """
    t1 = time()
    def residual_firstz(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs + v[1]*yrrs + v[2]*zrrs  + v[3]
        ytop = v[4]*xrrs + v[5]*yrrs + v[6]*zrrs  + v[7]
        bot = 1 # v[8]*xrrs + v[9]*yrrs + v[10]*zrrs + 1
    
        X,Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2
    
    def residual_secondz(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs  + v[1]*yrrs  + v[2]*zrrs  + v[3]  + v[4]*xrrs**2  + v[5]*yrrs**2  + v[6]*xrrs*yrrs  
        ytop = v[7]*xrrs  + v[8]*yrrs  + v[9]*zrrs  + v[10] + v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs 
        bot = 1 # v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1     + v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs
    
        X,Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2
    
    def residual_secondzb(v):
        xrrs, yrrs, zrrs = real_pos.T
        spxs, spys = pix_pos.T
        xtop = v[0]*xrrs  + v[1]*yrrs  + v[2]*zrrs  + v[3]  + v[4]*xrrs**2  + v[5]*yrrs**2  + v[6]*xrrs*yrrs
        ytop = v[7]*xrrs  + v[8]*yrrs  + v[9]*zrrs  + v[10] + v[11]*xrrs**2 + v[12]*yrrs**2 + v[13]*xrrs*yrrs
        bot =  v[14]*xrrs + v[15]*yrrs + v[16]*zrrs + 1     + v[17]*xrrs**2 + v[18]*yrrs**2 + v[19]*xrrs*yrrs
    
        X,Y = xtop / bot, ytop / bot
        return (X - spxs)**2 + (Y - spys)**2
    
    lal = least_squares(residual_firstz, [0,1,0,1,0,1,0,0] , method='lm', xtol=1e-12, max_nfev=10000)
    
    ls0 = np.zeros(14)
    ls0[:4],ls0[7:11] = lal.x[:4], lal.x[4:8]
    la2 = least_squares(residual_secondz, ls0 , method='lm', xtol=1e-12, max_nfev=10000)
    
    ls0 = np.zeros(20)
    ls0[:14] = la2.x[:14]
    lab = least_squares(residual_secondzb, ls0 , method='lm', xtol=1e-12, max_nfev=10000)
        
    t2 = time()
    print(t2-t1)
    
    if cal_check:
        points = np.zeros_like(real_pos)
        points[:,:2], points[:,2] = pix_pos, real_pos[:,2]
        Xb, Yb = calib_invzl(lab.x, points, order='2b')
        return lab, Xb, Yb

    else: return lab
    
def rotate_coords(points, coords, angle ):
    """
    Parameters
    ----------
    points : (N,3) array
        points to rotate.
    coords : (0,1 or 2) list
        which two coordinates to rotate. Eg. [1,2] rotates axis y and z
    angle : float
        angle in radians.

    Returns
    -------
    points_r : (N,3) array
        array of points rotated.
    """
    c1, c2 = points[:, coords[0]], points[:, coords[1]]
    c1r = np.cos(angle) * c1 - np.sin(angle) * c2
    c2r = np.sin(angle) * c1 + np.cos(angle) * c2
    
    points_r = np.copy(points)
    points_r[:, coords[0]], points_r[:, coords[1]] = c1r, c2r
    return points_r


#%%
t1 = time()

vidd1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9522.MOV', 'ffmpeg') # 8055 frames
vidd2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera down/DSC_9523.MOV', 'ffmpeg') # 4593 frames

vidu1 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6009.MOV', 'ffmpeg') # 8100 frames
vidu2 = imageio.get_reader('/Volumes/Ice blocks/Scan water channel/25-08-07/Camera up/DSC_6010.MOV', 'ffmpeg') # 4548 frames

ny,nx,_ = np.shape(vidd1.get_data(0))

backd = np.array( vidd2.get_data(4591) )
backd = 0.3 * backd[:,:,0] + 0.7 * backd[:,:,1]
backu = np.array( vidu2.get_data(4546) )
backu = 0.3 * backu[:,:,0] + 0.7 * backu[:,:,1]

t2 = time()
print(t2-t1)

#%%
t1 = time()

xmedd1, ymedd1, iesd1 = position_grid(vidd1, backd, -20, 8054, in_frame=0, lims=[None,None,1200,2450])
xmedd2, ymedd2, iesd2 = position_grid(vidd2, backd, -20, 4592, in_frame=0, lims=[None,None,1200,2450])
iesd2 += 8054
xmd, ymd, ied = np.hstack((xmedd1,xmedd2)), np.hstack((ymedd1,ymedd2)), np.hstack((iesd1,iesd2))

xmedu1, ymedu1, iesu1 = position_grid(vidu1, backu, -100, 8099, in_frame=0, lims=[130,None,1200,2450])
xmedu2, ymedu2, iesu2 = position_grid(vidu2, backu, -100, 4547, in_frame=0, lims=[130,None,1200,2450])
iesu2 += 8099
xmu, ymu, ieu = np.hstack((xmedu1,xmedu2)), np.hstack((ymedu1,ymedu2)), np.hstack((iesu1,iesu2))

framesd, pekd = get_frames(xmd, ied, 20, 5)
framesu, peku = get_frames(xmu, ieu, 15, 4)

t2 = time()
print(t2-t1)

#%%
framesd = np.array([ 540, 1100, 1840, 2380, 2920, 3630, 4550, 5290, 6090, 6790, 7420, 8074, 8814, 9764])
framesu = np.array([ 500, 1050, 1800, 2330, 2870, 3590, 4500, 5240, 6040, 6740, 7370, 8030, 8769, 9729])

imd = image_position(framesd, [vidd1,vidd2], [8054,4592])
imu = image_position(framesu, [vidu1,vidu2], [8099,4547])

#%%

# for i in range(14):
for i in [0,13]:
    imb = (imu[i]-backu)[130:,1200:2300]
    g1, pp, tt = get_points(imb, tol_f=0.01, tol_ga=0.3)

    plt.figure()
    plt.imshow( g1 , cmap='gray' )
    plt.title(i)
    plt.show()

#%%

knownsu = {0 :[1299,1297,1272],
           1 :[1300,1298,1273],
           2 :[1303,1301,1277],
           3 :[1299,1300,1273],
           4 :[1300,1301,1275],
           5 :[1299,1300,1273],
           6 :[1299,1300,1273],
           7 :[1299,1300,1274],
           8 :[1299,1300,1275],
           9 :[1303,1302,1277],
           10:[1302,1299,1275],
           11:[1299,1296,1271],
           12:[1301,1302,1276],
           13:[1297,1298,1271]}

knownsd = {0 :[98,99,71],
           1 :[102,103,75],
           2 :[105,106,78],
           3 :[105,106,79],
           4 :[106,107,80],
           5 :[108,109,82],
           6 :[112,113,86],
           7 :[114,115,88],
           8 :[117,118,91],
           9 :[121,122,95],
           10:[125,126,99],
           11:[129,130,103],
           12:[130,131,104],
           13:[130,131,104]}


# i = 0
for i in range(13,-1,-1):
# for i in [3]:

    t1 = time()
    imb = (imd[i]-backd)[:,1200:2300]
    # imb = (imu[i]-backu)[130:,1200:2300]
    g1, pp, tt = get_points(imb, tol_f=0.01, tol_ga=0.3)
    
    # plt.figure()
    # plt.imshow( g1, cmap='gray' )
    # plt.plot(pp[:,0], pp[:,1], 'r.', markersize=10)
    # for j in range(len(tt)):
    #     plt.text(pp[j,0],pp[j,1],tt[j],fontsize=9)
    # plt.title(i)
    # plt.grid()
    # plt.show()

    # known = knownsu[i]
    known = knownsd[i]
    out,gridpos = find_indices(pp, known, eps=0.2, lim_ind=[0,25,-1e5,1e5])
    
    t2 = time()
    print(t2-t1, end=' ')
    
    plt.figure()
    plt.gca().invert_yaxis()
    # plt.imshow( g1, cmap='gray' )

    plt.plot(pp[:,0], pp[:,1], 'r.', markersize=10)
    plt.plot(out[:,0], out[:,1], 'b.', markersize=8)
    for j in range(len(out)):
        texto = "({:.0f},{:.0f})".format(gridpos[j,0],gridpos[j,1])
        plt.text(out[j,0],out[j,1],texto,fontsize=9)
    plt.title(i)
    plt.grid()
    plt.show()



#%%
t1 = time()
knownsu = {0 :[1299,1297,1272], 1 :[1300,1298,1273], 2 :[1303,1301,1277], 3 :[1299,1300,1273], 4 :[1300,1301,1275], 5 :[1299,1300,1273], 6 :[1299,1300,1273], 
           7 :[1299,1300,1274], 8 :[1299,1300,1275], 9 :[1303,1302,1277], 10:[1302,1299,1275], 11:[1299,1296,1271], 12:[1301,1302,1276], 13:[1297,1298,1271]}

knownsd = {0 :[98,99,71], 1 :[102,103,75], 2 :[105,106,78], 3 :[105,106,79], 4 :[106,107,80], 5 :[108,109,82], 6 :[112,113,86],
           7 :[114,115,88], 8 :[117,118,91], 9 :[121,122,95], 10:[125,126,99], 11:[129,130,103], 12:[130,131,104], 13:[130,131,104]}

pgridd, pposd = [], []
pgridu, pposu = [], []
for i in tqdm(range(14)):
    imb = (imd[i]-backd)[:,1200:2300]
    g1, pp, tt = get_points(imb, tol_f=0.01, tol_ga=0.3)
    known = knownsd[i]
    outd, gridposd = find_indices(pp, known, eps=0.2, lim_ind=[0,25,-1e5,1e5])
    pposd.append( outd + [1200,0] ) #add pixels removed
    pgridd.append( np.hstack(( gridposd, np.zeros((len(gridposd),1))+13-i )) )    
for i in tqdm(range(14)):
    imb = (imu[i]-backu)[130:,1200:2300]
    g1, pp, tt = get_points(imb, tol_f=0.01, tol_ga=0.3)
    known = knownsu[i]
    outu, gridposu = find_indices(pp, known, eps=0.2, lim_ind=[0,25,-1e5,1e5])
    pposu.append( outu + [1200,130] ) #add pixels removed
    pgridu.append( np.hstack(( gridposu, np.zeros((len(gridposu),1))+13-i )) )    

t2 = time()
print(t2-t1)
#%%

d_pos = np.vstack( [pgridd[i] * [3,6,10] for i in range(14)] )
u_pos = np.vstack( [pgridu[i] * [3,6,10] for i in range(14)] )

d_pix = np.vstack( [pposd[i] for i in range(14)] )
u_pix = np.vstack( [pposu[i] for i in range(14)] )

mw = (-7.5+3)/(294+390)
angle_xy = np.arctan( np.abs(mw) )
wall_dist = np.cos(angle_xy) * -7.5 + np.sin(angle_xy) * 294
angle_yz = -0.004228629492427357

d_pos_r = rotate_coords(d_pos, [0,1], angle_xy)
u_pos_r = rotate_coords(u_pos, [0,1], angle_xy)
d_pos_r = rotate_coords(d_pos_r, [0,1], angle_yz)
u_pos_r = rotate_coords(u_pos_r, [0,1], angle_yz)

print( np.shape(d_pos),np.shape(d_pix) )

cutoff_d = [0] + list(np.where( np.diff(d_pos[:,2]) )[0] + 1) + [len(d_pos)]
# lab_d, Xb_d, Yb_d = calibration_fit(d_pos, d_pix, cal_check=True)
lab_d, Xb_d, Yb_d = calibration_fit(d_pos_r, d_pix, cal_check=True)

print( np.shape(u_pos),np.shape(u_pix) )

cutoff_u = [0] + list(np.where( np.diff(u_pos[:,2]) )[0] + 1) + [len(u_pos)]
# lab_u, Xb_u, Yb_u = calibration_fit(u_pos, u_pix, cal_check=True)
lab_u, Xb_u, Yb_u = calibration_fit(u_pos_r, u_pix, cal_check=True)

#%%
came = 'u'
if came == 'd':
    cutoff, lab, pix_pos, Xb ,Yb = cutoff_d, lab_d, d_pix, Xb_d, Yb_d
    # real_pos = d_pos
    real_pos = d_pos_r
    # ims = [imd[i][:,1200:2300] for i in range(len(imd))]
    ims = [imd[i][:,:] for i in range(len(imd))]
elif came == 'u':
    cutoff, lab, pix_pos, Xb ,Yb = cutoff_u, lab_u, u_pix, Xb_u, Yb_u
    # real_pos = u_pos
    real_pos = u_pos_r
    # ims = [imu[i][130:,1200:2300] for i in range(len(imd))]
    ims = [imu[i][:,:] for i in range(len(imd))]

# for i in range(14):
for i in [0,13]:
    posi = real_pos[cutoff[i]:cutoff[i+1]]
    pixi = pix_pos[cutoff[i]:cutoff[i+1]]
    px, py = pixi.T 
    xp, yp = cal_direct(lab.x, posi, '2b')    
    
    plt.figure()
    plt.imshow( ims[i], cmap='gray')
    # plt.gca().invert_yaxis()
    plt.plot( px , py , '.' )
    plt.plot( xp, yp, '.' )
    plt.title(i)
    plt.show()

# for i in range(14):
for i in [0,13]:
    ini, fin = cutoff[i], cutoff[i+1]
    
    plt.figure()
    plt.plot( real_pos[ini:fin,0], real_pos[ini:fin,1], '.' )
    plt.plot( Xb[ini:fin], Yb[ini:fin], '.' )
    plt.title(i)
    plt.show()

# dists = np.sqrt( (real_pos[:,0] - Xb)**2 + (real_pos[:,1] - Yb)**2 )  
# plt.figure()
# plt.hist(dists,bins=100, density=True)
# plt.xlabel('Distance (mm)')
# plt.show()

#%%
dists_d = np.sqrt( (d_pos[:,0] - Xb_d)**2 + (d_pos[:,1] - Yb_d)**2 )  
dists_u = np.sqrt( (u_pos[:,0] - Xb_u)**2 + (u_pos[:,1] - Yb_u)**2 )  
# dists_d = np.sqrt( (d_pos_r[:,0] - Xb_d)**2 + (d_pos_r[:,1] - Yb_d)**2 )  
# dists_u = np.sqrt( (u_pos_r[:,0] - Xb_u)**2 + (u_pos_r[:,1] - Yb_u)**2 )  

plt.figure()
plt.hist(dists_d,bins=100, density=True, alpha=0.5, label='Down camera')
plt.hist(dists_u,bins=100, density=True, alpha=0.5, label='Top camera')
plt.hist( np.hstack((dists_d,dists_u)),bins=100, density=True, alpha=0.5, label='Both cameras')
plt.xlabel('Distance (mm)')
plt.legend()
plt.show()

# msize = 3
# for i in [0,13]:
#     plt.figure()
    
#     ini, fin = cutoff_d[i], cutoff_d[i+1]
#     plt.plot( d_pos[ini:fin,0], d_pos[ini:fin,1], 'o', markersize=msize )
#     plt.plot( Xb_d[ini:fin], Yb_d[ini:fin], 'o', markersize=msize )
    
#     ini, fin = cutoff_u[i], cutoff_u[i+1]
#     plt.plot( u_pos[ini:fin,0], u_pos[ini:fin,1], 's', markersize=msize )
#     plt.plot( Xb_u[ini:fin], Yb_u[ini:fin], 's', markersize=msize )
    
#     plt.title(i)
#     plt.show()

#%%

came = 'u'
if came == 'd':
    cutoff, lab, pix_pos, Xb ,Yb = cutoff_d, lab_d, d_pix, Xb_d, Yb_d
    # real_pos = d_pos
    real_pos = d_pos_r
    # ims = [imd[i][:,1200:2300] for i in range(len(imd))]
    ims = [imd[i][:,:] for i in range(len(imd))]
elif came == 'u':
    cutoff, lab, pix_pos, Xb ,Yb = cutoff_u, lab_u, u_pix, Xb_u, Yb_u
    # real_pos = u_pos
    real_pos = u_pos_r
    # ims = [imu[i][130:,1200:2300] for i in range(len(imd))]
    ims = [imu[i][:,:] for i in range(len(imd))]


for i in [0,13]:
    xw = -5.565
    pos2 = np.zeros((55,3))
    pos2[:,0], pos2[:,1], pos2[:,2] = xw, np.arange(-30,294+1,6), (13-i)*10

    pos3 = np.zeros((55,3))
    pos3[:,0], pos3[:,1], pos3[:,2] = xw, np.arange(-30,294+1,6), -13
    
    # pos3 = np.zeros((200,3))
    # pos3[150:200,0], pos3[150:200,1], pos3[150:200,2] = 95, np.arange(-18,276+1,6), (13-i)*10
    # pos3[100:150,0], pos3[100:150,1], pos3[100:150,2] = 90, np.arange(-18,276+1,6), (13-i)*10
    # pos3[50:100,0], pos3[50:100,1], pos3[50:100,2] = 85, np.arange(-18,276+1,6), (13-i)*10
    # pos3[:50,0], pos3[:50,1], pos3[:50,2] = 80, np.arange(-18,276+1,6), (13-i)*10
    
    # print(i, pos2)
    
    posi = real_pos[cutoff[i]:cutoff[i+1]]
    pixi = pix_pos[cutoff[i]:cutoff[i+1]]
    px, py = pixi.T 
    xp, yp = cal_direct(lab.x, posi, '2b')    
    
    x2,y2 = cal_direct(lab.x, pos2, '2b')    
    x3,y3 = cal_direct(lab.x, pos3, '2b')    
    
    
    plt.figure()
    plt.imshow( ims[i], cmap='gray')
    # plt.gca().invert_yaxis()
    plt.plot( px , py , '.' )
    plt.plot( xp, yp, '.' )
    
    plt.plot( x2, y2, '.-' )
    plt.plot( x3, y3, '.-' )
    plt.title(i)
    plt.show()



#%%




plt.figure()
plt.plot(posi[:,1],'.-')
plt.grid()
plt.show()


#%%

ss = {1:[1,23], 2:'aa'}







