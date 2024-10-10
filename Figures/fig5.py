import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib
matplotlib.use('TkAgg')

mask = np.fromfile('/home/geunhyeong/frp/data/mask/frp_mask_final.gdat', np.float32)
mask = mask.reshape(90,180)
mask = np.ma.masked_equal(mask, 0)

t2m_rank1 = np.fromfile('/home/geunhyeong/frp/data/rank1/t2m_rank1.gdat', np.float32)
t2m_rank1[t2m_rank1==0] = -999
t2m_rank1[t2m_rank1==2] = -999
t2m_rank1 = t2m_rank1.reshape(90,180)
t2m_rank1 = np.ma.masked_equal(t2m_rank1, -999)

basemap_fontsize = 40
colorbar_fontsize = 40
title_fontsize = 50

## Data Load
# FRP by NN
frp1 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred1.gdat', np.float32).reshape(-1,90,180)
frp2 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred2.gdat', np.float32).reshape(-1,90,180)
frp3 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred3.gdat', np.float32).reshape(-1,90,180)
frp4 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred4.gdat', np.float32).reshape(-1,90,180)
frp5 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred5.gdat', np.float32).reshape(-1,90,180)

# delete leap year
frp1 = np.delete(frp1, 365*3+31+29, axis=0)
frp2 = np.delete(frp2, 365*3+31+29, axis=0)
frp3 = np.delete(frp3, 365*3+31+29, axis=0)
frp4 = np.delete(frp4, 365*3+31+29, axis=0)
frp5 = np.delete(frp5, 365*3+31+29, axis=0)

# LRP
lrp1 = np.fromfile('/home/geunhyeong/frp/data/LRP/lrp1.gdat', np.float32).reshape(-1,4,90,180)
lrp2 = np.fromfile('/home/geunhyeong/frp/data/LRP/lrp2.gdat', np.float32).reshape(-1,4,90,180)
lrp3 = np.fromfile('/home/geunhyeong/frp/data/LRP/lrp3.gdat', np.float32).reshape(-1,4,90,180)
lrp4 = np.fromfile('/home/geunhyeong/frp/data/LRP/lrp4.gdat', np.float32).reshape(-1,4,90,180)
lrp5 = np.fromfile('/home/geunhyeong/frp/data/LRP/lrp5.gdat', np.float32).reshape(-1,4,90,180)

# delete leap year
lrp1 = np.delete(lrp1, 365*3+31+29, axis=0)
lrp2 = np.delete(lrp2, 365*3+31+29, axis=0)
lrp3 = np.delete(lrp3, 365*3+31+29, axis=0)
lrp4 = np.delete(lrp4, 365*3+31+29, axis=0)
lrp5 = np.delete(lrp5, 365*3+31+29, axis=0)


## Data Preprocessing -----------------------------------------------------------------------------

frp = np.concatenate((frp1, frp2, frp3, frp4, frp5), axis=0)
del frp1, frp2, frp3 ,frp4 ,frp5

lrp = np.concatenate((lrp1, lrp2, lrp3, lrp4, lrp5), axis=0)
del lrp1, lrp2, lrp3, lrp4, lrp5

ws_lrp = lrp[:,3,:,:]
t2m_lrp = lrp[:,2,:,:]
rh_lrp = lrp[:,1,:,:]
prcp_lrp = lrp[:,0,:,:]
del lrp

# daily anomaly
frp = frp.reshape(20,365,90,180)
t2m_lrp = t2m_lrp.reshape(20,365,90,180)
ws_lrp = ws_lrp.reshape(20,365,90,180)
rh_lrp = rh_lrp.reshape(20,365,90,180)
prcp_lrp = prcp_lrp.reshape(20,365,90,180)

frp = np.ma.anom(frp, axis=0).reshape(-1,90,180)
t2m_lrp = np.ma.anom(t2m_lrp, axis=0).reshape(-1,90,180)
ws_lrp = np.ma.anom(ws_lrp, axis=0).reshape(-1,90,180)
rh_lrp = np.ma.anom(rh_lrp, axis=0).reshape(-1,90,180)
prcp_lrp = np.ma.anom(prcp_lrp, axis=0).reshape(-1,90,180)

## T2m Correlation --------------------------------------------------------------------------------

frp = np.ma.masked_equal(frp, -999)
t2m_lrp = np.ma.masked_equal(t2m_lrp, -999)

# cor
xy = np.mean(frp*t2m_lrp,axis=0)
xx = np.mean(frp**2,axis=0)
yy = np.mean(t2m_lrp**2,axis=0)

t2m_cor = xy/np.sqrt(xx*yy)
sig = (t2m_cor*np.sqrt(7298))/np.sqrt(1-t2m_cor*t2m_cor)

t2m_cor = t2m_cor.reshape(90,180)
sig = sig.reshape(90,180)

t2m_sig1 = sig.copy()
t2m_sig1[sig<1.96] = -999
t2m_sig1[sig>1.96] = 2
t2m_sig1 = np.ma.masked_equal(t2m_sig1, -999)

## WS Correlation ---------------------------------------------------------------------------------
frp = np.ma.masked_equal(frp, -999)
ws_lrp = np.ma.masked_equal(ws_lrp, -999)

# cor
xy = np.mean(frp*ws_lrp,axis=0)
xx = np.mean(frp**2,axis=0)
yy = np.mean(ws_lrp**2,axis=0)

ws_cor = xy/np.sqrt(xx*yy)
sig = (ws_cor*np.sqrt(7298))/np.sqrt(1-ws_cor*ws_cor)

ws_cor = ws_cor.reshape(90,180)
sig = sig.reshape(90,180)

ws_sig1 = sig.copy()
ws_sig1[sig<1.96] = -999
ws_sig1[sig>1.96] = 2
ws_sig1 = np.ma.masked_equal(ws_sig1, -999)

## RH Correlation ---------------------------------------------------------------------------------
frp = np.ma.masked_equal(frp, -999)
rh_lrp = np.ma.masked_equal(rh_lrp, -999)

# cor
xy = np.mean(frp*rh_lrp,axis=0)
xx = np.mean(frp**2,axis=0)
yy = np.mean(rh_lrp**2,axis=0)

rh_cor = xy/np.sqrt(xx*yy)
sig = (rh_cor*np.sqrt(7298))/np.sqrt(1-rh_cor*rh_cor)

rh_cor = rh_cor.reshape(90,180)
sig = sig.reshape(90,180)

rh_sig1 = sig.copy()
rh_sig1[sig<1.96] = -999
rh_sig1[sig>1.96] = 2
rh_sig1 = np.ma.masked_equal(rh_sig1, -999)

## PRCP Correlation -------------------------------------------------------------------------------
frp = np.ma.masked_equal(frp, -999)
prcp_lrp = np.ma.masked_equal(prcp_lrp, -999)

# cor
xy = np.mean(frp*prcp_lrp,axis=0)
xx = np.mean(frp**2,axis=0)
yy = np.mean(prcp_lrp**2,axis=0)

prcp_cor = xy/np.sqrt(xx*yy)
sig = (prcp_cor*np.sqrt(7298))/np.sqrt(1-prcp_cor*prcp_cor)

prcp_cor = prcp_cor.reshape(90,180)
sig = sig.reshape(90,180)

prcp_sig1 = sig.copy()
prcp_sig1[sig<1.96] = -999
prcp_sig1[sig>1.96] = 2
prcp_sig1 = np.ma.masked_equal(prcp_sig1, -999)

## figure -----------------------------------------------------------------------------------------

x_start1 = 0.05
x_start2 = 0.52

y_start1 = 0.95
y_start2 = y_start1 - 0.30*1
y_start3 = y_start1 - 0.30*2
y_start4 = y_start1 - 0.30*3

x_size = 0.45
y_size = 0.25

fig = plt.figure(figsize=(45,35))

#      x위치,y위치,x크기,y크기
ax1 = [x_start1,y_start1,x_size,y_size] #(a)
ax2 = [x_start2,y_start1,x_size,y_size] #(d)

ax3 = [x_start1,y_start2,x_size,y_size] #(b)
ax4 = [x_start2,y_start2,x_size,y_size] #(e)

# x1, y1 = np.meshgrid(lon, lat)

# ax1
plt.axes(ax1)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'
x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(rh_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.8,0.81],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.8,0.81,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.8,0.81,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,rh_sig1,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)

plt.title('(a) Correlation [RH score, NN FRP]',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

# ax2
plt.axes(ax2)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(prcp_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.8,0.81],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.8,0.81,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.8,0.81,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,prcp_sig1,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)

plt.title('(b) Correlation [prcp score, NN FRP]',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

# ax3
plt.axes(ax3)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(t2m_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.8,0.81],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.8,0.81,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.8,0.81,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,t2m_sig1,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)

plt.title('(c) Correlation [T2m score, NN FRP]',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

# ax4
plt.axes(ax4)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(ws_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.8,0.81],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.8,0.81,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.8,0.81,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,ws_sig1,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)

plt.title('(d) Correlation [WS score, NN FRP]',fontsize=title_fontsize, loc='left', pad=20)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)

plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig.5.png',format='png',dpi=300,bbox_inches='tight')


