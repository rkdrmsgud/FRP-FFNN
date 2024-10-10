import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')

basemap_fontsize = 26
colorbar_fontsize = 26
title_fontsize = 35
## data open --------------------------------------------------------------------------------------
dat_path = '/home/geunhyeong/frp/data/'
# Mask
mask = np.fromfile(dat_path + 'mask/frp_mask_final.gdat', np.float32)
mask = mask.reshape(90,180)
mask = np.ma.masked_equal(mask, 0)

# data
nn_rh = np.fromfile(dat_path + 'NN_sensitivity/NN_origin_rh_anom.gdat', np.float32)
nn_prcp = np.fromfile(dat_path + 'NN_sensitivity/NN_origin_prcp_anom.gdat', np.float32)

fwi_rh = np.fromfile(dat_path + 'FWI_sensitivity/FWI_origin_rh_anom.gdat', np.float32)
fwi_prcp = np.fromfile(dat_path + 'FWI_sensitivity/FWI_origin_prcp_anom.gdat', np.float32)

lrp = np.fromfile(dat_path + 'rank1/lrp_rank1.gdat', np.float32)
nn = np.fromfile(dat_path + 'rank1/nn_sensitivity_rank1.gdat', np.float32)
fwi = np.fromfile(dat_path + 'rank1/fwi_sensitivity_rank1.gdat', np.float32)

nn_rh = np.ma.masked_equal(nn_rh, -999)
nn_prcp = np.ma.masked_equal(nn_prcp, -999)

fwi_rh = np.ma.masked_equal(fwi_rh, -999)
fwi_prcp = np.ma.masked_equal(fwi_prcp, -999)

nn = np.ma.masked_equal(nn, -999)
fwi = np.ma.masked_equal(fwi, -999)

nn_rh = nn_rh.reshape(90,180)
nn_prcp = nn_prcp.reshape(90,180)

fwi_rh = fwi_rh.reshape(90,180)
fwi_prcp = fwi_prcp.reshape(90,180)

nn = nn.reshape(90,180)
fwi = fwi.reshape(90,180)


## figure -----------------------------------------------------------------------------------------

x_start1 = 0.05
x_start2 = 0.40
x_start3 = 0.75

y_start1 = 0.95
y_start2 = y_start1 - 0.14*1


x_size = 0.40
y_size = 0.11

fig = plt.figure(figsize=(38,45))

#      x위치,y위치,x크기,y크기
ax1 = [x_start1,y_start1,x_size,y_size] 
ax2 = [x_start2,y_start1,x_size,y_size] 
ax3 = [x_start3,y_start1,x_size,y_size] 

ax4 = [x_start1,y_start2,x_size,y_size] 
ax5 = [x_start2,y_start2,x_size,y_size] 
ax6 = [x_start3,y_start2,x_size,y_size] 


# ax1
plt.axes(ax1)

x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,92,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_rh,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.3,0.31],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.3,0.31,0.1))
cbar.ax.set_yticklabels(np.round(np.arange(-0.3,0.31,0.1),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title('(a) Org - RH2m Clim Corr. Diff., FFNNs',fontsize=title_fontsize, loc='left', pad=20)

# ax2
plt.axes(ax2)

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_prcp,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower',clim=[-0.3,0.31])
cbar = plt.colorbar(orientation='vertical', ticks=np.arange(-0.3,0.31,0.1))
cbar.ax.set_yticklabels(np.round(np.arange(-0.3,0.31,0.1),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title('(b) Org - prcp Clim Corr. Diff., FFNNs',fontsize=title_fontsize, loc='left', pad=20)

# ax4
plt.axes(ax4)

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi_rh,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.3,0.31],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.3,0.31,0.1))
cbar.ax.set_yticklabels(np.round(np.arange(-0.3,0.31,0.1),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title('(d) Org - RH2m Clim Corr. Diff., FWI',fontsize=title_fontsize, loc='left', pad=20)

# ax5
plt.axes(ax5)

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi_prcp,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower',clim=[-0.3,0.31])
cbar = plt.colorbar(orientation='vertical', ticks=np.arange(-0.3,0.31,0.1))
cbar.ax.set_yticklabels(np.round(np.arange(-0.3,0.31,0.1),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title('(e) Org - prcp Clim Corr. Diff., FWI',fontsize=title_fontsize, loc='left', pad=20)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ax3
plt.axes(ax3)

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

# rh gr
custom = [( 50 /255,  200 /255,   50 /255), # RH 
          (240 /255,  220 /255,   30 /255), # WS
          ( 10 /255,   80 /255,  210 /255), # prcp
          (240 /255,   20 /255,   20 /255)] # T2m

cmap = mcolors.ListedColormap(custom)
plt.imshow(np.flip(nn,axis=0), cmap=cmap, extent=[180,540,92,-90], origin='lower', clim = [0,4], alpha=0.9)
cbar = plt.colorbar(orientation='vertical',ticks=[0.5,1.5,2.5,3.5])
cbar.ax.set_yticklabels(['RH','WS','prcp','T2m'],fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.title('(c) NN Sensitivity Rank1',fontsize=title_fontsize,loc='left',pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)


# ax6
plt.axes(ax6)

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi,axis=0), cmap=cmap, extent=[180,540,92,-90], origin='lower', clim = [0,4], alpha=0.9)
cbar = plt.colorbar(orientation='vertical',ticks=[0.5,1.5,2.5,3.5])
cbar.ax.set_yticklabels(['RH','WS','prcp','T2m'],fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.title('(f) FWI Sensitivity Rank1',fontsize=title_fontsize,loc='left',pad=20)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig.4.png',format='png',dpi=300,bbox_inches='tight')
plt.close()
