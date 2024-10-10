import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')


mask = np.fromfile('frp_mask.gdat', np.float32)
mask = mask.reshape(90,180)

data_mask = mask.copy()
data_mask[data_mask==1] = -999

mask = np.ma.masked_equal(mask, 0)

### Chapter.1 -------------------------------------------------------------------------------------
# input : t2m, ws, rh, pr
# output : frp

basemap_fontsize = 35
colorbar_fontsize = 35
title_fontsize = 45


# NN result [1460 * 90 * 180]
nn_pred = np.fromfile('NN_Pred_2017-2020.gdat', np.float32)
nn_pred = np.ma.masked_equal(nn_pred, -999)

# FWI result [1460 * 90 * 180]
fwi_pred = np.fromfile('FWI_Pred_2017-2020.gdat', np.float32)

# True [1460 * 90 * 180]
true = np.fromfile('OBS_FRP_2017-2020.gdat', np.float32)
true = np.ma.masked_equal(true, -999)

true[true==0] = 1
true = np.log(true)


## True data preprocessing
true = true.reshape(-1,90,180)

# anomaly
true = true.reshape(-1,365,90,180)
true = np.ma.anom(true,axis=0).reshape(-1,90,180)   # (1460, 90, 180)                   


## Pred data preprocessing
nn_pred = nn_pred.reshape(-1,90,180)

# anomaly
nn_pred = nn_pred.reshape(-1,365,90,180)
nn_pred = np.ma.anom(nn_pred,axis=0).reshape(-1,90,180) # (1460, 90, 180)

print('nn pred')
print(nn_pred.shape)


# FWI
fwi_pred = fwi_pred.reshape(-1,90,180)

# anomaly
fwi_pred = fwi_pred.reshape(-1,365,90,180)
fwi_pred = np.ma.anom(fwi_pred,axis=0).reshape(-1,90,180)   # (1460, 90, 180)

print('fwi pred')
print(fwi_pred.shape)


## Correlation (nn_pred) --------------------------------------------------------------------------
nn_pred = nn_pred.reshape(-1,90,180)
true = true.reshape(-1,90,180)

# cor
xy = np.mean(true*nn_pred,axis=0)
xx = np.mean(true**2,axis=0)
yy = np.mean(nn_pred**2,axis=0)

nn_cor = xy/np.sqrt(xx*yy)
sig = (nn_cor*np.sqrt(1460-2)) / np.sqrt(1-nn_cor*nn_cor)

nn_cor = nn_cor.reshape(90,180)
sig = sig.reshape(90,180)

sig1 = sig.copy()
sig1[sig<1.96] = -999
sig1[sig>1.96] = 2
sig1 = np.ma.masked_equal(sig1, -999)


## RMSE
true = true.reshape(1460,-1)
nn_pred = nn_pred.reshape(1460,-1)

nn_rmse = np.zeros(90*180)
for i in range(90*180):
    if nn_pred[0,i] == -999:
        nn_rmse[i] = -999
    else:
        nn_rmse[i] = mean_squared_error(true[:,i], nn_pred[:,i], squared=False)**0.5

nn_rmse = nn_rmse.reshape(90,180)
nn_rmse = nn_rmse + data_mask
nn_rmse[nn_rmse<-900] = -999


## Correlation (fwi_pred) -------------------------------------------------------------------------
fwi_pred = fwi_pred.reshape(-1,90,180)
true = true.reshape(-1,90,180)

# cor
xy = np.mean(true*fwi_pred,axis=0)
xx = np.mean(true**2,axis=0)
yy = np.mean(fwi_pred**2,axis=0)

fwi_cor = xy/np.sqrt(xx*yy)

sig = (fwi_cor*np.sqrt(1460-2))/np.sqrt(1-fwi_cor*fwi_cor)

fwi_cor = fwi_cor.reshape(90,180)
sig = sig.reshape(90,180)

sig2 = sig.copy()
sig2[sig<1.96] = -999
sig2[sig>1.96] = 2
sig2 = np.ma.masked_equal(sig2, -999)


## RMSE
true = true.reshape(1460,-1)
fwi_pred = fwi_pred.reshape(1460,-1)

fwi_pred = np.array(fwi_pred)
true = np.array(true)
fwi_pred[fwi_pred==-0.] = -999

fwi_pred = np.nan_to_num(fwi_pred, nan=-999)
fwi_pred = np.ma.masked_equal(fwi_pred, -999)

fwi_rmse = np.zeros(90*180)
for i in range(90*180):
    if fwi_pred[0,i] == -999:
        fwi_rmse[i] = -999
    else:
        fwi_rmse[i] = mean_squared_error(true[:,i], fwi_pred[:,i], squared=False)**0.5

fwi_rmse = fwi_rmse.reshape(90,180)
fwi_rmse = np.ma.masked_equal(fwi_rmse, -999)


## sig_test
sig_test = np.fromfile('sig_test_org.gdat', np.float32).reshape(90,180)
cor_dif = nn_cor - fwi_cor
cor_dif[cor_dif<0] = -999
cor_dif[cor_dif>=0] = 0
sig_test = sig_test + cor_dif
sig_test[sig_test<0] = -999
sig_test = np.ma.masked_equal(sig_test, -999)
sig_test = np.ma.masked_where(sig_test>0.05, sig_test)


## plot -------------------------------------------------------------------------------------------
fig,ax = plt.subplots(1,1,figsize=(45,35))

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

ax5 = [x_start1,y_start3,x_size,y_size] #(c)
ax6 = [x_start2,y_start3,x_size,y_size] #(f)


# ax1
plt.axes(ax1)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'
x1, y1 = np.meshgrid(np.arange(180,540,2), np.arange(-90,90,2))

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.6,0.61],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.6,0.61,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.6,0.61,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,sig1,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title(f'(a) Cor. Skill of NN FRP forecast',fontsize=title_fontsize, loc='left', pad=20)

# ax2
plt.axes(ax2)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_rmse,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower',clim=[0.0,2.0])
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(0.00,2.01,0.25))
cbar.ax.set_yticklabels(np.round(np.arange(0.00,2.01,0.25),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title(f'(d) RMSE of NN FRP forecast',fontsize=title_fontsize, loc='left', pad=20)

# ax3
plt.axes(ax3)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.6,0.61],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.6,0.61,0.2))
cbar.ax.set_yticklabels(np.round(np.arange(-0.6,0.61,0.2),1),fontsize=colorbar_fontsize)

plt.pcolor(x1,y1,sig2,hatch='.',alpha=0., zorder=2)
plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title(f'(b) Cor. Skill of FWI FRP forecast',fontsize=title_fontsize, loc='left', pad=20)

# ax4
plt.axes(ax4)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(fwi_rmse,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower',clim=[0,2.0])
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(0.00,2.01,0.25))
cbar.ax.set_yticklabels(np.round(np.arange(0.00,2.01,0.25),1),fontsize=colorbar_fontsize)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2])
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title(f'(e) RMSE of FWI FRP forecast',fontsize=title_fontsize, loc='left', pad=20)


# ax5
plt.axes(ax5)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5,zorder=5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_cor - fwi_cor,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower', clim=[-0.3,0.31],zorder=1)
cbar = plt.colorbar(orientation='vertical',ticks=np.arange(-0.3,0.31,0.1))
cbar.ax.set_yticklabels(np.round(np.arange(-0.3,0.31,0.1),1),fontsize=colorbar_fontsize)
plt.pcolor(x1,y1,sig_test,hatch='.',alpha=0., zorder=2)

plt.imshow(mask,cmap=cm.gray,extent=[180,540,92,-90],clim=[0,2],zorder=4)
plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['top'].set_zorder(13)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['left'].set_zorder(13)
plt.gca().spines['right'].set_linewidth(3)
plt.gca().spines['right'].set_zorder(13)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['bottom'].set_zorder(13)
plt.title('(c) (a)-(b)',fontsize=title_fontsize, loc='left', pad=20)

# ax6
plt.axes(ax6)

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'black'

map1 = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,resolution='c',llcrnrlon=180,urcrnrlon=540)
map1.drawcoastlines(linewidth=0.5)
map1.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0],fontsize=basemap_fontsize,color='grey',linewidth=0.2)
map1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=basemap_fontsize,color='grey',linewidth=0.2)

plt.imshow(np.flip(nn_rmse - fwi_rmse,axis=0), cmap=plt.cm.RdBu_r,extent=[180,540,92,-90], origin='lower',clim=[-0.3,0.31])
cbar = plt.colorbar(orientation='vertical')
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
plt.title('(f) (d)-(e)',fontsize=title_fontsize, loc='left', pad=20)


plt.savefig('Figure2_2017-2020.png', dpi=300, bbox_inches='tight')
plt.close()
