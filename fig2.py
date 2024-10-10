import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')

mask = np.fromfile('/home/geunhyeong/frp/data/mask/frp_mask_final.gdat', np.float32)
mask = mask.reshape(90,180)
data_mask = mask.copy()
data_mask2 = mask.copy()
data_mask2[data_mask2==1] = -999
data_mask = data_mask.reshape(1,90,180)
data_mask = np.repeat(data_mask,7300,axis=0)
data_mask[data_mask==1] = -999 # 7300,90,180

mask = np.ma.masked_equal(mask, 0)


### Chapter.1 -------------------------------------------------------------------------------------
# input : t2m, ws, rh, pr
# output : frp

basemap_fontsize = 35
colorbar_fontsize = 35
title_fontsize = 50

nn_pred1 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred1.gdat', np.float32).reshape(-1,90,180) # 1461*90*180
nn_pred2 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred2.gdat', np.float32).reshape(-1,90,180)
nn_pred3 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred3.gdat', np.float32).reshape(-1,90,180)
nn_pred4 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred4.gdat', np.float32).reshape(-1,90,180)
nn_pred5 = np.fromfile('/home/geunhyeong/frp/output/NN_result/pred5.gdat', np.float32).reshape(-1,90,180)

nn_pred = np.concatenate((nn_pred1, nn_pred2, nn_pred3, nn_pred4, nn_pred5), axis=0)
nn_pred = np.ma.masked_equal(nn_pred, -999)

del nn_pred1, nn_pred2, nn_pred3, nn_pred4, nn_pred5

fwi_pred1 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred1.gdat', np.float32).reshape(-1,90,180)
fwi_pred2 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred2.gdat', np.float32).reshape(-1,90,180)
fwi_pred3 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred3.gdat', np.float32).reshape(-1,90,180)
fwi_pred4 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred4.gdat', np.float32).reshape(-1,90,180)
fwi_pred5 = np.fromfile('/home/geunhyeong/frp/output/FWI_result/fwi_pred5.gdat', np.float32).reshape(-1,90,180)

fwi_pred = np.concatenate((fwi_pred1, fwi_pred2, fwi_pred3, fwi_pred4, fwi_pred5), axis=0)

del fwi_pred1, fwi_pred2, fwi_pred3, fwi_pred4, fwi_pred5

true = np.fromfile('/home/geunhyeong/frp/data/ncl_2001_2020_ver2.gdat', np.float32).reshape(-1,90,180)
true = np.ma.masked_equal(true, -999)

true[true==0] = 1
true = np.log(true)

nn_pred = nn_pred.reshape(-1,90*180)
fwi_pred = fwi_pred.reshape(-1,90*180)
true = true.reshape(-1,90*180)


## True data preprocessing
true = true.reshape(-1,90,180)

# append
true1_1 = true[:365*3+31+28]                # (1154, 90, 180)
true1_2 = true[365*3+31+29:365*4+1]         # (306, 90, 180)
true1 = np.append(true1_1,true1_2,axis=0)   # (1460, 90, 180)

true2_1 = true[365*4+1:365*7+31+28+1]
true2_2 = true[365*7+31+29+1:365*8+2]
true2 = np.append(true2_1,true2_2,axis=0)   

true3_1 = true[365*8+2:365*11+31+28+2]      
true3_2 = true[365*11+31+29+2:365*12+3]     
true3 = np.append(true3_1,true3_2,axis=0)   

true4_1 = true[365*12+3:365*15+31+28+3]     
true4_2 = true[365*15+31+29+3:365*16+4]     
true4 = np.append(true4_1,true4_2,axis=0)   

true5_1 = true[365*16+4:365*19+31+28+4]     
true5_2 = true[365*19+31+29+4:]             
true5 = np.append(true5_1,true5_2,axis=0)   

true = np.concatenate((true1, true2, true3, true4, true5), axis=0)
del true1, true2, true3, true4, true5

# anomaly
true = true.reshape(-1,365,90,180)
true = np.ma.anom(true,axis=0)
true = true.reshape(-1,90,180)              # (7300, 90, 180)


## Pred data preprocessing
nn_pred = nn_pred.reshape(-1,90,180)

# append
nn_pred1_1 = nn_pred[:365*3+31+28]
nn_pred1_2 = nn_pred[365*3+31+29:365*4+1]
nn_pred1 = np.append(nn_pred1_1,nn_pred1_2,axis=0)

nn_pred2_1 = nn_pred[365*4+1:365*7+31+28+1]
nn_pred2_2 = nn_pred[365*7+31+29+1:365*8+2]
nn_pred2 = np.append(nn_pred2_1,nn_pred2_2,axis=0)

nn_pred3_1 = nn_pred[365*8+2:365*11+31+28+2]
nn_pred3_2 = nn_pred[365*11+31+29+2:365*12+3]
nn_pred3 = np.append(nn_pred3_1,nn_pred3_2,axis=0)

nn_pred4_1 = nn_pred[365*12+3:365*15+31+28+3]
nn_pred4_2 = nn_pred[365*15+31+29+3:365*16+4]
nn_pred4 = np.append(nn_pred4_1,nn_pred4_2,axis=0)

nn_pred5_1 = nn_pred[365*16+4:365*19+31+28+4]
nn_pred5_2 = nn_pred[365*19+31+29+4:]
nn_pred5 = np.append(nn_pred5_1,nn_pred5_2,axis=0)

nn_pred = np.concatenate((nn_pred1, nn_pred2, nn_pred3, nn_pred4, nn_pred5), axis=0)
del nn_pred1, nn_pred2, nn_pred3, nn_pred4, nn_pred5

# anomaly
nn_pred = nn_pred.reshape(-1,365,90,180)
nn_pred = np.ma.anom(nn_pred,axis=0)
nn_pred = nn_pred.reshape(-1,90,180)

print('pred')
print(nn_pred.shape)


fwi_pred = fwi_pred.reshape(-1,90,180)

# append
fwi_pred1_1 = fwi_pred[:365*3+31+28]
fwi_pred1_2 = fwi_pred[365*3+31+29:365*4+1]
fwi_pred1 = np.append(fwi_pred1_1,fwi_pred1_2,axis=0)

fwi_pred2_1 = fwi_pred[365*4+1:365*7+31+28+1]
fwi_pred2_2 = fwi_pred[365*7+31+29+1:365*8+2]
fwi_pred2 = np.append(fwi_pred2_1,fwi_pred2_2,axis=0)

fwi_pred3_1 = fwi_pred[365*8+2:365*11+31+28+2]
fwi_pred3_2 = fwi_pred[365*11+31+29+2:365*12+3]
fwi_pred3 = np.append(fwi_pred3_1,fwi_pred3_2,axis=0)

fwi_pred4_1 = fwi_pred[365*12+3:365*15+31+28+3]
fwi_pred4_2 = fwi_pred[365*15+31+29+3:365*16+4]
fwi_pred4 = np.append(fwi_pred4_1,fwi_pred4_2,axis=0)

fwi_pred5_1 = fwi_pred[365*16+4:365*19+31+28+4]
fwi_pred5_2 = fwi_pred[365*19+31+29+4:]
fwi_pred5 = np.append(fwi_pred5_1,fwi_pred5_2,axis=0)

fwi_pred = np.concatenate((fwi_pred1, fwi_pred2, fwi_pred3, fwi_pred4, fwi_pred5), axis=0)
del fwi_pred1, fwi_pred2, fwi_pred3, fwi_pred4, fwi_pred5

# anomaly
fwi_pred = fwi_pred.reshape(-1,365,90,180)
fwi_pred = np.ma.anom(fwi_pred,axis=0)
fwi_pred = fwi_pred.reshape(-1,90,180)

print('pred')
print(fwi_pred.shape)


## Correlation (nn_pred) --------------------------------------------------------------------------
nn_pred = nn_pred.reshape(-1,90,180)
true = true.reshape(-1,90,180)

# cor
xy = np.mean(true*nn_pred,axis=0)
xx = np.mean(true**2,axis=0)
yy = np.mean(nn_pred**2,axis=0)

nn_cor = xy/np.sqrt(xx*yy)
sig = (nn_cor*np.sqrt(7298))/np.sqrt(1-nn_cor*nn_cor)

nn_cor = nn_cor.reshape(90,180)
sig = sig.reshape(90,180)

sig1 = sig.copy()
sig1[sig<1.96] = -999
sig1[sig>1.96] = 2
sig1 = np.ma.masked_equal(sig1, -999)


## RMSE
true = true.reshape(7300,-1)
nn_pred = nn_pred.reshape(7300,-1)

nn_rmse = np.zeros(16200)
for i in range(16200):
    if nn_pred[0,i] == -999:
        nn_rmse[i] = -999
    else:
        nn_rmse[i] = mean_squared_error(true[:,i], nn_pred[:,i], squared=False)**0.5

nn_rmse = nn_rmse.reshape(90,180)
nn_rmse = nn_rmse + data_mask2
nn_rmse[nn_rmse<-900] = -999
nn_rmse = np.ma.masked_equal(nn_rmse,-999)

## Correlation (fwi_pred) -------------------------------------------------------------------------
fwi_pred = fwi_pred.reshape(-1,90,180)
true = true.reshape(-1,90,180)

# cor
xy = np.mean(true*fwi_pred,axis=0)
xx = np.mean(true**2,axis=0)
yy = np.mean(fwi_pred**2,axis=0)

fwi_cor = xy/np.sqrt(xx*yy)
sig = (fwi_cor*np.sqrt(7298))/np.sqrt(1-fwi_cor*fwi_cor)

fwi_cor = fwi_cor.reshape(90,180)
sig = sig.reshape(90,180)

sig2 = sig.copy()
sig2[sig<1.96] = -999
sig2[sig>1.96] = 2
sig2 = np.ma.masked_equal(sig2, -999)

## RMSE
true = true.reshape(7300,-1)
fwi_pred = fwi_pred.reshape(7300,-1)

fwi_pred = np.array(fwi_pred)
true = np.array(true)
fwi_pred[fwi_pred==-0.] = -999

fwi_rmse = np.zeros(16200)
for i in range(16200):
    if fwi_pred[0,i] == -999:
        fwi_rmse[i] = -999
    else:
        fwi_rmse[i] = mean_squared_error(true[:,i], fwi_pred[:,i], squared=False)**0.5

fwi_rmse = fwi_rmse.reshape(90,180)
fwi_rmse = np.ma.masked_equal(fwi_rmse, -999)


## sig_test
sig_test = np.fromfile('/home/geunhyeong/frp/data/cor_diff_sig_test.gdat', np.float32)
sig_test = sig_test.reshape(90,180)
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


plt.savefig('/home/geunhyeong/frp/fig/png/github/Fig2.png', dpi=300, bbox_inches='tight')
plt.close()
