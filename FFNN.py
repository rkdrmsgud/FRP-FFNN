#=================================================================================================
# Import Module
#=================================================================================================
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.models import Model
from sklearn.preprocessing import minmax_scale
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gc
import os

#=================================================================================================
# gpu setting 
# (if you can't using a GPU, please block this part)
#=================================================================================================
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
gpus = tf.config.experimental.list_physical_devices('GPU')

#=================================================================================================
# Data Load
#=================================================================================================

# set main director
main_dir = '/home/geunhyeong/frp/'

# load input [7305*90*180]
ws_org = np.fromfile('/home/geunhyeong/frp/data/wind_speed_2001_2020_ver2.gdat',np.float32)
t2m_org = np.fromfile('/home/geunhyeong/frp/data/t2m_2001_2020_ver2.gdat',np.float32)
rh_org = np.fromfile('/home/geunhyeong/frp/data/rh_2001_2020_ver2.gdat',np.float32)
prcp_org = np.fromfile('/home/geunhyeong/frp/data/prcp_2001_2020_ver2.gdat',np.float32)

# Load Label
frp_org = np.fromfile('/home/geunhyeong/frp/data/ncl_2001_2020_ver2.gdat',np.float32)

#=================================================================================================
# Data Preprocessing
#=================================================================================================

# Reshape
ws_org = ws_org.reshape(7305,-1)
t2m_org = t2m_org.reshape(7305,-1)
rh_org = rh_org.reshape(7305,-1)
prcp_org = prcp_org.reshape(7305,-1)
frp_org = frp_org.reshape(7305,-1)


for exp in np.arange(1,6):
  pred = np.zeros((1461,90*180))
  tf.keras.backend.clear_session()
  
  for i in range(90*180):
    if frp_org[0,i] == -999:
        pred[:,i] = -999
  
    else:
        ws = ws_org[:,i].reshape(-1,1) # 7305,1
        t2m = t2m_org[:,i].reshape(-1,1)
        rh = rh_org[:,i].reshape(-1,1)
        prcp = prcp_org[:,i].reshape(-1,1)
        frp = frp_org[:,i].reshape(-1)
        

        ## Input Data Preprocessing 
        # Normalization
        ws = minmax_scale(ws, axis=0, copy=True)
        t2m = minmax_scale(t2m, axis=0, copy=True)
        rh = minmax_scale(rh, axis=0, copy=True)
        prcp = minmax_scale(prcp, axis=0, copy=True)
              
        # Input datas append (7305,4)
        tmp = np.concatenate((ws, t2m, rh, prcp), axis=1)
        
        # Label Log Scaling
        frp[frp==0] = 1
        frp = np.log(frp)
        
        # Split the data into test and train
        if exp == 1 :
          # 1.
          train_x = tmp[365*4+1:365*18+4,:]     # 2005 - 2018
          train_y = frp[365*4+1:365*18+4]
          val_x = tmp[365*18+4:,:]              # 2019 - 2020
          val_y = frp[365*18+4:]
          test_x = tmp[:365*4+1,:]              # 2001 - 2004
          test_y = frp[:365*4+1]

        if exp == 2 :
          # 2.
          tmp1 = tmp[365*8+2:,:]                # 2009-2020
          tmp2 = tmp[:365*2,:]                  # 2001-2002
          train_x = np.append(tmp1,tmp2,axis=0)
          frp1 = frp[365*8+2:]
          frp2 = frp[:365*2]
          train_y = np.append(frp1,frp2,axis=0)
          val_x = tmp[365*2:365*4+1,:]          # 2003 - 2004
          val_y = frp[365*2:365*4+1]
          test_x = tmp[365*4+1:365*8+2,:]       # 2005 - 2008
          test_y = frp[365*4+1:365*8+2]

        if exp == 3 :
          # 3.
          tmp1 = tmp[365*12+3:,:]               # 2013 - 2020
          tmp2 = tmp[:365*6+1]                  # 2001 - 2006  
          train_x = np.append(tmp1,tmp2,axis=0)
          frp1 = frp[365*12+3:]
          frp2 = frp[:365*6+1]
          train_y = np.append(frp1,frp2,axis=0)
          val_x = tmp[365*6+1:365*8+2,:]        # 2007 - 2008
          val_y = frp[365*6+1:365*8+2]
          test_x = tmp[365*8+2:365*12+3,:]      # 2009 - 2012
          test_y = frp[365*8+2:365*12+3]

        if exp == 4 :
          # 4.
          tmp1 = tmp[365*16+4:,:]               # 2017 - 2020
          tmp2 = tmp[:365*10+2,:]               # 2001 - 2010
          train_x = np.append(tmp1,tmp2,axis=0)
          frp1 = frp[365*16+4:]
          frp2 = frp[:365*10+2]
          train_y = np.append(frp1,frp2,axis=0)
          val_x = tmp[365*10+2:365*12+3,:]      # 2011 - 2012
          val_y = frp[365*10+2:365*12+3]
          test_x = tmp[365*12+3:365*16+4,:]     # 2013 - 2016
          test_y = frp[365*12+3:365*16+4]

        if exp == 5 :
          # 5.
          train_x = tmp[:365*14+3,:]            # 2001 - 2014
          train_y = frp[:365*14+3]
          val_x  = tmp[365*14+3:365*16+4,:]     # 2015 - 2016
          val_y = frp[365*14+3:365*16+4]
          test_x = tmp[365*16+4:,:]             # 2017 - 2020
          test_y = frp[365*16+4:]
    
        del ws, t2m, rh, prcp, frp, tmp
        
        #=================================================================================================
        # Set Model
        #=================================================================================================
        
        input = Input(shape=(4,),name='input')
  
        dense = Dense(64, name='dense1')(input)
        dense = BatchNormalization()(dense)
        dense = Activation('relu', name='relu1')(dense)
        dense = Dropout(0.2)(dense)
  
        dense = Dense(32, name='dense2')(dense)
        dense = BatchNormalization()(dense)
        dense = Activation('relu', name='relu2')(dense)
        dense = Dropout(0.2)(dense)
  
        dense = Dense(16, name='dense3')(dense)
        dense = BatchNormalization()(dense)
        dense = Activation('relu', name='relu3')(dense)
        dense = Dropout(0.2)(dense)
  
        output_frp = Dense(1, activation='relu',name='output_frp')(dense)
        
        
        model = Model(inputs=input, outputs=output_frp)
        
        
        model.compile(optimizer='adam',
                      loss='mean_squared_error'
                     )
        
        EPOCHS = 1000
        BATCH_SIZE = 1024
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            verbose=1,
            patience=300,
            mode='min',
            restore_best_weights=True)
  
  
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = main_dir+'output/NN/weight'+str(exp)+'/weight'+str(i)+'.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
            )
        
        baseline_history = model.fit(
            train_x,
            train_y,
            validation_data=(val_x,val_y),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stopping, model_checkpoint] 
            )
        
        #=================================================================================================
        # Test
        #=================================================================================================
        y_pred = model.predict(test_x, batch_size=BATCH_SIZE, verbose=1)
        y_pred = y_pred.reshape(-1)
            
        tr_cost = baseline_history.history['loss']
        va_cost = baseline_history.history['val_loss']
        tr_cost = np.asarray(tr_cost)
        va_cost = np.asarray(va_cost)
        tr_cost.astype('float32').tofile(main_dir+'output/NN/cost'+str(exp)+'/tr_cost'+str(i)+'gdat')
        va_cost.astype('float32').tofile(main_dir+'output/NN/cost'+str(exp)+'/va_cost'+str(i)+'gdat')
  
  
        pred[:,i] = y_pred
  
        del train_x, train_y, val_x, val_y, test_x, test_y
        del model, baseline_history, early_stopping
        gc.collect()
        K.clear_session()
        
     
  pred[np.isnan(pred)] = 0
  pred.astype('float32').tofile(main_dir+'output/NN/pred'+str(exp)+'.gdat')