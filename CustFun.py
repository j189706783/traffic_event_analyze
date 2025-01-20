import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score
import optuna
import pickle
import joblib
import os

import tensorflow as tf
from tensorflow.keras import Sequential

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
    
def get_binary_label(data:pd.DataFrame,diff_scale=0.05):
    '''
    比較當月份各縣市event減去上月event的變化量是否 大於等於 或 當月份event * diff_scale的量
     1 => '>=' \n
     0 => '<'  \n
    -1 => nukonwn
    '''
    
    df_tmp = data[['year', 'month','county', 'event']].drop_duplicates().copy()

    lst_label = [-1]
    lst = df_tmp['event'].to_list()
    
    for i in range(1,len(lst)):

        diff = lst[i] - lst[i-1]

        if ((diff>0) & (abs(diff) >= (lst[i]*diff_scale))):
            lst_label.append(1)
        else:               
            lst_label.append(0)

    df_tmp['label'] = lst_label
    
    if 'label' in data.columns:
        data = data.drop(columns='label')

    df_tmp = pd.merge(data,df_tmp[['year', 'month','county','label']],how='left',on=['year','month','county'])

    df_tmp = df_tmp[[df_tmp.columns.to_list()[-1]]+df_tmp.columns.to_list()[:-1]]

    # df_tmp = df_tmp.loc[df_tmp['label']!=-1]

    return df_tmp

def get_three_label(data:pd.DataFrame,diff_scale=0.05):
    '''
    比較當月份各縣市event減去上月event的變化量是否 介於 當月份event * diff_scale的量
     2 = '>=' \n
     1 = 'between' \n
     0 = '<='  \n
    -1 = nukonwn
    '''
    
    df_tmp = data[['year', 'month','county', 'event']].drop_duplicates().copy()

    lst_label = [-1]
    lst = df_tmp['event'].to_list()
    
    for i in range(1,len(lst)):

        diff = lst[i] - lst[i-1]

        if ((diff>0) & (abs(diff) >= (lst[i]*diff_scale))):
            lst_label.append(2)
        elif ((diff<0) & (abs(diff) >= (lst[i]*diff_scale))):
            lst_label.append(0)
        else:
            lst_label.append(1)

    df_tmp['label'] = lst_label
    
    if 'label' in data.columns:
        data = data.drop(columns='label')

    df_tmp = pd.merge(data,df_tmp[['year', 'month','county','label']],how='left',on=['year','month','county'])

    df_tmp = df_tmp[[df_tmp.columns.to_list()[-1]]+df_tmp.columns.to_list()[:-1]]

    # df_tmp = df_tmp.loc[df_tmp['label']!=-1]

    return df_tmp

def compute_lable_count(label,is_binary=True,need_argmax=True):
    '''
    比較計算各類別的數量
    '''
    if is_binary:
        lst = [0,0]
    else:
        lst = [0,0,0]

    if need_argmax:
        for idx in np.argmax(label,axis=1):
            lst[idx]+=1
    else:
        for idx in label:
            lst[idx]+=1

    return lst

def single_row_onehot_encoding(county,county_list):

    ary = np.zeros((len(county_list)),dtype=int)
    ary[county_list.index(county)] = 1
    
    return ary

def _convert_to_rolling_windows_dataset(data,encode_cols,seq_length=12,is_binary=True,label_is_feature=False):

    X_unseq = []
    X_seq = []
    X_seq_encode = []
    y = []

    lst_countys = ['Changhua_County','Chiayi_City','Chiayi_County','Hsinchu_City','Hsinchu_County',
                   'Hualien_County','Kaohsiung_City','Keelung_City','Miaoli_County','Nantou_County',
                   'New_Taipei_City','Pingtung_County','Taichung_City','Tainan_City','Taipei_City',
                   'Taitung_County','Taoyuan_City','Yilan_County','Yunlin_County']
    
    for county,df in data.groupby('county'):

        df = df.sort_values(['year','month'])

        lst_x1 = []
        lst_x2 = []
        lst_x3 = []
        lst_y = []

        county_onthot_encoding = single_row_onehot_encoding(county,lst_countys)
        
        for k in range(seq_length,len(df)-1):

            lst_x1.append(county_onthot_encoding)
            
            if label_is_feature==True:
                lst_x2.append(df.iloc[k-seq_length:k][['label','month']].values)
                lst_x3.append(df.iloc[k-seq_length:k][encode_cols].values)
            else:
                lst_x2.append(df.iloc[k-seq_length:k][['month']].values)
                lst_x3.append(df.iloc[k-seq_length:k][encode_cols].values)
            
            if is_binary:
                lst_y.append(df.iloc[k-seq_length:k+1]['label'].values[-1])
            else:
                ary = [0,0,0]
                ary[df.iloc[k-seq_length:k+1]['label'].values[-1]]=1
                lst_y.append(ary)

        X_unseq.append(lst_x1)
        X_seq.append(lst_x2)
        X_seq_encode.append(lst_x3)
        y.append(lst_y)

    X_unseq = np.array(X_unseq)
    X_seq = np.array(X_seq)
    X_seq_encode = np.array(X_seq_encode)
    y = np.array(y)

    if label_is_feature==True:
        dic_feature_cols = {'onehot_cols':{'county':lst_countys},'seq_cols':['label','month'],'seq_encode_cols': encode_cols}
    else:
        dic_feature_cols = {'onehot_cols':{'county':lst_countys},'seq_cols':['month'],'seq_encode_cols': encode_cols}

    return X_unseq,X_seq,X_seq_encode,y,dic_feature_cols

def _split_train_test_data(X_unseq,X_seq,y,random_state=42,train_size=0.7,val_size=0.15):

    X_train_unseq = None
    X_train_seq = None
    y_train = None

    X_val_unseq = None
    X_val_seq = None
    y_val = None

    X_test_unseq = None
    X_test_seq = None
    y_test = None

    train_count = round(len(y[0])*train_size)
    val_count = round(len(y[0])*val_size)

    for i in range(0,len(X_seq)):
        if X_train_seq is None:

            X_train_unseq = X_unseq[i][:train_count]
            X_train_seq = X_seq[i][:train_count]
            y_train = y[i][:train_count]

            X_val_unseq = X_unseq[i][train_count:train_count+val_count]
            X_val_seq = X_seq[i][train_count:train_count+val_count]
            y_val = y[i][train_count:train_count+val_count]

            X_test_unseq = X_unseq[i][train_count+val_count:]
            X_test_seq = X_seq[i][train_count+val_count:]
            y_test = y[i][train_count+val_count:]

        else:

            X_train_unseq = np.append(X_train_unseq,X_unseq[i][:train_count], axis=0)
            X_train_seq = np.append(X_train_seq,X_seq[i][:train_count], axis=0)
            y_train = np.append(y_train,y[i][:train_count], axis=0)

            X_val_unseq = np.append(X_val_unseq,X_unseq[i][train_count:train_count+val_count], axis=0)
            X_val_seq = np.append(X_val_seq,X_seq[i][train_count:train_count+val_count], axis=0)
            y_val = np.append(y_val,y[i][train_count:train_count+val_count], axis=0)

            X_test_unseq = np.append(X_test_unseq,X_unseq[i][train_count+val_count:], axis=0)
            X_test_seq = np.append(X_test_seq,X_seq[i][train_count+val_count:], axis=0)
            y_test = np.append(y_test,y[i][train_count+val_count:], axis=0)

    
    rng = np.random.default_rng(seed=random_state)

    random = rng.choice(len(X_train_seq),len(X_train_seq),False)
    X_train_unseq = X_train_unseq[random]
    X_train_seq = X_train_seq[random]
    y_train = y_train[random]

    random = rng.choice(len(X_val_seq),len(X_val_seq),False)
    X_val_unseq = X_val_unseq[random]
    X_val_seq = X_val_seq[random]
    y_val = y_val[random]

    random = rng.choice(len(X_test_seq),len(X_test_seq),False)
    X_test_unseq = X_test_unseq[random]
    X_test_seq = X_test_seq[random]
    y_test = y_test[random]

    return [X_train_unseq,X_train_seq],y_train,[X_val_unseq,X_val_seq],y_val,[X_test_unseq,X_test_seq],y_test

def _minmax_transform(data,minmax_scaler = None):

    df = data.copy()

    lst_minmax_cols = df.columns.to_list()
    for col in ['label','year','month','county']:
        lst_minmax_cols.remove(col)

    if minmax_scaler is None:
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(df[lst_minmax_cols])

    df.loc[:,lst_minmax_cols] = minmax_scaler.fit_transform(df[lst_minmax_cols])

    df = df[['year','month','county','label'] + lst_minmax_cols]

    return df,minmax_scaler

def _tuningAutoEncoderOptunaOB(X_train,X_val,
                                encode_dims,
                                epochs,
                                loss_function,
                                verbose=0,
                                n_trials=20,
                                n_jobs = 1,
                                continue_train = False,
                                study_path = ""):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tf.config.experimental.list_physical_devices('GPU') 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    def hyperTuning(trial: optuna.Trial) -> float:

        dic_hyperparams = {
                'epochs':epochs,
                'encode_dims':encode_dims,
                'LSTM':trial.suggest_int('LSTM', 10,70),
                'lr':trial.suggest_float('lr', 0.00001, 0.005),
                'batch_size':trial.suggest_int('batch_size', 50,800),
             }
        
        encoder = Sequential([
            tf.keras.layers.LSTM(dic_hyperparams['LSTM'], activation='tanh',return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])),
            tf.keras.layers.LSTM(dic_hyperparams['encode_dims'], activation='tanh',return_sequences=True)
        ])
        encoder.build(input_shape=(X_train.shape[1],X_train.shape[2]))

        decoder = Sequential([
            tf.keras.layers.LSTM(dic_hyperparams['LSTM'], activation='tanh',return_sequences=True, input_shape=(X_train.shape[1],dic_hyperparams['encode_dims'])),
            tf.keras.layers.LSTM(X_train.shape[2],activation='sigmoid',return_sequences=True)
            # tf.keras.layers.LSTM(109,activation='sigmoid',return_sequences=True)
        ])
        decoder.build(input_shape=(X_train.shape[1],dic_hyperparams['encode_dims']))

        model = Sequential([encoder,decoder])

        model.compile(optimizer=tf.keras.optimizers.Adam(dic_hyperparams['lr']),
                      loss=loss_function,
                      metrics=['mape'])
        model.build(input_shape=(X_train.shape[1],X_train.shape[2]))

        history = model.fit(X_train, X_train,
                      epochs=dic_hyperparams['epochs'],batch_size=dic_hyperparams['batch_size'],verbose=verbose,
                      validation_data=(X_val, X_val))
        
        # loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        val_mape = history.history['val_mape'][-1]
        
        return val_loss,val_mape
    
    lst_directions = ["minimize","minimize"]

    if study_path != '':

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)

            if continue_train == True:

                study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])
            
                with open(study_path, "wb") as f:
                    pickle.dump(study, f)
                    print('save:',study_path)
            #else:
                # 直接回傳
        else:
            study = optuna.create_study(directions=lst_directions)

            study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])
            
            with open(study_path, "wb") as f:
                pickle.dump(study, f)
                print('save:',study_path)
    else:

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)
        else:
            study = optuna.create_study(directions=lst_directions)
                
        study = optuna.create_study(directions=lst_directions)
        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    return study

def _autoencoder(X_seq_encode,verbose=2):

    xy = None
    for x in X_seq_encode:
        if xy is None:
            xy = x
        else:
            xy = np.append(xy,x,axis=0)

    random = np.random.choice(len(xy),len(xy),False)
    xy = xy[random]
    X_train = xy[:round(len(xy)*0.6)]
    X_val = xy[len(X_train) : len(X_train) + round(len(xy)*0.2)]
    X_test = xy[len(X_train) + len(X_val):]

    loss_function = "mae"
    metrics=['mape']
    encode_dims = 30
    epochs = 100

    study = _tuningAutoEncoderOptunaOB(X_train,X_val,encode_dims,epochs,loss_function,verbose=2,n_trials=10,continue_train=False,study_path='./model/autoencoder.pkl')
    
    dic_params = study.best_trials[0].params

    # dic_params = {
    #     'batch_size':99,
    #     'LSTM':50,
    #     'lr':0.003
    #     }

    encoder = Sequential([
        tf.keras.layers.LSTM(dic_params['LSTM'], activation='tanh',return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])),
        tf.keras.layers.LSTM(encode_dims,activation='tanh',return_sequences=True)
    ])
    encoder.build(input_shape=(X_train.shape[1],X_train.shape[2]))

    decoder = Sequential([
        tf.keras.layers.LSTM(dic_params['LSTM'], activation='tanh',return_sequences=True, input_shape=(X_train.shape[1],encode_dims)),
        tf.keras.layers.LSTM(X_train.shape[2], activation='sigmoid',return_sequences=True)
        # tf.keras.layers.LSTM(109,return_sequences=True)
    ])
    decoder.build(input_shape=(X_train.shape[1],encode_dims))

    model = Sequential([encoder,decoder])

    model.compile(optimizer=tf.keras.optimizers.Adam(dic_params['lr']),
                  loss=loss_function,metrics=metrics)
    model.build(input_shape=(X_train.shape[1],X_train.shape[2]))

    _ = model.fit(X_train, X_train, epochs=epochs,batch_size=dic_params['batch_size'],verbose=verbose, validation_data=(X_val, X_val))

    print(f"[autoencoder] train's {loss_function}, {metrics[0]}",model.evaluate(X_train,X_train,verbose=0))
    print(f"[autoencoder]   val's {loss_function}, {metrics[0]}",model.evaluate(X_val,X_val,verbose=0))
    print(f"[autoencoder]  test's {loss_function}, {metrics[0]}",model.evaluate(X_test,X_test,verbose=0))

    return encoder,decoder

def get_rolling_and_seq_encode_data(data,
                                    seq_length=12,is_binary=True,label_is_feature=False,
                                    encoder_path = './encoder.pkl',decoder_path = './decoder.pkl',
                                    minmax_scaler = None):

    tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    df = data.copy()

    lst_encoder_cols = df.columns.to_list()
    for col in ['year', 'month', 'county','label']:
        lst_encoder_cols.remove(col)

    df,minmax_scaler = _minmax_transform(data,minmax_scaler)

    X_unseq,X_seq,X_seq_encode,y,dic_featrue_cols = _convert_to_rolling_windows_dataset(df,lst_encoder_cols,seq_length,is_binary,label_is_feature)

    if os.path.exists(encoder_path)==False:
        print('training model:auto-encoder')
        encoder,decoder = _autoencoder(X_seq_encode,verbose=2)
        joblib.dump(encoder, encoder_path)
        joblib.dump(decoder, decoder_path)
        
    else:
        encoder = joblib.load(encoder_path)

    lst = []
    for i,x in enumerate(X_seq_encode):
        ary = None
        for vals in x:
            if ary is None:
                ary = encoder.predict(vals.reshape(1,vals.shape[0],vals.shape[1]),verbose=0)
            else:
                ary = np.append(ary,encoder.predict(vals.reshape(1,vals.shape[0],vals.shape[1]),verbose=0),axis=0)

        lst.append(np.append(X_seq[i],ary,axis=2))

    X_seq = np.array(lst)

    return X_unseq,X_seq,y,dic_featrue_cols

def create_training_data(data,
                         seq_length=12,is_binary=True,label_is_feature=False,
                         random_state=42,
                         train_size=0.7,val_size = 0.15,
                         encoder_path = './encoder.pkl',decoder_path = './decoder.pkl',
                         minmax_scaler = None):

    X_unseq,X_seq,y,dic_featrue_cols = get_rolling_and_seq_encode_data(data,seq_length,is_binary,label_is_feature,
                                                                        encoder_path,decoder_path,
                                                                        minmax_scaler)

    X_train,y_train,X_val,y_val,X_test,y_test = _split_train_test_data(X_unseq,X_seq,y,random_state,train_size,val_size)

    return X_train,y_train,X_val,y_val,X_test,y_test,minmax_scaler,dic_featrue_cols

def _progress_callback(study, trial):  
    if study.best_trial.number == trial.number:
        print(f"[{trial.number}]Value:{trial.values},Params:{trial.params}")

def _progress_multi_callback(study, trial):
    if study.best_trials[-1].number == trial.number:
        print(f"[{trial.number}]Values:{trial.values},Params:{trial.params}")

def TuningUsageOptunaOB(X_train,y_train,
                        X_val,y_val,
                        epochs,
                        n_trials,
                        n_jobs = 1,
                        is_binary = True,
                        class_weight:dict={0:1,1:1},
                        continue_train = False,
                        study_path = "",
                        loss_weight=1,acc_weight=1,
                        auc_weight=1,recall_weight=1,
                        precision_weight=1):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tf.config.experimental.list_physical_devices('GPU') 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    def hyperTuning(trial: optuna.Trial) -> float:

        dic_hyperparams = {
                'unseq_Dense1':trial.suggest_int('unseq_Dense1', 2,15),
                'seq_LSTM':trial.suggest_int('seq_LSTM', 2,30),
                'seq_Dropout':trial.suggest_float('seq_Dropout', 0.01, 0.5),
                'merge_Dense1':trial.suggest_int('merge_Dense1', 2,15),

                'lr':trial.suggest_float('lr', 0.00001, 0.0005),
                'batch_size':trial.suggest_int('batch_size', 100,300),
             }
        
        if is_binary:
            loss_function = tf.keras.losses.binary_crossentropy
        else:
            loss_function = tf.keras.losses.categorical_crossentropy

        unseq_input = tf.keras.Input(shape=(X_train[0].shape[1]))
        unseq = tf.keras.layers.Dense(dic_hyperparams['unseq_Dense1'], activation="relu")(unseq_input)

        seq_input = tf.keras.Input(shape=(X_train[1].shape[1],X_train[1].shape[2]))
        seq = tf.keras.layers.LSTM(dic_hyperparams['seq_LSTM'], activation="tanh")(seq_input)
        seq = tf.keras.layers.Dropout(dic_hyperparams['seq_Dropout'])(seq)

        x = tf.keras.layers.Concatenate()([unseq,seq])
        x = tf.keras.layers.Dense(dic_hyperparams['merge_Dense1'], activation="relu")(x)

        if is_binary:
            output = tf.keras.layers.Dense(1,activation='sigmoid')(x)
        else:
            output = tf.keras.layers.Dense(3,activation='softmax')(x)

        model = tf.keras.Model(inputs=[unseq_input,seq_input], outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(dic_hyperparams['lr']),
                        loss=loss_function,
                        metrics=['accuracy','AUC','Recall','Precision'])
        
        _ = model.fit(X_train, y_train, epochs=epochs, batch_size=dic_hyperparams['batch_size'],verbose=0,validation_data=(X_val, y_val),
                      class_weight=class_weight)

        val_loss, val_acc,val_auc,val_recall,val_precision = model.evaluate(X_val, y_val,verbose=0)
        # val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

        val_loss = val_loss*loss_weight
        val_acc  = val_acc*acc_weight
        val_auc  = val_auc*auc_weight
        val_recall = val_recall*recall_weight
        val_precision = val_precision*precision_weight
        
        return val_loss,val_acc,val_auc,val_recall,val_precision
    
    lst_directions = ["minimize",'maximize', "maximize","maximize","maximize"]

    if study_path != '':

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)

            if continue_train == True:

                study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])
            
                with open(study_path, "wb") as f:
                    pickle.dump(study, f)
                    print('save:',study_path)
            #else:
                # 直接回傳
        else:
            study = optuna.create_study(directions=lst_directions)

            study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])
            
            with open(study_path, "wb") as f:
                pickle.dump(study, f)
                print('save:',study_path)
    else:

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)            
        else:
            study = optuna.create_study(directions=lst_directions)
                
        study = optuna.create_study(directions=lst_directions)
        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=n_jobs,show_progress_bar=True,callbacks=[_progress_multi_callback])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    return study

def train_model(X_train,y_train,X_val,y_val,X_test, y_test,
                params:dict,epochs=100,verbose=0,is_binary = True,class_weight:dict={0:1,1:1},
                show_result=True,show_history_plot=False):

    tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    if is_binary:
        loss_function = tf.keras.losses.binary_crossentropy
    else:
        loss_function = tf.keras.losses.categorical_crossentropy

    opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])

    unseq_input = tf.keras.Input(shape=(X_train[0].shape[1]))
    unseq = tf.keras.layers.Dense(params['unseq_Dense1'], activation="relu")(unseq_input)

    seq_input = tf.keras.Input(shape=(X_train[1].shape[1],X_train[1].shape[2]))
    seq = tf.keras.layers.LSTM(params['seq_LSTM'], activation="tanh")(seq_input)
    seq = tf.keras.layers.Dropout(params['seq_Dropout'])(seq)

    x = tf.keras.layers.Concatenate()([unseq,seq])
    x = tf.keras.layers.Dense(params['merge_Dense1'], activation="relu")(x)

    if is_binary:
        output = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    else:
        output = tf.keras.layers.Dense(3,activation='softmax')(x)

    model = tf.keras.Model(inputs=[unseq_input,seq_input], outputs=output)

    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy','AUC','Recall','Precision'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=params['batch_size'],verbose=verbose, validation_data=(X_val, y_val),
                        class_weight = class_weight)

    loss, acc,auc,recall,precision = model.evaluate(X_train, y_train,verbose=verbose)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    if show_result:
        print(
            f" train loss:   {loss:.4f}",  f"train accuracy:  {acc:.4f}\n",
            f"train recall: {recall:.4f}",f"train precision: {precision:.4f}\n",
            f"train auc:    {auc:.4f}",   f"train f1_score:  {f1_score:.4f}\n")

        loss, acc,auc,recall,precision = model.evaluate(X_val, y_val,verbose=verbose)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

        print(
            f" val loss:   {loss:.4f}",  f"val accuracy:  {acc:.4f}\n",
            f"val recall: {recall:.4f}",f"val precision: {precision:.4f}\n",
            f"val auc:    {auc:.4f}",   f"val f1_score:  {f1_score:.4f}\n")

        loss, acc,auc,recall,precision = model.evaluate(X_test, y_test,verbose=verbose)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        print(
            f" test loss:   {loss:.4f}",  f"test accuracy:  {acc:.4f}\n",
            f"test recall: {recall:.4f}",f"test precision: {precision:.4f}\n",
            f"test auc:    {auc:.4f}",   f"test f1_score:  {f1_score:.4f}\n")

    if show_history_plot:
        df_history = pd.DataFrame(history.history)

        df_history['f1_score'] = 2 * (df_history['precision'] * df_history['recall']) / (df_history['precision'] + df_history['recall'])
        df_history['val_f1_score'] = 2 * (df_history['val_precision'] * df_history['val_recall']) / (df_history['val_precision'] + df_history['val_recall'])
        df_history['epochs'] = df_history.index
        for m in ['loss','accuracy','recall','precision','auc','f1_score']:
            sns.lineplot(data=df_history,x='epochs',y=m,label=m)
            sns.lineplot(data=df_history,x='epochs',y=f'val_{m}',label=f'val_{m}')
            plt.show()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    return model

def _county_importance(dic,X,model,ori_preds,y_labels,ori_labels):

    dic['feature'] += ['county','county','county']
    dic['encode_feature_no']+= [-1,-1,-1]
    dic['class'] += [0,1,2]
    dic['true'] += y_labels
    dic['prediction'] += ori_labels

    x = X.copy()
    
    for i in range(1,11):
        x[0] = np.random.permutation(x[0])

        preds = model.predict(x,verbose=0)
        
        diff_preds = [p for p in np.mean(ori_preds - preds,axis=0)]
        preds_labels = compute_lable_count(np.argmax(preds,axis=1),False,False)

        dic[str(i)+'_diff']+=diff_preds
        dic[str(i)+'_prediction']+=preds_labels
        dic[str(i)+'_prediction_diff']+=[ori_labels[0] - preds_labels[0],
                                         ori_labels[1] - preds_labels[1],
                                         ori_labels[2] - preds_labels[2]]
    
    return dic

def _seq_importance(dic,X,model,label_is_fature,ori_preds,y_labels,ori_labels):
    lst_encode_random_idx = []

    j = 0
    bln = True
    for feature_idx in range(0,X[1].shape[-1]):
        if feature_idx == 0:
            dic['feature'] += ['month','month','month']
        else:
            if (label_is_fature and bln):
                dic['feature'] += ['label','label','label']
                bln = False
            else:
                dic['feature'] += [f'encode_{j}',f'encode_{j}',f'encode_{j}']
                j+=1
        
        dic['encode_feature_no']+= [-1 for _ in range(3)]
        dic['class'] += [0,1,2]
        dic['true'] += y_labels
        dic['prediction'] += ori_labels

        x = X.copy()

        for i in range(1,11):
            if j<1:
                x_tmp = x[1][:, :, feature_idx]
                shape1 = x_tmp.shape[0]
                shape2 = x_tmp.shape[1]
                x_tmp = x_tmp.reshape(-1,1)
                x_tmp = x_tmp[np.random.choice(len(x_tmp),len(x_tmp),replace=False)]
                x_tmp = x_tmp.reshape(shape1,shape2)
                x[1][:, :, feature_idx] = x_tmp
            else:
                x_tmp = x[1][:, :, feature_idx]
                shape1 = x_tmp.shape[0]
                shape2 = x_tmp.shape[1]
                x_tmp = x_tmp.reshape(-1,1)
                lst_encode_random_idx.append(np.random.choice(len(x_tmp),len(x_tmp),replace=False))
                x_tmp = x_tmp[lst_encode_random_idx[-1]]
                x_tmp = x_tmp.reshape(shape1,shape2)
                x[1][:, :, feature_idx] = x_tmp

            preds = model.predict(x,verbose=0)
            
            diff_preds = [p for p in np.mean(ori_preds - preds,axis=0)]
            preds_labels = compute_lable_count(np.argmax(preds,axis=1),False,False)

            dic[str(i)+'_diff']+=diff_preds
            dic[str(i)+'_prediction']+=preds_labels
            dic[str(i)+'_prediction_diff']+=[ori_labels[0] - preds_labels[0],
                                            ori_labels[1] - preds_labels[1],
                                            ori_labels[2] - preds_labels[2]]
    return dic,lst_encode_random_idx

def _compute_encode_feature_importance_ratio(ori_features,permutation_ori_features,diff_preds,ori_labels,preds_labels):

    ape = np.abs(ori_features - permutation_ori_features) / ori_features

    featrue_importance_ratio = np.sum(np.sum(ape,axis=0),axis=0)
    featrue_importance_ratio = featrue_importance_ratio / np.sum(featrue_importance_ratio)

    diff_preds        = np.array(diff_preds) * np.array(featrue_importance_ratio)[:, None]
    diff_preds_labels = np.array([ori_labels[i] - preds_labels [i] for i in range(len(ori_labels))]) * np.array(featrue_importance_ratio)[:, None]
    preds_labels      = np.array(preds_labels) * np.array(featrue_importance_ratio)[:, None]

    return diff_preds,preds_labels,diff_preds_labels

def _encode_feature_importance(dic,X,model,decoder,encode_fature_list,encode_random_idx,label_is_fature,ori_preds,y_labels,ori_labels):

    start_idx = 1

    if label_is_fature:
        start_idx = 2

    ori_features = decoder.predict(X[1][:, :,start_idx: ],verbose=0)
    j=0
    for feature_idx in range(start_idx,X[1].shape[-1]):
    
        for seq in range(1,11):
            x = X.copy()
            #沿用剛才encode特徵的打散順序
            x_tmp = x[1][:, :, feature_idx]
            shape1 = x_tmp.shape[0]
            shape2 = x_tmp.shape[1]
            x_tmp = x_tmp.reshape(-1,1)
            x_tmp = x_tmp[encode_random_idx[j]]
            x_tmp = x_tmp.reshape(shape1,shape2)
            x[1][:, :, feature_idx] = x_tmp

            # x[1][:, :, feature_idx] = x[1][:, :, feature_idx][encode_random_idx[j]]
            j+=1

            permutation_ori_features = decoder.predict(x[1][:, :,start_idx: ],verbose=0)

            preds = model.predict(x,verbose=0)
            sub_diff_preds = [p for p in np.mean(ori_preds - preds,axis=0)]
            sub_preds_labels = compute_lable_count(np.argmax(preds,axis=1),False,False)

            sub_diff_preds,sub_preds_labels,sub_preds_diff_labels = _compute_encode_feature_importance_ratio(ori_features,permutation_ori_features
                                                                                                             ,sub_diff_preds,ori_labels,sub_preds_labels)

            for i,vals in enumerate(sub_diff_preds):
                if seq == 1:
                    dic['feature'] += [encode_fature_list[i] for _ in range(3)]
                    dic['encode_feature_no'] += [feature_idx-start_idx for _ in range(3)]
                    dic['class'] += [0,1,2]
                    dic['true'] += y_labels
                    dic['prediction'] += ori_labels
                dic[str(seq)+'_diff']+=vals.tolist()
                dic[str(seq)+'_prediction'] += sub_preds_labels[i].tolist()
                dic[str(seq)+'_prediction_diff']+=sub_preds_diff_labels[i].tolist()

    return dic

def get_importance(data,model:tf.keras.Sequential,label_is_feature,minmax_scaler,encoder_path='./encoder.pkl',decoder_path='./decoder.pkl',data_type=0):

    '''
    data_type:\n
    0 => train\n
    1 => val\n
    2 => test\n
    '''

    df = data.copy()

    X_train,y_train,X_val,y_val,X_test,y_test,_,dic_featrue_cols = create_training_data(df,12,
                                                                                is_binary=False,
                                                                                minmax_scaler = minmax_scaler,
                                                                                encoder_path = encoder_path,
                                                                                train_size=0.6,val_size=0.2)
    
    decoder = joblib.load(decoder_path)

    if data_type==0:
        ori_preds = model.predict(X_train,verbose=0)
        y_labels = compute_lable_count(np.argmax(y_train,axis=1),False,False)
    elif data_type==1:
        ori_preds = model.predict(X_val,verbose=0)
        y_labels = compute_lable_count(np.argmax(y_val,axis=1),False,False)
    else:
        ori_preds = model.predict(X_test,verbose=0)
        y_labels = compute_lable_count(np.argmax(y_test,axis=1),False,False)

    ori_labels = compute_lable_count(np.argmax(ori_preds,axis=1),False,False)

    if data_type==0:
        X = X_train.copy()
    elif data_type==1:
        X = X_val.copy()
    else:
        X = X_test.copy()

    dic = {'feature':[],'encode_feature_no':[],'class':[],'true':[],'prediction':[]}
    for i in range(1,11):
        dic[str(i)+"_diff"] = []
        dic[str(i)+'_prediction'] = []
        dic[str(i)+'_prediction_diff'] = []

    dic = _county_importance(dic,X,model,ori_preds,y_labels,ori_labels)
    dic,lst_encode_random_idx = _seq_importance(dic,X,model,label_is_feature,ori_preds,y_labels,ori_labels)
    dic = _encode_feature_importance(dic,X,model,decoder,dic_featrue_cols['seq_encode_cols'],lst_encode_random_idx,label_is_feature,ori_preds,y_labels,ori_labels)

    df = pd.DataFrame(dic)

    return df

def get_sensitivity_rank(data,top=-1):
    df = data.copy()
    dic = {'feature':[],'sensitivity':[]}
    for f,d in df.groupby('feature'):
        d = d[[str(i)+'_diff' for i in range(1,11)]]
        sensitivity = np.mean(np.var(d.values))
        dic['feature'].append(f)
        dic['sensitivity'].append(sensitivity)
    df_sensitivity = pd.DataFrame(dic)
    df_sensitivity = df_sensitivity.sort_values('sensitivity',ascending=False)
    df_sensitivity['rank'] = [i for i in range(1,len(df_sensitivity)+1)]
    if top>0:
        df_sensitivity = df_sensitivity.head(top)

    ax = sns.barplot(data = df_sensitivity,x='sensitivity',y='feature',palette='Set2')

    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i])

    plt.show()
    return df_sensitivity
def get_importance_rank(data,title="",top=-1):
    df = data.copy()
    dic = {'feature':[],'importance':[]}
    for f,d in df.groupby('feature'):
        d = d[[str(i)+'_prediction_diff' for i in range(1,11)]]
        importance = np.sum(np.abs(d.values))
        
        dic['feature'].append(f)
        dic['importance'].append(importance)
    df_importance = pd.DataFrame(dic)
    df_importance = df_importance.sort_values('importance',ascending=False)
    df_importance['rank'] = [i for i in range(1,len(df_importance)+1)]
    if top>0:
        df_importance = df_importance.head(top)
        
    ax = sns.barplot(data = df_importance,x='importance',y='feature',palette='Set2')
    if title!="":
        ax.set_title(title)
    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i])
    plt.show()
    return df_importance

def show_sensitivity_plot(data,feature_name,show_title=True,colors = ['red','blue','orange'],label_name = ['class_0','class_1','class_2'],scale=1):
    df = data.copy()
    df = df.loc[(df['feature']==feature_name),[str(i)+'_diff' for i in range(1,11)]]  
    df = df.T
   
    df.columns = label_name
    df[label_name] = df[label_name]*scale
    for i,col in enumerate(label_name):
        sns.barplot(x=[i for i in range(1+i*10,11+i*10)],y=df[col].values,color=colors[i],label=label_name[i])

    plt.xticks([i for i in range(0,30)],["" for _ in range(0,30)])
    if show_title:
        plt.title(feature_name)
    plt.show()

def show_importance_lbl_plot(data,feature_name,show_title=True,colors = ['red','blue','orange'],label_name = ['class_0','class_1','class_2'],scale=1):
    df = data.copy()

    df = df.loc[(df['feature']==feature_name),[str(i)+'_prediction_diff' for i in range(1,11)]]
    
    for col in [str(i)+'_prediction_diff' for i in range(1,11)]:
        df[col] = df[col]

    df = df[[str(i)+'_prediction_diff' for i in range(1,11)]].T

    df.columns = label_name
    df[label_name] = df[label_name]*scale
    for i,col in enumerate(label_name):
        sns.barplot(x=[i for i in range(1+i*10,11+i*10)],y=df[col].values,color=colors[i],label=label_name[i])

    plt.xticks([i for i in range(0,31)],["" for _ in range(0,31)])
    if show_title:
        plt.title(feature_name)
    plt.show()

def _set_shap_value_table(shap_values,cols):

    lst_cols = ['seq']+['category']+cols

    data_count = len(shap_values[0])

    lst = []
    for data_idx in range(data_count):
        for category in range(3):
            lst.append([data_idx]+[category]+[np.sum(shap_values[0][data_idx][:,category])]+
                       [np.sum(shap_values[1][data_idx][:,features,category]) for features in range(31)])
        
    df = pd.DataFrame(lst,columns=lst_cols)
    df['category'] = df['category'].astype('int')
    
    return df

def get_Shap_value_rnak_bar(shap_values,cols):
    
    df = _set_shap_value_table(shap_values,cols)

    s = 'mean(|SHAP value|)'
    dic={'feature':[col for col in cols],s:[]}
    for col in cols:
        dic[s].append(np.mean(np.abs(df[[col]].values)))

    df = pd.DataFrame(data=dic).sort_values(by=s,ascending=False)
    g = sns.barplot(data=df,x=s,y='feature',palette='Set2')
    for i in range(len(g.containers)):
        g.bar_label(g.containers[i])
    plt.show()

def get_shap_value_table(shap_values,label_is_feature,countys):
    dic = {}

    lst = []
    for i in range(len(shap_values[0])):
        lst.append(pd.DataFrame(np.append([[i],[i],[i]],np.append([[0],[1],[2]],shap_values[0][i].T,axis=1),axis=1)))
    df = pd.concat(lst)
    df.columns = ['seq']+['category'] + countys
    df['seq'] = df['seq'].astype(int)
    df['category'] = df['category'].astype(int)
    dic['county'] = df
    if label_is_feature:
        cols = ['month','label']+[f'encode_{i}' for i in range(0,30)]
    else:
        cols = ['month']+[f'encode_{i}' for i in range(0,30)]

    for idx,f in enumerate(cols):
        lst = []
        for i in range(len(shap_values[1])):
            lst.append(pd.DataFrame(np.append([[i],[i],[i]],np.append([[0],[1],[2]],shap_values[1][i,:,idx].T,axis=1),axis=1)))
            df = pd.concat(lst)
            df.columns = ['seq']+['category']+[f'{i}' for i in range(12)]
            df['seq'] = df['seq'].astype(int)
            df['category'] = df['category'].astype(int)
        dic[f] = df

    return dic
