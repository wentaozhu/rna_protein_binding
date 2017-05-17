from keras.layers import Input, Dense, merge, Activation
from keras.models import Model
from keras import callbacks as cb
import keras.optimizers

inputDim = target.shape[1]

if denoise:
    trainTarget_ae = np.concatenate([source[toKeepS], target[toKeepT]], axis=0)
    np.random.shuffle(trainTarget_ae)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    encoded1 = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded1)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, nb_epoch=500, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])    
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
