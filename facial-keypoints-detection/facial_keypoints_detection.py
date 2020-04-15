import pandas as pd
import numpy as np

input_csv = pd.read_csv('training.csv', delimiter=',')

input_csv = input_csv.dropna() #Drops rows which has NaN as their values

input_csv['Image'] = input_csv['Image'].apply(lambda x: np.fromstring(x, sep=' '))

X = np.vstack(input_csv['Image'].values)/255.
X = X.astype(np.float32)

X = X.reshape(-1, 96, 96, 1)
print(X.shape)

y = input_csv[input_csv.columns[:-1]].values

y = (y-48)/48 #Scale the target region to -1 to 1

from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=42)

y = y.astype(np.float32)

from sklearn.model_selection import train_test_split

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

XValid, XTest, yValid, yTest = train_test_split(XTest, yTest, test_size=0.5)

print(XTrain.shape, XValid.shape, XTest.shape)

from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model
from keras.layers import Concatenate
import keras.layers

print(XTrain[0].shape)
#model.fit(XTrain, yTrain, batch_size=128, epochs=40, validation_data=(XValid, yValid))

def create_inception_like_model(input_shape):
  X_inp = Input(input_shape)

  X_1 = Conv2D(64, 1, padding='same')(X_inp)
  X_3 = Conv2D(128, 3, padding='same')(X_inp)
  X_5 = Conv2D(64, 5, padding='same')(X_inp)
  X_7 = Conv2D(32, 7, padding='same')(X_inp)
  X_pool = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(X_inp)
  X_pool_1 = Conv2D(32, 1, padding='same')(X_pool)

  X = keras.layers.concatenate([X_1, X_3, X_5, X_7, X_pool_1])
  X = MaxPooling2D(pool_size=(3,3))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  
  X_1 = Conv2D(64, 1, padding='same')(X)
  X_3 = Conv2D(128, 3, padding='same')(X)
  X_5 = Conv2D(32, 5, padding='same')(X)
  X_7 = Conv2D(32, 7, padding='same')(X)
  X_pool = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(X)
  X_pool_1 = Conv2D(32, 1, padding='same')(X_pool)

  X = keras.layers.concatenate([X_1, X_3, X_5, X_7, X_pool_1])
  X = MaxPooling2D(pool_size=(3,3))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)

  X_1 = Conv2D(64, 1, padding='same')(X)
  X_3 = Conv2D(128, 3, padding='same')(X)
  X_5 = Conv2D(32, 5, padding='same')(X)
  X_7 = Conv2D(32, 7, padding='same')(X)
  X_pool = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(X)
  X_pool_1 = Conv2D(32, 1, padding='same')(X_pool)

  X = keras.layers.concatenate([X_1, X_3, X_5, X_7, X_pool_1])
  X = MaxPooling2D(pool_size=(2,2))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  

  X_1 = Conv2D(64, 1, padding='same')(X)
  X_3 = Conv2D(128, 3, padding='same')(X)
  X_5 = Conv2D(32, 5, padding='same')(X)
  X_7 = Conv2D(32, 7, padding='same')(X)
  X_pool = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(X)
  X_pool_1 = Conv2D(32, 1, padding='same')(X_pool)

  X = keras.layers.concatenate([X_1, X_3, X_5, X_7, X_pool_1])
  X = BatchNormalization(axis = 3)(X)
  X = MaxPooling2D(pool_size=(3,3))(X)
  X = Activation('relu')(X)

  X = Flatten()(X)
  X = Dense(100, activation='relu')(X)
  X = Dense(30, activation='sigmoid')(X)

  model = Model(inputs=X_inp, outputs=X)
  return model

model = create_inception_like_model(XTrain.shape[1:])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#print(model.summary())

X_Train = np.concatenate((XTrain, XTest, XValid))
y_Train = np.concatenate((yTrain, yTest, yValid))
print(X_Train.shape, y_Train.shape)

model.fit(X_Train, y_Train, batch_size=128, epochs=50)

model.save('model2.h5')
print("Model Saved!!")

print(model.evaluate(XTest, yTest))

test_csv = pd.read_csv('test.csv', delimiter=',')

image_id = test_csv['ImageId']

test_csv['Image'] = test_csv['Image'].apply(lambda x: np.fromstring(x, sep=' '))
X = np.vstack(test_csv['Image'].values)/255.
X = X.astype(np.float32)
X = X.reshape(-1, 96, 96, 1)
print(X.shape)

Y = model.predict(X)


Y = Y*48 + 48
Y.clip(0, 96)
'''
li = input_csv.columns[:-1]

row = []
img = []
fea = []
loc = []
count = 1

for i in range(len(Y)):
  for _ in range(len(li)):
    row.append(count)
    img.append(image_id[i])
    fea.append(li[_])
    loc.append(Y[i][_])
    count += 1

len(row)

row[0], img[0], fea[0], loc[0]

di = {'RowId': row, 'ImageId': img, 'FeatureName': fea, 'Location':loc}

df = pd.DataFrame(di) 
  
# saving the dataframe 
df.to_csv('submission.csv')
'''

lookid_data = pd.read_csv("IdLookupTable.csv")
samplesubmission = pd.read_csv("SampleSubmission.csv")

lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(Y)
rowid = lookid_data['RowId']
rowid=list(rowid)
feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))

preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])

rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('submission.csv',index = False)

