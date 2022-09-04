# CNN-Digit-Image-Processing
## Introduction
This is a Convolutional Neural Network for digits recognition trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend). Firstly, I will prepare the data (handwritten digits images) then i will focus on the CNN modeling and evaluation.


# I. Import lib
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import MaxPool2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from typing import Union, cast, List
import math
```

```
def show(X: Union[pd.DataFrame, np.ndarray], indexes: Union[List[int], range], 
         y: Union[pd.Series, np.ndarray] = None, preds: np.ndarray = None):

    for nrows in range(1, int(math.sqrt(len(indexes)))+1):
        ncols = int(8/5 * nrows) # 8/5 column to row ratio
        if nrows * ncols > len(indexes): 
            ncols = math.ceil(len(indexes) / nrows)
            break  
    y = y.values if type(y) == pd.Series else y
    X = X.values if type(X) == pd.DataFrame else X

    assert preds is None or type(preds) == np.ndarray

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(8, 10))
    ax = ax.flatten()
    for i, index in enumerate(indexes):
        img = X[index].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        if preds is not None and y is not None:
            y = cast(np.ndarray, y)
            if len(indexes) <=25:
                title = f'label: {y[index]} pred: {preds[index]}'
            else:
                title = f'{y[index]} p:{preds[index]}'
            ax[i].set_title(title)
    for i in range(len(indexes), nrows*ncols):
        ax[i].set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
    
def getData(filename):
    csv = os.path.join('../input/mnist-data', filename)
    return pd.read_csv(csv)
```

# II. Data preparation
```
train_set = getData('train.csv')
test = getData('test.csv')

# train_set.shape (42000, 785)
Y_train = train_set["label"]
X_train = train_set.drop(labels=["label"], axis=1)  # Drop 'label' column


# free space
del train_set 
```


```
# Count value label
print(Y_train.value_counts())
```

<img width="225" alt="image" src="https://user-images.githubusercontent.com/63381043/188302552-00424bb0-32fc-49b3-b214-95dab0023eea.png">

```
# Print check for null or missing values
print(X_train.isnull().sum().any())
```
<img width="77" alt="image" src="https://user-images.githubusercontent.com/63381043/188302581-8c2965ce-9d8e-4dbc-a153-0e4c6a857efa.png">

```
pd.DataFrame(X_train.iloc[10].values.reshape(28, 28))
```
<img width="667" alt="image" src="https://user-images.githubusercontent.com/63381043/188302632-ceab6cc9-bfee-4731-915d-7a4290e54ff7.png">

```
plt.imshow(X_train.iloc[10].values.reshape(28, 28), cmap='Greys')
```
<img width="421" alt="image" src="https://user-images.githubusercontent.com/63381043/188302686-f17b1e2a-9767-4002-abe1-f3309cdcd1ef.png">

```
# show first instance of each digit
show(X_train, [Y_train.tolist().index(n) for n in range(10)])
```
<img width="702" alt="image" src="https://user-images.githubusercontent.com/63381043/188302709-b3e6b083-ed26-4231-9ddf-31db702e697e.png">

```
# Grayscale normalization from [0..255] to [0..1]
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3D (28px * 28px * 1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Label encoding one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]), 
# without it the model got a low accuracy
Y_train = to_categorical(Y_train, num_classes=10)
```
```
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)
```

# III.Create Model
```
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

adam = Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()
```
<img width="707" alt="image" src="https://user-images.githubusercontent.com/63381043/188302765-af7f580e-803d-4d2b-a3e0-50a85aca517f.png">

# IV. Training
```
history = model.fit(X_train,Y_train, batch_size=128, epochs=1)  # Turn epochs to 50 to get high accuracy
```
```
# make predictions on evaluation data
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

# show accuracy
print("Accuracy score: {}".format(accuracy_score(Y_true, Y_pred_classes)))
show(X_val, range(25), Y_true, Y_pred_classes)
```
<img width="693" alt="image" src="https://user-images.githubusercontent.com/63381043/188302807-172fc5a7-244d-4353-ac97-ee9a5482292f.png">

```
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)
# Errors are difference between predicted labels and true labels
Y_pred_classes_errors = Y_pred_classes[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]
show(X_val_errors, range(25), Y_true_errors, Y_pred_classes_errors)
```
<img width="768" alt="image" src="https://user-images.githubusercontent.com/63381043/188302827-2bb967c5-f7a3-434f-9797-7667313deb0b.png">
