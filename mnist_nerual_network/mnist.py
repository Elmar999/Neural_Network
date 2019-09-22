import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


train  = pd.read_csv('./train.csv' , encoding='utf-8')
test  = pd.read_csv('./test.csv')
y = train.label
x = train.drop(columns='label' , axis = 1 )


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the images.
x_train = (x_train / 255) - 0.5
x_test = (x_test / 255) - 0.5

# Flatten the images.
''' our data is already in a reshaped form . There are mnist datas which is shown in 
	a 28*28 form for each pixel . use this reshaping method in this case.
	train_images = train_images.reshape((-1, 784))
	test_images = test_images.reshape((-1, 784))
	 
'''


''' we will have input , two hidden and one output layer '''
model = Sequential([
  Dense(64, activation='relu' , input_shape = (784,)),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

'''we'll use adam gradient optimization and as we use softmax we'll need to measure loss
	with categorical_crossentropy
 '''
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  x_train, # training data
  to_categorical(y_train), # training targets . use to categorical fucntion to convert answer to binary form
  epochs=10, # number of iterations
  batch_size=32, 
)

model.evaluate(
  x_test,
  to_categorical(y_test) 
)

#save the model to use it later
model.save_weights('model.h5')

# Predict on the first 5 test images.
'''
predictions = model.predict(x_test[:5])
# Print our  predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the truths.
print(y_test[:5]) # [7, 2, 1, 0, 4]
'''

results = model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_da.csv",index=False)



