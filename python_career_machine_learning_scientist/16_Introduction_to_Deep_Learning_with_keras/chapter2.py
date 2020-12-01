#Exploring dollar bills
#
#You will practice building classification models in Keras with the Banknote Authentication dataset.
#
#Your goal is to distinguish between real and fake dollar bills. In order to do this, the dataset comes with 4 features: variance,skewness,kurtosis and entropy. These features are calculated by applying mathematical operations over the dollar bill images. The labels are found in the dataframe's class column.
#
#A pandas DataFrame named banknotes is ready to use, let's do some data exploration!

# Import seaborn
import seaborn   as sns

# Use pairplot and set the hue to be our class column
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())

#A binary classification model
#
#Now that you know what the Banknote Authentication dataset looks like, we'll build a simple model to distinguish between real and fake bills.
#
#You will perform binary classification by using a single neuron as an output. The input layer will have 4 neurons since we have 4 features in our dataset. The model's output will be a value constrained between 0 and 1.
#
#We will interpret this output number as the probability of our input variables coming from a fake dollar bill, with 1 meaning we are certain it's a fake bill.


# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()

#
#<script.py> output:
#    Model: "sequential_1"
#    _________________________________________________________________
#    Layer (type)                 Output Shape              Param #   
#    =================================================================
#    dense_1 (Dense)              (None, 1)                 5         
#    =================================================================
#    Total params: 5
#    Trainable params: 5
#    Non-trainable params: 0
#    _________________________________________________________________


#
#Is this dollar bill fake ?
#
#You are now ready to train your model and check how well it performs when classifying new bills! The dataset has already been partitioned into features: X_train & X_test, and labels: y_train & y_test.


# Train your model for 20 epochs
model.fit(X_train, y_train, epochs = 20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


#<script.py> output:
    #Epoch 1/20
#    
     #32/960 [>.............................] - ETA: 4s - loss: 0.7641 - acc: 0.5000
    #960/960 [==============================] - 0s 193us/step - loss: 0.6655 - acc: 0.6531
    #Epoch 2/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.6201 - acc: 0.5938
    #896/960 [===========================>..] - ETA: 0s - loss: 0.6364 - acc: 0.6741
    #960/960 [==============================] - 0s 61us/step - loss: 0.6429 - acc: 0.6698
    #Epoch 3/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.7369 - acc: 0.5312
    #960/960 [==============================] - 0s 52us/step - loss: 0.6223 - acc: 0.6875
    #Epoch 4/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5655 - acc: 0.8125
    #960/960 [==============================] - 0s 51us/step - loss: 0.6036 - acc: 0.7208
    #Epoch 5/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5576 - acc: 0.7188
    #960/960 [==============================] - 0s 53us/step - loss: 0.5865 - acc: 0.7385
    #Epoch 6/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4129 - acc: 0.9062
    #960/960 [==============================] - 0s 48us/step - loss: 0.5707 - acc: 0.7479
    #Epoch 7/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.7094 - acc: 0.7188
    #960/960 [==============================] - 0s 47us/step - loss: 0.5561 - acc: 0.7573
    #Epoch 8/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5858 - acc: 0.6562
    #960/960 [==============================] - 0s 48us/step - loss: 0.5425 - acc: 0.7635
    #Epoch 9/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5396 - acc: 0.7812
    #960/960 [==============================] - 0s 47us/step - loss: 0.5298 - acc: 0.7677
    #Epoch 10/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4492 - acc: 0.9375
    #960/960 [==============================] - 0s 49us/step - loss: 0.5179 - acc: 0.7729
    #Epoch 11/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.3611 - acc: 0.9062
    #960/960 [==============================] - 0s 47us/step - loss: 0.5066 - acc: 0.7771
    #Epoch 12/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4393 - acc: 0.7812
    #960/960 [==============================] - 0s 40us/step - loss: 0.4960 - acc: 0.7792
    #Epoch 13/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4759 - acc: 0.7812
    #960/960 [==============================] - 0s 40us/step - loss: 0.4858 - acc: 0.7854
    #Epoch 14/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4394 - acc: 0.7812
    #960/960 [==============================] - 0s 40us/step - loss: 0.4762 - acc: 0.7896
    #Epoch 15/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5614 - acc: 0.7188
    #960/960 [==============================] - 0s 52us/step - loss: 0.4671 - acc: 0.7948
    #Epoch 16/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4566 - acc: 0.8125
    #960/960 [==============================] - 0s 56us/step - loss: 0.4583 - acc: 0.8000
    #Epoch 17/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.6000 - acc: 0.6875
    #960/960 [==============================] - 0s 54us/step - loss: 0.4499 - acc: 0.8094
    #Epoch 18/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5346 - acc: 0.7812
    #960/960 [==============================] - 0s 53us/step - loss: 0.4419 - acc: 0.8188
    #Epoch 19/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.5203 - acc: 0.7500
    #960/960 [==============================] - 0s 52us/step - loss: 0.4341 - acc: 0.8240
    #Epoch 20/20
#    
     #32/960 [>.............................] - ETA: 0s - loss: 0.4655 - acc: 0.7812
    #960/960 [==============================] - 0s 54us/step - loss: 0.4266 - acc: 0.8250
#    
     #32/412 [=>............................] - ETA: 0s
    #412/412 [==============================] - 0s 64us/step
    #Accuracy: 0.8252427167105443





#A multi-class model
#
#You're going to build a model that predicts who threw which dart only based on where that dart landed! (That is the dart's x and y coordinates on the board.)
#
#This problem is a multi-class classification problem since each dart can only be thrown by one of 4 competitors. So classes/labels are mutually exclusive, and therefore we can build a neuron with as many output as competitors and use the softmax activation function to achieve a total sum of probabilities of 1 over all competitors.
#
#Keras Sequential model and Dense layer are already loaded for you to use.

# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#Prepare your dataset
#
#In the console you can check that your labels, darts.competitor are not yet in a format to be understood by your network. They contain the names of the competitors as strings. You will first turn these competitors into unique numbers,then use the to_categorical() function from keras.utils to turn these numbers into their one-hot encoded representation.
#
#This is useful for multi-class classification problems, since there are as many output neurons as classes and for every observation in our dataset we just want one of the neurons to be activated.
#
#The dart's dataset is loaded as darts. Pandas is imported as pd. Let's prepare this dataset!

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Import to_categorical from keras utils module
from keras.utils import to_categorical

coordinates = darts.drop(['competitor'], axis=1)
# Use to_categorical on your labels
competitors = to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n',competitors)




#Training on dart throwers
#
#Your model is now ready, just as your dataset. It's time to train!
#
#The coordinates features and competitors labels you just transformed have been partitioned into coord_train,coord_test and competitors_train,competitors_test.
#
#Your model is also loaded. Feel free to visualize your training data or model.summary() in the console.
#
#Let's find out who threw which dart just by looking at the board!


# Fit your model to the training data for 200 epochs
model.fit(coord_train,competitors_train,epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)





#Softmax predictions
#
#Your recently trained model is loaded for you. This model is generalizing well!, that's why you got a high accuracy on the test set.
#
#Since you used the softmax activation function, for every input of 2 coordinates provided to your model there's an output vector of 4 numbers. Each of these numbers encodes the probability of a given dart being thrown by one of the 4 possible competitors.
#
#When computing accuracy with the model's .evaluate() method, your model takes the class with the highest probability as the prediction. np.argmax() can help you do this since it returns the index with the highest value in an array. 


# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))



#
#An irrigation machine
#
#You're going to automate the watering of farm parcels by making an intelligent irrigation machine. Multi-label classification problems differ from multi-class problems in that each observation can be labeled with zero or more classes. So classes/labels are not mutually exclusive, you could water all, none or any combination of farm parcels based on the inputs.
#
#To account for this behavior what we do is have an output layer with as many neurons as classes but this time, unlike in multi-class problems, each output neuron has a sigmoid activation function. This makes each neuron in the output layer able to output a number between 0 and 1 independently.
#
#Keras Sequential() model and Dense() layers are preloaded. It's time to build an intelligent irrigation machine!


# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with binary crossentropy loss
model.compile(optimizer='adam',
           loss = 'binary_crossentropy',
           metrics=['accuracy'])

model.summary()

#
#<script.py> output:
#    Model: "sequential_1"
#    _________________________________________________________________
#    Layer (type)                 Output Shape              Param #   
#    =================================================================
#    dense_1 (Dense)              (None, 64)                1344      
#    _________________________________________________________________
#    dense_2 (Dense)              (None, 3)                 195       
#    =================================================================
#    Total params: 1,539
#    Trainable params: 1,539
#    Non-trainable params: 0
#    _________________________________________________________________



#Training with multiple labels
#
#An output of your multi-label model could look like this: [0.76 , 0.99 , 0.66 ]. If we round up probabilities higher than 0.5, this observation will be classified as containing all 3 possible labels [1,1,1]. For this particular problem, this would mean watering all 3 parcels in your farm is the right thing to do, according to the network, given the input sensor measurements.
#
#You will now train and predict with the model you just built. sensors_train, parcels_train, sensors_test and parcels_test are already loaded for you to use.
#
#Let's see how well your intelligent machine performs!

# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100, validation_split = 0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)




#The history callback
#
#The history callback is returned by default every time you train a model with the .fit() method. To access these metrics you can access the history dictionary parameter inside the returned h_callback object with the corresponding keys.
#
#The irrigation machine model you built in the previous lesson is loaded for you to train, along with its features and labels now loaded as X_train, y_train, X_test, y_test. This time you will store the model's historycallback and use the validation_data parameter as it trains.
#
#You will plot the results stored in history with plot_accuracy() and plot_loss(), two simple matplotlib functions. You can check their code in the console by pasting show_code(plot_loss).
#
#Let's see the behind the scenes of our training!


def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()


# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs = 50,
               validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['acc'], h_callback.history['val_acc'])




#Early stopping your model
#
#The early stopping callback is useful since it allows for you to stop the model training if it no longer improves after a given number of epochs. To make use of this functionality you need to pass the callback inside a list to the model's callback parameter in the .fit() method.
#
#The model you built to detect fake dollar bills is loaded for you to train, this time with early stopping. X_train, y_train, X_test and y_test are also available for you to use.


# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                       patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test, y_test),
           callbacks= [monitor_val_acc])



#A combination of callbacks
#
#Deep learning models can take a long time to train, especially when you move to deeper architectures and bigger datasets. Saving your model every time it improves as well as stopping it when it no longer does allows you to worry less about choosing the number of epochs to train for. You can also restore a saved model anytime and resume training where you left it.
#
#The model training and validation data are available in your workspace as X_train, X_test, y_train, and y_test.
#
#Use the EarlyStopping() and the ModelCheckpoint() callbacks so that you can go eat a jar of cookies while you leave your computer to work!


# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience = 3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train,
                    epochs = 1000000000000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))