#Hello nets!
#
#You're going to build a simple neural network to get a feeling of how quickly it is to accomplish this in Keras.
#
#You will build a network that takes two numbers as an input, passes them through a hidden layer of 10 neurons, and finally outputs a single non-constrained number.
#
#A non-constrained output can be obtained by avoiding setting an activation function in the output layer. This is useful for problems like regression, when we want our output to be able to take any non-constrained value.

# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()





#Counting parameters
#
#You've just created a neural network. But you're going to create a new one now, taking some time to think about the weights of each layer. The Keras Dense layer and the Sequential model are already loaded for you to use.
#
#This is the network you will be creating: 

# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()


#
#<script.py> output:
#    Model: "sequential_1"
#    _________________________________________________________________
#    Layer (type)                 Output Shape              Param #   
#    =================================================================
#    dense_1 (Dense)              (None, 5)                 20        
#    _________________________________________________________________
#    dense_2 (Dense)              (None, 1)                 6         
#    =================================================================
#    Total params: 26
#    Trainable params: 26
#    Non-trainable params: 0
#    _________________________________________________________________



#
#Build as shown!
#
#You will take on a final challenge before moving on to the next lesson. Build the network shown in the picture below. Prove your mastered Keras basics in no time!



from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape=(2,)))

# Add the ouput layer
model.add(Dense(1))




#Specifying a model
#
#You will build a simple regression model to predict the orbit of the meteor!
#
#Your training data consist of measurements taken at time steps from -10 minutes before the impact region to +10 minutes after. Each time step can be viewed as an X coordinate in our graph, which has an associated position Y for the meteor orbit at that time step.
#
#Note that you can view this problem as approximating a quadratic function via the use of neural networks.
#
#
#This data is stored in two numpy arrays: one called time_steps , what we call features, and another called y_positions, with the labels. Go on and build your model! It should be able to predict the y positions for the meteor orbit at future time steps.
#
#Keras Sequential model and Dense layers are available for you to use.


# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))



#Training
#
#You're going to train your first model in this course, and for a good cause!
#
#Remember that before training your Keras models you need to compile them. This can be done with the .compile() method. The .compile() method takes arguments such as the optimizer, used for weight updating, and the loss function, which is what we want to minimize. Training your model is as easy as calling the .fit() method, passing on the features, labels and a number of epochs to train for.
#
#The regression model you built in the previous exercise is loaded for you to use, along with the time_steps and y_positions data. Train it and evaluate it on this very same data, let's see if your model can learn the meteor's trajectory.

# Compile your model
model.compile(optimizer = 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps,y_positions, epochs = 30)

# Evaluate your model 
print("Final loss value:",model.evaluate(time_steps,y_positions))

#
#<script.py> output:
#    Training started..., this can take a while:
#    Epoch 1/30
    #
#      32/2000 [..............................] - ETA: 13s - loss: 2465.2439
#     992/2000 [=============>................] - ETA: 0s - loss: 1796.7567 
#    1984/2000 [============================>.] - ETA: 0s - loss: 1376.7267
#    2000/2000 [==============================] - 0s 166us/step - loss: 1368.0267
#    Epoch 2/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 513.5734
#    1024/2000 [==============>...............] - ETA: 0s - loss: 249.6394
#    1952/2000 [============================>.] - ETA: 0s - loss: 200.5413
#    2000/2000 [==============================] - 0s 55us/step - loss: 198.6538
#    Epoch 3/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 173.3387
#     736/2000 [==========>...................] - ETA: 0s - loss: 140.0724
#    1632/2000 [=======================>......] - ETA: 0s - loss: 135.9449
#    2000/2000 [==============================] - 0s 62us/step - loss: 131.8093
#    Epoch 4/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 135.7519
#     992/2000 [=============>................] - ETA: 0s - loss: 118.9052
#    1952/2000 [============================>.] - ETA: 0s - loss: 114.5442
#    2000/2000 [==============================] - 0s 54us/step - loss: 113.8239
#    Epoch 5/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 143.1160
#     960/2000 [=============>................] - ETA: 0s - loss: 100.4160
#    1856/2000 [==========================>...] - ETA: 0s - loss: 93.5010 
#    2000/2000 [==============================] - 0s 56us/step - loss: 92.2799
#    Epoch 6/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 85.0505
#     992/2000 [=============>................] - ETA: 0s - loss: 73.7383
#    1952/2000 [============================>.] - ETA: 0s - loss: 68.6156
#    2000/2000 [==============================] - 0s 54us/step - loss: 68.3369
#    Epoch 7/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 52.2588
#     992/2000 [=============>................] - ETA: 0s - loss: 51.9828
#    1888/2000 [===========================>..] - ETA: 0s - loss: 45.9222
#    2000/2000 [==============================] - 0s 58us/step - loss: 45.5017
#    Epoch 8/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 26.7489
#     704/2000 [=========>....................] - ETA: 0s - loss: 31.7832
#    1344/2000 [===================>..........] - ETA: 0s - loss: 30.1516
#    1760/2000 [=========================>....] - ETA: 0s - loss: 28.9274
#    2000/2000 [==============================] - 0s 95us/step - loss: 27.8420
#    Epoch 9/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 20.0875
#     800/2000 [===========>..................] - ETA: 0s - loss: 20.7555
#    1760/2000 [=========================>....] - ETA: 0s - loss: 17.7102
#    2000/2000 [==============================] - 0s 63us/step - loss: 17.4158
#    Epoch 10/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 9.9479
#     768/2000 [==========>...................] - ETA: 0s - loss: 12.4351
#    1536/2000 [======================>.......] - ETA: 0s - loss: 12.0094
#    2000/2000 [==============================] - 0s 65us/step - loss: 11.3851
#    Epoch 11/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 11.1571
#     992/2000 [=============>................] - ETA: 0s - loss: 8.2785 
#    1888/2000 [===========================>..] - ETA: 0s - loss: 7.3622
#    2000/2000 [==============================] - 0s 56us/step - loss: 7.2779
#    Epoch 12/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 6.5058
#     960/2000 [=============>................] - ETA: 0s - loss: 5.2942
#    1728/2000 [========================>.....] - ETA: 0s - loss: 5.3385
#    2000/2000 [==============================] - 0s 61us/step - loss: 5.2924
#    Epoch 13/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 4.1943
#     992/2000 [=============>................] - ETA: 0s - loss: 4.3302
#    1888/2000 [===========================>..] - ETA: 0s - loss: 3.7979
#    2000/2000 [==============================] - 0s 56us/step - loss: 3.6720
#    Epoch 14/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 2.9375
#     992/2000 [=============>................] - ETA: 0s - loss: 2.8813
#    1920/2000 [===========================>..] - ETA: 0s - loss: 2.6264
#    2000/2000 [==============================] - 0s 54us/step - loss: 2.5767
#    Epoch 15/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 2.9971
#     992/2000 [=============>................] - ETA: 0s - loss: 1.9842
#    1952/2000 [============================>.] - ETA: 0s - loss: 1.8787
#    2000/2000 [==============================] - 0s 54us/step - loss: 1.9011
#    Epoch 16/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 2.0429
#     896/2000 [============>.................] - ETA: 0s - loss: 1.5418
#    1248/2000 [=================>............] - ETA: 0s - loss: 1.4281
#    1664/2000 [=======================>......] - ETA: 0s - loss: 1.4436
#    2000/2000 [==============================] - 0s 87us/step - loss: 1.4887
#    Epoch 17/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 1.0549
#     640/2000 [========>.....................] - ETA: 0s - loss: 1.3947
#    1312/2000 [==================>...........] - ETA: 0s - loss: 1.3325
#    2000/2000 [==============================] - 0s 72us/step - loss: 1.3266
#    Epoch 18/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 1.6148
#     960/2000 [=============>................] - ETA: 0s - loss: 1.1735
#    1792/2000 [=========================>....] - ETA: 0s - loss: 1.0514
#    2000/2000 [==============================] - 0s 57us/step - loss: 1.0483
#    Epoch 19/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.1750
#     960/2000 [=============>................] - ETA: 0s - loss: 0.7825
#    1888/2000 [===========================>..] - ETA: 0s - loss: 0.7486
#    2000/2000 [==============================] - 0s 55us/step - loss: 0.7410
#    Epoch 20/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.4081
#     992/2000 [=============>................] - ETA: 0s - loss: 0.6941
#    1920/2000 [===========================>..] - ETA: 0s - loss: 0.6669
#    2000/2000 [==============================] - 0s 54us/step - loss: 0.6527
#    Epoch 21/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.3950
#     960/2000 [=============>................] - ETA: 0s - loss: 0.4801
#    1856/2000 [==========================>...] - ETA: 0s - loss: 0.5146
#    2000/2000 [==============================] - 0s 57us/step - loss: 0.5059
#    Epoch 22/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 1.5598
#     960/2000 [=============>................] - ETA: 0s - loss: 0.5544
#    1856/2000 [==========================>...] - ETA: 0s - loss: 0.4975
#    2000/2000 [==============================] - 0s 56us/step - loss: 0.4829
#    Epoch 23/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.3158
#     960/2000 [=============>................] - ETA: 0s - loss: 0.4469
#    1920/2000 [===========================>..] - ETA: 0s - loss: 0.4388
#    2000/2000 [==============================] - 0s 54us/step - loss: 0.4328
#    Epoch 24/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.3297
#     960/2000 [=============>................] - ETA: 0s - loss: 0.4230
#    1824/2000 [==========================>...] - ETA: 0s - loss: 0.3677
#    2000/2000 [==============================] - 0s 56us/step - loss: 0.3642
#    Epoch 25/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.3345
#     992/2000 [=============>................] - ETA: 0s - loss: 0.3392
#    1952/2000 [============================>.] - ETA: 0s - loss: 0.3574
#    2000/2000 [==============================] - 0s 53us/step - loss: 0.3556
#    Epoch 26/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.1934
#     992/2000 [=============>................] - ETA: 0s - loss: 0.2785
#    1952/2000 [============================>.] - ETA: 0s - loss: 0.2492
#    2000/2000 [==============================] - 0s 53us/step - loss: 0.2491
#    Epoch 27/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.3330
#    1024/2000 [==============>...............] - ETA: 0s - loss: 0.2554
#    1984/2000 [============================>.] - ETA: 0s - loss: 0.2302
#    2000/2000 [==============================] - 0s 53us/step - loss: 0.2286
#    Epoch 28/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.0871
#     992/2000 [=============>................] - ETA: 0s - loss: 0.1726
#    1952/2000 [============================>.] - ETA: 0s - loss: 0.2241
#    2000/2000 [==============================] - 0s 53us/step - loss: 0.2251
#    Epoch 29/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.1072
#    1024/2000 [==============>...............] - ETA: 0s - loss: 0.2169
#    2000/2000 [==============================] - 0s 52us/step - loss: 0.1958
#    Epoch 30/30
    #
#      32/2000 [..............................] - ETA: 0s - loss: 0.1343
#     992/2000 [=============>................] - ETA: 0s - loss: 0.2094
#    1984/2000 [============================>.] - ETA: 0s - loss: 0.1803
#    2000/2000 [==============================] - 0s 53us/step - loss: 0.1807
    #
#      32/2000 [..............................] - ETA: 1s
#    2000/2000 [==============================] - 0s 33us/step
#    Final loss value: 0.12419465046259574



#Predicting the orbit!
#
#You've already trained a model that approximates the orbit of the meteor approaching Earth and it's loaded for you to use.
#
#Since you trained your model for values between -10 and 10 minutes, your model hasn't yet seen any other values for different time steps. You will now visualize how your model behaves on unseen data.
#
#If you want to check the source code of plot_orbit, paste show_code(plot_orbit) into the console.
#
#Hurry up, the Earth is running out of time!
#
#Remember np.arange(x,y) produces a range of values from x to y-1. That is the [x, y) interval.


# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit 
plot_orbit(twenty_min_orbit)


# Predict the eighty minute orbit
eighty_min_orbit = model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit 
plot_orbit(eighty_min_orbit)