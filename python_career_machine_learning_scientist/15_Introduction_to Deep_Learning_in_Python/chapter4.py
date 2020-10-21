#Changing optimization parameters
#
#It's time to get your hands dirty with optimization. You'll now try optimizing a model at a very low learning rate, a very high learning rate, and a "just right" learning rate. You'll want to look at the results after running this exercise, remembering that a low value for the loss function is good.
#
#For these exercises, we've pre-loaded the predictors and target values from your previous classification models (predicting who would survive on the Titanic). You'll want the optimization to start from scratch every time you change the learning rate, to give a fair comparison of how each learning rate did in your results. So we have created a function get_new_model() that creates an unoptimized model to optimize.


def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape = input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)


# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1.0]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    # Compile the model
    model.compile(my_optimizer, loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)



<script.py> output:
    
    
    Testing model with learning rate: 0.000001
    
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.6053
704/891 [======================>.......] - ETA: 0s - loss: 3.6753
891/891 [==============================] - 0s - loss: 3.6057     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.5751
736/891 [=======================>......] - ETA: 0s - loss: 3.5123
891/891 [==============================] - 0s - loss: 3.5656     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.6692
736/891 [=======================>......] - ETA: 0s - loss: 3.5492
891/891 [==============================] - 0s - loss: 3.5255     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.0058
704/891 [======================>.......] - ETA: 0s - loss: 3.4634
891/891 [==============================] - 0s - loss: 3.4854     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.5452
704/891 [======================>.......] - ETA: 0s - loss: 3.4019
891/891 [==============================] - 0s - loss: 3.4454     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.4446
544/891 [=================>............] - ETA: 0s - loss: 3.4580
891/891 [==============================] - 0s - loss: 3.4056     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 4.1073
736/891 [=======================>......] - ETA: 0s - loss: 3.4082
891/891 [==============================] - 0s - loss: 3.3659     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.0972
704/891 [======================>.......] - ETA: 0s - loss: 3.2714
891/891 [==============================] - 0s - loss: 3.3263     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.7464
672/891 [=====================>........] - ETA: 0s - loss: 3.2302
891/891 [==============================] - 0s - loss: 3.2867     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.3862
704/891 [======================>.......] - ETA: 0s - loss: 3.1384
891/891 [==============================] - 0s - loss: 3.2473     
    
    
    Testing model with learning rate: 0.010000
    
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 1s - loss: 1.0910
704/891 [======================>.......] - ETA: 0s - loss: 1.5968
891/891 [==============================] - 0s - loss: 1.4069     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.1145
704/891 [======================>.......] - ETA: 0s - loss: 0.7233
891/891 [==============================] - 0s - loss: 0.7036     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.5716
704/891 [======================>.......] - ETA: 0s - loss: 0.6517
891/891 [==============================] - 0s - loss: 0.6469     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6275
704/891 [======================>.......] - ETA: 0s - loss: 0.6263
891/891 [==============================] - 0s - loss: 0.6175     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.4938
704/891 [======================>.......] - ETA: 0s - loss: 0.6233
891/891 [==============================] - 0s - loss: 0.6242     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6611
704/891 [======================>.......] - ETA: 0s - loss: 0.6089
891/891 [==============================] - 0s - loss: 0.6002     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6244
704/891 [======================>.......] - ETA: 0s - loss: 0.6019
891/891 [==============================] - 0s - loss: 0.5980     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6077
704/891 [======================>.......] - ETA: 0s - loss: 0.5881
891/891 [==============================] - 0s - loss: 0.6025     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6535
704/891 [======================>.......] - ETA: 0s - loss: 0.5921
891/891 [==============================] - 0s - loss: 0.5915     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6415
704/891 [======================>.......] - ETA: 0s - loss: 0.5763
891/891 [==============================] - 0s - loss: 0.5818     
    
    
    Testing model with learning rate: 1.000000
    
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 1s - loss: 1.0273
672/891 [=====================>........] - ETA: 0s - loss: 5.6615
891/891 [==============================] - 0s - loss: 5.9885     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 4.5332
640/891 [====================>.........] - ETA: 0s - loss: 6.1954
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 7.0517
704/891 [======================>.......] - ETA: 0s - loss: 6.2961
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 6.0443
704/891 [======================>.......] - ETA: 0s - loss: 6.1588
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 9.0664
672/891 [=====================>........] - ETA: 0s - loss: 5.9483
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 6.0443
704/891 [======================>.......] - ETA: 0s - loss: 6.1817
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.0369
704/891 [======================>.......] - ETA: 0s - loss: 6.2732
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.0369
704/891 [======================>.......] - ETA: 0s - loss: 6.0672
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.5406
704/891 [======================>.......] - ETA: 0s - loss: 6.0672
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.5406
704/891 [======================>.......] - ETA: 0s - loss: 6.2046
891/891 [==============================] - 0s - loss: 6.1867     
I



#
#Evaluating model accuracy on validation dataset
#
#Now it's your turn to monitor model accuracy with a validation data set. A model definition has been provided as model. Your job is to add the code to compile it and then fit it. You'll check the validation score in each epoch.

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)




#Early stopping: Optimizing the optimization
#
#Now that you know how to monitor your model performance throughout optimization, you can use early stopping to stop optimization when it isn't helping any more. Since the optimization stops automatically when it isn't helping, you can also set a high value for epochs in your call to .fit(), as Dan showed in the video.
#
#The model you'll optimize has been specified as model. As before, the data is pre-loaded as predictors and target


# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
hist = model.fit(predictors, target,
 validation_split=0.3, epochs=30,
 callbacks=[early_stopping_monitor] )





#Experimenting with wider networks
#
#Now you know everything you need to begin experimenting with different models!
#
#A model called model_1 has been pre-loaded. You can see a summary of this model printed in the IPython Shell. This is a relatively small network, with only 10 units in each hidden layer.
#
#In this exercise you'll create a new model called model_2 which is similar to model_1, except it has 100 units in each hidden layer.
#
#After you create model_2, both models will be fitted, and a graph showing both models loss score at each epoch will be shown. We added the argument verbose=False in the fitting commands to print out fewer updates, since you will look at these graphically instead of as text.
#
#Because you are fitting two models, it will take a moment to see the outputs after you hit run, so be patient.


model_1.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22        
=================================================================
Total params: 242.0
Trainable params: 242
Non-trainable params: 0.0
_________________________________________________________________


# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape = input_shape))
model_2.add(Dense(100, activation='relu'))
# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


#The blue model is the one you made, the red is the original model. Your model had a lower loss value, so it is the better model. Nice job!



#Adding layers to a network
#
#You've seen how to experiment with wider networks. In this exercise, you'll try a deeper network (more hidden layers).
#
#Once again, you have a baseline model called model_1 as a starting point. It has 1 hidden layer, with 50 units. You can see a summary of that model's structure printed out. You will create a similar network with 3 hidden layers (still keeping 50 units in each layer).
#
#This will again take a moment to fit both models, so you'll need to wait a few seconds to see the results after you run your code.

model_1.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 50)                550       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 102       
=================================================================
Total params: 652.0
Trainable params: 652
Non-trainable params: 0.0



# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu', input_shape = input_shape))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()



#Building your own digit recognition model
#
#You've reached the final exercise of the course - you now know everything you need to build an accurate model to recognize handwritten digits!
#
#We've already done the basic manipulation of the MNIST dataset shown in the video, so you have X and y loaded and ready to model with. Sequential and Dense from keras are also pre-imported.
#
#To add an extra challenge, we've loaded only 2500 images, rather than 60000 which you will see in some published results. Deep learning models perform better with more data, however, they also take longer to train, especially when they start becoming more complex.
#
#If you have a computer with a CUDA compatible GPU, you can take advantage of it to improve computation time. If you don't have a GPU, no problem! You can set up a deep learning environment in the cloud that can run your models on a GPU. Here is a blog post by Dan that explains how to do this - check it out after completing this exercise! It is a great next step as you continue your deep learning journey.
#
#Ready to take your deep learning to the next level? Check out Advanced Deep Learning with Keras in Python to see how the Keras functional API lets you build domain knowledge to solve new types of problems. Once you know how to use the functional API, take a look at "Convolutional Neural Networks for Image Processing" to learn image-specific applications of Keras.



# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=20, validation_split=0.3)

