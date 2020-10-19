#The sequential model in Keras
#
#In chapter 3, we used components of the keras API in tensorflow to define a neural network, but we stopped short of using its full capabilities to streamline model definition and training. In this exercise, you will use the keras sequential model API to define a neural network that can be used to classify images of sign language letters. You will also use the .summary() method to print the model's architecture, including the shape and number of parameters associated with each layer.
#
#Note that the images were reshaped from (28, 28) to (784,), so that they could be used as inputs to a dense layer. Additionally, note that keras has been imported from tensorflow for you.

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation= 'softmax'))

# Print the model architecture
print(model.summary())


<script.py> output:
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 16)                12560     
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 136       
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 36        
    =================================================================
    Total params: 12,732
    Trainable params: 12,732
    Non-trainable params: 0
    _________________________________________________________________
    None





#Compiling a sequential model
#
#In this exercise, you will work towards classifying letters from the Sign Language MNIST dataset; however, you will adopt a different network architecture than what you used in the previous exercise. There will be fewer layers, but more nodes. You will also apply dropout to prevent overfitting. Finally, you will compile the model to use the adam optimizer and the categorical_crossentropy loss. You will also use a method in keras to summarize your model's architecture. Note that keras has been imported from tensorflow for you and a sequential keras model has been defined as model.


# Define the first dense layer
model.add(keras.layers.Dense(16, activation= 'sigmoid', input_shape = (784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation= 'softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())


<script.py> output:
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 16)                12560     
    _________________________________________________________________
    dropout (Dropout)            (None, 16)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 68        
    =================================================================
    Total params: 12,628
    Trainable params: 12,628
    Non-trainable params: 0
    _________________________________________________________________
    None




#Defining a multiple input model
#
#In some cases, the sequential API will not be sufficiently flexible to accommodate your desired model architecture and you will need to use the functional API instead. If, for instance, you want to train two models with different architectures jointly, you will need to use the functional API to do this. In this exercise, we will see how to do this. We will also use the .summary() method to examine the joint model's architecture.
#
#Note that keras has been imported from tensorflow for you. Additionally, the input layers of the first and second models have been defined as m1_inputs and m2_inputs, respectively. Note that the two models have the same architecture, but one of them uses a sigmoid activation in the first layer and the other uses a relu.


# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())


 

<script.py> output:
    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 784)]        0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, 784)]        0                                            
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 12)           9420        input_1[0][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 12)           9420        input_2[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 4)            52          dense[0][0]                      
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 4)            52          dense_2[0][0]                    
    __________________________________________________________________________________________________
    add (Add)                       (None, 4)            0           dense_1[0][0]                    
                                                                     dense_3[0][0]                    
    ==================================================================================================
    Total params: 18,944
    Trainable params: 18,944
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None


#Training with Keras
#
#In this exercise, we return to our sign language letter classification problem. We have 2000 images of four letters--A, B, C, and D--and we want to classify them with a high level of accuracy. We will complete all parts of the problem, including the model definition, compilation, and training.
#
#Note that keras has been imported from tensorflow for you. Additionally, the features are available as sign_language_features and the targets are available as sign_language_labels.

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)


<script.py> output:
    Epoch 1/5
    
 1/32 [..............................] - ETA: 0s - loss: 2.1080
28/32 [=========================>....] - ETA: 0s - loss: 1.4180
32/32 [==============================] - 0s 2ms/step - loss: 1.4111
    Epoch 2/5
    
 1/32 [..............................] - ETA: 0s - loss: 1.3885
30/32 [===========================>..] - ETA: 0s - loss: 1.3111
32/32 [==============================] - 0s 2ms/step - loss: 1.3103
    Epoch 3/5
    
 1/32 [..............................] - ETA: 0s - loss: 1.3164
26/32 [=======================>......] - ETA: 0s - loss: 1.2532
32/32 [==============================] - 0s 2ms/step - loss: 1.2468
    Epoch 4/5
    
 1/32 [..............................] - ETA: 0s - loss: 1.1445
31/32 [============================>.] - ETA: 0s - loss: 1.1555
32/32 [==============================] - 0s 2ms/step - loss: 1.1554
    Epoch 5/5
    
 1/32 [..............................] - ETA: 0s - loss: 1.0753
32/32 [==============================] - ETA: 0s - loss: 1.0666
32/32 [==============================] - 0s 2ms/step - loss: 1.0666




#Metrics and validation with Keras
#
#We trained a model to predict sign language letters in the previous exercise, but it is unclear how successful we were in doing so. In this exercise, we will try to improve upon the interpretability of our results. Since we did not use a validation split, we only observed performance improvements within the training set; however, it is unclear how much of that was due to overfitting. Furthermore, since we did not supply a metric, we only saw decreases in the loss function, which do not have any clear interpretation.
#
#Note that keras has been imported for you from tensorflow.



# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation= 'sigmoid', input_shape = (784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)



<script.py> output:
    Epoch 1/10
    
 1/29 [>.............................] - ETA: 0s - loss: 1.4231 - accuracy: 0.3125
29/29 [==============================] - 0s 16ms/step - loss: 1.2651 - accuracy: 0.4171 - val_loss: 1.2242 - val_accuracy: 0.2900
    Epoch 2/10
    
 1/29 [>.............................] - ETA: 0s - loss: 1.3104 - accuracy: 0.2500
29/29 [==============================] - 0s 2ms/step - loss: 0.9925 - accuracy: 0.6841 - val_loss: 1.0445 - val_accuracy: 0.5900
    Epoch 3/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.9369 - accuracy: 0.7812
29/29 [==============================] - 0s 2ms/step - loss: 0.8422 - accuracy: 0.7709 - val_loss: 0.8102 - val_accuracy: 0.7000
    Epoch 4/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.8435 - accuracy: 0.7188
29/29 [==============================] - 0s 2ms/step - loss: 0.7009 - accuracy: 0.8098 - val_loss: 0.7993 - val_accuracy: 0.6900
    Epoch 5/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.7524 - accuracy: 0.7812
29/29 [==============================] - 0s 2ms/step - loss: 0.6137 - accuracy: 0.8632 - val_loss: 0.6350 - val_accuracy: 0.7300
    Epoch 6/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.7238 - accuracy: 0.7188
29/29 [==============================] - 0s 2ms/step - loss: 0.5383 - accuracy: 0.8821 - val_loss: 0.7719 - val_accuracy: 0.5900
    Epoch 7/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.8442 - accuracy: 0.5000
29/29 [==============================] - 0s 2ms/step - loss: 0.4764 - accuracy: 0.9077 - val_loss: 0.4521 - val_accuracy: 0.9800
    Epoch 8/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.4912 - accuracy: 0.9375
29/29 [==============================] - 0s 2ms/step - loss: 0.4262 - accuracy: 0.9244 - val_loss: 0.5926 - val_accuracy: 0.7100
    Epoch 9/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.4262 - accuracy: 0.8125
29/29 [==============================] - 0s 2ms/step - loss: 0.3777 - accuracy: 0.9433 - val_loss: 0.3892 - val_accuracy: 0.9400
    Epoch 10/10
    
 1/29 [>.............................] - ETA: 0s - loss: 0.3319 - accuracy: 1.0000
29/29 [==============================] - 0s 2ms/step - loss: 0.3376 - accuracy: 0.9511 - val_loss: 0.3641 - val_accuracy: 0.8900





#Overfitting detection
#
#In this exercise, we'll work with a small subset of the examples from the original sign language letters dataset. A small sample, coupled with a heavily-parameterized model, will generally lead to overfitting. This means that your model will simply memorize the class of each example, rather than identifying features that generalize to many examples.
#
#You will detect overfitting by checking whether the validation sample loss is substantially higher than the training sample loss and whether it increases with further training. With a small sample and a high learning rate, the model will struggle to converge on an optimum. You will set a low learning rate for the optimizer, which will make it easier to identify overfitting.
#
#Note that keras has been imported from tensorflow.


# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation= 'relu', input_shape = (784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)




<script.py> output:
    Epoch 1/50
    
1/1 [==============================] - ETA: 0s - loss: 1.3461 - accuracy: 0.3077
1/1 [==============================] - 0s 442ms/step - loss: 1.3461 - accuracy: 0.3077 - val_loss: 3.0842 - val_accuracy: 0.3846
    Epoch 2/50
    
1/1 [==============================] - ETA: 0s - loss: 2.4663 - accuracy: 0.2308
1/1 [==============================] - 0s 23ms/step - loss: 2.4663 - accuracy: 0.2308 - val_loss: 4.2784 - val_accuracy: 0.3846
    Epoch 3/50
    
1/1 [==============================] - ETA: 0s - loss: 2.1878 - accuracy: 0.6154
1/1 [==============================] - 0s 21ms/step - loss: 2.1878 - accuracy: 0.6154 - val_loss: 5.6272 - val_accuracy: 0.3077
    Epoch 4/50
    
1/1 [==============================] - ETA: 0s - loss: 3.6705 - accuracy: 0.3846
1/1 [==============================] - 0s 15ms/step - loss: 3.6705 - accuracy: 0.3846 - val_loss: 4.5968 - val_accuracy: 0.3077
    Epoch 5/50
    
1/1 [==============================] - ETA: 0s - loss: 2.5616 - accuracy: 0.6923
1/1 [==============================] - 0s 15ms/step - loss: 2.5616 - accuracy: 0.6923 - val_loss: 4.3052 - val_accuracy: 0.0769
    Epoch 6/50
    
1/1 [==============================] - ETA: 0s - loss: 2.3775 - accuracy: 0.6154
1/1 [==============================] - 0s 15ms/step - loss: 2.3775 - accuracy: 0.6154 - val_loss: 2.8193 - val_accuracy: 0.0769
    Epoch 7/50
    
1/1 [==============================] - ETA: 0s - loss: 1.5247 - accuracy: 0.6154
1/1 [==============================] - 0s 15ms/step - loss: 1.5247 - accuracy: 0.6154 - val_loss: 1.0500 - val_accuracy: 0.5385
    Epoch 8/50
    
1/1 [==============================] - ETA: 0s - loss: 0.6675 - accuracy: 0.7692
1/1 [==============================] - 0s 22ms/step - loss: 0.6675 - accuracy: 0.7692 - val_loss: 0.9406 - val_accuracy: 0.6154
    Epoch 9/50
    
1/1 [==============================] - ETA: 0s - loss: 1.2254 - accuracy: 0.5385
1/1 [==============================] - 0s 16ms/step - loss: 1.2254 - accuracy: 0.5385 - val_loss: 1.1072 - val_accuracy: 0.4615
    Epoch 10/50
    
1/1 [==============================] - ETA: 0s - loss: 1.6209 - accuracy: 0.5385
1/1 [==============================] - 0s 18ms/step - loss: 1.6209 - accuracy: 0.5385 - val_loss: 0.9477 - val_accuracy: 0.6923
    Epoch 11/50
    
1/1 [==============================] - ETA: 0s - loss: 1.3329 - accuracy: 0.5385
1/1 [==============================] - 0s 14ms/step - loss: 1.3329 - accuracy: 0.5385 - val_loss: 0.8459 - val_accuracy: 0.6923
    Epoch 12/50
    
1/1 [==============================] - ETA: 0s - loss: 0.8220 - accuracy: 0.6154
1/1 [==============================] - 0s 14ms/step - loss: 0.8220 - accuracy: 0.6154 - val_loss: 1.0359 - val_accuracy: 0.5385
    Epoch 13/50
    
1/1 [==============================] - ETA: 0s - loss: 0.5389 - accuracy: 0.8462
1/1 [==============================] - 0s 16ms/step - loss: 0.5389 - accuracy: 0.8462 - val_loss: 1.5257 - val_accuracy: 0.3846
    Epoch 14/50
    
1/1 [==============================] - ETA: 0s - loss: 0.6948 - accuracy: 0.6923
1/1 [==============================] - 0s 16ms/step - loss: 0.6948 - accuracy: 0.6923 - val_loss: 2.0301 - val_accuracy: 0.3846
    Epoch 15/50
    
1/1 [==============================] - ETA: 0s - loss: 0.9550 - accuracy: 0.6923
1/1 [==============================] - 0s 15ms/step - loss: 0.9550 - accuracy: 0.6923 - val_loss: 2.0480 - val_accuracy: 0.3846
    Epoch 16/50
    
1/1 [==============================] - ETA: 0s - loss: 0.9436 - accuracy: 0.6923
1/1 [==============================] - 0s 15ms/step - loss: 0.9436 - accuracy: 0.6923 - val_loss: 1.6047 - val_accuracy: 0.3846
    Epoch 17/50
    
1/1 [==============================] - ETA: 0s - loss: 0.6777 - accuracy: 0.6923
1/1 [==============================] - 0s 18ms/step - loss: 0.6777 - accuracy: 0.6923 - val_loss: 1.1002 - val_accuracy: 0.4615
    Epoch 18/50
    
1/1 [==============================] - ETA: 0s - loss: 0.4211 - accuracy: 0.8462
1/1 [==============================] - 0s 14ms/step - loss: 0.4211 - accuracy: 0.8462 - val_loss: 0.8775 - val_accuracy: 0.6923
    Epoch 19/50
    
1/1 [==============================] - ETA: 0s - loss: 0.4486 - accuracy: 0.8462
1/1 [==============================] - 0s 15ms/step - loss: 0.4486 - accuracy: 0.8462 - val_loss: 0.8258 - val_accuracy: 0.6923
    Epoch 20/50
    
1/1 [==============================] - ETA: 0s - loss: 0.6064 - accuracy: 0.6154
1/1 [==============================] - 0s 15ms/step - loss: 0.6064 - accuracy: 0.6154 - val_loss: 0.7554 - val_accuracy: 0.6923
    Epoch 21/50
    
1/1 [==============================] - ETA: 0s - loss: 0.6040 - accuracy: 0.6154
1/1 [==============================] - 0s 14ms/step - loss: 0.6040 - accuracy: 0.6154 - val_loss: 0.6710 - val_accuracy: 0.7692
    Epoch 22/50
    
1/1 [==============================] - ETA: 0s - loss: 0.4510 - accuracy: 0.7692
1/1 [==============================] - 0s 14ms/step - loss: 0.4510 - accuracy: 0.7692 - val_loss: 0.7021 - val_accuracy: 0.7692
    Epoch 23/50
    
1/1 [==============================] - ETA: 0s - loss: 0.3448 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.3448 - accuracy: 1.0000 - val_loss: 0.8841 - val_accuracy: 0.6154
    Epoch 24/50
    
1/1 [==============================] - ETA: 0s - loss: 0.3646 - accuracy: 1.0000
1/1 [==============================] - 0s 14ms/step - loss: 0.3646 - accuracy: 1.0000 - val_loss: 1.0934 - val_accuracy: 0.4615
    Epoch 25/50
    
1/1 [==============================] - ETA: 0s - loss: 0.4279 - accuracy: 0.7692
1/1 [==============================] - 0s 14ms/step - loss: 0.4279 - accuracy: 0.7692 - val_loss: 1.1553 - val_accuracy: 0.4615
    Epoch 26/50
    
1/1 [==============================] - ETA: 0s - loss: 0.4289 - accuracy: 0.6923
1/1 [==============================] - 0s 14ms/step - loss: 0.4289 - accuracy: 0.6923 - val_loss: 1.0227 - val_accuracy: 0.5385
    Epoch 27/50
    
1/1 [==============================] - ETA: 0s - loss: 0.3458 - accuracy: 0.6923
1/1 [==============================] - 0s 14ms/step - loss: 0.3458 - accuracy: 0.6923 - val_loss: 0.8419 - val_accuracy: 0.6923
    Epoch 28/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2693 - accuracy: 0.9231
1/1 [==============================] - 0s 14ms/step - loss: 0.2693 - accuracy: 0.9231 - val_loss: 0.7599 - val_accuracy: 0.6923
    Epoch 29/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2741 - accuracy: 0.9231
1/1 [==============================] - 0s 14ms/step - loss: 0.2741 - accuracy: 0.9231 - val_loss: 0.7528 - val_accuracy: 0.6923
    Epoch 30/50
    
1/1 [==============================] - ETA: 0s - loss: 0.3189 - accuracy: 0.8462
1/1 [==============================] - 0s 16ms/step - loss: 0.3189 - accuracy: 0.8462 - val_loss: 0.7428 - val_accuracy: 0.7692
    Epoch 31/50
    
1/1 [==============================] - ETA: 0s - loss: 0.3236 - accuracy: 0.8462
1/1 [==============================] - 0s 15ms/step - loss: 0.3236 - accuracy: 0.8462 - val_loss: 0.7158 - val_accuracy: 0.6923
    Epoch 32/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2771 - accuracy: 0.9231
1/1 [==============================] - 0s 15ms/step - loss: 0.2771 - accuracy: 0.9231 - val_loss: 0.7075 - val_accuracy: 0.6923
    Epoch 33/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2321 - accuracy: 1.0000
1/1 [==============================] - 0s 16ms/step - loss: 0.2321 - accuracy: 1.0000 - val_loss: 0.7444 - val_accuracy: 0.6923
    Epoch 34/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2246 - accuracy: 1.0000
1/1 [==============================] - 0s 24ms/step - loss: 0.2246 - accuracy: 1.0000 - val_loss: 0.8045 - val_accuracy: 0.6154
    Epoch 35/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2386 - accuracy: 1.0000
1/1 [==============================] - 0s 25ms/step - loss: 0.2386 - accuracy: 1.0000 - val_loss: 0.8353 - val_accuracy: 0.5385
    Epoch 36/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2408 - accuracy: 1.0000
1/1 [==============================] - 0s 27ms/step - loss: 0.2408 - accuracy: 1.0000 - val_loss: 0.8052 - val_accuracy: 0.5385
    Epoch 37/50
    
1/1 [==============================] - ETA: 0s - loss: 0.2196 - accuracy: 1.0000
1/1 [==============================] - 0s 20ms/step - loss: 0.2196 - accuracy: 1.0000 - val_loss: 0.7360 - val_accuracy: 0.6923
    Epoch 38/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1944 - accuracy: 1.0000
1/1 [==============================] - 0s 23ms/step - loss: 0.1944 - accuracy: 1.0000 - val_loss: 0.6752 - val_accuracy: 0.6923
    Epoch 39/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1877 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.1877 - accuracy: 1.0000 - val_loss: 0.6398 - val_accuracy: 0.7692
    Epoch 40/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1952 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.1952 - accuracy: 1.0000 - val_loss: 0.6147 - val_accuracy: 0.7692
    Epoch 41/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1948 - accuracy: 1.0000
1/1 [==============================] - 0s 14ms/step - loss: 0.1948 - accuracy: 1.0000 - val_loss: 0.5950 - val_accuracy: 0.7692
    Epoch 42/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1787 - accuracy: 1.0000
1/1 [==============================] - 0s 14ms/step - loss: 0.1787 - accuracy: 1.0000 - val_loss: 0.5959 - val_accuracy: 0.7692
    Epoch 43/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1610 - accuracy: 1.0000
1/1 [==============================] - 0s 23ms/step - loss: 0.1610 - accuracy: 1.0000 - val_loss: 0.6252 - val_accuracy: 0.7692
    Epoch 44/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1552 - accuracy: 1.0000
1/1 [==============================] - 0s 21ms/step - loss: 0.1552 - accuracy: 1.0000 - val_loss: 0.6674 - val_accuracy: 0.7692
    Epoch 45/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1578 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.1578 - accuracy: 1.0000 - val_loss: 0.6989 - val_accuracy: 0.7692
    Epoch 46/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1573 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.1573 - accuracy: 1.0000 - val_loss: 0.7078 - val_accuracy: 0.6923
    Epoch 47/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1495 - accuracy: 1.0000
1/1 [==============================] - 0s 15ms/step - loss: 0.1495 - accuracy: 1.0000 - val_loss: 0.6976 - val_accuracy: 0.6923
    Epoch 48/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1394 - accuracy: 1.0000
1/1 [==============================] - 0s 14ms/step - loss: 0.1394 - accuracy: 1.0000 - val_loss: 0.6791 - val_accuracy: 0.7692
    Epoch 49/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1334 - accuracy: 1.0000
1/1 [==============================] - 0s 16ms/step - loss: 0.1334 - accuracy: 1.0000 - val_loss: 0.6602 - val_accuracy: 0.6923
    Epoch 50/50
    
1/1 [==============================] - ETA: 0s - loss: 0.1323 - accuracy: 1.0000
1/1 [==============================] - 0s 23ms/step - loss: 0.1323 - accuracy: 1.0000 - val_loss: 0.6412 - val_accuracy: 0.6923



# Excellent work! You may have noticed that the validation loss, val_loss, was substantially higher than the training loss, loss. Furthermore, if val_loss started to increase before the training process was terminated, then we may have overfitted. When this happens, you will want to try decreasing the number of epochs.




#Evaluating models
#
#Two models have been trained and are available: large_model, which has many parameters; and small_model, which has fewer parameters. Both models have been trained using train_features and train_labels, which are available to you. A separate test set, which consists of test_features and test_labels, is also available.
#
#Your goal is to evaluate relative model performance and also determine whether either model exhibits signs of overfitting. You will do this by evaluating large_model and small_model on both the train and test sets. For each model, you can do this by applying the .evaluate(x, y) method to compute the loss for features x and labels y. You will then compare the four losses generated.


# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))


 Small - Train: 0.16981548070907593, Test: 0.2848725914955139
    Large - Train: 0.03957207128405571, Test: 0.14543527364730835

#
#    Great job! Notice that the gap between the test and train set losses is high for large_model, suggesting that overfitting may be an issue. Furthermore, both test and train set performance is better for large_model. This suggests that we may want to use large_model, but reduce the number of training epochs.




#Preparing to train with Estimators
#
#For this exercise, we'll return to the King County housing transaction dataset from chapter 2. We will again develop and train a machine learning model to predict house prices; however, this time, we'll do it using the estimator API.
#
#Rather than completing everything in one step, we'll break this procedure down into parts. We'll begin by defining the feature columns and loading the data. In the next exercise, we'll define and train a premade estimator. Note that feature_column has been imported for you from tensorflow. Additionally, numpy has been imported as np, and the Kings County housing dataset is available as a pandas DataFrame: housing.


# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]
housing.columns
def input_fn():
	# Define the labels
	labels = np.array(housing.price)
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels



#Defining Estimators
#
#In the previous exercise, you defined a list of feature columns, feature_list, and a data input function, input_fn(). In this exercise, you will build on that work by defining an estimator that makes use of input data.

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)


# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)