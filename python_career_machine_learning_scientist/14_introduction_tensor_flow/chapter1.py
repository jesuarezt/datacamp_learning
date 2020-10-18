#Defining data as constants
#
#Throughout this course, we will use tensorflow version 2.3 and will exclusively import the submodules needed to complete each exercise. This will usually be done for you, but you will do it in this exercise by importing constant from tensorflow.
#
#After you have imported constant, you will use it to transform a numpy array, credit_numpy, into a tensorflow constant, credit_constant. This array contains feature columns from a dataset on credit card holders and is previewed in the image below. We will return to this dataset in later chapters.
#
#Note that tensorflow 2 allows you to use data as either a numpy array or a tensorflow constant object. Using a constant will ensure that any operations performed with that object are done in tensorflow. 


# Import constant from TensorFlow
from tensorflow import constant

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('The datatype is:', credit_constant.dtype)

# Print constant shape
print('The shape is:', credit_constant.shape)


#Defining variables
#
#Unlike a constant, a variable's value can be modified. This will be quite useful when we want to train a model by updating its parameters. Constants can't be used for this purpose, so variables are the natural choice.
#
#Let's try defining and working with a variable. Note that Variable(), which is used to create a variable tensor, has been imported from tensorflow and is available to use in the exercise.


# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print(B1)




#Performing element-wise multiplication
#
#Element-wise multiplication in TensorFlow is performed using two tensors with identical shapes. This is because the operation multiplies elements in corresponding positions in the two tensors. An example of an element-wise multiplication, denoted by the ⊙
#
#symbol, is shown below:
#
#[1221]⊙[3215]=[3425]
#
#In this exercise, you will perform element-wise multiplication, paying careful attention to the shape of the tensors you multiply. Note that multiply(), constant(), and ones_like() have been imported for you.


# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)


# Print the tensors C1 and C23
print('C1: {}'.format(C1.numpy()))
print('C23: {}'.format(C23.numpy()))



#Making predictions with matrix multiplication
#
#In later chapters, you will learn to train linear regression models. This process will yield a vector of parameters that can be multiplied by the input data to generate predictions. In this exercise, you will use input data, features, and a target vector, bill, which are taken from a credit card dataset we will use later in the course.
#
#features=⎡⎣⎢⎢⎢222124265737⎤⎦⎥⎥⎥
#, bill=⎡⎣⎢⎢⎢39132682861764400⎤⎦⎥⎥⎥, params=[1000150]
#
#The matrix of input data, features, contains two columns: education level and age. The target vector, bill, is the size of the credit card borrower's bill.
#
#Since we have not trained the model, you will enter a guess for the values of the parameter vector, params. You will then use matmul() to perform matrix multiplication of features by params to generate predictions, billpred, which you will compare with bill. Note that we have imported matmul() and constant().


# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())



#Summing over tensor dimensions
#
#You've been given a matrix, wealth. This contains the value of bond and stock wealth for five individuals in thousands of dollars.
#
#wealth = ⎡⎣⎢⎢⎢⎢⎢⎢117432550260010⎤⎦⎥⎥⎥⎥⎥⎥
#
#The first column corresponds to bonds and the second corresponds to stocks. Each row gives the bond and stock wealth for a single individual. Use wealth, reduce_sum(), and .numpy() to determine which statements are correct about wealth.



#<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
#array([[11, 50],
#       [ 7,  2],
#       [ 4, 60],
#       [ 3,  0],
#       [25, 10]], dtype=int32)>



In [1]: wealth
Out[1]: 
<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
array([[11, 50],
       [ 7,  2],
       [ 4, 60],
       [ 3,  0],
       [25, 10]], dtype=int32)>

In [2]: reduce_sum(wealth)
Out[2]: <tf.Tensor: shape=(), dtype=int32, numpy=172>

In [3]: reduce_sum(wealth,0)
Out[3]: <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 50, 122], dtype=int32)>

In [4]: reduce_sum(wealth,1)
Out[4]: <tf.Tensor: shape=(5,), dtype=int32, numpy=array([61,  9, 64,  3, 35], dtype=int32)>


#Combined, the 5 individuals hold $50,000 in bonds.

#
#Reshaping tensors
#
#Later in the course, you will classify images of sign language letters using a neural network. In some cases, the network will take 1-dimensional tensors as inputs, but your data will come in the form of images, which will either be either 2- or 3-dimensional tensors, depending on whether they are grayscale or color images.
#
#The figure below shows grayscale and color images of the sign language letter A. The two images have been imported for you and converted to the numpy arrays gray_tensor and color_tensor. Reshape these arrays into 1-dimensional vectors using the reshape operation, which has been imported for you from tensorflow. Note that the shape of gray_tensor is 28x28 and the shape of color_tensor is 28x28x3.



# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352, 1))


#
#Optimizing with gradients
#
#You are given a loss function, y=x2
#, which you want to minimize. You can do this by computing the slope using the GradientTape() operation at different values of x. If the slope is positive, you can decrease the loss by lowering x. If it is negative, you can decrease it by increasing x. This is how gradient descent works.


#The image shows a plot of y equals x squared. It also shows the gradient at x equals -1, x equals 0, and x equals 1.
#
#In practice, you will use a high level tensorflow operation to perform gradient descent automatically. In this exercise, however, you will compute the slope at x values of -1, 1, and 0. The following operations are available: GradientTape(), multiply(), and Variable().


def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))





#Working with image data
#
#You are given a black-and-white image of a letter, which has been encoded as a tensor, letter. You want to determine whether the letter is an X or a K. You don't have a trained neural network, but you do have a simple model, model, which can be used to classify letter.
#
#The 3x3 tensor, letter, and the 1x3 tensor, model, are available in the Python shell. You can determine whether letter is a K by multiplying letter by model, summing over the result, and then checking if it is equal to 1. As with more complicated models, such as neural networks, model is a collection of weights, arranged in a tensor.
#
#Note that the functions reshape(), matmul(), and reduce_sum() have been imported from tensorflow and are available for use.


#In [3]: letter
#Out[3]: 
#array([[1., 0., 1.],
#       [1., 1., 0.],
#       [1., 0., 1.]], dtype=float32)
#
#In [5]: model
#Out[5]: array([[ 1.,  0., -1.]], dtype=float32)
#



# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())