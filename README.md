## Imports      
```python  
import numpy as np  
import tensorflow as tf
```  

* `numpy`: A powerful library for numerical computations. Here, it’s used to create and manipulate the XOR input and output data arrays.      
* `tensorflow`: A deep learning framework used to build, train, and evaluate the neural network. It simplifies creating neural networks, handling data, and performing automatic differentiation during training.    

## Data for XOR problem    
```python  
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
```   
* X: This is the input data for the XOR problem. It’s a 2D array with four rows and two columns, where each row represents a different combination of binary inputs (X1, X2)    
* Y: This is the corresponding output for the XOR problem. It’s a 2D array where each element is the output of the XOR operation for the corresponding row in X.  
Both X and Y are converted to the float32 data type because TensorFlow expects inputs to be in floating-point format.    

## Defining the Neural Network Model    
```python  
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='relu'), # Hidden layers with 2 neurons
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer with 1 neuron
])
```
`tf.keras.Sequential`: This is a way of defining a neural network as a sequence of layers. In this case, the model consists of two layers: a hidden layer and an output layer.  
### Hidden layer:  
* `Dense(2)`: This is a fully connected (dense) layer with 2 neurons (units). Each neuron in this layer will be connected to both input features.  
* `input_dim=2`: This indicates that the input layer has 2 features (i.e., X1 and X2 for the XOR problem).  
* `activation='relu'`: The Rectified Linear Unit (ReLU) activation function is applied to the outputs of the 2 neurons. ReLU outputs the input directly if it’s positive, and outputs zero otherwise. This introduces non-linearity into the model, enabling it to learn complex patterns, such as XOR.  
### Output layer:  
* `Dense(1)`: This is a fully connected output layer with 1 neuron. The output is a single value because we are dealing with binary classification.  
* `activation='sigmoid'`: The Sigmoid activation function is applied to the output neuron. Sigmoid squashes the output between 0 and 1, which makes it ideal for binary classification problems like XOR. The output can be interpreted as the probability of class 1.  
  
## Compiling the model  
```python  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```  
* `optimizer='adam'`: Adam (Adaptive Moment Estimation) is an optimization algorithm that is commonly used in training neural networks. It’s an extension of Stochastic Gradient Descent (SGD) that adapts the learning rate for each parameter based on estimates of first and second moments of the gradients. It’s well-suited for most problems.  
* `loss='binary_crossentropy'`: Binary Crossentropy is the loss function used for binary classification problems. It measures the difference between the true label and the predicted probability. The goal during training is to minimize this loss function.  
* `metrics=['accuracy']`: This specifies that we want to track the accuracy of the model during training. Accuracy is the proportion of correctly classified examples.  

## Training the model  
```python  
model.fit(X, Y, epochs=5, verbose=1)
```
* `fit(X, Y)`: This function starts the training process. The input data X and output labels Y are fed into the model.  
* `epochs=5`: The model is trained for 5 epochs. One epoch means the model will go through the entire dataset once. By using 5 epochs, we allow the model to repeatedly adjust the weights to minimize the loss.  
* `verbose=1`: This shows the progress of each epoch.  

## Evaluating the model  
```python  
loss, accuracy = model.evaluate(X, Y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```
* `evaluate(X, Y)`: This function evaluates the performance of the model on the given data (in this case, the XOR problem). It returns the loss and the accuracy of the model.  
* `print(f"Model Accuracy: {accuracy * 100:.2f}%")`: This line prints the accuracy of the model as a percentage.  

## Making predictions  
```python  
predictions = model.predict(X)
print('Prediction:')
print(np.round(predictions))
```  
* `predict(X)`: This function makes predictions on the input data X. The model will output predicted probabilities for each input.  
* `np.round(predictions)`: The predicted outputs are probabilities between 0 and 1 due to the Sigmoid activation function. np.round() rounds these probabilities to 0 or 1 to match the binary outputs of the XOR problem.  

## Output  
The model will predict values close to 0 or 1 for the XOR inputs. After rounding, it should produce something similar to this:  
![alt](https://github.com/user-attachments/assets/9414ad28-32bf-42c2-9a23-c33954bcd686)    
This does not match the correct XOR outputs, so we need to increase the number of epochs to adjust the weights to minimize the loss.  
Here we set epochs to 200 and verbose to 0 (to disable the detailed output of the training progress):    
![alt](https://github.com/user-attachments/assets/be59888c-59b0-433d-b3d9-d4953787ac6e)   
This still does not match the correct XOR outputs, but the accuracy increases and the loss decreases.  
We set epochs to 500:  
![alt](https://github.com/user-attachments/assets/b810ab56-30a7-4570-a00c-52418c03b3c4)  
This matches the correct XOR outputs, demonstrating that the network has successfully learned the XOR function.  

