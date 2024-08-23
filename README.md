1) Imports    
![image](https://github.com/user-attachments/assets/2aeb4d8f-9825-468f-9136-62dcc68efba0)    
- numpy: A powerful library for numerical computations. Here, it’s used to create and manipulate the XOR input and output data arrays.    
- tensorflow: A deep learning framework used to build, train, and evaluate the neural network. It simplifies creating neural networks, handling data, and performing automatic differentiation during training.  

2) Data for XOR problem  
![image](https://github.com/user-attachments/assets/5d1fcb6e-7b08-4be8-ae73-bbe2cadc456b) 
- X: This is the input data for the XOR problem. It’s a 2D array with four rows and two columns, where each row represents a different combination of binary inputs (X1, X2)  
- Y: This is the corresponding output for the XOR problem. It’s a 2D array where each element is the output of the XOR operation for the corresponding row in X.
  Both X and Y are converted to the float32 data type because TensorFlow expects inputs to be in floating-point format.  

3) Defining the Neural Network Model
![image](https://github.com/user-attachments/assets/e56281af-6164-406d-a9cd-bec5c499c2b6)
tf.keras.Sequential: This is a way of defining a neural network as a sequence of layers. In this case, the model consists of two layers: a hidden layer and an output layer.
- Hidden layer:
  + Dense(2): This is a fully connected (dense) layer with 2 neurons (units). Each neuron in this layer will be connected to both input features.
  + input_dim=2: This indicates that the input layer has 2 features (i.e., X1 and X2 for the XOR problem).
  + activation='relu': The Rectified Linear Unit (ReLU) activation function is applied to the outputs of the 2 neurons. ReLU outputs the input directly if it’s positive, and outputs zero otherwise. This introduces non-linearity into the model, enabling it to learn complex patterns, such as XOR.
- Output layer:
  + Dense(1): This is a fully connected output layer with 1 neuron. The output is a single value because we are dealing with binary classification.
  + activation='sigmoid': The Sigmoid activation function is applied to the output neuron. Sigmoid squashes the output between 0 and 1, which makes it ideal for binary classification problems like XOR. The output can be interpreted as the probability of class 1.
  
4) Compiling the model  
![image](https://github.com/user-attachments/assets/db7c5e52-b42e-4d82-87a9-c1b0fc626649)
- optimizer='adam': Adam (Adaptive Moment Estimation) is an optimization algorithm that is commonly used in training neural networks. It’s an extension of Stochastic Gradient Descent (SGD) that adapts the learning rate for each parameter based on estimates of first and second moments of the gradients. It’s well-suited for most problems.
- loss='binary_crossentropy': Binary Crossentropy is the loss function used for binary classification problems. It measures the difference between the true label and the predicted probability. The goal during training is to minimize this loss function.
- metrics=['accuracy']: This specifies that we want to track the accuracy of the model during training. Accuracy is the proportion of correctly classified examples.

5) Training the model  
![image](https://github.com/user-attachments/assets/319ec018-2176-47f3-9d28-1849e11225e8)  
- fit(X, Y): This function starts the training process. The input data X and output labels Y are fed into the model.
- epochs=5: The model is trained for 5 epochs. One epoch means the model will go through the entire dataset once. By using 5 epochs, we allow the model to repeatedly adjust the weights to minimize the loss.
- verbose=1: This shows the progress of each epoch.

6) Evaluating the model  
![image](https://github.com/user-attachments/assets/d5c919bd-08af-4f75-8042-b64c89a7810e)
- evaluate(X, Y): This function evaluates the performance of the model on the given data (in this case, the XOR problem). It returns the loss and the accuracy of the model.  
- print(f"Model Accuracy: {accuracy * 100:.2f}%"): This line prints the accuracy of the model as a percentage.

7) Making predictions  
![image](https://github.com/user-attachments/assets/a6ed88b7-9e4c-4482-b40b-89047d854609)
- predict(X): This function makes predictions on the input data X. The model will output predicted probabilities for each input.
- np.round(predictions): The predicted outputs are probabilities between 0 and 1 due to the Sigmoid activation function. np.round() rounds these probabilities to 0 or 1 to match the binary outputs of the XOR problem.

8) Output  
The model will predict values close to 0 or 1 for the XOR inputs. After rounding, it should produce something similar to this:  
![Screenshot 2024-08-23 114217](https://github.com/user-attachments/assets/7c155eab-24ed-4e03-8643-84803d26728c)  
This does not match the correct XOR outputs, so we need to increase the number of epochs to adjust the weights to minimize the loss.  
Here we set epochs to 200 and verbose to 0 (to disable the detailed output of the training progress):    
![Screenshot 2024-08-23 115258](https://github.com/user-attachments/assets/ab8e3039-9f2f-4b23-b86c-a9ee2b69e5bf)  
This still does not match the correct XOR outputs, but the accuracy increases and the loss decreases.  
We set epochs to 500:  
![image](https://github.com/user-attachments/assets/dc42edec-2b34-4ed6-aaa5-fb2c19132e74)  
This matches the correct XOR outputs, demonstrating that the network has successfully learned the XOR function.

