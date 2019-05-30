# Machine Learning Terminology

When studying any field we come across many different terms. Knowing what these terms are, what do they mean and how do they relate to each other is an essential part of building a good understanding of the subjects in this field. 


### 1. Artificial Intelligence
A field of computer science that aims to make computers achieve human-style intelligence. There are many approaches to reaching this goal, including machine learning and deep learning.
### 2. Machine Learning
A technique in which computers are trained to perform a particular task rather than by explicitly programming them.
### 3. Deep Learning
A subfield of machine learning that uses multi-layered neural networks.
### 4. Neural Network
A construct in Machine Learning inspired by the network of neurons (nerve cells) in the biological brain.
### 5. Supervised Learning
In supervised learning we know what we want to teach the computer.
### 6. UnSupervised Learning
Unsupervised learning is about letting the computer figure out what can be learned.
### 7. Features
The input(s) to the machine learning model.
### 8. Labels
The output of the machine learning model.
### 9. Examples
An input/output pair used for training.
### 10. Model
A math function or a specific representation of the neural network (if the model is based on neural networks).
### 11. Dense/Fully Connected Layer
A neural network layer in which each node within it is connected to each node in the previous layer.
### 12. Gradient Decent
An algorithm that adjusts the internal variables a bit at a time to gradually reduce the loss function.
### 13. Optimizer
A specific implementation of the gradient descent algorithm.
### 14. Learning Rate
The step size taken when adjusting values in the model for loss improvement during training. If the value is too small, it will take too many iterations to train the model. Too large, the accuracy may do down. Finding a good value often involves some trial and error, but the range is usually within 0.001 (default), and 0.1.
### 15. Batch
The set of examples used during training of the neural network.
### 16. Epoch
A full pass over the entire training dataset.
### 17. Forward Pass
The computation of output values from input.
### 18. Backward Pass (Backpropagation)
The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.
### 19. Optimizer
A way of adjusting internal values in order to reduce the loss.
### 20. Loss
The discrepancy between the desired output and the actual output.
### 21. Loss Function
A way of measuring how far off predictions are from the desired outcome. 
### 22. Flattening
The process of converting a 2d image into 1d vector.
### 23. Classification
A model that outputs a probability distribution across several categories.
### 24. Regression
A model that outputs a single value. For example, an estimate of a house’s value.
### 25. Training Set
The data used for training the neural network.
### 26. Test Set
The data used for testing the final performance of our neural network.
### 27. Validation Set
### 28. Test Set
### 29. Activation Function
### 30. Convolution
The process of applying a kernel (filter) to an image.
### 31. Convolutional Neural Network (CNN)
### 32. Pooling
The process of reducing the size of an image through down sampling.
### 33. Kernal
A matrix which is smaller than the input, used to transform the input into chunks.
### 34. Padding
Adding pixels of some value, usually 0, around the input image.
### 35. Stride
The number of pixels to slide the kernel (filter) across the image.
### 36. Down Sampling
The act of reducing the size of an image.
### 36. Training a Model
Ahe act of calculating the current loss of a model and then improving it.



## Activation Functions
### 1. ReLU
### 2. ELU
### 3. Sigmoid
### 4. Tanh
### 5. Softmax
A function that provides probabilities for each possible output class.



## Optimizer Algorithms
### 1. Adam (ADAptive with Momentum)



## Pooling Methods
### 1. Max Pooling
A pooling process in which many values are converted into a single value by taking the maximum value from among them.
### 2. Avg Pooling
A pooling process in which many values are converted into a single value by taking the average of the values.



## Loss Functions
### 1.MSE (Mean Squared Error)
A type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
### 2.Sparse Categorical Crossentropy



## Questions to ask when looking at an ML problem
1. Is it a supervised or unsupervised learning problem?
2. Is it a regresstion or classification problem?