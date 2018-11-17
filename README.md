# Auto_MPG---TensorFlow
Using TensorFlow to access the features which best determine "power to weight" of cars.
The script identifies possible missing values, looks at the relations betwee the different features using their correaltions. 
Creates a new feature "power_to_weight" from the original features and uses the  to forecast the "power_to_weight" values. 
Furthermore, the performance of the model is evaluated by using the mean-squared Loss Function over each epoch.
In the forward propogration, the Sigmoid activation fuction in both layers and the standard gradient-descent algorithm is used
in the backpropogation. 
