# Data-driven RANS using Tensorflow

Please ensure that the TensorFlow C API and OpenFOAM 5 have been installed properly before following the steps in this tutorial. This may be found in the root `README.md`. In this document we will rely on the approach laid out in [arXiv:1910.10878](https://arxiv.org/pdf/1910.10878) where a deep neural network is utilized to predict steady-state turbulent viscosities of the Spalart-Allmaras (SA) model. The goal of this article is to avoid solving the one-extra equation that SA requires but replace it with a point deep-learning framework for acceleration. The training data for this experiment corresponds to inputs of initial conditions (given by potentialFoam), mesh coordinates and step height and outputs of the steady-state turbulent viscosity. We will try to train a neural network to learn $M_2$ in Equation 4 of the article. 

## Step 0: Convert OpenFOAM field output data to numpy array

You can perform this by using the `training_data_maker.py` function in `Data_generation/`. It reads in OpenFOAM standard output formal fields (depending on if they are a vector or scalar) and stacks them into a numpy array. The data may be extracted from `Data_generation/OF_Data.zip` within the same directory. Following this, the execution of the code generates a `Total_dataset.npy` which can be copied into the `Training/` directory. 

## Step 1: Train and export a model in Python

While the C API of TensorFlow can be used to train models, it is far easier to use the Python API for training and porting a model to OpenFOAM using an immutable set of trained model parameters. From the aforementioned step, grab the training data `Total_dataset.npy` and place it in the `Training/` directory. Our model is defined in the `get_model` function of `ML_Model.py`. An inspection shows us that this is a simple fully-connected architecture defined using `tensorflow.keras`:
```
# Define model architecture here
ph_input = keras.Input(shape=(num_inputs,),name='input_placeholder')
hidden_layer = keras.layers.Dense(num_neurons,activation='relu')(ph_input)

for layer in range(num_layers):
    hidden_layer = keras.layers.Dense(num_neurons,activation='relu')(hidden_layer)

output = keras.layers.Dense(num_outputs,activation='linear',name='output_value')(hidden_layer)

model = keras.Model(inputs=[ph_input],outputs=[output])
keras.utils.plot_model(model, 'ml_model.png', show_shapes=True)
my_adam = keras.optimizers.Adam(lr=0.001, decay=0.0)

model.compile(optimizer=my_adam,loss={'output_value': 'mean_squared_error'},metrics=[r2_2])
```
Feel free to tweak this architecture and assess its effects on the quality of the learning. The model can be trained by invoking `python ML_Model.py` from a terminal (with the right python environment active) following which a `model.h5` is generated with the best trainable parameters. Some _a prior_ assessment of the learning can be performed by running `python Training_Assessment.py` which generates several plots showing the scatter accuracy, convergence of training and probability density functions of true versus predicted turbulent eddy viscosities (our targets).

An important function in `ML_Model.py` is the `freeze_session()` function which is instrumental for saving the trained model to disk as a `*.pb` file. This may then be read in by the TensorFlow C API in OpenFOAM. Our trained model (both in `.h5` and `.pb` format along with plots are available in the original `Original/` subdirectory.)


## Step 2: Make modifications to Spalart-Allmaras turbulence model

Once our machine learned turbulence model is trained and exported to disk as a `.pb` file, we need to read it into OpenFOAM to compute steady-state turbulent eddy viscosities with it. The following material is based on tutorials for compiling new turbulence models in OpenFOAM and some preliminary reading is encouraged from the following links if you are not super familiar with this:
1. [How to implement your own turbulence model - by H. Nilsson](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=6&cad=rja&uact=8&ved=2ahUKEwjw-8WEuKXpAhVHHs0KHYEfCBcQFjAFegQIBRAB&url=http%3A%2F%2Fwww.tfd.chalmers.se%2F~hani%2Fkurser%2FOS_CFD_2010%2FimplementTurbulenceModel.pdf&usg=AOvVaw2MvuKXW200b75D7tNAmhZg)
2. [How to add a turbulence model in OpenFOAM-3.0.0 - by H. Kassem](http://hassankassem.me/posts/newturbulencemodel/)
3. [How to add a turbulence model in OpenFOAM-5.0.0 - by H. Kassem](http://hassankassem.me/posts/newturbulencemodel6/)

Now that you know how to go about with the addition of a new turbulence model to OpenFOAM - we can go ahead and make additions to it to interact with TensorFlow 1.15. Within `OF_Model/` you can find our novel turbulence model set up and ready to go. We are going to call this model `ML_SA_CG` ("Machine-learned Spalart-Allmaras for Cross-Geometry applications"). You will notice the following three lines (291-293) in the constructor of the `ML_SA_CG.C` file
```
graph_ = tf_utils::LoadGraph("./ML_SA_CG.pb");
input_ph_ = {TF_GraphOperationByName(graph_, "input_placeholder"), 0};
output_ = {TF_GraphOperationByName(graph_, "output_value/BiasAdd"), 0};
```
The first `graph_` tells OpenFOAM to load in the neural network model as graph, the second line defines an operation `input_ph_` for funneling data into the network to make predictions and `output_` tells TensorFlow about the operation it should run to retrieve the predictions of the framework. These variables are defined in lines 87-89 of `ML_SA_CG.H` following OpenFOAM convention. Note that TensorFlow functionality is embedded into this new turbulence model in line 43 of the header file by stating `#include "tf_utils.H"`. Both `tf_utils.C` and `tf_utils.H` are based on this fantastic repository provided for constructing neural networks using the TF C API by [Daniil Goncharov](https://github.com/Neargye). Also note the contents of `Make/files` where we have also included `tf_utils.C` to ensure TensorFlow functionality is properly recognized and line 13 of `Make/options` where `libtensorflow.so` is provided for linking during compile time. Note that the absolute path to `libtensorflow.so` must be provided if you did not install TensorFlow in the default location.

## Step 3: Compile ML turbulence model

Now all that is left to do is to run `wclean && wmake libso .` from the `OF_Model/` directory. This will create a `ML_SA_CG.so` file in `$FOAM_USER_LIBBIN`.

## Step 4: Make changes to simpleFoam

The original framework in this project required a minor modification to `simpleFoam` to switch turbulence model deployment at each iteration towards steady-state. This is because the machine learning framework has been trained to _predict_ the steady-state turbulent eddy viscosity. To that end, we modify `simpleFoam` to `simpleFoam_ML` to make an eddy-viscosity prediction at `t=0` and then solve only pressure and velocity equations after. Once again, recommended reading for understanding this may be found in [OpenFOAM programming tutorial by Tommaso Lucchini](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=2ahUKEwjKy6-XvqXpAhVIQ80KHYisDikQFjANegQICRAB&url=http%3A%2F%2Fwww.tfd.chalmers.se%2F~hani%2Fkurser%2FOS_CFD_2009%2FprogrammingTutorial.pdf&usg=AOvVaw2H5hYleW_cI6CN0inVnBWQ).

To create a new version of `simpleFoam` that predicts the eddy viscosity only once and then solves solely pressure and velocity, we have followed the steps in the previous link and have created a new solver saved in `simpleFOAM_ML/`. You may incorporate this model into OpenFOAM as follows:
1. From `ML_RANS/` copy `simpleFOAM_ML` to `$WM_PROJECT_DIR` by running `cp -r simpleFoam_ML/ $WM_PROJECT_DIR/applications/solvers/simpleFoam_ML`.
2. Compile this new solver using `cd $WM_PROJECT_DIR/applications/solvers/simpleFoam_ML && wclean && wmake`
3. Test that the new solver is compiled and ready to use with `simpleFoam_ML -help`

## Step 5: Deploy for test case

Finally, now that a new solver and a new turbulence model are ready to deploy. We can run a case to test them. This case is provided in `Testing/`. To run the case successfully, we have made a few changes to incorporate the new solver and the neural network model. These are
1. Line 18 in `system/controlDict` where the application is now `simpleFoam_ML`
2. Line 58 in `system/controlDict` where we link to `ML_SA_CG.so`, our new turbulence model, during run time. 
3. Line 22 in `constant/turbulenceProperties` where our `RASModel` is now `ML_SA_CG`.
4. Line 30 in `constant/turbulenceProperties` has an input parameter `hh_` which corresponds to the step-height (an external input to the network).
5. Finally, we have a text file `means` in the `Testing/` which corresponds to the ML scaling factors for preprocessing our physical inputs. This is read by line 388 of `ML_SA_CG.C`. The values within this file have to manually be set by the means and standard deviations of the training inputs and training outputs. These numbers are produced in `Training/` when the ML model is trained in Python in a similarly named `means` file. Note that these numbers may be copied into the OpenFOAM read-friendly format located within the case directory.
6. The case directory should also have `ML_SA_CG.pb` created within `Training/` (i.e., our neural network model saved to disk)
7. If the previous steps in this tutorial have been followed accurately, you may now run `simpleFoam_ML` from the case directory and reproduce the results shown in the second half of the paper. 