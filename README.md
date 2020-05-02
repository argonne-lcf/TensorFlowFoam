# TensorFlowFoam
A turbulence model built using a deep neural network trained in Tensorflow 1.15.

The following steps will get you started with a data-driven turbulence model deployed in OpenFOAM 5. I am assuming you have already installed and successfully run OpenFOAM 5 prior to this. Visit XX for more information on downloading and installing OpenFOAM 5. Also, this tutorial will be based on Linux (Ubuntu 18.04) at this point of time.

While training on the fly is also possible using this procedure - it would need some MPI magic to segregate resources for training and inference at the same time. That is an active topic of research - stay tuned. 

## Step 1: Download the Tensorflow C API

You can download the TensorFlow C API at `https://www.tensorflow.org/install/lang_c`. Follow instructions there to install on your machine. This tutorial/model is designed for the **Linux CPU only** release. Briefly, the instructions to install are:

1. `sudo tar -C /usr/local -xzf (downloaded file)`
2. `sudo ldconfig`

and you are good to go. If you want to install the API to an off-nominal location please consult the documentation at the previously mentioned link. 

## Step 2: Test that C API is running using Test_TF_C code

After Step 1 is complete test that your API is configured correctly by executing the following code (which you can save in `hello_tf.cpp`
```
//Working on some tensorflow and c++ implementations

#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());

  return 0;
}
```
by using 
```g++ hello_tf.cpp -ltensorflow```
and 
```./a.out```
to obtain the following output
```
Hello from TensorFlow C library version 1.15.0
```
If you have reached this point - congratulations you are ready to use TensorFlow 1.15 *within* OpenFOAM 5. 

## Step 3: Train a model in the TensorFlow Python API

For this step - it is assumed that you have access to Python 3.6.8 with the following dependencies:
1. Numpy 1.18
2. TensorFlow 1.15 (this is TensorFlow for the Python training)
3. Matplotlib

However, before you can run your simulation - you need to train your data-driven turbulence model. For this tutorial we will rely on the approach laid out in [arXiv:1910.10878](https://arxiv.org/pdf/1910.10878) where a deep neural network was utilized to predict steady-state turbulent viscosities of the Spalart-Allmaras (SA) model. The goal of this article was to avoid solving the one-extra equation that SA requires but replace it with a point deep-learning framework for acceleration.

For this step - you need not worry about the training and testing data. They are available [here]() and correspond to inputs of initial conditions, mesh coordinates and step height and outputs of the steady-state turbulent viscosity. Note that this corresponds to $M_2$ in Equation X.X of the article. To train this map go to the folder `Training/` and execute the training file by running `python ML_Model.py`. A quick peek inside this python script will tell us several things: the function `load_data` is for load and rescaling data to zero mean and unit variance; the function `get_model` creates a new `tensorflow.keras.model` for a feed-forward neural network and `fit_model` trains this model 

## Step 4: Export model to disk

## Step 5: Make modifications to standard OpenFOAM RANS (or LES) model case to call TensorFlow operations

## Step 6: Compile with wmake

## Step 7: Make changes to case 

## Step 8: Make changes to simpleFoam (if needed)

## Step 9: Deploy
