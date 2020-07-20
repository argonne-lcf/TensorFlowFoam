# TensorFlowFoam

![](/misc/repo_logo.png)

A turbulence model built using a deep neural network trained in Tensorflow 1.15.

## Contents
1. `ML_RANS/`: A tutorial for setting up an artificial neural network surrogate for a linear eddy-viscosity RANS model.

2. `ML_LES/`: A tutorial for setting up an artificial neural network surrogate for dynamic Smagorinsky coefficient calculation (i.e., bypass test-filtering).

3. `IN_SITU/`: A tutorial for setting up a neural network training _from within_ OpenFOAM to avoid disk IO.

## Installation

The following steps will get you started with a data-driven turbulence model deployed in OpenFOAM 5. I am assuming you have already installed and successfully run OpenFOAM 5 prior to this. Also, this tutorial will be based on Linux (Ubuntu 18.04) at this point of time.

While training on the fly is also possible using this procedure - it would need some MPI magic to segregate resources for training and inference at the same time. That is an active topic of research - stay tuned. 

### Step 1: Install python environment

We suggest the creation of a new virtual environment (either in conda or venv) and the installation of relevant packages for this tutorial using
```
pip install -r requirements.txt
```

### Step 2: Download the Tensorflow C API

You can download the TensorFlow C API at `https://www.tensorflow.org/install/lang_c`. Follow instructions there to install on your machine. This tutorial/model is designed for the **Linux CPU only** release. Briefly, the instructions to install (if you have root access) are:

1. `sudo tar -C /usr/local -xzf (downloaded file)`
2. `sudo ldconfig`

and you are good to go. If you do not have root access and need to place the TensorFlow C API at an arbitrary location you may use the following steps:

1. `tar -C /path/to/api -xzf (downloaded file)`
2. `export LIBRARY_PATH=$LIBRARY_PATH:/path/to/api/lib`
3. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/api/lib`

Note that steps 2 and 3 either need to be added to your `.bashrc` file or need to be executed each time a new terminal session is started. 

### Step 3: Test that C API is running using Test_TF_C code

After Step 2 is complete test that your API is configured correctly by executing the following code (which you can save in `hello_tf.cpp`
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
1. `g++ hello_tf.cpp -ltensorflow`
2. `./a.out`
to obtain the following output
```
Hello from TensorFlow C library version 1.15.0
```
If you face an error here (perhaps due to not linking appropriately in the absence of root priveleges) try using:
```
g++ -I/path/to/api/include -L/path/to/api/lib hello_tf.cpp -ltensorflow
```
for step 2.

If you have reached this point - congratulations you are ready to use TensorFlow 1.15 *within* OpenFOAM 5. You may utilize the individual READMEs from `ML_RANS/`, `ML_LES` and `IN-SITU` (documentation for the latter two are in-progress) to construct a neural network based turbulence model for training/deployment in OpenFOAM.

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov), Himanshu Sharma (himanshu.sharma@pnnl.gov), Saumil Patel (spatel@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. 

If you have found this framework informative and useful for any of your research, please cite us
```
@article{maulik2019accelerating,
  title={Accelerating RANS simulations using a data-driven framework for eddy-viscosity emulation},
  author={Maulik, Romit and Sharma, Himanshu and Patel, Saumil and Lusch, Bethany and Jennings, Elise},
  journal={arXiv preprint arXiv:1910.10878},
  year={2019}
}
```
