# TensorFlowFoam
A turbulence model built using a deep neural network trained in Tensorflow 1.15.

The following steps will get you started with a data-driven turbulence model deployed in OpenFOAM 5. I am assuming you have already installed and successfully run OpenFOAM 5 prior to this. Visit XX for more information on downloading and installing OpenFOAM 5. Also, this tutorial will be based on Linux (Ubuntu 18.04) at this point of time.

While training on the fly is also possible using this procedure - it would need some MPI magic to segregate resources for training and inference at the same time. That is an active topic of research - stay tuned. 

# Step 1: Download the Tensorflow C API

# Step 2: Test that C API is running using Test_TF_C code

# Step 3: Train a model in the TensorFlow Python API

# Step 4: Export model to disk

# Step 5: Make modifications to standard OpenFOAM RANS (or LES) model case to call TensorFlow operations

# Step 6: Compile with wmake

# Step 7: Make changes to case 

# Step 8: Make changes to simpleFoam (if needed)

# Step 9: Deploy
