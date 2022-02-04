
source /opt/openfoam8/etc/bashrc
export TF_C_PATH=$PWD/tf_c_api
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TF_C_PATH/lib
export LIBRARY_PATH=$LIBRARY_PATH:$TF_C_PATH/lib

echo "Using TF C API at "$TF_C_PATH


# Information about coupling TF_C_API and OpenFOAM8
# 1. The above commands ensure that OpenFOAM is source and tensorflow is available in the path
# 2. Within SOURCE/ you have APPS/ and LIBS/ - the former contains solvers and the latter is for turbulence models. 
# 3. To develop a new incompressible RAS turbulence model, make a copy of a RAS model in `Your_Path/SOURCE/LIBS/myMomentumTransportModels/momentumTransportModels/RAS` and make the required changes for the C and H files.
# 4. Go to `Your_Path/SOURCE/LIBS/myMomentumTransportModels/incompressible/kinematicMomentumTransportModels` and add your model to kinematicMomentumTransportModels.C
# 5. Use wclean and wmake at both `Your_Path/SOURCE/LIBS/myMomentumTransportModels/incompressible/` and at `/SOURCE/LIBS/myMomentumTransportModels/momentumTransportModels/`
# 6. Make sure to carefully inspect Make/files and Make/options at these two places to ensure that tensorflow include and lib information is accurate.
# 7. Compile a new solver simpleFOAM_ML to test our first example - to do so go to `Your_Path/SOURCE/APPS/SimpleFOAM_ML` and hit wclean and wmake.
# 8. Run the EXAMPLES/BFS case with simpleFOAM_ML to validate everything is working okay.