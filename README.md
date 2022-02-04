# TensorFlowFoam - work in progress for OpenFOAM 8 Compatibility

![](/misc/repo_logo.png)

A turbulence model built using a deep neural network trained in Tensorflow 1.15.

## Instructions

1. Ensure OpenFOAM is sourced and tensorflow is available in the path - Extract TF API zip file to `tf_c_api/` at the base of the directory.
2. Within SOURCE/ you have APPS/ and LIBS/ - the former contains solvers and the latter is for turbulence models. 
3. To develop a new incompressible RAS turbulence model, make a copy of a RAS model in `Your_Path/SOURCE/LIBS/myMomentumTransportModels/momentumTransportModels/RAS` and make the required changes for the C and H files.
4. Go to `Your_Path/SOURCE/LIBS/myMomentumTransportModels/incompressible/kinematicMomentumTransportModels` and add your model to kinematicMomentumTransportModels.C
5. Use wclean and wmake at both `Your_Path/SOURCE/LIBS/myMomentumTransportModels/incompressible/` and at `/SOURCE/LIBS/myMomentumTransportModels/momentumTransportModels/`
6. Make sure to carefully inspect Make/files and Make/options at these two places to ensure that tensorflow include and lib information is accurate.
7. Compile a new solver simpleFOAM_ML to test our first example - to do so go to `Your_Path/SOURCE/APPS/SimpleFOAM_ML` and hit wclean and wmake.
8. Run the EXAMPLES/BFS case with simpleFOAM_ML to validate everything is working okay.


## Tips
Use `prep_env.sh` to initialize variables and path to TF C API for setting up development environment.
Use `SOURCE/LIBS/compile_new_models.sh` to compile new turbulence models.

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. 

If you have found this framework informative and useful for any of your research, please cite us
```
@inproceedings{maulik2021deploying,
  title={Deploying deep learning in OpenFOAM with TensorFlow},
  author={Maulik, Romit and Sharma, Himanshu and Patel, Saumil and Lusch, Bethany and Jennings, Elise},
  booktitle={AIAA Scitech 2021 Forum},
  pages={1485},
  year={2021}
}
```

## LICENSE

[MIT](LICENSE)
