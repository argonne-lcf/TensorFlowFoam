# Work in progress
Training a surrogate model _within_ OpenFOAM without having to checkpoint to disk.
Runs with `pimpleFoam`
Channel flow at ReTau=395.

# Important
1. Remove /checkpoints from run directory if you change your model definition or want to restart training from scratch
