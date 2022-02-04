cd ../../ 
source prep_env.sh

cd SOURCE/LIBS/myMomentumTransportModels/momentumTransportModels
wclean
wmake

cd ../incompressible
wclean
wmake

cd ../../