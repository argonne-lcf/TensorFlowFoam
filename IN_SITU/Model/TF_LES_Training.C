/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.


\*---------------------------------------------------------------------------*/

#include "TF_LES_Training.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "TF_Model.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void TF_LES_Training<BasicTurbulenceModel>::correctNut
(
    const tmp<volTensorField>& gradU
)
{
    this->nut_ = max(Cs_*sqr(this->delta())*mag(dev(symm(gradU))),-1.0*this->nu());
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void TF_LES_Training<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
TF_LES_Training<BasicTurbulenceModel>::TF_LES_Training
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    Cs_
    (
        IOobject
        (
            IOobject::groupName("Cs", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar ("Cs", dimless,SMALL)
    ),
    simpleFilter_(U.mesh()),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_()),
    y_(wallDist::New(this->mesh_).y())
{

    if (type == typeName)
    {
        this->printCoeffs(type);
    }

    // Loading or restoring model from graph
    restore = DirectoryExists("checkpoints");
    printf("Loading graph\n");
    if (!ModelCreate(&model, graph_def_filename)) printf("Something wrong with loading graph\n");

    if (restore) {
    printf(
        "Restoring weights from checkpoint (remove the checkpoints directory "
        "to reset)\n");
    if (!ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) printf("Something wrong with loading checkpoint\n");
    } else {
    printf("Initializing model weights\n");
    if (!ModelInit(&model)) printf("Something wrong with initializing weights\n");
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool TF_LES_Training<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());        

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
void TF_LES_Training<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    // const surfaceScalarField& phi = this->phi_;
    const volVectorField& U = this->U_;
    // fv::options& fvOptions(fv::options::New(this->mesh_)); // Source term

    LESeddyViscosity<BasicTurbulenceModel>::correct(); // For changing mesh - coming virtually from turbulenceModel.C

    tmp<volTensorField> tgradU(fvc::grad(U));
    const volTensorField& gradU = tgradU();

    volSymmTensorField S(dev(symm(gradU)));
    volScalarField magS(mag(S));

    volVectorField Uf(filter_(U));

    volSymmTensorField Sf(filter_(S));  
//    volSymmTensorField Sf(dev(symm(fvc::grad(Uf))));
    
    volScalarField magSf(mag(Sf));
          
    
    volSymmTensorField LL =
    (dev(filter_(sqr(U)) - (sqr(filter_(U)))));

    volSymmTensorField MM
    (
        sqr(this->delta())*(filter_(magS*S) - 4.0*magSf*Sf)
    );
    
    volScalarField MMMM = fvc::average(magSqr(MM));
    MMMM.max(VSMALL);

    Cs_= 0.5* fvc::average(LL && MM)/MMMM;

    volScalarField KK =
    0.5*(filter_(magSqr(U)) - magSqr(filter_(U)));
    
    volScalarField mm
    (
        sqr(this->delta())*(4.0*sqr(mag(Sf)) - filter_(sqr(magS)))
       
    );

    volScalarField mmmm = fvc::average(magSqr(mm));
    mmmm.max(VSMALL);

    k_ = fvc::average(KK*mm)/mmmm * sqr(this->delta())*magSqr(S);

    correctNut(gradU);

    volScalarField filter_width(this->delta());

    if(this->runTime_.outputTime())
    {
        const char* Cs_name = "Cs";
        Cs_.rename(Cs_name);
        Cs_.write();

        const char* Uf_name = "Uf";
        Uf.rename(Uf_name);
        Uf.write();

        const char* S_name = "S_ij";
        S.rename(S_name);
        S.write();

        const char* delta_name = "del";
        filter_width.rename(delta_name);
        filter_width.write();

        const char* y_name = "yw";
        y_.rename(y_name);
        y_.write();

        const char* nut_name = "nut";
        this->nut_.rename(nut_name);
        this->nut_.write();
    }

    LESeddyViscosity<BasicTurbulenceModel>::correct(); // For changing mesh - coming from turbulenceModel.C through virtual shit

    run_ml_graph(); // We will train a surrogate model for the Smagorinsky coefficient using this function call
}

template<class BasicTurbulenceModel>
void TF_LES_Training<BasicTurbulenceModel>::run_ml_graph()
{
    // Structure for tensors in ML
    int num_cells = this->mesh_.cells().size();
    int num_inputs = 9;
    int num_outputs = 1;

    float input_vals[num_cells][num_inputs];
    float target_vals[num_cells][num_outputs];
    float mean_array[num_inputs];
    float std_array[num_inputs];

    // GRAB DATA FOR SENDING TO ANN
    tmp<volTensorField> tgradU(fvc::grad(this->U_));
    const volTensorField& gradU = tgradU();

    volSymmTensorField S(dev(symm(gradU)));
    volScalarField s11 = S.component(tensor::XX);
    volScalarField s12 = S.component(tensor::XY);
    volScalarField s13 = S.component(tensor::XZ);
    volScalarField s22 = S.component(tensor::YY);
    volScalarField s23 = S.component(tensor::YZ);
    volScalarField s33 = S.component(tensor::ZZ);


    volScalarField u_ = this->U_.component(vector::X);
    volScalarField v_ = this->U_.component(vector::Y);
    volScalarField w_ = this->U_.component(vector::Z);

    volScalarField filter_width(this->delta());

    // Scaling the inputs to zero mean and unit variance
    forAll(s11.internalField(), id) // for boundary field use u_.boundaryField()
    {
        
        mean_array[0] = mean_array[0] + s11[id];
        mean_array[1] = mean_array[1] + s12[id];
        mean_array[2] = mean_array[2] + s13[id];
        mean_array[3] = mean_array[3] + s22[id];
        mean_array[4] = mean_array[4] + s23[id];
        mean_array[5] = mean_array[5] + s33[id];
        mean_array[6] = mean_array[6] + u_[id];
        mean_array[7] = mean_array[7] + v_[id];
        mean_array[8] = mean_array[8] + w_[id];
    }

    for (int i = 0; i < num_inputs; ++i)
    {
        mean_array[i] = mean_array[i]/num_cells;
    }

    forAll(s11.internalField(), id) // for boundary field use u_.boundaryField()
    {
        std_array[0] = std_array[0] + pow(s11[id]-mean_array[0],2);
        std_array[1] = std_array[1] + pow(s12[id]-mean_array[1],2);
        std_array[2] = std_array[2] + pow(s13[id]-mean_array[2],2);
        std_array[3] = std_array[3] + pow(s22[id]-mean_array[3],2);
        std_array[4] = std_array[4] + pow(s23[id]-mean_array[4],2);
        std_array[5] = std_array[5] + pow(s33[id]-mean_array[5],2);
        std_array[6] = std_array[6] + pow(u_[id]-mean_array[6],2);
        std_array[7] = std_array[7] + pow(v_[id]-mean_array[7],2);
        std_array[8] = std_array[8] + pow(w_[id]-mean_array[8],2);
    }

    for (int i = 0; i < num_inputs; ++i)
    {
        std_array[i] = sqrt(std_array[i]/num_cells);
    }
   
    // Store data in memory for tensorflow - can be made more efficient for OK for now
    forAll(s11.internalField(), id) // for boundary field use u_.boundaryField()
    {
        float i1 = (s11[id] - mean_array[0])/(std_array[0]);
        float i2 = (s12[id] - mean_array[1])/(std_array[1]);
        float i3 = (s13[id] - mean_array[2])/(std_array[2]);
        float i4 = (s22[id] - mean_array[3])/(std_array[3]);
        float i5 = (s23[id] - mean_array[4])/(std_array[4]);
        float i6 = (s33[id] - mean_array[5])/(std_array[5]);
        float i7 = (u_[id] - mean_array[6])/(std_array[6]);
        float i8 = (v_[id] - mean_array[7])/(std_array[7]);
        float i9 = (w_[id] - mean_array[8])/(std_array[8]);
               
        input_vals[id][0] = i1;
        input_vals[id][1] = i2;
        input_vals[id][2] = i3;    
        input_vals[id][3] = i4;
        input_vals[id][4] = i5;
        input_vals[id][5] = i6;
        input_vals[id][6] = i7;
        input_vals[id][7] = i8;
        input_vals[id][8] = i9;

        target_vals[id][0] = this->Cs_[id];
    }
       
    // printf("Initial predictions\n");
    // if (!ModelPredict(&model, &testdata[0][0], batch_size)) printf("Something wrong with predictions\n");

    float testdata[3][9] = {0.0}; // Initializing to zero right now just for simplicity - you can set up a for loop here to test on some real data
    int batch_size = 3;

    printf("Training for a few steps\n");
    for (int i = 0; i < 2000; ++i) {
    if (!ModelRunTrainStep(&model,&input_vals[0][0],&target_vals[0][0])) printf("Something wrong with training\n");
    } // This step may segfault if the model initialization causes a floating point overflow - rerun if that happens

    printf("Test predictions\n");
    if (!ModelPredict(&model, &testdata[0][0], batch_size)) printf("Something wrong with updating predictions\n");

    printf("Saving checkpoint\n");
    if (!ModelCheckpoint(&model, checkpoint_prefix, SAVE)) printf("Something wrong with saving checkpoint\n");

    // ModelDestroy(&model);

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //