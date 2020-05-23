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

#include "ML_LES.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "scope_guard.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void ML_LES<BasicTurbulenceModel>::correctNut
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
void ML_LES<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
ML_LES<BasicTurbulenceModel>::ML_LES
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
    y_(wallDist::New(this->mesh_).y()),
    // Follow convention for graph operations - append '_ph' for placeholder, simply '_' for output, '_op' for an operation like backprop
    // Constructor loads a frozen *.pb graph from the path 
    graph_
    (
        tf_utils::LoadGraph("./ML_LES.pb")
    ),

    // Defines an input operation - this must correspond to the name specified in python API
    input_ph_
    (
        {TF_GraphOperationByName(graph_, "input_placeholder"), 0} // the operation would look like "input_placeholder:0" in model.inputs[0] (keras)

    ),
    // Defines an output operation - this must correspond to the name specified in python API
    output_
    (
        {TF_GraphOperationByName(graph_, "output_value/BiasAdd"), 0} // the operation would look like "output_value/BiasAdd:0" in model.outputs[0] (keras)
    )
{
//    bound(k_, this->kMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool ML_LES<BasicTurbulenceModel>::read()
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
void ML_LES<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    // const surfaceScalarField& phi = this->phi_;
    // const volVectorField& U = this->U_;
    // fv::options& fvOptions(fv::options::New(this->mesh_)); // Source term

    LESeddyViscosity<BasicTurbulenceModel>::correct(); // For changing mesh - coming from turbulenceModel.C through virtual shit

    int num_inputs = 9;
    int num_outputs = 1;
    // The following lines may cause problems with ICPC compilers
    // convert to "float mean_array [10] = {....};"
    float mean_array [num_inputs+num_outputs] = {4.948654692193851069e-05,-1.416845935576197153e-03,1.695804398322601982e-04,-4.909234209068177434e-05,7.200956380997814788e-04,-3.949331152012949186e-07,1.155548212380012041e-01,-1.447936297672789625e-05,-1.249577196433397854e-05,4.991843687885162174e-03};
    float std_array [num_inputs+num_outputs] = {5.156828074413144503e-02,4.477068164228664160e-01,7.504360316118742491e-02,5.840131322144209713e-02,8.381264536219842909e-02,5.977095870302145258e-02,3.844051667887211921e-02,5.081242374826853286e-03,7.321494585983414141e-03,1.157813043774712919e-02};

    run_ml_graph(mean_array,std_array,num_inputs,num_outputs);
}

template<class BasicTurbulenceModel>
void ML_LES<BasicTurbulenceModel>::run_ml_graph(float* mean_array, float* std_array, int num_inputs, int num_outputs)
{
    // omp_set_num_threads(omp_num_threads_);
    tmp<volTensorField> tgradU(fvc::grad(this->U_));
    const volTensorField& gradU = tgradU();

    // Structure for tensors in ML
    int num_cells = this->mesh_.cells().size();

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

    // Some tensorflow pointer requirements
    auto status_ = TF_NewStatus();
    SCOPE_EXIT{ TF_DeleteStatus(status_); }; // Auto-delete on scope exit.
    
    auto options_ = TF_NewSessionOptions();
    SCOPE_EXIT{ TF_DeleteSessionOptions(options_); }; // Auto-delete on scope exit.

    auto sess_ = TF_NewSession(graph_, options_, status_);
    SCOPE_EXIT{ tf_utils::DeleteSession(sess_); }; // Auto-delete on scope exit.

    {
    
    float input_vals[num_cells][num_inputs];
    const std::vector<std::int64_t> input_dims = {num_cells, num_inputs};

    // #pragma omp parallel for
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
        
        // float i7 = (filter_width[id] - mean_array[6])/(std_array[6]);
        // float i8 = (y_[id] - mean_array[7])/(std_array[7]);
        
        input_vals[id][0] = i1;
        input_vals[id][1] = i2;
        input_vals[id][2] = i3;    
        input_vals[id][3] = i4;
        input_vals[id][4] = i5;
        input_vals[id][5] = i6;
        input_vals[id][6] = i7;
        input_vals[id][7] = i8;
        input_vals[id][8] = i9;
    }

    // Set up TF C API stuff
    TF_Tensor* output_tensor_ = nullptr;
    TF_Tensor* input_tensor_ = tf_utils::CreateTensor(TF_FLOAT,
                                          input_dims.data(), input_dims.size(),
                                          &input_vals, num_cells*num_inputs*sizeof(float));

    SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor_); }; // Auto-delete on scope exit.
    SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor_); }; // Auto-delete on scope exit.
    
    // // Arrays of tensors
    // TF_Tensor* input_tensors_[1] = {input_tensor_};
    // TF_Tensor* output_tensors_[1] = {output_tensor_};
    // // Arrays of operations
    // TF_Output inputs[1] = {input_ph_};
    // TF_Output outputs[1] = {output_};
    
    TF_SessionRun(sess_,
                nullptr, // Run options.
                &input_ph_, &input_tensor_, 1, // Input tensor ops, input tensor values, number of inputs.
                &output_, &output_tensor_, 1, // Output tensor ops, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status_ // Output status.
                );

    const auto data = static_cast<float*>(TF_TensorData(output_tensor_));
    for (int i = 0; i < num_cells; i++)
    {
        this->Cs_[i] = data[num_outputs*i]*std_array[num_inputs] + mean_array[num_inputs]; // Funnel changes back into OF - row major order   
    }

    this->Cs_ = filter_(this->Cs_);
    correctNut(gradU);
    // printf("%s\n", "We are using TF for Smagorinsky coefficient prediction");
    // this->nut_.correctBoundaryConditions();
    // fv::options::New(this->mesh_).correct(this->nut_);

    } // Scope for deleting local variable input_vals
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //