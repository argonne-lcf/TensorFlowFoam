/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::LESModels::TF_LES_Training

Group
    grpLESTurbulence

Description
    Dynamic SGS model with local averaging, based on Germano and Lilly works. In-situ ML training to build surrogate for Smagorinsky coefficient.
    The current version has been developed for version 5.0.

    Original dynamic Smagorinsky implementation for OF 4.1: Ehsan Asgari, e.asgari@aut.ac.ir

    Modified for in-situ training of machine learning models: Romit Maulik, rmaulik@anl.gov

    # For details of Training using TensorFlow 1.15 C API
    https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2

SourceFiles
    TF_LES_Training.C

\*---------------------------------------------------------------------------*/

#ifndef TF_LES_Training_H
#define TF_LES_Training_H

#include <memory>
#include "LESeddyViscosity.H"
#include "simpleFilter.H"
#include "TF_Model.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                       Class TF_LES_Training Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class TF_LES_Training
:
    public LESeddyViscosity<BasicTurbulenceModel>
{
    // Private Member Functions

        // Disallow default bitwise copy construct and assignment
        TF_LES_Training(const TF_LES_Training&);
        void operator=(const TF_LES_Training&);


protected:

    // Protected data
        volScalarField k_;
        volScalarField Cs_;
        
//       volScalarField beta_;
        simpleFilter simpleFilter_;
        autoPtr<LESfilter> filterPtr_;
        LESfilter& filter_;
        volScalarField y_;

        // Protected Member Functions

        //- Update sub-grid eddy-viscosity
        void correctNut(const tmp<volTensorField>& gradU);

        virtual void correctNut();

        void run_ml_graph();

        // TF API Stuff
        const char* graph_def_filename = "graph.pb";
        const char* checkpoint_prefix = "./checkpoints/checkpoint";
        int restore;

        model_t model;


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    //- Runtime type information
    TypeName("TF_LES_Training");


    // Constructors

        //- Construct from components
        TF_LES_Training
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~TF_LES_Training()
    {}


    // Member Functions

        //- Read model coefficients if they have changed
        virtual bool read();

        //- Return SGS kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField("DkEff", this->nut_ + this->nu())
            );
        }

        //- Correct Eddy-Viscosity and related properties
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "TF_LES_Training.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //