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
    Foam::LESModels::D_SMA

Group
    grpLESTurbulence

Description
    Dynamic SGS model with local averaging, based on Germano and Lilly works.
    
    An older version of this code had been developed for version 2.3.1 and older by Alberto Passalacqua.
    
    The current version has been developed for version 4.1

    Ehsan Asgari, e.asgari@aut.ac.ir

SourceFiles
    D_SMA.C

\*---------------------------------------------------------------------------*/

#ifndef D_SMA_H
#define D_SMA_H

#include "LESeddyViscosity.H"
#include "simpleFilter.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                       Class D_SMA Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class D_SMA
:
    public LESeddyViscosity<BasicTurbulenceModel>
{
    // Private Member Functions

        // Disallow default bitwise copy construct and assignment
        D_SMA(const D_SMA&);
        void operator=(const D_SMA&);


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


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    //- Runtime type information
    TypeName("D_SMA");


    // Constructors

        //- Construct from components
        D_SMA
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
    virtual ~D_SMA()
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
    #include "D_SMA.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //