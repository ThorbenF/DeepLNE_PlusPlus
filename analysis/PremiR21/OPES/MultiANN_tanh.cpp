/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2021 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "/home/borsatto@farma.unige.ch/plumed2-master/src/function/Function.h"
#include "/home/borsatto@farma.unige.ch/plumed2-master/src/function/ActionRegister.h"
#include "/home/borsatto@farma.unige.ch/plumed2-master/src/tools/IFile.h"
#include "/home/borsatto@farma.unige.ch/plumed2-master/src/tools/Matrix.h"

#include <cmath>
#include <algorithm>

using namespace std;

namespace PLMD {
namespace function {

//+PLUMEDOC FUNCTION COMBINE
/*
Calculate a polynomial combination of a set of other variables.

The functional form of this function is
\f[
C=\sum_{i=1}^{N_{arg}} c_i (x_i-a_i)^{p_i}
\f]

The coefficients c, the parameters a and the powers p are provided as vectors.

Notice that COMBINE is not able to predict which will be periodic domain
of the computed value automatically. The user is thus forced to specify it
explicitly. Use PERIODIC=NO if the resulting variable is not periodic,
and PERIODIC=A,B where A and B are the two boundaries if the resulting variable
is periodic.



\par Examples

The following input tells plumed to print the distance between atoms 3 and 5
its square (as computed from the x,y,z components) and the distance
again as computed from the square root of the square.
\plumedfile
DISTANCE LABEL=dist      ATOMS=3,5 COMPONENTS
COMBINE  LABEL=distance2 ARG=dist.x,dist.y,dist.z POWERS=2,2,2 PERIODIC=NO
COMBINE  LABEL=distance  ARG=distance2 POWERS=0.5 PERIODIC=NO
PRINT ARG=distance,distance2
\endplumedfile
(See also \ref PRINT and \ref DISTANCE).

The following input tells plumed to add a restraint on the
cube of a dihedral angle. Notice that since the angle has a
periodic domain
-pi,pi its cube has a domain -pi**3,pi**3.
\plumedfile
t: TORSION ATOMS=1,3,5,7
c: COMBINE ARG=t POWERS=3 PERIODIC=-31.0062766802998,31.0062766802998
RESTRAINT ARG=c KAPPA=10 AT=0
\endplumedfile



*/
//+ENDPLUMEDOC


class MultiANN :
  public Function
{
  enum class Activation {
    Softplus, // here we could enumerate more options
    Tanh
  };
  unsigned groupby;
  // weights
  std::vector<Matrix<double>> weights;
  // transpose matrices (this is to avoid taking the transpose in calculate()
  std::vector<Matrix<double>> weights_transposed;
  // biases
  std::vector<std::vector<double>> biases;
  // activation functions (custom for each node of each layer)
  std::vector<std::vector<Activation>> activation;
  // value of each layer
  std::vector<Matrix<double>> layers;
  // derivative of result wrt values at each layer
  std::vector<Matrix<double>> dlayers;
  // derivative of the activation function (stored in the forward loop for efficienty)
  std::vector<Matrix<double>> dactivation;
public:
  explicit MultiANN(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};


PLUMED_REGISTER_ACTION(MultiANN,"MULTI_ANN")

void MultiANN::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","PARAMETERS","ANN.dat","parameter file");
  keys.add("compulsory","GROUPBY","1","groups");
}

MultiANN::MultiANN(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  groupby=1;
  parse("GROUPBY",groupby);
  log<<"  groupby: "<<groupby<<"\n";
  plumed_assert(getNumberOfArguments()%groupby==0);
  std::string parfile="";
  parse("PARAMETERS",parfile);
  log<<"  parameters file: "<<parfile<<"\n";
  plumed_assert(parfile.length()>0);
  IFile ifile;
  ifile.link(*this);
  ifile.open(parfile);
  unsigned previous_layer=groupby;
  layers.push_back(Matrix<double>(getNumberOfArguments()/groupby,groupby));
  dlayers.push_back(Matrix<double>(getNumberOfArguments()/groupby,groupby));
  dactivation.push_back(Matrix<double>(getNumberOfArguments()/groupby,groupby));
  while(true) { // layers
    std::vector<std::vector<double>> weights_rows;
    log<<"  weights:\n";
    while(true) { // weights
      std::vector<double> weights_row;
      if(!ifile.FieldExist("w0")) break;
      log<<"  *";
      for(unsigned i=0; i<previous_layer; i++) {
        std::string item;
        Tools::convert(i,item);
        item="w"+item;
        double w=0.0;
        ifile.scanField(item,w);
        weights_row.push_back(w);
        log<<" "<<w;
      }
      log<<"\n";
      ifile.scanField(); // end of line
      if(weights_rows.size()>1) plumed_assert(weights_rows[0].size()==weights_row.size());
      weights_rows.push_back(weights_row);
    }
    plumed_assert(weights_rows.size()>0);
    auto neww=Matrix<double>(weights_rows[0].size(),weights_rows.size());
    for(unsigned i=0; i<weights_rows.size(); i++) {
      for(unsigned j=0; j<weights_rows[0].size(); j++) {
        neww(j,i)=weights_rows[i][j];
      }
    }
    weights.push_back(neww);
    previous_layer=weights_rows.size();
    std::vector<double> biases_row;
    plumed_assert(ifile.FieldExist("b0"));
    log<< "  biases: ";
    for(unsigned i=0; i<previous_layer; i++) {
      std::string item;
      Tools::convert(i,item);
      item="b"+item;
      double b=0.0;
      ifile.scanField(item,b);
      log<<" "<<b;
      biases_row.push_back(b);
    }
    log<<"\n";
    ifile.scanField(); // end of line;
    biases.push_back(biases_row);
    std::vector<Activation> activation_row;
    if(!ifile.FieldExist("activation0")) break;
    log<<"  activation:";
    for(unsigned i=0; i<previous_layer; i++) {
      std::string item;
      Tools::convert(i,item);
      item="activation"+item;
      std::string a;
      if(!ifile.scanField(item,a)) break;
      Activation aa;
      if(a=="softplus") {
        aa=Activation::Softplus;
      } else if (a == "tanh") {
        aa = Activation::Tanh;
      } else {
        plumed_error();
      }
      log<<" "<<a;
      activation_row.push_back(aa);
    }
    log<<"\n";
    ifile.scanField(); // end of line;
    plumed_assert(activation_row.size()==previous_layer);
    activation.push_back(activation_row);
  }

  for(unsigned i=0; i<weights.size(); i++) {
    layers.push_back(Matrix<double>(getNumberOfArguments()/groupby,weights[i].ncols()));
    dlayers.push_back(Matrix<double>(getNumberOfArguments()/groupby,weights[i].ncols()));
    dactivation.push_back(Matrix<double>(getNumberOfArguments()/groupby,weights[i].ncols()));
    Matrix<double> tw;
    transpose(weights[i],tw);
    weights_transposed.push_back(tw);
  }

  addValueWithDerivatives();
  getPntrToComponent(0)->setNotPeriodic();
  checkRead();

}

void MultiANN::calculate() {

  // init
  for(unsigned i=0; i<getNumberOfArguments(); i++) {
    layers[0](i/groupby,i%groupby)=getArgument(i);
  }

  // forward loop
  for(unsigned i=0; i<weights.size(); i++) {

    mult(layers[i],weights[i],layers[i+1]);

    for(unsigned j=0; j<layers[i+1].nrows(); j++) {
      for(unsigned k=0; k<layers[i+1].ncols(); k++) {
        layers[i+1](j,k)+=biases[i][k];
      }
    }
    if(i<weights.size()-1) {
      for(unsigned k=0; k<layers[i+1].ncols(); k++) {
        switch(activation[i][k]) {
        case Activation::Softplus:
        {
          // compute both activation and dactivation, to save on the exponential
          for(unsigned j=0; j<layers[i+1].nrows(); j++) {
            auto x=layers[i+1](j,k);
            auto expabsx=std::exp(-std::abs(x));
            auto y=std::log1p(expabsx) + std::max(x,0.0);
            auto dy=0.0;
            if(x>=0) dy=1.0/(1.0+expabsx);
            else     dy=expabsx/(1.0+expabsx);
            layers[i+1](j,k)=y;
            dactivation[i+1](j,k)=dy;
          }
        }
        case Activation::Tanh:
        {
            // compute both activation and dactivation using tanh and its derivative
            for(unsigned j=0; j<layers[i+1].nrows(); j++) {
                auto x = layers[i+1](j,k);
                auto y = std::tanh(x);
                auto dy = 1.0 - y*y; // tanh derivative: 1 - tanh^2(x)
                layers[i+1](j,k) = y;
                dactivation[i+1](j,k) = dy;
            }
        }
        break;
        default:
          plumed_error();
        }
      }
    }
  }
  double combine=0.0;
  for(unsigned i=0; i<layers[layers.size()-1].nrows(); i++) combine+=layers[layers.size()-1](i,0);


  // init
  for(unsigned i=0; i<layers[layers.size()-1].nrows(); i++) dlayers[layers.size()-1](i,0)=1.0;

  // backward loop
  for(unsigned i=layers.size()-1; i>0; i--) {
    if(i<layers.size()-1) {
      for(unsigned k=0; k<dlayers[i].nrows(); k++) for(unsigned l=0; l<dlayers[i].ncols(); l++)
          dlayers[i](k,l)*=dactivation[i](k,l);
    }
    mult(dlayers[i],weights_transposed[i-1],dlayers[i-1]);
  }

  setValue(combine);
  for(unsigned i=0; i<getNumberOfArguments(); i++) {
    setDerivative(i,dlayers[0](i/groupby,i%groupby));
  }

}

}
}


