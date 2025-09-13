%module WireStandalone
%include <stdint.i>
// %include "WireStandalone.h"
%{
#define SWIG_FILE_WITH_INIT
#include "WireStandalone.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    // (int dim_np, double* node_pos),
    (int dim_qp, double* qf_pas)
};
// %include <std_vector.i>
%include "WireStandalone.h" 