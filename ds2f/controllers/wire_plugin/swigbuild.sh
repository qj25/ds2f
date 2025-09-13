#!/bin/bash
set -e

SWIG=swig
PYTHON=python3
CXX=g++
PYTHON_INCLUDE=$($PYTHON -c "from sysconfig import get_paths as gp; print(gp()['include'])")
NUMPY_INCLUDE=$($PYTHON -c "import numpy; print(numpy.get_include())")
MUJOCO_INCLUDE=$HOME/mujoco/include
MUJOCO_LIB=$HOME/mujoco/build/lib
EIGEN_INCLUDE=/usr/include/eigen3

# 1. Run SWIG to generate the wrapper
$SWIG -c++ -python -I$MUJOCO_INCLUDE -I. -o WireStandalone_wrap.cpp WireStandalone.i

# 2. Compile all sources to object files
$CXX -c -fPIC -std=c++17 -O2 \
    -I$PYTHON_INCLUDE -I$NUMPY_INCLUDE -I$MUJOCO_INCLUDE -I$EIGEN_INCLUDE \
    WireStandalone.cpp WireStandalone_wrap.cpp wire.cc wire_utils.cc

# 3. Link all object files into a shared library
$CXX -shared WireStandalone.o WireStandalone_wrap.o wire.o wire_utils.o \
    -o _WireStandalone.so -L$MUJOCO_LIB -lmujoco -lstdc++fs

echo "Build complete: _WireStandalone.so" 