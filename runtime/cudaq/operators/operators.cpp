/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "common/EigenSparse.h"
#include "common/FmtCore.h"
#include <cudaq/operators.h>
#include <stdint.h>
#include <unsupported/Eigen/KroneckerProduct>
#ifdef CUDAQ_HAS_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <utility>

namespace cudaq {


} // end namespace