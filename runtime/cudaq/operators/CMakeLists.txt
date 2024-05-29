# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-operators)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_OPERATOR_SRC
  hamiltonian.cpp
  operators.cpp
)

find_package(OpenMP)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_OPERATOR_SRC})
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

if (OpenMP_CXX_FOUND)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_OPENMP)
  target_link_libraries(${LIBRARY_NAME} PRIVATE fmt::fmt-header-only OpenMP::OpenMP_CXX)
else()
  target_link_libraries(${LIBRARY_NAME} PRIVATE fmt::fmt-header-only)
endif()

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-operator-targets DESTINATION lib)

install(EXPORT cudaq-operator-targets
        FILE CUDAQOperatorTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
