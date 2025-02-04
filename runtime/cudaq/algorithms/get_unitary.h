/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include "cudaq/utils/tensor.h"
#include "nvqir/Gates.h"

namespace cudaq {

namespace __internal__ {
matrix_2 get_unitary(const Trace &trace);
} // namespace __internal__

namespace details {

/// @brief execute the kernel functor (with optional arguments) and return the
/// trace of the execution path.
template <typename KernelFunctor, typename... Args>
cudaq::Trace traceFromKernel(KernelFunctor &&kernel, Args &&...args) {
  // Get the platform.
  auto &platform = cudaq::get_platform();

  // Create an execution context, indicate this is for tracing the execution
  // path
  ExecutionContext context("tracer");

  // set the context, execute and then reset
  platform.set_exec_ctx(&context);
  kernel(args...);
  platform.reset_exec_ctx();

  return context.kernelTrace;
}

} // namespace details

#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
#else
template <
    typename QuantumKernel, typename... Args,
    typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
#endif
matrix_2 get_unitary(QuantumKernel &&kernel, Args &&...args) {
  return __internal__::get_unitary(
      details::traceFromKernel(kernel, std::forward<Args>(args)...));
}

} // namespace cudaq