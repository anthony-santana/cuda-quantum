/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "py_qubit_qis.h"
#include "cudaq/qis/qubit_qis.h"
#include <pybind11/stl.h>

namespace cudaq {

namespace details {

template <typename QuantumOp>
void bindQuantumOperation(py::module &mod) {
  QuantumOp op;
  py::class_<QuantumOp>(mod, op.name.c_str(), "")
      .def(py::init<>())
      .def_static("__call__",
                  [](py::args &args) {
                    std::vector<std::size_t> mapped;
                    if (args.size() == 1 &&
                        py::isinstance<qreg<dyn, 2>>(args[0])) {
                      // this is a qreg broadcast
                      auto &casted = args[0].cast<qreg<dyn, 2> &>();
                      for (auto &qubit : casted)
                        mapped.push_back(qubit.id());

                    } else if (args.size() == 1 &&
                               py::isinstance<qspan<dyn, 2>>(args[0])) {
                      // this is a qspan broadcast
                      auto &casted = args[0].cast<qspan<dyn, 2> &>();
                      for (auto &qubit : casted)
                        mapped.push_back(qubit.id());
                    } else {
                      // There are n qubits here
                      for (auto &arg : args)
                        mapped.emplace_back(arg.cast<qubit &>().id());
                    }

                    QuantumOp()(mapped);
                  })
      .def_static(
          "ctrl",
          [](py::args &qubits) {
            std::vector<std::size_t> mapped;
            std::vector<bool> isNegated;
            for (auto &arg : qubits) {
              mapped.emplace_back(arg.cast<qubit &>().id());
              isNegated.emplace_back(arg.cast<qubit &>().is_negative());
              if (isNegated.back())
                arg.cast<qubit &>().negate();
            }
            QuantumOp op;
            op.ctrl(mapped, isNegated);
          },
          "");
}

template <typename QuantumOp>
void bindQuantumOperationWithParameter(py::module &mod) {
  QuantumOp op;
  py::class_<QuantumOp>(mod, op.name.c_str(), "")
      .def(py::init<>())
      .def_static("__call__",
                  [](double angle, py::args &args) {
                    std::vector<std::size_t> mapped;
                    for (auto &arg : args) {
                      mapped.emplace_back(arg.cast<qubit &>().id());
                    }
                    QuantumOp()(angle, mapped);
                  })
      .def_static(
          "ctrl",
          [](double angle, py::args &qubits) {
            std::vector<std::size_t> mapped;
            std::vector<bool> isNegated;
            for (auto &arg : qubits) {
              mapped.emplace_back(arg.cast<qubit &>().id());
              isNegated.emplace_back(arg.cast<qubit &>().is_negative());
              if (isNegated.back())
                arg.cast<qubit &>().negate();
            }
            QuantumOp op;
            op.ctrl(angle, mapped, isNegated);
          },
          "");
}
} // namespace details

void bindQIS(py::module &mod) {

  py::class_<qubit>(mod, "qubit", "")
      .def(py::init<>())
      .def(
          "id", [](qubit &self) { return self.id(); }, "");
  py::class_<qspan<dyn, 2>>(mod, "qspan", "")
      .def("size", [](qspan<dyn, 2> &self) { return self.size(); })
      .def("__getitem__", &qspan<dyn, 2>::operator[],
           py::return_value_policy::reference, "");

  py::class_<qreg<dyn, 2>>(mod, "qvector", "")
      .def(py::init<std::size_t>())
      .def("size", [](qreg<dyn, 2> &self) { return self.size(); })
      .def("front",
           [](qreg<dyn, 2> &self, std::size_t n) { return self.front(n); })
      .def(
          "back", [](qreg<dyn, 2> &self) -> qubit & { return self.back(); },
          py::return_value_policy::reference)
      .def("__getitem__", &qreg<dyn, 2>::operator[],
           py::return_value_policy::reference, "");

  details::bindQuantumOperation<cudaq::types::h>(mod);
  details::bindQuantumOperation<cudaq::types::x>(mod);
  details::bindQuantumOperation<cudaq::types::y>(mod);
  details::bindQuantumOperation<cudaq::types::z>(mod);
  details::bindQuantumOperation<cudaq::types::t>(mod);
  details::bindQuantumOperation<cudaq::types::s>(mod);

  details::bindQuantumOperationWithParameter<cudaq::types::rx>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::ry>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::rz>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::r1>(mod);

  mod.def("swap", [](qubit& q, qubit& r){
    swap(q, r);
  });

  mod.def(
      "mz", [](qubit &q, const std::string &regName) { return mz(q, regName); },
      py::arg("target"), py::arg("register_name") = "", "");
  mod.def(
      "mz", [](qreg<dyn, 2> &q) { return mz(q); }, py::arg("target"), "");
  mod.def(
      "mz", [](qspan<dyn, 2> &q) { return mz(q); }, py::arg("target"), "");

  mod.def("control", [](py::object kernel, py::list &controlQubits,
                        py::args &args) {
    std::vector<std::size_t> controlIds;
    for (std::size_t i = 0; i < controlQubits.size(); i++)
      controlIds.push_back(controlQubits[i].attr("id")().cast<std::size_t>());
    cudaq::control([&]() { kernel(*args); }, controlIds);
  });
}
} // namespace cudaq