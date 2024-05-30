#include "nvqir/AnalogSimulator.h"

#include <iostream>

// A purely CPU implementation of an analog simulation backend.
// This doesn't use any master equation solver, but instead uses
// a Trotter solution to approximate the time propagator for the
// provided system. This was developed for neither its performance,
// but instead as a stand-in for other analog quantum simulators
// we may target.

namespace nvqir {

class CustomAnalogSimulator : public nvqir::AnalogSimulatorBase {
protected:
  void evolve_state(/*TODO*/) override {
    // pass
  }

public:
  // we need from the user:
  //     (1) the system hamiltonian
  //     (2) the initial state (hard-code for now)
  //     (3) the time steps for evolution
  //     (4) an expectation value to take w.r.t

  cudaq::observe_result observe(/*TODO*/) override {
    // pass
  }
};

}; // namespace nvqir