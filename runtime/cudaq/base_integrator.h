/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <map>
#include <vector>
#include <memory>
#include "schedule.h"
#include "operators.h"

namespace cudaq {
template <typename TState>
class BaseIntegrator {
protected:
    std::map<std::string, double> integrator_options;
    TState state;
    double t;
    std::map<int, int> dimensions;
    std::shared_ptr<Schedule> schedule;
    std::shared_ptr<Operator> hamiltonian;
    std::shared_ptr<BaseTimeStepper<Tstate>> stepper;
    std::vector<std::shared_ptr<Operator>> collapse_operators;

    virtual void post_init() = 0;

public:
    virtual ~BaseIntegrator() = default;

    void set_state(const TState &initial_state, double t0 = 0.0) {
        state = initial_state;
        t = t0;
    }

    void set_system(
        const std::map<int, int> &dimensions,
        std::shared_ptr<Schedule> schedule,
        std::shared_ptr<Operator> hamiltonian,
        std::vector<std::shared_ptr<Operator>> collapse_operators = {}
    ) {
        this->dimensions = dimensions;
        this->schedule = schedule;
        this->hamiltonian = hamiltonian;
        this->collapse_operators = collapse_operators;
    }

    virtual void integrate(double t) = 0;

    std::pair<double, TState> get_state() const {
        return {t, state};
    }
};
}
