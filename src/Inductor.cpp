// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Inductor.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"

#include <utility>

using namespace JoSIM;

/*
  Llabel V+ V- L

  V - (3*L)/(2*h)Io = -(2*L)/(h)In-1 + (L)/(2*h)In-2

  [ 0  0            1 ] [ V+ ]   [                              0 ]
  | 0  0           -1 | | V- | = |                              0 |
  [ 1 -1 -(3*L)/(2*h) ] [ Io ]   [ -(2*L)/(h)In-1 + (L)/(2*h)In-2 ]

  (PHASE)
  phi - L(2e/hbar)Io = 0

  [ 0  0           1 ] [ phi+ ]   [ 0 ]
  | 0  0          -1 | | phi- | = | 0 |
  [ 1 -1 -L(2e/hbar) ] [  Io  ]   [ 0 ]
*/

Inductor::Inductor(const std::pair<tokens_t, string_o>& s,
                   const NodeConfig&                    ncon,
                   const nodemap&                       nm,
                   std::unordered_set<std::string>&     lm,
                   nodeconnections&                     nc,
                   Input&                               iObj,
                   Spread&                              spread,
                   int64_t&                             bi) {
    double                spr     = 1.0;
    std::optional<double> fluxVal = std::nullopt;
    std::optional<double> phiVal  = std::nullopt;
    for (size_t idx = 4; idx < s.first.size(); ++idx) {
        const auto& i = s.first.at(idx);
        if (i.rfind("SPREAD=", 0) == 0) {
            spr = parse_param(i.substr(7), iObj.parameters, s.second);
        } else if (i.rfind("FLUX=", 0) == 0) {
            fluxVal = parse_param(i.substr(5), iObj.parameters, s.second);
        } else if (i.rfind("PHI0=", 0) == 0) {
            fluxVal = parse_param(i.substr(5), iObj.parameters, s.second);
        } else if (i.rfind("PHI=", 0) == 0) {
            phiVal = parse_param(i.substr(4), iObj.parameters, s.second);
        } else if (i.rfind("PHIOFF=", 0) == 0) {
            phiVal = parse_param(i.substr(7), iObj.parameters, s.second);
        } else if (i.rfind("IC=", 0) == 0) {
            icCurrent_ = parse_param(i.substr(3), iObj.parameters, s.second);
        }
    }
    at_  = iObj.argAnal;
    // Set previous current value
    In2_ = 0.0;
    // Check if the label has already been defined
    if (lm.count(s.first.at(0)) != 0) {
        Errors::invalid_component_errors(ComponentErrors::DUPLICATE_LABEL, s.first.at(0));
    }
    // Set the label
    netlistInfo.label_ = s.first.at(0);
    // Add the label to the known labels list
    lm.emplace(s.first.at(0));
    // Set the value (Inductance), this should be the 4th token
    nominalValue_      = parse_param(s.first.at(3), iObj.parameters, s.second);
    spread_            = spr;
    netlistInfo.value_ = spread.spread_value(nominalValue_, Spread::IND, spread_);
    if (fluxVal.has_value()) {
        phiOffset_ = 2.0 * Constants::PI * fluxVal.value();
    } else if (phiVal.has_value()) {
        phiOffset_ = phiVal.value();
    }
    if (icCurrent_.has_value()) {
        In2_ = In3_ = In4_ = icCurrent_.value();
        if (!fluxVal.has_value() && !phiVal.has_value() && at_ == AnalysisType::Phase) {
            phiOffset_ = -(netlistInfo.value_ / Constants::SIGMA) * icCurrent_.value();
        }
    } else if (phiOffset_ != 0.0 && netlistInfo.value_ != 0.0 && at_ == AnalysisType::Voltage) {
        icCurrent_ = -(Constants::SIGMA / netlistInfo.value_) * phiOffset_;
        In2_ = In3_ = In4_ = icCurrent_.value();
    }
    // Set the node configuration type
    indexInfo.nodeConfig_   = ncon;
    // Set current index and increment it
    indexInfo.currentIndex_ = bi++;
    // Set te node indices, using token 2 and 3
    set_node_indices(tokens_t(s.first.begin() + 1, s.first.begin() + 3), nm, nc);
    // Set the non zero, column index and row pointer vectors
    set_matrix_info();
    // Append the value to the non zero vector
    if (at_ == AnalysisType::Voltage) {
        // If voltage mode analysis then append -(3/2) * (L/h)
        matrixInfo.nonZeros_.emplace_back(-(3.0 / 2.0) * (netlistInfo.value_ / iObj.transSim.tstep()));
    } else if (at_ == AnalysisType::Phase) {
        // If phase mdoe analysis then append -(L/σ)
        matrixInfo.nonZeros_.emplace_back(-netlistInfo.value_ / Constants::SIGMA);
    }
}

void Inductor::add_mutualInductance(const double& m, const AnalysisType& at, const double& h, const int64_t& ci) {
    // Adds the mutual inductance to the non zero vector
    if (at == AnalysisType::Voltage) {
        // If voltage mode then add -(3/2) * (m/h)
        matrixInfo.nonZeros_.emplace_back(-(3.0 / 2.0) * (m / h));
    } else if (at == AnalysisType::Phase) {
        // If phase mode then add -(m/σ)
        matrixInfo.nonZeros_.emplace_back(-m / Constants::SIGMA);
    }
    // Append the column index of the current branch of the second inductor
    matrixInfo.columnIndex_.emplace_back(ci);
    // Increase the row pointer vector by 1
    matrixInfo.rowPointer_.back()++;
}

// Update timestep based on a scalar factor i.e 0.5 for half the timestep
void Inductor::update_timestep(const double& factor) {
    if (at_ == AnalysisType::Voltage) { matrixInfo.nonZeros_.back() = (1.0 / factor) * matrixInfo.nonZeros_.back(); }
}
