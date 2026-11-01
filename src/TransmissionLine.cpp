// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/TransmissionLine.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring> // for memcmp
#include <functional>
#include <iostream>
#include <locale>
#include <string>

using namespace JoSIM;

/*
  Tlabel Vi+ Vi- Vo+ Vo- td=value z0=value

  V1 - Z I1 = Z I2n-k + V2n-k
  V2 - Z I2 = Z I1n-k + V1n-k

  k = td/h

  [ 0  0  0  0  1  0 ] [ V1+ ]   [              0 ]
  | 0  0  0  0 -1  0 | | V1- |   |              0 |
  | 0  0  0  0  0  1 | | V2+ | = |              0 |
  | 0  0  0  0  0 -1 | | V2- |   |              0 |
  | 1 -1  0  0 -Z  0 | |  I1 |   | ZI2n-k + V2n-k |
  [ 0  0  1 -1  0 -Z ] [  I2 ]   [ ZI1n-k + V1n-k ]

  (PHASE)
  phi1 - Z(2e/hbar)(2h/3)I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)phi1n-1 - (1/3)phi1n-2 +
                              phi2n-k - (4/3)phi2n-k-1 + (1/3)phi2n-k-2
  phi2 - Z(2e/hbar)(2h/3)I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)phi2n-1 - (1/3)phi2n-2 +
                              phi1n-k - (4/3)phi1n-k-1 + (1/3)phi1n-k-2

  [ 0  0  0  0                -1                  0 ] [ phio+ ]   [    0 ]
  | 0  0  0  0                 1                  0 | | phio- |   |    0 |
  | 0  0  0  0                 0                  1 | | phic+ | = |    0 |
  | 0  0  0  0                 0                 -1 | | phic- |   |    0 |
  | 1 -1  0  0 -Z(2e/hbar)(2h/3)                  0 | |  Ic   |   |  RHS1|
  [ 0  0  1 -1                 0  -Z(2e/hbar)(2h/3) ] [  Io   ]   [  RHS2]

  RHS1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)phi1n-1 - (1/3)phi1n-2 +
         phi2n-k - (4/3)phi2n-k-1 + (1/3)phi2n-k-2
  RHS2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)phi2n-1 - (1/3)phi2n-2 +
         phi1n-k - (4/3)phi1n-k-1 + (1/3)phi1n-k-2
*/

TransmissionLine::TransmissionLine(const std::pair<tokens_t, string_o>& s,
                                   const NodeConfig&                    ncon,
                                   const std::optional<NodeConfig>&     ncon2,
                                   const nodemap&                       nm,
                                   std::unordered_set<std::string>&     lm,
                                   nodeconnections&                     nc,
                                   const param_map&                     pm,
                                   const AnalysisType&                  at,
                                   const double&                        h,
                                   int64_t&                             bi,
                                   bool                                 txInterp) {
    at_ = at;
    // Check if the label has already been defined
    if (lm.count(s.first.at(0)) != 0) {
        Errors::invalid_component_errors(ComponentErrors::DUPLICATE_LABEL, s.first.at(0));
    }
    // Set the label
    netlistInfo.label_ = s.first.at(0);
    // Add the label to the known labels list
    lm.emplace(s.first.at(0));
    bool   useInterp = txInterp;
    double rawTd     = 0.0;
    bool   hasTd     = false;
    bool   hasAlpha  = false;
    double alphaVal  = 0.0;
    bool   hasLen    = false;
    double lenVal    = 0.0;
    // Find the parts that contain the impedance, time delay, and optional attenuation
    for (auto i = 5; i < s.first.size(); ++i) {
        const std::string& tok = s.first.at(i);
        // Impedance
        if (tok.size() >= 3 && ::memcmp(tok.c_str(), "Z0=", 3) == 0) {
            if (tok.length() < 4) {
                Errors::invalid_component_errors(ComponentErrors::INVALID_TX_DEFINED, Misc::vector_to_string(s.first));
            }
            netlistInfo.value_ = parse_param(tok.substr(3), pm, s.second);
        }
        // Time Delay
        if (tok.size() >= 3 && ::memcmp(tok.c_str(), "TD=", 3) == 0) {
            if (tok.length() < 4) {
                Errors::invalid_component_errors(ComponentErrors::INVALID_TX_DEFINED, Misc::vector_to_string(s.first));
            }
            rawTd = parse_param(tok.substr(3), pm, s.second);
            hasTd = true;
        }
        // Attenuation / Loss keywords
        if (tok.size() > 3 && ::memcmp(tok.c_str(), "DB=", 3) == 0) {
            const double db = std::abs(parse_param(tok.substr(3), pm, s.second));
            atten_ *= std::pow(10.0, -db / 20.0);
        } else if (tok.size() > 8 && ::memcmp(tok.c_str(), "LOSS_DB=", 8) == 0) {
            const double db = std::abs(parse_param(tok.substr(8), pm, s.second));
            atten_ *= std::pow(10.0, -db / 20.0);
        } else if (tok.size() > 9 && ::memcmp(tok.c_str(), "ATTEN_DB=", 9) == 0) {
            const double db = std::abs(parse_param(tok.substr(9), pm, s.second));
            atten_ *= std::pow(10.0, -db / 20.0);
        } else if (tok.size() > 5 && ::memcmp(tok.c_str(), "LOSS=", 5) == 0) {
            const double lossFrac = parse_param(tok.substr(5), pm, s.second);
            atten_ *= std::clamp(1.0 - lossFrac, 0.0, 1.0);
        } else if (tok.size() > 6 && ::memcmp(tok.c_str(), "ATTEN=", 6) == 0) {
            const double g = parse_param(tok.substr(6), pm, s.second);
            atten_ *= std::clamp(g, 0.0, 1.0);
        } else if (tok.size() > 6 && ::memcmp(tok.c_str(), "GAMMA=", 6) == 0) {
            const double g = parse_param(tok.substr(6), pm, s.second);
            atten_ *= std::clamp(g, 0.0, 1.0);
        } else if (tok.size() > 5 && ::memcmp(tok.c_str(), "GAIN=", 5) == 0) {
            const double g = parse_param(tok.substr(5), pm, s.second);
            atten_ *= std::clamp(g, 0.0, 1.0);
        } else if (tok.size() > 6 && ::memcmp(tok.c_str(), "ALPHA=", 6) == 0) {
            alphaVal = parse_param(tok.substr(6), pm, s.second);
            hasAlpha = true;
        } else if (tok.size() > 4 && ::memcmp(tok.c_str(), "LEN=", 4) == 0) {
            lenVal = parse_param(tok.substr(4), pm, s.second);
            hasLen = true;
        } else if (tok.size() > 3 && ::memcmp(tok.c_str(), "NL=", 3) == 0) {
            lenVal = parse_param(tok.substr(3), pm, s.second);
            hasLen = true;
        }
        if (tok == "INTERP=1" || tok == "INTERP=TRUE") {
            useInterp = true;
        } else if (tok == "INTERP=0" || tok == "INTERP=FALSE") {
            useInterp = false;
        }
    }
    if (hasAlpha) {
        const double effDist = hasLen ? std::abs(lenVal) : std::abs(rawTd);
        atten_ *= std::exp(-std::abs(alphaVal) * effDist);
    }
    atten_ = std::clamp(atten_, 0.0, 1.0);
    if (hasTd) {
        if (useInterp) {
            const double exactSteps = rawTd / h;
            timestepDelay_          = static_cast<int64_t>(std::floor(exactSteps + 1e-12));
            fracDelay_              = exactSteps - static_cast<double>(timestepDelay_);
            if (fracDelay_ < 1e-12) { fracDelay_ = 0.0; }
            if (1.0 - fracDelay_ < 1e-12) {
                timestepDelay_ += 1;
                fracDelay_      = 0.0;
            }
        } else {
            timestepDelay_ = static_cast<int64_t>(std::round(rawTd / h));
            fracDelay_     = 0.0;
        }
        if (timestepDelay_ >= 1) {
            histSize_ = std::max<int64_t>(timestepDelay_ + 4, 8);
            hist_.assign(static_cast<size_t>(histSize_), TxHistEntry{});
        }
    }
    // Set the node configuration type
    indexInfo.nodeConfig_   = ncon;
    // Set current index and increment it
    indexInfo.currentIndex_ = bi++;
    // Set second current index and increment it
    currentIndex2_          = bi++;
    // Set te node indices, using token 2 and 3
    set_node_indices(tokens_t(s.first.begin() + 1, s.first.begin() + 3), nm, nc);
    // Set the node configuration type for secondary nodes
    nodeConfig2_ = ncon2.value();
    // Set te node indices, using token 4 and 5
    set_secondary_node_indices(tokens_t(s.first.begin() + 3, s.first.begin() + 5), nm, nc);
    // Set the non zero, column index and row pointer vectors
    set_matrix_info();
    // Append the value to the non zero vector
    if (at == AnalysisType::Voltage) {
        // If voltage mode analysis then append -Z0
        matrixInfo.nonZeros_.emplace_back(-netlistInfo.value_);
    } else if (at == AnalysisType::Phase) {
        // If phase mdoe analysis then append -(2*h/3) * (Z0/σ)
        matrixInfo.nonZeros_.emplace_back(-(2.0 * h / 3.0) * (netlistInfo.value_ / Constants::SIGMA));
    }
    hDepPos_ = matrixInfo.nonZeros_.size() - 1;
    // Set the non zero, column index and row pointer vectors
    set_secondary_matrix_info();
    // Append the value to the non zero vector
    if (at == AnalysisType::Voltage) {
        // If voltage mode analysis then append -Z0
        matrixInfo.nonZeros_.emplace_back(-netlistInfo.value_);
    } else if (at == AnalysisType::Phase) {
        // If phase mdoe analysis then append -(2*h/3) * (Z0/σ)
        matrixInfo.nonZeros_.emplace_back(-(2.0 * h / 3.0) * (netlistInfo.value_ / Constants::SIGMA));
    }
}

void TransmissionLine::set_secondary_node_indices(const tokens_t& t, const nodemap& nm, nodeconnections& nc) {
    nodes.insert(nodes.end(), t.begin(), t.end());
    // Transmission lines have 4 nodes, this sets the 2nd pair of the 4
    switch (nodeConfig2_) {
        case NodeConfig::POSGND:
            posIndex2_ = nm.at(t.at(0));
            nc.at(nm.at(t.at(0))).emplace_back(std::make_pair(1, currentIndex2_));
            break;
        case NodeConfig::GNDNEG:
            negIndex2_ = nm.at(t.at(1));
            nc.at(nm.at(t.at(1))).emplace_back(std::make_pair(-1, currentIndex2_));
            break;
        case NodeConfig::POSNEG:
            posIndex2_ = nm.at(t.at(0));
            negIndex2_ = nm.at(t.at(1));
            if (posIndex2_.value() != negIndex2_.value()) {
                nc.at(nm.at(t.at(0))).emplace_back(std::make_pair(1, currentIndex2_));
                nc.at(nm.at(t.at(1))).emplace_back(std::make_pair(-1, currentIndex2_));
            }
            break;
        case NodeConfig::GND: break;
    }
}

void TransmissionLine::set_secondary_matrix_info() {
    // Tranmission lines have two current branches, this sets the second branch
    switch (nodeConfig2_) {
        case NodeConfig::POSGND:
            matrixInfo.nonZeros_.emplace_back(1);
            matrixInfo.columnIndex_.emplace_back(posIndex2_.value());
            matrixInfo.rowPointer_.emplace_back(2);
            break;
        case NodeConfig::GNDNEG:
            matrixInfo.nonZeros_.emplace_back(-1);
            matrixInfo.columnIndex_.emplace_back(negIndex2_.value());
            matrixInfo.rowPointer_.emplace_back(2);
            break;
        case NodeConfig::POSNEG:
            if (posIndex2_.value() != negIndex2_.value()) {
                matrixInfo.nonZeros_.emplace_back(1);
                matrixInfo.nonZeros_.emplace_back(-1);
                matrixInfo.columnIndex_.emplace_back(posIndex2_.value());
                matrixInfo.columnIndex_.emplace_back(negIndex2_.value());
                matrixInfo.rowPointer_.emplace_back(3);
            } else {
                matrixInfo.rowPointer_.emplace_back(1);
            }
            break;
        case NodeConfig::GND: matrixInfo.rowPointer_.emplace_back(1); break;
    }
    matrixInfo.columnIndex_.emplace_back(currentIndex2_);
}

// Update timestep based on a scalar factor i.e 0.5 for half the timestep
void TransmissionLine::update_timestep(const double& factor) {
    if (at_ == AnalysisType::Phase) {
        matrixInfo.nonZeros_.at(hDepPos_) = factor * matrixInfo.nonZeros_.at(hDepPos_);
        matrixInfo.nonZeros_.back()       = factor * matrixInfo.nonZeros_.back();
    }
}
