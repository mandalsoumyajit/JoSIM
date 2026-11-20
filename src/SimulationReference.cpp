// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
//
// Reference MNA transient path: upstream JoSIM 3.1's trans_sim and step helpers, unchanged except for names.
// The fork's solvers (nodal, partitioned, GPU, Monte Carlo, the contiguous-array MNA loop) model the devices
// of JoSIM <= 2.7. Decks that use what only upstream 3.x models -- gate-controlled (GC=) or Ic-noise
// (ICNOISE=) junctions, nonlinear inductors (ALPHA=), capacitor IC= and colored-noise sources -- run here,
// with upstream's numerics, on the device state held in the matrix's components. Transmission lines and
// nanowires use the fork's handlers. Speed is not a concern for these decks.

#include "JoSIM/Simulation.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Noise.hpp"
#include "JoSIM/ProgressBar.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace JoSIM;

void Simulation::trans_sim_reference(Matrix& mObj) {
    // Ensure time axis is cleared
    results.timeAxis.clear();
    ProgressBar bar;
    if (!minOut_) {
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating");
        bar.set_total((float) simSize_);
        bar.create_thread();
    }
    // Initialize the b matrix
    b_.resize(mObj.rp.size(), 0.0);
    capacitorIC_ = initialize_capacitors(mObj);
    if (capacitorIC_) {
        // The fork's line handler keeps its state in the solver's device array, copied in setup(); give it the
        // initial histories initialize_capacitors() just wrote into the components.
        for (size_t k = 0; k < txlines_.size(); ++k) {
            txlines_[k] = std::get<TransmissionLine>(mObj.components.devices.at(mObj.components.txIndices.at(k)));
        }
    }
    if (startup_) {
        // Stabilize the simulation before starting at t=0
        const double proposed = 2 * pow(10, (abs(log10(stepSize_)) - 12) * 2 + 1);
        const int64_t startup = proposed >= 1000 ? 1000 : static_cast<int64_t>(proposed);
        if (!std::isfinite(static_cast<double>(startup) * stepSize_)) {
            Errors::control_errors(ControlErrors::TRANS_ERROR, "Startup interval exceeds the supported range");
        }
        for (int64_t i = -startup; i < 0; ++i) {
            // Setup the b matrix
            ref_setup_b(mObj, i, i * stepSize_);
            if (needsTR_) { return; }
            if (!solve_step(mObj)) { return; }
            for (auto& history : results.phaseBefore) {
                const auto node = x_.at(history.first);
                const double phase = atyp_ == AnalysisType::Phase ? node
                    : (2.0 * stepSize_ / (3.0 * Constants::SIGMA)) * node
                      + (4.0 / 3.0) * history.second[0] - (1.0 / 3.0) * history.second[1];
                history.second[1] = history.second[0];
                history.second[0] = phase;
            }
        }
    }
    // Start the simulation loop
    for (int64_t i = capacitorIC_ ? 1 : 0; i < simSize_; ++i) {
        double step = i * stepSize_;
        // If not minimal printing report progress
        if (!minOut_) { bar.update(static_cast<float>(i)); }
        // Setup the b matrix
        ref_setup_b(mObj, i, i * stepSize_);
        if (needsTR_) { return; }
        if (!solve_step(mObj)) { return; }
        // Store results (only requested, to prevent massive memory usage)
        for (auto j = 0; j < results.xVector.size(); ++j) {
            if (results.xVector.at(j)) { results.xVector.at(j).value().emplace_back(x_.at(j)); }
        }
        // Store the time step
        results.timeAxis.emplace_back(step);
    }
    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
}

void Simulation::ref_setup_b(Matrix& mObj, int64_t i, double step, double factor) {
    // Clear b matrix and reset
    b_.clear();
    b_.resize(mObj.rp.size(), 0.0);
    // Assemble flux histories before shifting any device state.
    ref_handle_inductors(mObj, factor);
    if (needsTR_) { return; }
    // Handle jj
    ref_handle_jj(mObj, i, step, factor);
    if (needsTR_) { return; }
    // Fork: nanowires/cryotrons keep their state in the solver's device array.
    handle_nanowires(mObj, i, step, factor);
    if (needsLU_) { factorize(mObj); }
    // Handle current sources
    ref_handle_cs(mObj, step, i);
    // Handle resistors
    ref_handle_resistors(mObj, step);
    // Handle capacitors
    ref_handle_capacitors(mObj);
    // Handle voltage sources
    ref_handle_vs(mObj, i, step, factor);
    // Handle phase sources
    ref_handle_ps(mObj, i, step, factor);
    // Handle ccvs
    ref_handle_ccvs(mObj);
    // Handle vccs
    ref_handle_vccs(mObj);
    // Transmission lines: the fork's ring-buffer lines (their state is in the solver's device array).
    handle_tx(mObj, i, step);
}

void Simulation::ref_handle_cs(Matrix& mObj, double& step, const int64_t& i) {
    for (const auto& j : mObj.components.currentsources) {
        const double value = mObj.sourcegen.at(j.sourceIndex_).value(step);
        if (i >= 0 && !results.sourceCurrents.empty() && results.sourceCurrents.at(j.sourceIndex_)) {
            results.sourceCurrents.at(j.sourceIndex_).value().emplace_back(value);
        }
        if (j.indexInfo.nodeConfig_ == NodeConfig::POSGND) {
            b_.at(j.indexInfo.posIndex_.value()) -= value;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::GNDNEG) {
            b_.at(j.indexInfo.negIndex_.value()) += value;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::POSNEG) {
            b_.at(j.indexInfo.posIndex_.value()) -= value;
            b_.at(j.indexInfo.negIndex_.value()) += value;
        }
    }
}

void Simulation::ref_handle_resistors(Matrix& mObj, double& step) {
    double noiseVal;
    for (const auto& j : mObj.components.resistorIndices) {
        auto&       temp = std::get<Resistor>(mObj.components.devices.at(j));
        if (temp.thermalNoise) {
            noiseVal = temp.thermalNoise.value().value(step);
        } else {
            noiseVal = 0.0;
        }
        NodeConfig& nc   = temp.indexInfo.nodeConfig_;
        if (nc == NodeConfig::POSGND) {
            b_.at(temp.indexInfo.posIndex_.value()) -= noiseVal;
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()));
        } else if (nc == NodeConfig::GNDNEG) {
            b_.at(temp.indexInfo.negIndex_.value()) += noiseVal;
            temp.pn1_ = (-x_.at(temp.indexInfo.negIndex_.value()));
        } else if (nc == NodeConfig::POSNEG) {
            b_.at(temp.indexInfo.posIndex_.value()) -= noiseVal;
            b_.at(temp.indexInfo.negIndex_.value()) += noiseVal;
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()) - x_.at(temp.indexInfo.negIndex_.value()));
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Phase) {
            // 4/3 φp1 - 1/3 φp2
            b_.at(temp.indexInfo.currentIndex_.value()) = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                   = temp.pn3_;
            temp.pn3_                                   = temp.pn2_;
            temp.pn2_                                   = temp.pn1_;
        }
    }
}

void Simulation::ref_handle_inductors(Matrix& mObj, double factor) {
    for (const auto& j : mObj.components.inductorIndices) {
        auto& temp = std::get<Inductor>(mObj.components.devices.at(j));
        if (!nonlinearInductors_.empty()) {
            const double current = x_.at(*temp.indexInfo.currentIndex_);
            const double flux = temp.flux(current), slope = temp.differential_inductance(current);
            temp.fluxHistory_ = 0.0;
            if (atyp_ == AnalysisType::Voltage) {
                double totalFlux = flux;
                for (const auto& mutual : temp.mutualInductances_) {
                    const auto& other = std::get<Inductor>(mObj.components.devices.at(mutual.first));
                    totalFlux += mutual.second * x_.at(*other.indexInfo.currentIndex_);
                }
                temp.fluxHistory_ = -(4.0 / 3.0) * totalFlux + (1.0 / 3.0) * temp.previousFlux_;
                temp.previousFlux_ = totalFlux;
            }
            const double scale = atyp_ == AnalysisType::Voltage ? 1.5 / (stepSize_ * factor) : 1.0 / Constants::SIGMA;
            const double rhs = scale * (temp.fluxHistory_ + (temp.alpha_ > 0 ? flux - slope * current : 0.0));
            if (!std::isfinite(rhs) || !std::isfinite(slope) || !std::isfinite(scale * slope)) {
                nonlinearFailure_ = temp.netlistInfo.label_ + ": flux or differential inductance exceeds the numerical range";
                needsTR_ = true;
                return;
            }
            if (temp.alpha_ > 0) { needsLU_ = temp.linearize(current, stepSize_ * factor) || needsLU_; }
            b_.at(*temp.indexInfo.currentIndex_) = rhs;
        } else if (atyp_ == AnalysisType::Voltage) {
            // -2L/h Ip + L/2h Ip2
            b_.at(temp.indexInfo.currentIndex_.value())
                    = -(2.0 * temp.netlistInfo.value_ / (stepSize_ * factor))
                              * x_.at(temp.indexInfo.currentIndex_.value())
                      + ((temp.netlistInfo.value_ / (2.0 * (stepSize_ * factor))) * temp.In2_);
            // -2M/h Im + M/2h Im2
            for (const auto& m : temp.mutualInductances_) {
                Inductor& mi = std::get<Inductor>(mObj.components.devices.at(m.first));
                b_.at(temp.indexInfo.currentIndex_.value())
                        += (-((2 * m.second) / (stepSize_ * factor)) * x_.at(mi.indexInfo.currentIndex_.value())
                            + (m.second / (2.0 * (stepSize_ * factor))) * mi.In2_);
            }
        }
    }
    if (atyp_ == AnalysisType::Voltage) {
        // Every coupled branch must read the same preceding current histories.
        for (const auto& j : mObj.components.inductorIndices) {
            auto& temp = std::get<Inductor>(mObj.components.devices.at(j));
            temp.In4_ = temp.In3_;
            temp.In3_ = temp.In2_;
            temp.In2_ = x_.at(temp.indexInfo.currentIndex_.value());
        }
    }
}

void Simulation::ref_handle_capacitors(Matrix& mObj) {
    for (const auto& j : mObj.components.capacitorIndices) {
        auto& temp = std::get<Capacitor>(mObj.components.devices.at(j));
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()));
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = (-x_.at(temp.indexInfo.negIndex_.value()));
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()) - x_.at(temp.indexInfo.negIndex_.value()));
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Voltage) {
            // 4/3 Vp1 - 1/3 Vp2
            b_.at(temp.indexInfo.currentIndex_.value()) = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        } else if (atyp_ == AnalysisType::Phase) {
            // (8/3)φn-1 - (22/9)φn-2 + (8/9)φn-3 - (1/9)φn-4
            b_.at(temp.indexInfo.currentIndex_.value()) = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                          + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
            temp.pn7_ = temp.pn6_;
            temp.pn6_ = temp.pn5_;
            temp.pn5_ = temp.pn4_;
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
        }
        temp.pn2_ = temp.pn1_;
    }
}

void Simulation::ref_handle_jj(Matrix& mObj, int64_t& i, double& step, double factor) {
    for (const auto& j : mObj.components.junctionIndices) {
        auto&       temp  = std::get<JJ>(mObj.components.devices.at(j));
        const auto& model = temp.model_;

        // ---- Node voltage/phase "pn1_" and add thermal noise (as current injection) ----
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()));
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = (-x_.at(temp.indexInfo.negIndex_.value()));
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()) - x_.at(temp.indexInfo.negIndex_.value()));
        } else {
            temp.pn1_ = 0.0;
        }

        // Use the previous solved phase/voltage during startup and at t=0,
        // as well as later samples. Zero-state runs already start with x_=0.
        if (atyp_ == AnalysisType::Voltage) {
            temp.vn1_ = temp.pn1_;
            temp.pn1_ = x_.at(temp.variableIndex_);
        } else if (atyp_ == AnalysisType::Phase) {
            temp.vn1_ = x_.at(temp.variableIndex_);
            // temp.pn1_ already holds the node phase difference
        }

        // ---- Guess voltage (V0) and phase (phi0) ----
        double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
        temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                     + ((1.0 / Constants::SIGMA) * ((2.0 * (stepSize_)) / 3.0)) * v0;

        // ---- Timestep stability check ----
        const double dphi = temp.phi0_ - temp.pn1_;
        if ((double) i / (double) simSize_ > 0.01) {
            if (std::abs(dphi) > (0.20 * Constants::TWO_PI)) {
                needsTR_ = true;
                return;
            }
        }

        // ---- RHS1 (phase/voltage equation row) ----
        if (atyp_ == AnalysisType::Voltage) {
            b_.at(temp.variableIndex_)
                    = (Constants::SIGMA) * (-(2.0 / (stepSize_)) * temp.pn1_ + (1.0 / (2.0 * (stepSize_))) * temp.pn2_);
        } else if (atyp_ == AnalysisType::Phase) {
            b_.at(temp.variableIndex_) = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        }

        // ---- Shift history (phase + voltage) ----
        temp.pn4_ = temp.pn3_;
        temp.pn3_ = temp.pn2_;
        temp.pn2_ = temp.pn1_;

        temp.vn6_ = temp.vn5_;
        temp.vn5_ = temp.vn4_;
        temp.vn4_ = temp.vn3_;
        temp.vn3_ = temp.vn2_;

        // ---- Update junction transition (may trigger LU refactor) ----
        if (model.rtype() == 1) {
            const bool testLU = temp.update_value(v0);
            if (testLU && !needsLU_) { needsLU_ = true; }
        }
        // Apply one sample using the conductance selected for this step.
        if (temp.thermalNoise) {
            const double noise = temp.thermalNoise.value().value(step);
            if (temp.indexInfo.posIndex_) { b_.at(*temp.indexInfo.posIndex_) -= noise; }
            if (temp.indexInfo.negIndex_) { b_.at(*temp.indexInfo.negIndex_) += noise; }
        }
        double gateFactor = 1.0;
        if (temp.gateIndex_) {
            const double signal = x_.at(temp.gateIndex_.value());
            double voltage = signal;
            if (atyp_ == AnalysisType::Phase) {
                voltage = Constants::SIGMA * (3.0 * signal - 4.0 * temp.gatePn1_ + temp.gatePn2_)
                          / (2.0 * stepSize_);
                temp.gatePn2_ = temp.gatePn1_;
                temp.gatePn1_ = signal;
            }
            gateFactor = std::clamp(1.0 - std::abs(voltage) / model.gwidth(), 0.0, 1.0);
        }

        // Fractional critical-current noise: δ(t) dimensionless, generated by ICNOISE(...)
        double dIc = 0.0;
        if (temp.icNoise) { dIc = temp.icNoise.value().value(step); }
        if (dIc < -0.99) { dIc = -0.99; }

        // ---- RHS2 (junction current equation row) ----
        b_.at(temp.indexInfo.currentIndex_.value())
                = temp.matrixInfo.nonZeros_.back()
                  * (temp.supercurrent(temp.phi0_, gateFactor * (1.0 + dIc))
                     - ((2.0 * model.c()) / stepSize_) * temp.vn1_
                     + (model.c() / (2.0 * stepSize_)) * temp.vn2_ + temp.it_);

        // ---- Final history update (as in your original code) ----
        temp.vn2_ = temp.vn1_;
    }
}

void Simulation::ref_handle_vs(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (const auto& j : mObj.components.vsIndices) {
        auto&              temp = std::get<VoltageSource>(mObj.components.devices.at(j));
        JoSIM::NodeConfig& nc   = temp.indexInfo.nodeConfig_;
        if (atyp_ == AnalysisType::Voltage) {
            // Vn
            b_.at(temp.indexInfo.currentIndex_.value()) = (mObj.sourcegen.at(temp.sourceIndex_).value(step));
        } else if (atyp_ == AnalysisType::Phase) {
            if (nc == NodeConfig::POSGND) {
                temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()));
            } else if (nc == NodeConfig::GNDNEG) {
                temp.pn1_ = (-x_.at(temp.indexInfo.negIndex_.value()));
            } else if (nc == NodeConfig::POSNEG) {
                temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()) - x_.at(temp.indexInfo.negIndex_.value()));
            } else {
                // This branch is only taken as a result of a GND-GND short-circuited voltage source.
                // In this case, the matrix is singular and the program will throw an error.
                // Without this branch, the program could crash due to std::bad_optional_access.
                temp.pn1_ = 0.0;
            }
            // (2e/hbar)(2h/3)Vn + (4/3)φn-1 - (1/3)φn-2
            b_.at(temp.indexInfo.currentIndex_.value()) = ((2 * stepSize_ * factor) / (3 * Constants::SIGMA))
                                                                  * (mObj.sourcegen.at(temp.sourceIndex_).value(step))
                                                          + (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
            temp.pn2_ = temp.pn1_;
        }
    }
}

void Simulation::ref_handle_ps(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (const auto& j : mObj.components.psIndices) {
        auto& temp = std::get<PhaseSource>(mObj.components.devices.at(j));
        const double phase = mObj.sourcegen.at(temp.sourceIndex_).value(step);
        if (atyp_ == AnalysisType::Phase) {
            // φn
            b_.at(temp.indexInfo.currentIndex_.value()) = phase;
        } else if (atyp_ == AnalysisType::Voltage) {
            b_.at(temp.indexInfo.currentIndex_.value()) = (Constants::SIGMA / (stepSize_ * factor))
                * (1.5 * phase - 2.0 * temp.pn1_ + 0.5 * temp.pn2_);
        }
        temp.pn2_ = temp.pn1_;
        temp.pn1_ = phase;
    }
}

void Simulation::ref_handle_ccvs(Matrix& mObj) {
    for (const auto& j : mObj.components.ccvsIndices) {
        auto& temp = std::get<CCVS>(mObj.components.devices.at(j));
        if (atyp_ == AnalysisType::Phase) {
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()));
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = (-x_.at(temp.indexInfo.negIndex_.value()));
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = (x_.at(temp.indexInfo.posIndex_.value()) - x_.at(temp.indexInfo.negIndex_.value()));
            } else {
                temp.pn1_ = 0.0;
            }
            b_.at(temp.indexInfo.currentIndex_.value()) = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                   = temp.pn3_;
            temp.pn3_                                   = temp.pn2_;
            temp.pn2_                                   = temp.pn1_;
        }
    }
}

void Simulation::ref_handle_vccs(Matrix& mObj) {
    for (const auto& j : mObj.components.vccsIndices) {
        auto& temp = std::get<VCCS>(mObj.components.devices.at(j));
        if (atyp_ == AnalysisType::Phase) {
            if (temp.posIndex2_ && !temp.negIndex2_) {
                temp.pn1_ = (x_.at(temp.posIndex2_.value()));
            } else if (!temp.posIndex2_ && temp.negIndex2_) {
                temp.pn1_ = (-x_.at(temp.negIndex2_.value()));
            } else if (temp.posIndex2_ && temp.negIndex2_) {
                temp.pn1_ = (x_.at(temp.posIndex2_.value()) - x_.at(temp.negIndex2_.value()));
            } else {
                temp.pn1_ = 0.0;
            }
            b_.at(temp.indexInfo.currentIndex_.value()) = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                   = temp.pn3_;
            temp.pn3_                                   = temp.pn2_;
            temp.pn2_                                   = temp.pn1_;
        }
    }
}
