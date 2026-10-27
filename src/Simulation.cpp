// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Simulation.hpp"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/Components.hpp"
#include "JoSIM/Constants.hpp"
#include "JoSIM/Matrix.hpp"
#include "JoSIM/Model.hpp"
#include "JoSIM/ProgressBar.hpp"
#include "JoSIM/Rng.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace JoSIM;

Simulation::Simulation(Input& iObj, Matrix& mObj) {
    while (needsTR_) {
        // Do generic simulation setup for given step size
        setup(iObj, mObj);
        // Do solver setup
#ifdef SLU
        // SLU setup
        lu.create_matrix(mObj.rp.size() - 1, mObj.nz, mObj.ci, mObj.rp);
        lu.factorize();
#else
        // KLU setup
        simOK_ = klu_l_defaults(&Common_);
        assert(simOK_);
        Symbolic_ = klu_l_analyze(mObj.rp.size() - 1, &mObj.rp.front(), &mObj.ci.front(), &Common_);
        Numeric_  = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
#endif

        // Run transient simulation
        trans_sim(mObj);
        // If step size is too large, reduce and try again
        if (needsTR_) { reduce_step(iObj, mObj); }

        // Do solver cleanup
#ifdef SLU
        // SLU cleanup
        lu.free();
#else
        // KLU cleanup
        klu_l_free_symbolic(&Symbolic_, &Common_);
        klu_l_free_numeric(&Numeric_, &Common_);
#endif
    }
}

void Simulation::setup(Input& iObj, Matrix& mObj) {
    // Simulation setup
    simSize_                 = iObj.transSim.simsize();
    atyp_                    = iObj.argAnal;
    minOut_                  = iObj.argMin;
    parallel_                = iObj.argPar;
    hasJJThermalNoise_       = false;
    hasResistorThermalNoise_ = false;
    needsLU_                 = false;
    needsTR_                 = false;
    stepSize_                = iObj.transSim.tstep();
    prstep_                  = iObj.transSim.prstep();
    prstart_                 = iObj.transSim.prstart();
    startup_                 = iObj.transSim.startup();

    // Populate contiguous homogeneous device arrays for cache-friendly hot loops
    jjs_.clear();
    jjs_.reserve(mObj.components.junctionIndices.size());
    for (const auto& idx : mObj.components.junctionIndices) {
        const auto& jj = std::get<JJ>(mObj.components.devices[idx]);
        if (jj.thermalNoise) { hasJJThermalNoise_ = true; }
        jjs_.push_back(jj);
    }

    resistors_.clear();
    resistors_.reserve(mObj.components.resistorIndices.size());
    for (const auto& idx : mObj.components.resistorIndices) {
        const auto& res = std::get<Resistor>(mObj.components.devices[idx]);
        if (res.thermalNoise) { hasResistorThermalNoise_ = true; }
        resistors_.push_back(res);
    }

    std::unordered_map<int64_t, int64_t> indIdxMap;
    indIdxMap.reserve(mObj.components.inductorIndices.size());
    inductors_.clear();
    inductors_.reserve(mObj.components.inductorIndices.size());
    for (int64_t i = 0; i < static_cast<int64_t>(mObj.components.inductorIndices.size()); ++i) {
        int64_t devIdx    = mObj.components.inductorIndices[i];
        indIdxMap[devIdx] = i;
        inductors_.push_back(std::get<Inductor>(mObj.components.devices[devIdx]));
    }
    for (auto& ind : inductors_) {
        for (auto& m : ind.mutualInductances_) { m.first = indIdxMap.at(m.first); }
    }

    capacitors_.clear();
    capacitors_.reserve(mObj.components.capacitorIndices.size());
    for (const auto& idx : mObj.components.capacitorIndices) {
        capacitors_.push_back(std::get<Capacitor>(mObj.components.devices[idx]));
    }

    txlines_.clear();
    txlines_.reserve(mObj.components.txIndices.size());
    for (const auto& idx : mObj.components.txIndices) {
        txlines_.push_back(std::get<TransmissionLine>(mObj.components.devices[idx]));
    }

    vsources_.clear();
    vsources_.reserve(mObj.components.vsIndices.size());
    for (const auto& idx : mObj.components.vsIndices) {
        vsources_.push_back(std::get<VoltageSource>(mObj.components.devices[idx]));
    }

    psources_.clear();
    psources_.reserve(mObj.components.psIndices.size());
    for (const auto& idx : mObj.components.psIndices) {
        psources_.push_back(std::get<PhaseSource>(mObj.components.devices[idx]));
    }

    ccvss_.clear();
    ccvss_.reserve(mObj.components.ccvsIndices.size());
    for (const auto& idx : mObj.components.ccvsIndices) {
        ccvss_.push_back(std::get<CCVS>(mObj.components.devices[idx]));
    }

    vccss_.clear();
    vccss_.reserve(mObj.components.vccsIndices.size());
    for (const auto& idx : mObj.components.vccsIndices) {
        vccss_.push_back(std::get<VCCS>(mObj.components.devices[idx]));
    }

    b_.assign(mObj.rp.size(), 0.0);
    x_.assign(mObj.rp.size(), 0.0);
    if (!mObj.relevantTraces.empty()) {
        results.xVector.resize(mObj.branchIndex);
        for (const auto& i : mObj.relevantIndices) {
            results.xVector.at(i).emplace();
            results.xVector.at(i)->reserve(simSize_);
        }
    } else {
        results.xVector.resize(mObj.branchIndex, std::vector<double>(0));
        for (auto& v : results.xVector) { v->reserve(simSize_); }
    }
}

void Simulation::trans_sim(Matrix& mObj) {
    // Ensure time axis is cleared and pre-allocated
    results.timeAxis.clear();
    results.timeAxis.reserve(simSize_);
    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating");
        bar.set_total((float) simSize_);
    }
    // Ensure the b vector is sized once
    b_.assign(mObj.rp.size(), 0.0);
    if (startup_) {
        // Stabilize the simulation before starting at t=0
        int64_t startup = static_cast<int64_t>(2 * pow(10, (abs(log10(stepSize_)) - 12) * 2 + 1));
        if (startup > 1000) { startup = 1000; }
        for (int64_t i = -startup; i < 0; ++i) {
            // Setup the b matrix
            setup_b(mObj, i, i * stepSize_);
            if (needsTR_) { return; }
            // Assign x_prev the new b without reallocating
            std::copy(b_.begin(), b_.end(), x_.begin());
            // Solve Ax=b, storing the results in x_
#ifdef SLU
            lu.solve(x_);
#else
            simOK_ = klu_l_tsolve(Symbolic_, Numeric_, mObj.rp.size() - 1, 1, &x_.front(), &Common_);
            // If anything is a amiss, complain about it
            if (!simOK_) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
#endif
        }
    }
    const bool hasRelevantTraces = !mObj.relevantTraces.empty();
    // Start the simulation loop
    for (int64_t i = 0; i < simSize_; ++i) {
        double step = i * stepSize_;
        // If not minimal printing report progress
        if (!minOut_) { bar.update(static_cast<float>(i)); }
        // Setup the b matrix
        setup_b(mObj, i, step);
        if (needsTR_) { return; }
        // Assign x_prev the new b without reallocating
        std::copy(b_.begin(), b_.end(), x_.begin());
        // Solve Ax=b, storing the results in x_
#ifdef SLU
        lu.solve(x_);
#else
        simOK_ = klu_l_tsolve(Symbolic_, Numeric_, mObj.rp.size() - 1, 1, &x_.front(), &Common_);
        // If anything is a amiss, complain about it
        if (!simOK_) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
#endif
        // Store results (only requested, to prevent massive memory usage)
        if (hasRelevantTraces) {
            for (const auto& j : mObj.relevantIndices) { results.xVector[j]->emplace_back(x_[j]); }
        } else {
            for (size_t j = 0; j < results.xVector.size(); ++j) {
                if (results.xVector[j]) { results.xVector[j]->emplace_back(x_[j]); }
            }
        }
        // Store the time step
        results.timeAxis.emplace_back(step);
    }

    // Sync device states back to mObj.components.devices for callers that reuse mObj (e.g. IV)
    for (size_t idx = 0; idx < mObj.components.junctionIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.junctionIndices[idx]] = jjs_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.resistorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.resistorIndices[idx]] = resistors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.inductorIndices.size(); ++idx) {
        auto& dst = std::get<Inductor>(mObj.components.devices[mObj.components.inductorIndices[idx]]);
        dst.In2_  = inductors_[idx].In2_;
        dst.In3_  = inductors_[idx].In3_;
        dst.In4_  = inductors_[idx].In4_;
    }
    for (size_t idx = 0; idx < mObj.components.capacitorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.capacitorIndices[idx]] = capacitors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.txIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.txIndices[idx]] = txlines_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vsIndices[idx]] = vsources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.psIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.psIndices[idx]] = psources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.ccvsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.ccvsIndices[idx]] = ccvss_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vccsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vccsIndices[idx]] = vccss_[idx];
    }

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
}

void Simulation::reduce_step(Input& iObj, Matrix& mObj) {
    iObj.transSim.tstep(iObj.transSim.tstep() / 2);
    Rng::rewind();
    bool tempMinOut = iObj.argMin;
    if (!iObj.argMin) { iObj.argMin = true; }
    Matrix newmObj;
    mObj = newmObj;
    // Create the matrix in csr format
    for (auto& i : mObj.sourcegen) { i.clearMisc(); }
    mObj.create_matrix(iObj);
    find_relevant_traces(iObj, mObj);
    //// Dump expanded Netlist since it is no longer needed
    // iObj.netlist.expNetlist.clear();
    // iObj.netlist.expNetlist.shrink_to_fit();
    if (!tempMinOut) { iObj.argMin = tempMinOut; }
    results.xVector.clear();
    results.timeAxis.clear();
}

void Simulation::setup_b(Matrix& mObj, int64_t i, double step, double factor) {
    // Reset b vector in place without deallocating capacity
    std::fill(b_.begin(), b_.end(), 0.0);
    // Handle jj
    handle_jj(mObj, i, step, factor);
    if (needsTR_) { return; }
    // Re-factorize the LU if any jj transitions (mObj.nz is updated in-place in handle_jj)
    if (needsLU_) {
#ifdef SLU
        lu.factorize(true);
#else
        simOK_ = klu_l_refactor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, Numeric_, &Common_);
        if (!simOK_) {
            klu_l_free_numeric(&Numeric_, &Common_);
            Numeric_ = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
        }
#endif
        needsLU_ = false;
    }
    // Handle current sources
    handle_cs(mObj, step, i);
    // Handle resistors
    handle_resistors(mObj, step);
    // Handle inductors
    handle_inductors(mObj, factor);
    // Handle capacitors
    handle_capacitors(mObj);
    // Handle voltage sources
    handle_vs(mObj, i, step, factor);
    // Handle phase sources
    handle_ps(mObj, i, step, factor);
    // Handle ccvs
    handle_ccvs(mObj);
    // Handle vccs
    handle_vccs(mObj);
    // Handle transmission lines
    handle_tx(mObj, i, step);
}

void Simulation::handle_cs(Matrix& mObj, double& step, const int64_t& i) {
    for (const auto& j : mObj.components.currentsources) {
        double srcVal = mObj.sourcegen[j.sourceIndex_].value(step);
        if (j.indexInfo.nodeConfig_ == NodeConfig::POSGND) {
            b_[j.indexInfo.posIndex_.value()] -= srcVal;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::GNDNEG) {
            b_[j.indexInfo.negIndex_.value()] += srcVal;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::POSNEG) {
            b_[j.indexInfo.posIndex_.value()] -= srcVal;
            b_[j.indexInfo.negIndex_.value()] += srcVal;
        }
    }
}

void Simulation::handle_resistors(Matrix& mObj, double& step) {
    if (resistors_.empty()) { return; }
    if (!hasResistorThermalNoise_) {
        if (atyp_ != AnalysisType::Phase) { return; }
        const int64_t nRes = static_cast<int64_t>(resistors_.size());
#ifdef _OPENMP
        if (parallel_ && nRes >= 8192) {
            const int nThreads = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nRes / 4096)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t idx = 0; idx < nRes; ++idx) {
                auto&       temp = resistors_[idx];
                NodeConfig& nc   = temp.indexInfo.nodeConfig_;
                if (nc == NodeConfig::POSGND) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                } else if (nc == NodeConfig::GNDNEG) {
                    temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                } else if (nc == NodeConfig::POSNEG) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                } else {
                    temp.pn1_ = 0.0;
                }
                b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                temp.pn4_                                = temp.pn3_;
                temp.pn3_                                = temp.pn2_;
                temp.pn2_                                = temp.pn1_;
            }
            return;
        }
#endif
        for (int64_t idx = 0; idx < nRes; ++idx) {
            auto&       temp = resistors_[idx];
            NodeConfig& nc   = temp.indexInfo.nodeConfig_;
            if (nc == NodeConfig::POSGND) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (nc == NodeConfig::POSNEG) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            // 4/3 φp1 - 1/3 φp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                = temp.pn3_;
            temp.pn3_                                = temp.pn2_;
            temp.pn2_                                = temp.pn1_;
        }
        return;
    }

    for (auto& temp : resistors_) {
        NodeConfig& nc = temp.indexInfo.nodeConfig_;
        if (nc == NodeConfig::POSGND) {
            if (temp.thermalNoise) { b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step); }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (nc == NodeConfig::GNDNEG) {
            if (temp.thermalNoise) { b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step); }
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (nc == NodeConfig::POSNEG) {
            if (temp.thermalNoise) {
                b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step);
                b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step);
            }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Phase) {
            // 4/3 φp1 - 1/3 φp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                = temp.pn3_;
            temp.pn3_                                = temp.pn2_;
            temp.pn2_                                = temp.pn1_;
        }
    }
}

void Simulation::handle_inductors(Matrix& mObj, double factor) {
    if (atyp_ != AnalysisType::Voltage) { return; }
    const double h = stepSize_ * factor;
    for (auto& temp : inductors_) {
        const int64_t curIdx = temp.indexInfo.currentIndex_.value();
        // -2L/h Ip + L/2h Ip2
        b_[curIdx]           = -(2.0 * temp.netlistInfo.value_ / h) * x_[curIdx]
                     + ((temp.netlistInfo.value_ / (2.0 * h)) * temp.In2_);
        // -2M/h Im + M/2h Im2
        for (const auto& m : temp.get_mutualInductance()) {
            Inductor& mi  = inductors_[m.first];
            b_[curIdx]   += (-((2 * m.second) / h) * x_[mi.indexInfo.currentIndex_.value()]
                           + (m.second / (2.0 * h)) * mi.In2_);
        }
        temp.In4_ = temp.In3_;
        temp.In3_ = temp.In2_;
        temp.In2_ = x_[curIdx];
    }
}

void Simulation::handle_capacitors(Matrix& mObj) {
    if (capacitors_.empty()) { return; }
    const int64_t nCap = static_cast<int64_t>(capacitors_.size());
#ifdef _OPENMP
    if (parallel_ && nCap >= 8192) {
        const int nThreads = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nCap / 4096)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
        for (int64_t idx = 0; idx < nCap; ++idx) {
            auto& temp = capacitors_[idx];
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            if (atyp_ == AnalysisType::Voltage) {
                b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            } else if (atyp_ == AnalysisType::Phase) {
                b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                           + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
                temp.pn7_ = temp.pn6_;
                temp.pn6_ = temp.pn5_;
                temp.pn5_ = temp.pn4_;
                temp.pn4_ = temp.pn3_;
                temp.pn3_ = temp.pn2_;
            }
            temp.pn2_ = temp.pn1_;
        }
        return;
    }
#endif
    for (int64_t idx = 0; idx < nCap; ++idx) {
        auto& temp = capacitors_[idx];
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Voltage) {
            // 4/3 Vp1 - 1/3 Vp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        } else if (atyp_ == AnalysisType::Phase) {
            // (8/3)φn-1 - (22/9)φn-2 + (8/9)φn-3 - (1/9)φn-4
            b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
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

void Simulation::handle_jj(Matrix& mObj, int64_t& i, double& step, double factor) {
    if (jjs_.empty()) { return; }
    const int64_t nJJ        = static_cast<int64_t>(jjs_.size());
    const bool    checkPhase = ((double) i / (double) simSize_ > 0.01);

#ifdef _OPENMP
    if (parallel_ && !hasJJThermalNoise_ && nJJ >= 8192) {
        const int nThreads     = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nJJ / 4096)));
        bool      localNeedsTR = false;
        bool      localNeedsLU = false;
#pragma omp parallel for schedule(static) num_threads(nThreads) reduction(|| : localNeedsTR, localNeedsLU)
        for (int64_t idx = 0; idx < nJJ; ++idx) {
            if (localNeedsTR) { continue; }
            auto&       temp  = jjs_[idx];
            const auto& model = temp.model_;
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            if (i > 0) {
                if (atyp_ == AnalysisType::Voltage) {
                    temp.vn1_ = temp.pn1_;
                    temp.pn1_ = x_[temp.variableIndex_];
                } else if (atyp_ == AnalysisType::Phase) {
                    temp.vn1_ = x_[temp.variableIndex_];
                }
            }
            double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
            temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                         + ((1.0 / Constants::SIGMA) * ((2.0 * (stepSize_)) / 3.0)) * v0;
            if (checkPhase) {
                if (abs(temp.phi0_ - temp.pn1_) > (0.20 * 2 * Constants::PI)) {
                    localNeedsTR = true;
                    continue;
                }
            }
            if (atyp_ == AnalysisType::Voltage) {
                b_[temp.variableIndex_] = (Constants::SIGMA)
                                          * (-(2.0 / (stepSize_)) * temp.pn1_ + (1.0 / (2.0 * (stepSize_))) * temp.pn2_);
            } else if (atyp_ == AnalysisType::Phase) {
                b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            }
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
            temp.pn2_ = temp.pn1_;
            temp.vn6_ = temp.vn5_;
            temp.vn5_ = temp.vn4_;
            temp.vn4_ = temp.vn3_;
            temp.vn3_ = temp.vn2_;
            if (model.rtype() == 1) {
                auto testLU = temp.update_value(v0);
                if (testLU) {
                    mObj.nz[temp.nzBackIndex_] = temp.matrixInfo.nonZeros_.back();
                    localNeedsLU               = true;
                }
            }
            const auto&  cpr        = model.cpr();
            const double phiOff     = model.phiOff();
            const double ic         = model.ic();
            double       ic_sin_phi = 0.0;
            if (cpr.size() == 1) {
                ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
            } else {
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
                }
            }
            if (!model.tDep()) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (temp.matrixInfo.nonZeros_.back())
                          * (ic_sin_phi - (((2 * model.c()) / (stepSize_)) * temp.vn1_)
                             + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_) + temp.it_);
            } else {
                double sin2_half_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
                }
                sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                double sin_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
                }
                double sqrt_part                         = sqrt(1 - model.d() * sin2_half_phi);
                b_[temp.indexInfo.currentIndex_.value()] = (temp.matrixInfo.nonZeros_.back())
                                                           * (((((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn()))
                                                                * (sin_phi / sqrt_part)
                                                                * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t())
                                                                       * sqrt_part))
                                                               - (((2 * model.c()) / stepSize_) * temp.vn1_)
                                                               + ((model.c() / (2.0 * stepSize_)) * temp.vn2_)
                                                               + temp.it_));
            }
            temp.vn2_ = temp.vn1_;
        }
        if (localNeedsTR) {
            needsTR_ = true;
            return;
        }
        if (localNeedsLU) { needsLU_ = true; }
        return;
    }
#endif

    for (int64_t idx = 0; idx < nJJ; ++idx) {
        auto&       temp  = jjs_[idx];
        const auto& model = temp.model_;
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) { b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step); }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) { b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step); }
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) {
                b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step);
                b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step);
            }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (i > 0) {
            if (atyp_ == AnalysisType::Voltage) {
                temp.vn1_ = temp.pn1_;
                temp.pn1_ = x_[temp.variableIndex_];
            } else if (atyp_ == AnalysisType::Phase) {
                temp.vn1_ = x_[temp.variableIndex_];
                temp.pn1_ = temp.pn1_;
            }
        }
        // Guess voltage (V0)
        double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
        // Phase guess (P0)
        temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                     + ((1.0 / Constants::SIGMA) * ((2.0 * (stepSize_)) / 3.0)) * v0;
        // Ensure timestep is not too large
        if (checkPhase) {
            if (abs(temp.phi0_ - temp.pn1_) > (0.20 * 2 * Constants::PI)) {
                needsTR_ = true;
                return;
                /*Errors::simulation_errors(
                  SimulationErrors::PHASEGUESS_TOO_LARGE, temp.netlistInfo.label_);*/
            }
        }
        // (hbar / 2 * e) ( -(2 / h) φp1 + (1 / 2h) φp2 )
        if (atyp_ == AnalysisType::Voltage) {
            b_[temp.variableIndex_]
                    = (Constants::SIGMA) * (-(2.0 / (stepSize_)) * temp.pn1_ + (1.0 / (2.0 * (stepSize_))) * temp.pn2_);
            // (4 / 3) φp1 - (1/3) φp2
        } else if (atyp_ == AnalysisType::Phase) {
            b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        }
        temp.pn4_ = temp.pn3_;
        temp.pn3_ = temp.pn2_;
        temp.pn2_ = temp.pn1_;
        temp.vn6_ = temp.vn5_;
        temp.vn5_ = temp.vn4_;
        temp.vn4_ = temp.vn3_;
        temp.vn3_ = temp.vn2_;
        // Update junction transition
        if (model.rtype() == 1) {
            auto testLU = temp.update_value(v0);
            if (testLU) {
                mObj.nz[temp.nzBackIndex_] = temp.matrixInfo.nonZeros_.back();
                if (!needsLU_) { needsLU_ = true; }
            }
        }
        // Ic * sin (phi * (φ0 - φ))
        const auto&  cpr        = model.cpr();
        const double phiOff     = model.phiOff();
        const double ic         = model.ic();
        double       ic_sin_phi = 0.0;
        if (cpr.size() == 1) {
            ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
        } else {
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
            }
        }
        if (!model.tDep()) {
            // -(hR / h + 2RC) * (Ic sin (φ0) - 2C / h Vp1 + C/2h Vp2 + It)
            b_[temp.indexInfo.currentIndex_.value()]
                    = (temp.matrixInfo.nonZeros_.back())
                      * (ic_sin_phi - (((2 * model.c()) / (stepSize_)) * temp.vn1_)
                         + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_) + temp.it_);
        } else {
            double sin2_half_phi = 0.0;
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
            }
            sin2_half_phi  = sin2_half_phi * sin2_half_phi;
            double sin_phi = 0.0;
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
            }
            double sqrt_part                            = sqrt(1 - model.d() * sin2_half_phi);
            b_[temp.indexInfo.currentIndex_.value()] =
                    // -(hR / h + 2RC) *(
                    (temp.matrixInfo.nonZeros_.back())
                    * ((
                               // (π * Δ / 2 * e * Rn)
                               ((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn()))
                               // * (sin(φ0 - φ) / √(1 - D * sin²((φ0 - φ) / 2))
                               * (sin_phi / sqrt_part)
                               // * tanh(Δ / (2 * kB * T) * √(1 - D * sin²((φ0 - φ) / 2)))
                               * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                       // - 2C / h Vp1
                       - (((2 * model.c()) / stepSize_) * temp.vn1_)
                       // + C/2h Vp2
                       + ((model.c() / (2.0 * stepSize_)) * temp.vn2_)
                       // + It)
                       + temp.it_);
        }
        temp.vn2_ = temp.vn1_;
    }
}

void Simulation::handle_vs(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (auto& temp : vsources_) {
        JoSIM::NodeConfig& nc = temp.indexInfo.nodeConfig_;
        if (atyp_ == AnalysisType::Voltage) {
            // Vn
            b_[temp.indexInfo.currentIndex_.value()] = mObj.sourcegen[temp.sourceIndex_].value(step);
        } else if (atyp_ == AnalysisType::Phase) {
            if (nc == NodeConfig::POSGND) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (nc == NodeConfig::POSNEG) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                // This branch is only taken as a result of a GND-GND short-circuited voltage source.
                // In this case, the matrix is singular and the program will throw an error.
                // Without this branch, the program could crash due to std::bad_optional_access.
                temp.pn1_ = 0.0;
            }
            // (2e/hbar)(2h/3)Vn + (4/3)φn-1 - (1/3)φn-2
            b_[temp.indexInfo.currentIndex_.value()] = ((2 * stepSize_ * factor) / (3 * Constants::SIGMA))
                                                               * (mObj.sourcegen[temp.sourceIndex_].value(step))
                                                       + (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
            temp.pn2_ = temp.pn1_;
        }
    }
}

void Simulation::handle_ps(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (auto& temp : psources_) {
        if (atyp_ == AnalysisType::Phase) {
            // φn
            b_[temp.indexInfo.currentIndex_.value()] = mObj.sourcegen[temp.sourceIndex_].value(step);
        } else if (atyp_ == AnalysisType::Voltage) {
            if (i == 0) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step)));
            } else if (i == 1) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step))
                             - 2.0 * (mObj.sourcegen[temp.sourceIndex_].value(step - (stepSize_ * factor))));
            } else {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step))
                             - 2.0 * (mObj.sourcegen[temp.sourceIndex_].value(step - (stepSize_ * factor)))
                             + 0.5 * (mObj.sourcegen[temp.sourceIndex_].value(step - (2 * stepSize_ * factor))));
            }
        }
    }
}

void Simulation::handle_ccvs(Matrix& mObj) {
    if (atyp_ != AnalysisType::Phase) { return; }
    for (auto& temp : ccvss_) {
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        temp.pn4_                                = temp.pn3_;
        temp.pn3_                                = temp.pn2_;
        temp.pn2_                                = temp.pn1_;
    }
}

void Simulation::handle_vccs(Matrix& mObj) {
    if (atyp_ != AnalysisType::Phase) { return; }
    for (auto& temp : vccss_) {
        if (temp.posIndex2_ && !temp.negIndex2_) {
            temp.pn1_ = x_[temp.posIndex2_.value()];
        } else if (!temp.posIndex2_ && temp.negIndex2_) {
            temp.pn1_ = -x_[temp.negIndex2_.value()];
        } else if (temp.posIndex2_ && temp.negIndex2_) {
            temp.pn1_ = x_[temp.posIndex2_.value()] - x_[temp.negIndex2_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        temp.pn4_                                = temp.pn3_;
        temp.pn3_                                = temp.pn2_;
        temp.pn2_                                = temp.pn1_;
    }
}

void Simulation::handle_tx(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (auto& temp : txlines_) {
        // Z0
        double&            Z    = temp.netlistInfo.value_;
        // Td == k
        int64_t&           k    = temp.timestepDelay_;
        // Shorthands
        JoSIM::NodeConfig &nc = temp.indexInfo.nodeConfig_, &nc2 = temp.nodeConfig2_;
        int_o &posInd = temp.indexInfo.posIndex_, &negInd = temp.indexInfo.negIndex_, &posInd2 = temp.posIndex2_,
              &negInd2  = temp.negIndex2_;
        int64_t &curInd = temp.indexInfo.currentIndex_.value(), &curInd2 = temp.currentIndex2_;
        if (atyp_ == AnalysisType::Voltage) {
            if (i >= k) {
                // φ1n-k
                if (nc == NodeConfig::POSGND) {
                    temp.nk_1_ = (*results.xVector[posInd.value()])[i - k];
                } else if (nc == NodeConfig::GNDNEG) {
                    temp.nk_1_ = -(*results.xVector[negInd.value()])[i - k];
                } else if (nc == NodeConfig::POSNEG) {
                    temp.nk_1_ = (*results.xVector[posInd.value()])[i - k] - (*results.xVector[negInd.value()])[i - k];
                } else {
                    temp.nk_1_ = 0.0;
                }
                // φ2n-k
                if (nc2 == NodeConfig::POSGND) {
                    temp.nk_2_ = (*results.xVector[posInd2.value()])[i - k];
                } else if (nc2 == NodeConfig::GNDNEG) {
                    temp.nk_2_ = -(*results.xVector[negInd2.value()])[i - k];
                } else if (nc2 == NodeConfig::POSNEG) {
                    temp.nk_2_
                            = (*results.xVector[posInd2.value()])[i - k] - (*results.xVector[negInd2.value()])[i - k];
                } else {
                    temp.nk_2_ = 0.0;
                }
                // I1n-k
                double& I1nk = (*results.xVector[curInd])[i - k];
                // I2n-k
                double& I2nk = (*results.xVector[curInd2])[i - k];
                // I1 = ZI2n-k + V2n-k
                b_[curInd]   = Z * I2nk + temp.nk_2_;
                // I2 = ZI1n-k + V1n-k
                b_[curInd2]  = Z * I1nk + temp.nk_1_;
            }
        } else if (atyp_ == AnalysisType::Phase) {
            if (i > 0) {
                temp.n2_1_ = temp.n1_1_;
                temp.n2_2_ = temp.n1_2_;
                // φ1n-1
                if (nc == NodeConfig::POSGND) {
                    temp.n1_1_ = x_[posInd.value()];
                } else if (nc == NodeConfig::GNDNEG) {
                    temp.n1_1_ = -x_[negInd.value()];
                } else if (nc == NodeConfig::POSNEG) {
                    temp.n1_1_ = x_[posInd.value()] - x_[negInd.value()];
                } else {
                    temp.n1_1_ = 0.0;
                }
                // φ2n-1
                if (nc2 == NodeConfig::POSGND) {
                    temp.n1_2_ = x_[posInd2.value()];
                } else if (nc2 == NodeConfig::GNDNEG) {
                    temp.n1_2_ = -x_[negInd2.value()];
                } else if (nc2 == NodeConfig::POSNEG) {
                    temp.n1_2_ = x_[posInd2.value()] - x_[negInd2.value()];
                } else {
                    temp.n1_2_ = 0.0;
                }
            }
            if (i >= k) {
                // φ1n-k
                if (nc == NodeConfig::POSGND) {
                    temp.nk_1_ = (*results.xVector[posInd.value()])[i - k];
                } else if (nc == NodeConfig::GNDNEG) {
                    temp.nk_1_ = -(*results.xVector[negInd.value()])[i - k];
                } else if (nc == NodeConfig::POSNEG) {
                    temp.nk_1_ = (*results.xVector[posInd.value()])[i - k] - (*results.xVector[negInd.value()])[i - k];
                } else {
                    temp.nk_1_ = 0.0;
                }
                // φ2n-k
                if (nc2 == NodeConfig::POSGND) {
                    temp.nk_2_ = (*results.xVector[posInd2.value()])[i - k];
                } else if (nc2 == NodeConfig::GNDNEG) {
                    temp.nk_2_ = -(*results.xVector[negInd2.value()])[i - k];
                } else if (nc2 == NodeConfig::POSNEG) {
                    temp.nk_2_
                            = (*results.xVector[posInd2.value()])[i - k] - (*results.xVector[negInd2.value()])[i - k];
                } else {
                    temp.nk_2_ = 0.0;
                }
                // I1n-k
                double& I1nk = (*results.xVector[curInd])[i - k];
                // I2n-k
                double& I2nk = (*results.xVector[curInd2])[i - k];
                if (i == k) {
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 + φ2n-k
                    b_[curInd]  = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 + φ1n-k
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_;
                } else if (i == k + 1) {
                    // φ1n-k-1
                    if (nc == NodeConfig::POSGND) {
                        temp.nk1_1_ = (*results.xVector[posInd.value()])[i - k - 1];
                    } else if (nc == NodeConfig::GNDNEG) {
                        temp.nk1_1_ = -(*results.xVector[negInd.value()])[i - k - 1];
                    } else if (nc == NodeConfig::POSNEG) {
                        temp.nk1_1_ = (*results.xVector[posInd.value()])[i - k - 1]
                                      - (*results.xVector[negInd.value()])[i - k - 1];
                    } else {
                        temp.nk1_1_ = 0.0;
                    }
                    // φ2n-k-1
                    if (nc2 == NodeConfig::POSGND) {
                        temp.nk1_2_ = (*results.xVector[posInd2.value()])[i - k - 1];
                    } else if (nc2 == NodeConfig::GNDNEG) {
                        temp.nk1_2_ = -(*results.xVector[negInd2.value()])[i - k - 1];
                    } else if (nc2 == NodeConfig::POSNEG) {
                        temp.nk1_2_ = (*results.xVector[posInd2.value()])[i - k - 1]
                                      - (*results.xVector[negInd2.value()])[i - k - 1];
                    } else {
                        temp.nk1_2_ = 0.0;
                    }
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 +
                    //      φ2n-k - (4/3)φ2n-k-1
                    b_[curInd]  = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 +
                    //      φ1n-k - (4/3)φ1n-k-1
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_;
                } else if (i > k + 1) {
                    // φ1n-k-1
                    if (nc == NodeConfig::POSGND) {
                        temp.nk1_1_ = (*results.xVector[posInd.value()])[i - k - 1];
                    } else if (nc == NodeConfig::GNDNEG) {
                        temp.nk1_1_ = -(*results.xVector[negInd.value()])[i - k - 1];
                    } else if (nc == NodeConfig::POSNEG) {
                        temp.nk1_1_ = (*results.xVector[posInd.value()])[i - k - 1]
                                      - (*results.xVector[negInd.value()])[i - k - 1];
                    } else {
                        temp.nk1_1_ = 0.0;
                    }
                    // φ2n-k-1
                    if (nc2 == NodeConfig::POSGND) {
                        temp.nk1_2_ = (*results.xVector[posInd2.value()])[i - k - 1];
                    } else if (nc2 == NodeConfig::GNDNEG) {
                        temp.nk1_2_ = -(*results.xVector[negInd2.value()])[i - k - 1];
                    } else if (nc2 == NodeConfig::POSNEG) {
                        temp.nk1_2_ = (*results.xVector[posInd2.value()])[i - k - 1]
                                      - (*results.xVector[negInd2.value()])[i - k - 1];
                    } else {
                        temp.nk1_2_ = 0.0;
                    }
                    // φ1n-k-2
                    if (nc == NodeConfig::POSGND) {
                        temp.nk2_1_ = (*results.xVector[posInd.value()])[i - k - 2];
                    } else if (nc == NodeConfig::GNDNEG) {
                        temp.nk2_1_ = -(*results.xVector[negInd.value()])[i - k - 2];
                    } else if (nc == NodeConfig::POSNEG) {
                        temp.nk2_1_ = (*results.xVector[posInd.value()])[i - k - 2]
                                      - (*results.xVector[negInd.value()])[i - k - 2];
                    } else {
                        temp.nk2_1_ = 0.0;
                    }
                    // φ2n-k-2
                    if (nc2 == NodeConfig::POSGND) {
                        temp.nk2_2_ = (*results.xVector[posInd2.value()])[i - k - 2];
                    } else if (nc2 == NodeConfig::GNDNEG) {
                        temp.nk2_2_ = -(*results.xVector[negInd2.value()])[i - k - 2];
                    } else if (nc2 == NodeConfig::POSNEG) {
                        temp.nk2_2_ = (*results.xVector[posInd2.value()])[i - k - 2]
                                      - (*results.xVector[negInd2.value()])[i - k - 2];
                    } else {
                        temp.nk2_2_ = 0.0;
                    }
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 +
                    //      φ2n-k - (4/3)φ2n-k-1 + (1/3)φ2n-k-2
                    b_[curInd]  = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_ + (1.0 / 3.0) * temp.nk2_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 +
                    //      φ1n-k - (4/3)φ1n-k-1 + (1/3)φ1n-k-2
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_ + (1.0 / 3.0) * temp.nk2_1_;
                }
            } else {
                // I1 = (4/3)φ1n-1 - (1/3)φ1n-2
                b_[curInd]  = (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_;
                // I2 = (4/3)φ2n-1 - (1/3)φ2n-2
                b_[curInd2] = (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_;
            }
        }
    }
}
