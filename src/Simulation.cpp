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
#include <cstdint>
#include <fstream>
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
    fixedStep_               = iObj.argFixedStep;
    warnStep_                = iObj.argWarnStep;
    verb_                    = iObj.argVerb;
    correctorIter_           = iObj.argCorrector;
    correctorTol_            = iObj.argCorrectorTol;
    saveStateFile_           = iObj.saveStateFile;
    loadStateFile_           = iObj.loadStateFile;
    offendingJJ_.clear();
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
    if (loadStateFile_.has_value()) {
        load_state(loadStateFile_.value(), mObj);
    } else if (startup_) {
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
            if (correctorIter_ > 0 && !jjs_.empty()) { corrector_jj(mObj); }
        }
    }
    const bool hasRelevantTraces = !mObj.relevantTraces.empty();
    bool       stateSaved        = false;
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
        if (correctorIter_ > 0 && !jjs_.empty()) { corrector_jj(mObj); }
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
        if (!stateSaved && saveStateFile_.has_value() && saveStateFile_->second.has_value()
            && step >= saveStateFile_->second.value()) {
            save_state(saveStateFile_->first, mObj);
            stateSaved = true;
        }
    }
    if (!stateSaved && saveStateFile_.has_value()) {
        save_state(saveStateFile_->first, mObj);
        stateSaved = true;
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
    if (fixedStep_) {
        Errors::simulation_errors(SimulationErrors::PHASEGUESS_TOO_LARGE,
                                  offendingJJ_.empty() ? "unknown" : offendingJJ_);
    }
    const double oldStep = iObj.transSim.tstep();
    const double newStep = oldStep / 2.0;
    ++stepHalvings_;
    if (verb_ >= 1 || warnStep_) {
        std::cout << "JoSIM: Step halving #" << stepHalvings_ << " triggered by "
                  << (offendingJJ_.empty() ? "JJ" : offendingJJ_) << " (tstep " << oldStep << " -> " << newStep
                  << ")\n";
    }
    iObj.transSim.tstep(newStep);
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
            temp.cTerm_ = -(((2 * model.c()) / (stepSize_)) * temp.vn1_)
                          + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_);
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
                offendingJJ_ = temp.netlistInfo.label_;
                needsTR_     = true;
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
        temp.cTerm_ = -(((2 * model.c()) / (stepSize_)) * temp.vn1_)
                      + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_);
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

void Simulation::corrector_jj(Matrix& mObj) {
    const int64_t nJJ = static_cast<int64_t>(jjs_.size());
    for (int64_t iter = 0; iter < correctorIter_; ++iter) {
        double maxDiff     = 0.0;
        bool   corrNeedsLU = false;
        for (int64_t idx = 0; idx < nJJ; ++idx) {
            auto&  temp    = jjs_[idx];
            double nodeVal = 0.0;
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                nodeVal = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                nodeVal = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                nodeVal = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            }
            const double phiNew = (atyp_ == AnalysisType::Phase) ? nodeVal : x_[temp.variableIndex_];
            const double vNew   = (atyp_ == AnalysisType::Phase) ? x_[temp.variableIndex_] : nodeVal;
            const double diff   = std::abs(phiNew - temp.phi0_);
            if (diff > maxDiff) { maxDiff = diff; }
            temp.phi0_        = 0.5 * (temp.phi0_ + phiNew);
            const auto& model = temp.model_;
            if (model.rtype() == 1) {
                if (temp.update_value(vNew)) {
                    mObj.nz[temp.nzBackIndex_] = temp.matrixInfo.nonZeros_.back();
                    corrNeedsLU                = true;
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
                        = (temp.matrixInfo.nonZeros_.back()) * (ic_sin_phi + temp.cTerm_ + temp.it_);
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
                double sqrt_part = sqrt(1 - model.d() * sin2_half_phi);
                b_[temp.indexInfo.currentIndex_.value()]
                        = (temp.matrixInfo.nonZeros_.back())
                          * ((((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn())) * (sin_phi / sqrt_part)
                              * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                             + temp.cTerm_ + temp.it_);
            }
        }
        if (maxDiff <= correctorTol_ && !corrNeedsLU) { break; }
        if (corrNeedsLU) {
#ifdef SLU
            lu.factorize(true);
#else
            simOK_ = klu_l_refactor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, Numeric_, &Common_);
            if (!simOK_) {
                klu_l_free_numeric(&Numeric_, &Common_);
                Numeric_ = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
            }
#endif
        }
        std::copy(b_.begin(), b_.end(), x_.begin());
#ifdef SLU
        lu.solve(x_);
#else
        simOK_ = klu_l_tsolve(Symbolic_, Numeric_, mObj.rp.size() - 1, 1, &x_.front(), &Common_);
        if (!simOK_) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
#endif
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
    auto sample_port = [&](JoSIM::NodeConfig nc, const int_o& pInd, const int_o& nInd, int64_t idx) -> double {
        if (idx < 0) { return 0.0; }
        if (nc == NodeConfig::POSGND) { return (*results.xVector[pInd.value()])[idx]; }
        if (nc == NodeConfig::GNDNEG) { return -(*results.xVector[nInd.value()])[idx]; }
        if (nc == NodeConfig::POSNEG) {
            return (*results.xVector[pInd.value()])[idx] - (*results.xVector[nInd.value()])[idx];
        }
        return 0.0;
    };
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
        if (temp.fracDelay_ > 0.0) {
            const double alpha = temp.fracDelay_;
            const double oma   = 1.0 - alpha;
            auto         interp_port
                    = [&](JoSIM::NodeConfig cfg, const int_o& pI, const int_o& nI, int64_t m) -> double {
                if (m < 0) { return 0.0; }
                return oma * sample_port(cfg, pI, nI, m) + alpha * sample_port(cfg, pI, nI, m - 1);
            };
            auto interp_cur = [&](int64_t cIdx, int64_t m) -> double {
                if (m < 0) { return 0.0; }
                const auto& vec = *results.xVector[cIdx];
                return oma * vec[m] + (m >= 1 ? alpha * vec[m - 1] : 0.0);
            };
            if (atyp_ == AnalysisType::Voltage) {
                if (i >= k) {
                    const int64_t m    = i - k;
                    temp.nk_1_         = interp_port(nc, posInd, negInd, m);
                    temp.nk_2_         = interp_port(nc2, posInd2, negInd2, m);
                    const double  I1nk = interp_cur(curInd, m);
                    const double  I2nk = interp_cur(curInd2, m);
                    b_[curInd]         = Z * I2nk + temp.nk_2_;
                    b_[curInd2]        = Z * I1nk + temp.nk_1_;
                }
            } else if (atyp_ == AnalysisType::Phase) {
                if (i > 0) {
                    temp.n2_1_ = temp.n1_1_;
                    temp.n2_2_ = temp.n1_2_;
                    if (nc == NodeConfig::POSGND) {
                        temp.n1_1_ = x_[posInd.value()];
                    } else if (nc == NodeConfig::GNDNEG) {
                        temp.n1_1_ = -x_[negInd.value()];
                    } else if (nc == NodeConfig::POSNEG) {
                        temp.n1_1_ = x_[posInd.value()] - x_[negInd.value()];
                    } else {
                        temp.n1_1_ = 0.0;
                    }
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
                    const int64_t m    = i - k;
                    temp.nk_1_         = interp_port(nc, posInd, negInd, m);
                    temp.nk_2_         = interp_port(nc2, posInd2, negInd2, m);
                    temp.nk1_1_        = interp_port(nc, posInd, negInd, m - 1);
                    temp.nk1_2_        = interp_port(nc2, posInd2, negInd2, m - 1);
                    temp.nk2_1_        = interp_port(nc, posInd, negInd, m - 2);
                    temp.nk2_2_        = interp_port(nc2, posInd2, negInd2, m - 2);
                    const double  I1nk = interp_cur(curInd, m);
                    const double  I2nk = interp_cur(curInd2, m);
                    b_[curInd]         = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_ + (1.0 / 3.0) * temp.nk2_2_;
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_ + (1.0 / 3.0) * temp.nk2_1_;
                } else {
                    b_[curInd]  = (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_;
                    b_[curInd2] = (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_;
                }
            }
            continue;
        }
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

void Simulation::save_state(const std::string& filename, const Matrix& mObj) const {
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    const char     magic[8] = {'J', 'O', 'S', 'I', 'M', 'S', 'T', 'A'};
    const uint32_t version  = 1;
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    auto write_u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(uint64_t)); };
    auto write_f64 = [&](double v) { out.write(reinterpret_cast<const char*>(&v), sizeof(double)); };

    write_u64(static_cast<uint64_t>(x_.size()));
    out.write(reinterpret_cast<const char*>(x_.data()), static_cast<std::streamsize>(x_.size() * sizeof(double)));

    write_u64(static_cast<uint64_t>(jjs_.size()));
    for (const auto& jj : jjs_) {
        const double vals[13] = {jj.pn1_,
                                 jj.pn2_,
                                 jj.pn3_,
                                 jj.pn4_,
                                 jj.vn1_,
                                 jj.vn2_,
                                 jj.vn3_,
                                 jj.vn4_,
                                 jj.vn5_,
                                 jj.vn6_,
                                 jj.phi0_,
                                 jj.it_,
                                 jj.matrixInfo.nonZeros_.back()};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(inductors_.size()));
    for (const auto& ind : inductors_) {
        const double vals[3] = {ind.In2_, ind.In3_, ind.In4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(resistors_.size()));
    for (const auto& res : resistors_) {
        const double vals[4] = {res.pn1_, res.pn2_, res.pn3_, res.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(capacitors_.size()));
    for (const auto& cap : capacitors_) {
        const double vals[7] = {cap.pn1_, cap.pn2_, cap.pn3_, cap.pn4_, cap.pn5_, cap.pn6_, cap.pn7_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(vsources_.size()));
    for (const auto& vs : vsources_) {
        const double vals[4] = {vs.pn1_, vs.pn2_, vs.pn3_, vs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(ccvss_.size()));
    for (const auto& ccvs : ccvss_) {
        const double vals[4] = {ccvs.pn1_, ccvs.pn2_, ccvs.pn3_, ccvs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(vccss_.size()));
    for (const auto& vccs : vccss_) {
        const double vals[4] = {vccs.pn1_, vccs.pn2_, vccs.pn3_, vccs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(txlines_.size()));
    for (const auto& tx : txlines_) {
        const double vals[10] = {tx.n1_1_,
                                 tx.n1_2_,
                                 tx.n2_1_,
                                 tx.n2_2_,
                                 tx.nk_1_,
                                 tx.nk_2_,
                                 tx.nk1_1_,
                                 tx.nk1_2_,
                                 tx.nk2_1_,
                                 tx.nk2_2_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }
    (void) write_f64;
    (void) mObj;
}

void Simulation::load_state(const std::string& filename, Matrix& mObj) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        Errors::input_errors(InputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    char     magic[8] = {};
    uint32_t version  = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (std::string(magic, 8) != "JOSIMSTA" || version != 1) {
        Errors::input_errors(InputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(uint64_t));
        return v;
    };

    const uint64_t xLen = read_u64();
    if (xLen == x_.size()) {
        in.read(reinterpret_cast<char*>(x_.data()), static_cast<std::streamsize>(xLen * sizeof(double)));
    } else {
        std::vector<double> tmp(xLen);
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(xLen * sizeof(double)));
        const size_t copyLen = std::min<size_t>(x_.size(), tmp.size());
        std::copy(tmp.begin(), tmp.begin() + copyLen, x_.begin());
    }

    const uint64_t nJJ      = read_u64();
    bool           nzChange = false;
    for (uint64_t i = 0; i < nJJ; ++i) {
        double vals[13] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < jjs_.size()) {
            auto& jj = jjs_[i];
            jj.pn1_  = vals[0];
            jj.pn2_  = vals[1];
            jj.pn3_  = vals[2];
            jj.pn4_  = vals[3];
            jj.vn1_  = vals[4];
            jj.vn2_  = vals[5];
            jj.vn3_  = vals[6];
            jj.vn4_  = vals[7];
            jj.vn5_  = vals[8];
            jj.vn6_  = vals[9];
            jj.phi0_ = vals[10];
            jj.it_   = vals[11];
            if (jj.matrixInfo.nonZeros_.back() != vals[12]) {
                jj.matrixInfo.nonZeros_.back() = vals[12];
                mObj.nz[jj.nzBackIndex_]       = vals[12];
                nzChange                       = true;
            }
        }
    }

    const uint64_t nInd = read_u64();
    for (uint64_t i = 0; i < nInd; ++i) {
        double vals[3] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < inductors_.size()) {
            inductors_[i].In2_ = vals[0];
            inductors_[i].In3_ = vals[1];
            inductors_[i].In4_ = vals[2];
        }
    }

    const uint64_t nRes = read_u64();
    for (uint64_t i = 0; i < nRes; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < resistors_.size()) {
            resistors_[i].pn1_ = vals[0];
            resistors_[i].pn2_ = vals[1];
            resistors_[i].pn3_ = vals[2];
            resistors_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nCap = read_u64();
    for (uint64_t i = 0; i < nCap; ++i) {
        double vals[7] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < capacitors_.size()) {
            capacitors_[i].pn1_ = vals[0];
            capacitors_[i].pn2_ = vals[1];
            capacitors_[i].pn3_ = vals[2];
            capacitors_[i].pn4_ = vals[3];
            capacitors_[i].pn5_ = vals[4];
            capacitors_[i].pn6_ = vals[5];
            capacitors_[i].pn7_ = vals[6];
        }
    }

    const uint64_t nVS = read_u64();
    for (uint64_t i = 0; i < nVS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < vsources_.size()) {
            vsources_[i].pn1_ = vals[0];
            vsources_[i].pn2_ = vals[1];
            vsources_[i].pn3_ = vals[2];
            vsources_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nCCVS = read_u64();
    for (uint64_t i = 0; i < nCCVS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < ccvss_.size()) {
            ccvss_[i].pn1_ = vals[0];
            ccvss_[i].pn2_ = vals[1];
            ccvss_[i].pn3_ = vals[2];
            ccvss_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nVCCS = read_u64();
    for (uint64_t i = 0; i < nVCCS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < vccss_.size()) {
            vccss_[i].pn1_ = vals[0];
            vccss_[i].pn2_ = vals[1];
            vccss_[i].pn3_ = vals[2];
            vccss_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nTX = read_u64();
    for (uint64_t i = 0; i < nTX; ++i) {
        double vals[10] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < txlines_.size()) {
            txlines_[i].n1_1_  = vals[0];
            txlines_[i].n1_2_  = vals[1];
            txlines_[i].n2_1_  = vals[2];
            txlines_[i].n2_2_  = vals[3];
            txlines_[i].nk_1_  = vals[4];
            txlines_[i].nk_2_  = vals[5];
            txlines_[i].nk1_1_ = vals[6];
            txlines_[i].nk1_2_ = vals[7];
            txlines_[i].nk2_1_ = vals[8];
            txlines_[i].nk2_2_ = vals[9];
        }
    }

    if (nzChange) {
#ifdef SLU
        lu.factorize(true);
#else
        simOK_ = klu_l_refactor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, Numeric_, &Common_);
        if (!simOK_) {
            klu_l_free_numeric(&Numeric_, &Common_);
            Numeric_ = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
        }
#endif
    }
}
