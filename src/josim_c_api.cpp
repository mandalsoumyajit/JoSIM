// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#define JOSIM_C_EXPORTS
#include "JoSIM/josim_c_api.h"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/CliOptions.hpp"
#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/IV.hpp"
#include "JoSIM/Input.hpp"
#include "JoSIM/Matrix.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Model.hpp"
#include "JoSIM/Output.hpp"
#include "JoSIM/Parameters.hpp"
#include "JoSIM/RelevantTrace.hpp"
#include "JoSIM/Simulation.hpp"
#include "JoSIM/Transient.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace JoSIM;

struct CosimPortEntry {
    std::string posNode;
    std::string negNode;
    int32_t     mode          = 0;    // 0: V_in_I_out, 1: I_in_V_out, 2: Wave_Branin
    double      z0            = 50.0;
    std::string synthLabel;
    int64_t     sourceIdx     = -1;
    int64_t     curRowIdx     = -1;
    int64_t     posRowIdx     = -1;
    int64_t     negRowIdx     = -1;
    double      phiCommitted0 = 0.0;
    double      phiCommitted1 = 0.0;
    double      phiCommitted2 = 0.0;
    double      phiEval0      = 0.0;
    double      phiEval1      = 0.0;
    double      phiEval2      = 0.0;
};

struct josim_session {
    CliOptions                                                              cliOpts;
    Input                                                                   baseInput;
    bool                                                                    hasNetlist = false;
    std::string                                                             netlistPath;
    std::string                                                             netlistText;
    std::string                                                             baseDir;
    std::vector<std::tuple<std::string, std::optional<std::string>, double>> paramOverrides;

    Input                                                                   activeInput;
    Matrix                                                                  activeMatrix;
    Output                                                                  activeOutput;

    // Co-simulation state (Layer 1)
    std::vector<CosimPortEntry>                                             cosimPorts;
    std::unique_ptr<Simulation>                                             cosimSim;
    RollingCheckpoint                                                       ckptCommitted;
    RollingCheckpoint                                                       ckptEval;
    int64_t                                                                 committedStep = 0;
    double                                                                  committedTime = 0.0;
    int64_t                                                                 evalStep      = 0;
    double                                                                  evalTime      = 0.0;
    bool                                                                    hasEvalState  = false;
    int64_t                                                                 rollbackCount = 0;
    std::vector<double>                                                     committedInputs;
    std::vector<double>                                                     committedOutputs;
    std::vector<double>                                                     evalInputs;
    std::vector<double>                                                     evalOutputs;
    std::vector<double>                                                     evalJacobian;
    std::vector<double>                                                     windowTimePts;
    std::vector<double>                                                     windowPortOutFlat;

    std::vector<std::string>                                                cleanTraceNames;
    std::vector<double>                                                     colMajorMatrix;
    std::vector<double>                                                     mcTraceOut;
    int64_t                                                                 mcTrials = 1;
    int64_t                                                                 mcRows   = 0;
    int64_t                                                                 mcSteps  = 0;

    std::vector<josim_sfq_event_t>                                          events;
    std::vector<std::string>                                                eventTraceNames;
    std::vector<std::pair<std::string, double>>                             paramList;
    std::string                                                             lastError;

    josim_session() {
        cliOpts.minimal = true;
        baseInput.argMin = true;
    }
};

namespace {

std::string unquote(std::string s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

void extract_session_events(josim_session* s, const Input& iObj, const Matrix& mObj, const Simulation& sObj) {
    s->events.clear();
    s->eventTraceNames.clear();

    const int64_t B = std::max<int64_t>(1, sObj.results.mcRuns);
    if (sObj.eventsOnly_ && !sObj.mcEventTrackers_.empty()) {
        for (size_t tIdx = 0; tIdx < mObj.relevantTraces.size(); ++tIdx) {
            const auto& tr = mObj.relevantTraces[tIdx];
            if (tr.storageType != StorageType::Phase) { continue; }
            const int32_t traceId = static_cast<int32_t>(s->eventTraceNames.size());
            s->eventTraceNames.push_back(unquote(tr.deviceLabel.value_or("P")));
            for (int64_t b = 0; b < B; ++b) {
                if (static_cast<size_t>(b) < sObj.mcEventTrackers_.size()
                    && tIdx < sObj.mcEventTrackers_[static_cast<size_t>(b)].size()) {
                    for (const auto& ev : sObj.mcEventTrackers_[static_cast<size_t>(b)][tIdx].events) {
                        josim_sfq_event_t cev{};
                        cev.time     = ev.time;
                        cev.t_peak   = ev.tPeak;
                        cev.v_peak   = ev.vPeak;
                        cev.fwhm     = ev.fwhm;
                        cev.dt_pi    = ev.dtPi;
                        cev.area     = ev.area;
                        cev.phase    = ev.phase;
                        cev.polarity = ev.polarity;
                        cev.winding  = ev.winding;
                        cev.trace_id = traceId;
                        cev.trial    = static_cast<int32_t>(b);
                        s->events.push_back(cev);
                    }
                }
            }
        }
        return;
    }

    const auto& tAxis = sObj.results.timeAxis;
    if (tAxis.size() < 2) { return; }
    const size_t nSteps = tAxis.size();

    // Collect phase traces from relevantTraces or all junctions in the circuit
    struct PhaseTarget {
        std::string            name;
        std::optional<int64_t> i1;
        std::optional<int64_t> i2;
        std::optional<int64_t> vi;
        bool                   isJJ = false;
    };
    std::vector<PhaseTarget> targets;
    std::unordered_set<std::string> seen;

    for (const auto& tr : mObj.relevantTraces) {
        if (tr.storageType != StorageType::Phase) { continue; }
        std::string nm = unquote(tr.deviceLabel.value_or("P"));
        if (seen.insert(nm).second) {
            const bool isJJ = (nm.size() > 2 && nm[0] == 'P' && nm[1] == '(' && nm[2] == 'B');
            targets.push_back({nm, tr.index1, tr.index2, tr.variableIndex, isJJ});
        }
    }
    for (const auto& jIdx : mObj.components.junctionIndices) {
        const auto& jj = std::get<JJ>(mObj.components.devices.at(static_cast<size_t>(jIdx)));
        std::string nm = "P(" + jj.netlistInfo.label_ + ")";
        if (seen.insert(nm).second) {
            targets.push_back({nm, jj.indexInfo.posIndex_, jj.indexInfo.negIndex_, jj.variableIndex_, true});
        }
    }

    for (const auto& tgt : targets) {
        const int32_t traceId = static_cast<int32_t>(s->eventTraceNames.size());
        s->eventTraceNames.push_back(tgt.name);

        if (B > 1 && !sObj.results.mcTraceOut.empty()) {
            const int64_t nRows = sObj.results.mcNumRows;
            const int64_t nKept = sObj.results.mcNumKeptSteps;
            const int32_t s1    = (tgt.i1 && *tgt.i1 >= 0 && *tgt.i1 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                      ? sObj.results.mcRowToSlot[*tgt.i1]
                                      : -1;
            const int32_t s2    = (tgt.i2 && *tgt.i2 >= 0 && *tgt.i2 < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                      ? sObj.results.mcRowToSlot[*tgt.i2]
                                      : -1;
            const int32_t sv    = (tgt.vi && *tgt.vi >= 0 && *tgt.vi < static_cast<int64_t>(sObj.results.mcRowToSlot.size()))
                                      ? sObj.results.mcRowToSlot[*tgt.vi]
                                      : -1;
            const size_t  kMax  = std::min<size_t>(nSteps, static_cast<size_t>(nKept));
            for (int64_t b = 0; b < B; ++b) {
                const double* p1 = (s1 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s1) * nKept) : nullptr;
                const double* p2 = (s2 >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + s2) * nKept) : nullptr;
                const double* pv = (sv >= 0) ? (sObj.results.mcTraceOut.data() + (b * nRows + sv) * nKept) : nullptr;
                if (!p1 && !p2 && !pv) { continue; }
                InLoopEventTracker tracker;
                tracker.traceId = traceId;
                tracker.init_metrics(EVENT_METRIC_ALL);
                for (size_t k = 0; k < kMax; ++k) {
                    double phi = 0.0;
                    if (iObj.argAnal == AnalysisType::Voltage && tgt.isJJ && pv) {
                        phi = pv[k];
                    } else {
                        phi = (p1 ? p1[k] : 0.0) - (p2 ? p2[k] : 0.0);
                    }
                    tracker.step(tAxis[k], phi, EVENT_METRIC_ALL);
                }
                tracker.finalize(EVENT_METRIC_ALL);
                for (const auto& ev : tracker.events) {
                    josim_sfq_event_t cev{};
                    cev.time     = ev.time;
                    cev.t_peak   = ev.tPeak;
                    cev.v_peak   = ev.vPeak;
                    cev.fwhm     = ev.fwhm;
                    cev.dt_pi    = ev.dtPi;
                    cev.area     = ev.area;
                    cev.phase    = ev.phase;
                    cev.polarity = ev.polarity;
                    cev.winding  = ev.winding;
                    cev.trace_id = traceId;
                    cev.trial    = static_cast<int32_t>(b);
                    s->events.push_back(cev);
                }
            }
            continue;
        }

        const auto& x = sObj.results.xVector;
        const std::vector<double>* v1 =
            (tgt.i1 && *tgt.i1 >= 0 && static_cast<size_t>(*tgt.i1) < x.size() && x[*tgt.i1]) ? &(*x[*tgt.i1]) : nullptr;
        const std::vector<double>* v2 =
            (tgt.i2 && *tgt.i2 >= 0 && static_cast<size_t>(*tgt.i2) < x.size() && x[*tgt.i2]) ? &(*x[*tgt.i2]) : nullptr;
        const std::vector<double>* vv =
            (tgt.vi && *tgt.vi >= 0 && static_cast<size_t>(*tgt.vi) < x.size() && x[*tgt.vi]) ? &(*x[*tgt.vi]) : nullptr;
        if (!v1 && !v2 && !vv) { continue; }

        InLoopEventTracker tracker;
        tracker.traceId = traceId;
        tracker.init_metrics(EVENT_METRIC_ALL);
        for (size_t k = 0; k < nSteps; ++k) {
            double phi = 0.0;
            if (iObj.argAnal == AnalysisType::Voltage && tgt.isJJ && vv && k < vv->size()) {
                phi = (*vv)[k];
            } else {
                phi = (v1 && k < v1->size() ? (*v1)[k] : 0.0) - (v2 && k < v2->size() ? (*v2)[k] : 0.0);
            }
            tracker.step(tAxis[k], phi, EVENT_METRIC_ALL);
        }
        tracker.finalize(EVENT_METRIC_ALL);
        for (const auto& ev : tracker.events) {
            josim_sfq_event_t cev{};
            cev.time     = ev.time;
            cev.t_peak   = ev.tPeak;
            cev.v_peak   = ev.vPeak;
            cev.fwhm     = ev.fwhm;
            cev.dt_pi    = ev.dtPi;
            cev.area     = ev.area;
            cev.phase    = ev.phase;
            cev.polarity = ev.polarity;
            cev.winding  = ev.winding;
            cev.trace_id = traceId;
            cev.trial    = 0;
            s->events.push_back(cev);
        }
    }
}

int reload_base_input(josim_session* s) {
    if (!s->hasNetlist) { return 0; }
    try {
        s->baseInput = Input(s->cliOpts);
        if (!s->netlistPath.empty()) {
            s->baseInput.parse_input(s->netlistPath);
        } else {
            s->baseInput.parse_input_string(
                s->netlistText, s->baseDir.empty() ? std::nullopt : string_o(s->baseDir));
        }
        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

} // namespace

extern "C" {

JOSIM_API const char* josim_version(void) {
#ifdef VERSION
    return VERSION;
#else
    return "2.7";
#endif
}

JOSIM_API josim_session_t* josim_session_create(void) {
    return new josim_session();
}

JOSIM_API void josim_session_destroy(josim_session_t* s) {
    delete s;
}

JOSIM_API const char* josim_session_last_error(const josim_session_t* s) {
    if (!s) { return "Null session pointer"; }
    return s->lastError.c_str();
}

JOSIM_API int josim_set_cli_args(josim_session_t* s, int argc, const char* const* argv) {
    if (!s) { return -1; }
    try {
        std::vector<const char*> fullArgv;
        fullArgv.push_back("josim");
        for (int i = 0; i < argc; ++i) {
            if (argv && argv[i]) { fullArgv.push_back(argv[i]); }
        }
        fullArgv.push_back("-m");
        fullArgv.push_back("1");
        fullArgv.push_back("in_memory.cir");
        s->cliOpts = CliOptions::parse(static_cast<int64_t>(fullArgv.size()), fullArgv.data());
        s->cliOpts.cir_file_name = std::nullopt;
        if (s->hasNetlist) { return reload_base_input(s); }
        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

JOSIM_API int josim_load_netlist_file(josim_session_t* s, const char* path) {
    if (!s || !path) { return -1; }
    try {
        s->netlistPath = path;
        s->netlistText.clear();
        s->hasNetlist  = true;
        return reload_base_input(s);
    } catch (const std::exception& e) {
        s->lastError  = e.what();
        s->hasNetlist = false;
        return -1;
    }
}

JOSIM_API int josim_load_netlist_string(josim_session_t* s, const char* netlist_text, const char* base_dir) {
    if (!s || !netlist_text) { return -1; }
    try {
        s->netlistText = netlist_text;
        s->netlistPath.clear();
        s->baseDir    = base_dir ? base_dir : "";
        s->hasNetlist = true;
        return reload_base_input(s);
    } catch (const std::exception& e) {
        s->lastError  = e.what();
        s->hasNetlist = false;
        return -1;
    }
}

JOSIM_API int josim_set_parameter(josim_session_t* s, const char* name, double value, const char* subckt) {
    if (!s || !name) { return -1; }
    std::string pName = name;
    std::transform(pName.begin(), pName.end(), pName.begin(), ::toupper);
    std::optional<std::string> subOpt;
    if (subckt && subckt[0] != '\0') {
        std::string sc = subckt;
        std::transform(sc.begin(), sc.end(), sc.begin(), ::toupper);
        subOpt = sc;
    }
    for (auto& entry : s->paramOverrides) {
        if (std::get<0>(entry) == pName && std::get<1>(entry) == subOpt) {
            std::get<2>(entry) = value;
            return 0;
        }
    }
    s->paramOverrides.emplace_back(pName, subOpt, value);
    return 0;
}

JOSIM_API int josim_clear_parameter_overrides(josim_session_t* s) {
    if (!s) { return -1; }
    s->paramOverrides.clear();
    return 0;
}

JOSIM_API int josim_run(josim_session_t* s) {
    if (!s || !s->hasNetlist) {
        if (s) { s->lastError = "No netlist loaded into session"; }
        return -1;
    }
    try {
        s->lastError.clear();
        s->colMajorMatrix.clear();
        s->cleanTraceNames.clear();
        s->events.clear();
        s->eventTraceNames.clear();
        s->paramList.clear();

        // Copy pre-parsed netlist from baseInput without touching disk or re-tokenizing
        s->activeInput        = s->baseInput;
        s->activeMatrix       = Matrix();
        s->activeOutput       = Output();
        s->activeInput.netlist.parametersPtr = &s->activeInput.parameters;
        if (!s->cliOpts.events) { s->activeInput.argEventsOnly = false; }

        // Apply parameter overrides
        for (const auto& ov : s->paramOverrides) {
            Parameter          p;
            std::ostringstream oss;
            oss << std::uppercase << std::scientific << std::setprecision(12) << std::get<2>(ov);
            p.set_expression(oss.str());
            p.set_value(std::get<2>(ov));
            s->activeInput.parameters[ParameterName(std::get<0>(ov), std::get<1>(ov))] = p;
        }

        if (!s->activeInput.parameters.empty()) { parse_parameters(s->activeInput.parameters); }
        s->activeInput.netlist.models_new.clear();
        for (const auto& m : s->activeInput.netlist.models) {
            Model::parse_model(
                std::make_pair(m.second, m.first.second), s->activeInput.netlist.models_new, s->activeInput.parameters);
        }
        s->activeInput.netlist.expand_subcircuits();
        s->activeInput.netlist.expand_maindesign();
        Transient::identify_simulation(
            s->activeInput.controls, s->activeInput.transSim, s->activeInput.parameters);
        IV ivObj(s->activeInput);

        s->activeMatrix.create_matrix(s->activeInput);
        find_relevant_traces(s->activeInput, s->activeMatrix);

        // Ensure junction indices are recorded so SFQ events are always available in memory
        if (!s->activeInput.argEventsOnly && !s->activeMatrix.relevantTraces.empty()) {
            for (const auto& jIdx : s->activeMatrix.components.junctionIndices) {
                const auto& jj = std::get<JJ>(s->activeMatrix.components.devices.at(static_cast<size_t>(jIdx)));
                if (jj.indexInfo.posIndex_.has_value()) {
                    s->activeMatrix.relevantIndices.push_back(*jj.indexInfo.posIndex_);
                }
                if (jj.indexInfo.negIndex_.has_value()) {
                    s->activeMatrix.relevantIndices.push_back(*jj.indexInfo.negIndex_);
                }
                if (jj.variableIndex_ >= 0) {
                    s->activeMatrix.relevantIndices.push_back(jj.variableIndex_);
                }
            }
            s->activeMatrix.relevantIndices.erase(
                uniquify(s->activeMatrix.relevantIndices.begin(), s->activeMatrix.relevantIndices.end()),
                s->activeMatrix.relevantIndices.end());
        }

        Simulation sObj(s->activeInput, s->activeMatrix);

        // Save batched MC matrix if present
        s->mcTrials   = sObj.results.mcRuns;
        s->mcRows     = sObj.results.mcNumRows;
        s->mcSteps    = sObj.results.mcNumKeptSteps;
        s->mcTraceOut = sObj.results.mcTraceOut;

        // Extract in-memory SFQ events before Output clears xVector
        extract_session_events(s, s->activeInput, s->activeMatrix, sObj);

        // Run all post-simulation analyses and format traces into s->activeOutput.traces
        if (s->activeInput.argMargins && !s->activeInput.marginTargets.empty()) {
            s->activeOutput.write_margins(s->activeInput, s->activeMatrix, sObj);
        }
        if (s->activeInput.verifyEventsFile.has_value() && !s->activeInput.verifyEventsFile->empty()) {
            s->activeOutput.verify_golden_events(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.assertions.empty()) {
            s->activeOutput.verify_assertions(s->activeInput, s->activeMatrix, sObj);
        }
        if (s->activeInput.argCheckLoops) {
            s->activeOutput.write_loop_audit(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.jitterSpecs.empty()) {
            s->activeOutput.evaluate_jitter(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.measurements.empty()) {
            s->activeOutput.evaluate_measurements(s->activeInput, s->activeMatrix, sObj);
        } else if (s->activeInput.argSens && !s->activeInput.marginTargets.empty()) {
            s->activeOutput.write_sensitivity(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.argEventsOnly) {
            s->activeOutput.write_output(s->activeInput, s->activeMatrix, sObj);
        }

        s->cleanTraceNames.reserve(s->activeOutput.traces.size());
        for (const auto& tr : s->activeOutput.traces) {
            s->cleanTraceNames.push_back(unquote(tr.name_));
        }

        // Snapshot all top-level parameters (including exported IV_*, SQUID_*, BETA_L_*, JITTER_*, and .meas values)
        for (const auto& kv : s->activeInput.parameters) {
            if (!kv.first.subcircuit().has_value() && kv.second.get_value().has_value()) {
                s->paramList.emplace_back(kv.first.name(), *kv.second.get_value());
            }
        }
        std::sort(s->paramList.begin(), s->paramList.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

JOSIM_API int64_t josim_num_traces(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int64_t>(s->activeOutput.traces.size());
}

JOSIM_API int64_t josim_num_steps(const josim_session_t* s) {
    if (!s || s->activeOutput.traces.empty()) { return 0; }
    return static_cast<int64_t>(s->activeOutput.traces.front().data_.size());
}

JOSIM_API const char* josim_trace_name(const josim_session_t* s, int64_t trace_idx) {
    if (!s || trace_idx < 0 || static_cast<size_t>(trace_idx) >= s->cleanTraceNames.size()) { return nullptr; }
    return s->cleanTraceNames[static_cast<size_t>(trace_idx)].c_str();
}

JOSIM_API char josim_trace_type(const josim_session_t* s, int64_t trace_idx) {
    if (!s || trace_idx < 0 || static_cast<size_t>(trace_idx) >= s->activeOutput.traces.size()) { return '\0'; }
    return s->activeOutput.traces[static_cast<size_t>(trace_idx)].type_;
}

JOSIM_API const double* josim_trace_data(const josim_session_t* s, int64_t trace_idx) {
    if (!s || trace_idx < 0 || static_cast<size_t>(trace_idx) >= s->activeOutput.traces.size()) { return nullptr; }
    return s->activeOutput.traces[static_cast<size_t>(trace_idx)].data_ptr();
}

JOSIM_API const double* josim_trace_data_by_name(const josim_session_t* s, const char* name, int64_t* out_len) {
    if (out_len) { *out_len = 0; }
    if (!s || !name) { return nullptr; }
    std::string target = unquote(name);
    std::string targetUpper = target;
    std::transform(targetUpper.begin(), targetUpper.end(), targetUpper.begin(), ::toupper);

    for (size_t i = 0; i < s->cleanTraceNames.size(); ++i) {
        std::string candUpper = s->cleanTraceNames[i];
        std::transform(candUpper.begin(), candUpper.end(), candUpper.begin(), ::toupper);
        if (s->cleanTraceNames[i] == target || candUpper == targetUpper) {
            if (out_len) { *out_len = static_cast<int64_t>(s->activeOutput.traces[i].data_.size()); }
            return s->activeOutput.traces[i].data_ptr();
        }
    }
    return nullptr;
}

JOSIM_API const double* josim_traces_matrix_col_major(josim_session_t* s, int64_t* out_cols, int64_t* out_rows) {
    if (out_cols) { *out_cols = 0; }
    if (out_rows) { *out_rows = 0; }
    if (!s || s->activeOutput.traces.empty()) { return nullptr; }
    const size_t nCols = s->activeOutput.traces.size();
    const size_t nRows = s->activeOutput.traces.front().data_.size();
    if (out_cols) { *out_cols = static_cast<int64_t>(nCols); }
    if (out_rows) { *out_rows = static_cast<int64_t>(nRows); }
    if (nCols == 0 || nRows == 0) { return nullptr; }
    if (s->colMajorMatrix.size() != nCols * nRows) {
        s->colMajorMatrix.resize(nCols * nRows);
        for (size_t c = 0; c < nCols; ++c) {
            const double* src = s->activeOutput.traces[c].data_ptr();
            if (src && s->activeOutput.traces[c].data_.size() >= nRows) {
                std::memcpy(s->colMajorMatrix.data() + c * nRows, src, nRows * sizeof(double));
            }
        }
    }
    return s->colMajorMatrix.data();
}

JOSIM_API const double* josim_mc_trace_matrix(
    const josim_session_t* s, int64_t* out_trials, int64_t* out_rows, int64_t* out_steps) {
    if (out_trials) { *out_trials = s ? s->mcTrials : 0; }
    if (out_rows) { *out_rows = s ? s->mcRows : 0; }
    if (out_steps) { *out_steps = s ? s->mcSteps : 0; }
    if (!s || s->mcTraceOut.empty()) { return nullptr; }
    return s->mcTraceOut.data();
}

JOSIM_API int64_t josim_num_events(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int64_t>(s->events.size());
}

JOSIM_API const josim_sfq_event_t* josim_events_data(const josim_session_t* s) {
    if (!s || s->events.empty()) { return nullptr; }
    return s->events.data();
}

JOSIM_API int64_t josim_num_event_traces(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int64_t>(s->eventTraceNames.size());
}

JOSIM_API const char* josim_event_trace_name(const josim_session_t* s, int32_t trace_id) {
    if (!s || trace_id < 0 || static_cast<size_t>(trace_id) >= s->eventTraceNames.size()) { return nullptr; }
    return s->eventTraceNames[static_cast<size_t>(trace_id)].c_str();
}

JOSIM_API int64_t josim_num_parameters(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int64_t>(s->paramList.size());
}

JOSIM_API const char* josim_parameter_name(const josim_session_t* s, int64_t idx) {
    if (!s || idx < 0 || static_cast<size_t>(idx) >= s->paramList.size()) { return nullptr; }
    return s->paramList[static_cast<size_t>(idx)].first.c_str();
}

JOSIM_API double josim_parameter_value(const josim_session_t* s, int64_t idx) {
    if (!s || idx < 0 || static_cast<size_t>(idx) >= s->paramList.size()) { return 0.0; }
    return s->paramList[static_cast<size_t>(idx)].second;
}

JOSIM_API int josim_get_parameter(const josim_session_t* s, const char* name, double* out_val) {
    if (!s || !name || !out_val) { return -1; }
    std::string pName = name;
    std::transform(pName.begin(), pName.end(), pName.begin(), ::toupper);
    auto it = s->activeInput.parameters.find(ParameterName(pName, std::nullopt));
    if (it != s->activeInput.parameters.end() && it->second.get_value().has_value()) {
        *out_val = *it->second.get_value();
        return 0;
    }
    return -1;
}

JOSIM_API int64_t josim_num_measurements(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int64_t>(s->activeOutput.measResults.size());
}

JOSIM_API const char* josim_measurement_name(const josim_session_t* s, int64_t idx) {
    if (!s || idx < 0 || static_cast<size_t>(idx) >= s->activeOutput.measResults.size()) { return nullptr; }
    return s->activeOutput.measResults[static_cast<size_t>(idx)].first.c_str();
}

JOSIM_API double josim_measurement_value(const josim_session_t* s, int64_t idx) {
    if (!s || idx < 0 || static_cast<size_t>(idx) >= s->activeOutput.measResults.size()) { return 0.0; }
    return s->activeOutput.measResults[static_cast<size_t>(idx)].second;
}

JOSIM_API const double* josim_measurement_trial_values(const josim_session_t* s, int64_t idx, int64_t* out_trials) {
    if (out_trials) { *out_trials = 0; }
    if (!s || idx < 0 || static_cast<size_t>(idx) >= s->activeOutput.measTrialValues.size()) { return nullptr; }
    const auto& vec = s->activeOutput.measTrialValues[static_cast<size_t>(idx)];
    if (out_trials) { *out_trials = static_cast<int64_t>(vec.size()); }
    return vec.empty() ? nullptr : vec.data();
}

namespace {

std::string normalize_node_name(const char* raw) {
    if (!raw || raw[0] == '\0') { return "0"; }
    std::string n = raw;
    std::transform(n.begin(), n.end(), n.begin(), ::toupper);
    if (n == "GND") { return "0"; }
    return n;
}

void apply_port_input_to_source(josim_session* s, const CosimPortEntry& pe, double u) {
    if (pe.sourceIdx < 0 || static_cast<size_t>(pe.sourceIdx) >= s->activeMatrix.sourcegen.size()) { return; }
    auto&  amps = s->activeMatrix.sourcegen[static_cast<size_t>(pe.sourceIdx)].ampValues();
    double val  = u;
    if (pe.mode == JOSIM_PORT_CURRENT_IN_VOLTAGE_OUT) {
        val = -u; // Positive u = current injected into posNode
    } else if (pe.mode == JOSIM_PORT_WAVE_BRANIN) {
        val = -(2.0 / pe.z0) * u; // Norton current source for incident voltage wave u = V_inc
    }
    if (amps.size() >= 2) {
        amps[0] = val;
        amps[1] = val;
    } else {
        amps.assign(2, val);
    }
}

double read_port_node_diff(const Simulation& sim, const CosimPortEntry& pe) {
    const double vPos = (pe.posRowIdx >= 0) ? sim.cosim_read_x(pe.posRowIdx) : 0.0;
    const double vNeg = (pe.negRowIdx >= 0) ? sim.cosim_read_x(pe.negRowIdx) : 0.0;
    return vPos - vNeg;
}

double compute_port_output(
    const Simulation& sim, const CosimPortEntry& pe, double u, double phi0, double phi1, double phi2) {
    if (pe.mode == JOSIM_PORT_VOLTAGE_IN_CURRENT_OUT) {
        return (pe.curRowIdx >= 0) ? (-sim.cosim_read_x(pe.curRowIdx)) : 0.0;
    }
    double vPort = phi0;
    if (sim.cosim_analysis_type() == AnalysisType::Phase) {
        const double h = sim.cosim_step_size();
        vPort          = ((3.0 * Constants::SIGMA) / (2.0 * h)) * (phi0 - (4.0 / 3.0) * phi1 + (1.0 / 3.0) * phi2);
    }
    if (pe.mode == JOSIM_PORT_CURRENT_IN_VOLTAGE_OUT) {
        return vPort;
    }
    // JOSIM_PORT_WAVE_BRANIN: outgoing reflected voltage wave V_ref = V_port - V_inc
    return vPort - u;
}

void run_cosim_substeps(josim_session*             s,
                        int64_t                    targetStep,
                        const std::vector<double>& trialInputs,
                        std::vector<double>&       outPortVals,
                        bool                       recordHistory) {
    Simulation&   sim   = *s->cosimSim;
    const size_t  nP    = s->cosimPorts.size();
    const int64_t start = s->committedStep;
    const int64_t M     = std::max<int64_t>(1, targetStep - start);
    const double  h     = sim.cosim_step_size();

    std::vector<double> phi0(nP), phi1(nP), phi2(nP);
    for (size_t p = 0; p < nP; ++p) {
        phi0[p] = s->cosimPorts[p].phiCommitted0;
        phi1[p] = s->cosimPorts[p].phiCommitted1;
        phi2[p] = s->cosimPorts[p].phiCommitted2;
    }

    for (int64_t k = 1; k <= M; ++k) {
        const int64_t stepIdx = start + k;
        const double  alpha   = static_cast<double>(k) / static_cast<double>(M);
        for (size_t p = 0; p < nP; ++p) {
            const double uStep = (1.0 - alpha) * s->committedInputs[p] + alpha * trialInputs[p];
            apply_port_input_to_source(s, s->cosimPorts[p], uStep);
            phi2[p] = phi1[p];
            phi1[p] = phi0[p];
        }
        const double stepTime = static_cast<double>(stepIdx) * h;
        sim.cosim_step_one(s->activeMatrix, stepIdx, stepTime);
        if (recordHistory) {
            sim.cosim_record_step(s->activeMatrix, stepIdx, stepTime);
        }
        for (size_t p = 0; p < nP; ++p) {
            phi0[p] = read_port_node_diff(sim, s->cosimPorts[p]);
        }
    }

    outPortVals.resize(nP);
    for (size_t p = 0; p < nP; ++p) {
        outPortVals[p] = compute_port_output(sim, s->cosimPorts[p], trialInputs[p], phi0[p], phi1[p], phi2[p]);
        if (recordHistory) {
            s->cosimPorts[p].phiEval0 = phi0[p];
            s->cosimPorts[p].phiEval1 = phi1[p];
            s->cosimPorts[p].phiEval2 = phi2[p];
        }
    }
}

double compute_single_step_port_jacobian(josim_session* s, const CosimPortEntry& pe) {
    Simulation&  sim  = *s->cosimSim;
    const double h    = sim.cosim_step_size();
    const auto   atyp = sim.cosim_analysis_type();
    if (pe.mode == JOSIM_PORT_VOLTAGE_IN_CURRENT_OUT) {
        const double rhsScale = (atyp == AnalysisType::Voltage) ? 1.0 : ((2.0 * h) / (3.0 * Constants::SIGMA));
        return sim.cosim_solve_unit_rhs(s->activeMatrix, pe.curRowIdx, rhsScale, -1, 0.0, -1, pe.curRowIdx);
    }
    if (pe.mode == JOSIM_PORT_CURRENT_IN_VOLTAGE_OUT) {
        const double outScale = (atyp == AnalysisType::Voltage) ? 1.0 : ((3.0 * Constants::SIGMA) / (2.0 * h));
        return outScale
               * sim.cosim_solve_unit_rhs(
                       s->activeMatrix, pe.posRowIdx, 1.0, pe.negRowIdx, -1.0, pe.posRowIdx, pe.negRowIdx);
    }
    return 1.0 / pe.z0;
}

} // namespace

JOSIM_API int32_t josim_cosim_add_port(
    josim_session_t* s, const char* pos_node, const char* neg_node, int32_t mode, double z0) {
    if (!s || !pos_node) { return -1; }
    CosimPortEntry pe;
    pe.posNode = normalize_node_name(pos_node);
    pe.negNode = normalize_node_name(neg_node);
    pe.mode    = mode;
    pe.z0      = (z0 > 0.0) ? z0 : 50.0;
    const int32_t portId = static_cast<int32_t>(s->cosimPorts.size());
    s->cosimPorts.push_back(std::move(pe));
    return portId;
}

JOSIM_API int josim_cosim_clear_ports(josim_session_t* s) {
    if (!s) { return -1; }
    s->cosimPorts.clear();
    return 0;
}

JOSIM_API int32_t josim_cosim_num_ports(const josim_session_t* s) {
    if (!s) { return 0; }
    return static_cast<int32_t>(s->cosimPorts.size());
}

JOSIM_API int josim_cosim_init(josim_session_t* s) {
    if (!s || !s->hasNetlist) {
        if (s) { s->lastError = "No netlist loaded into session"; }
        return -1;
    }
    try {
        s->lastError.clear();
        s->colMajorMatrix.clear();
        s->cleanTraceNames.clear();
        s->events.clear();
        s->eventTraceNames.clear();
        s->paramList.clear();

        s->activeInput                       = s->baseInput;
        s->activeMatrix                      = Matrix();
        s->activeOutput                      = Output();
        s->activeInput.netlist.parametersPtr = &s->activeInput.parameters;
        s->activeInput.argEventsOnly         = false;

        for (const auto& ov : s->paramOverrides) {
            Parameter          p;
            std::ostringstream oss;
            oss << std::uppercase << std::scientific << std::setprecision(12) << std::get<2>(ov);
            p.set_expression(oss.str());
            p.set_value(std::get<2>(ov));
            s->activeInput.parameters[ParameterName(std::get<0>(ov), std::get<1>(ov))] = p;
        }

        if (!s->activeInput.parameters.empty()) { parse_parameters(s->activeInput.parameters); }
        s->activeInput.netlist.models_new.clear();
        for (const auto& m : s->activeInput.netlist.models) {
            Model::parse_model(
                std::make_pair(m.second, m.first.second), s->activeInput.netlist.models_new, s->activeInput.parameters);
        }
        s->activeInput.netlist.expand_subcircuits();
        s->activeInput.netlist.expand_maindesign();
        Transient::identify_simulation(
            s->activeInput.controls, s->activeInput.transSim, s->activeInput.parameters);

        // Synthesize boundary elements in expNetlist for each registered co-simulation port
        for (size_t p = 0; p < s->cosimPorts.size(); ++p) {
            auto&             pe   = s->cosimPorts[p];
            const std::string tEnd = std::to_string(p + 1);
            if (pe.mode == JOSIM_PORT_VOLTAGE_IN_CURRENT_OUT) {
                pe.synthLabel = "V$COSIM$" + std::to_string(p);
                tokens_t line = {pe.synthLabel, pe.posNode, pe.negNode, "PWL(0", "0", tEnd, "0)"};
                s->activeInput.netlist.expNetlist.emplace_back(std::move(line), std::nullopt);
            } else if (pe.mode == JOSIM_PORT_CURRENT_IN_VOLTAGE_OUT) {
                pe.synthLabel = "I$COSIM$" + std::to_string(p);
                tokens_t line = {pe.synthLabel, pe.posNode, pe.negNode, "PWL(0", "0", tEnd, "0)"};
                s->activeInput.netlist.expNetlist.emplace_back(std::move(line), std::nullopt);
            } else {
                const std::string rLabel = "R$COSIM$" + std::to_string(p);
                std::ostringstream zOss;
                zOss << std::setprecision(12) << pe.z0;
                tokens_t rLine = {rLabel, pe.posNode, pe.negNode, zOss.str()};
                s->activeInput.netlist.expNetlist.emplace_back(std::move(rLine), std::nullopt);

                pe.synthLabel = "I$COSIM$" + std::to_string(p);
                tokens_t iLine = {pe.synthLabel, pe.posNode, pe.negNode, "PWL(0", "0", tEnd, "0)"};
                s->activeInput.netlist.expNetlist.emplace_back(std::move(iLine), std::nullopt);
            }
        }

        s->activeMatrix.create_matrix(s->activeInput);
        find_relevant_traces(s->activeInput, s->activeMatrix);

        // Resolve MNA row and source indices for each port
        for (auto& pe : s->cosimPorts) {
            pe.posRowIdx = (pe.posNode != "0" && s->activeMatrix.nm.count(pe.posNode) != 0)
                                   ? s->activeMatrix.nm.at(pe.posNode)
                                   : -1;
            pe.negRowIdx = (pe.negNode != "0" && s->activeMatrix.nm.count(pe.negNode) != 0)
                                   ? s->activeMatrix.nm.at(pe.negNode)
                                   : -1;
            if (pe.mode == JOSIM_PORT_VOLTAGE_IN_CURRENT_OUT) {
                auto dIt = s->activeMatrix.components.deviceMap.find(pe.synthLabel);
                if (dIt != s->activeMatrix.components.deviceMap.end()) {
                    const auto& vs = std::get<VoltageSource>(s->activeMatrix.components.devices.at(dIt->second));
                    pe.curRowIdx   = vs.indexInfo.currentIndex_.value_or(-1);
                    pe.sourceIdx   = static_cast<int64_t>(vs.sourceIndex_);
                }
            } else {
                auto cIt = s->activeMatrix.components.csMap.find(pe.synthLabel);
                if (cIt != s->activeMatrix.components.csMap.end()) {
                    const auto& cs = s->activeMatrix.components.currentsources.at(cIt->second);
                    pe.sourceIdx   = static_cast<int64_t>(cs.sourceIndex_);
                }
            }
        }

        // Ensure junction and port indices are recorded in xVector
        if (!s->activeMatrix.relevantTraces.empty()) {
            for (const auto& jIdx : s->activeMatrix.components.junctionIndices) {
                const auto& jj = std::get<JJ>(s->activeMatrix.components.devices.at(static_cast<size_t>(jIdx)));
                if (jj.indexInfo.posIndex_.has_value()) {
                    s->activeMatrix.relevantIndices.push_back(*jj.indexInfo.posIndex_);
                }
                if (jj.indexInfo.negIndex_.has_value()) {
                    s->activeMatrix.relevantIndices.push_back(*jj.indexInfo.negIndex_);
                }
                if (jj.variableIndex_ >= 0) {
                    s->activeMatrix.relevantIndices.push_back(jj.variableIndex_);
                }
            }
            for (const auto& pe : s->cosimPorts) {
                if (pe.posRowIdx >= 0) { s->activeMatrix.relevantIndices.push_back(pe.posRowIdx); }
                if (pe.negRowIdx >= 0) { s->activeMatrix.relevantIndices.push_back(pe.negRowIdx); }
                if (pe.curRowIdx >= 0) { s->activeMatrix.relevantIndices.push_back(pe.curRowIdx); }
            }
            s->activeMatrix.relevantIndices.erase(
                uniquify(s->activeMatrix.relevantIndices.begin(), s->activeMatrix.relevantIndices.end()),
                s->activeMatrix.relevantIndices.end());
        }

        s->cosimSim = std::make_unique<Simulation>();
        s->cosimSim->cosim_initialize(s->activeInput, s->activeMatrix);

        const size_t nP = s->cosimPorts.size();
        for (size_t p = 0; p < nP; ++p) {
            const double phi0             = read_port_node_diff(*s->cosimSim, s->cosimPorts[p]);
            s->cosimPorts[p].phiCommitted0 = phi0;
            s->cosimPorts[p].phiCommitted1 = phi0;
            s->cosimPorts[p].phiCommitted2 = phi0;
            s->cosimPorts[p].phiEval0      = phi0;
            s->cosimPorts[p].phiEval1      = phi0;
            s->cosimPorts[p].phiEval2      = phi0;
        }

        s->committedStep = 0;
        s->committedTime = 0.0;
        s->evalStep      = 0;
        s->evalTime      = 0.0;
        s->hasEvalState  = false;
        s->rollbackCount = 0;
        s->committedInputs.assign(nP, 0.0);
        s->committedOutputs.assign(nP, 0.0);
        s->evalInputs.assign(nP, 0.0);
        s->evalOutputs.assign(nP, 0.0);
        s->evalJacobian.assign(nP, 0.0);
        for (size_t p = 0; p < nP; ++p) {
            s->evalJacobian[p] = compute_single_step_port_jacobian(s, s->cosimPorts[p]);
        }

        s->cosimSim->cosim_save_ckpt(s->ckptCommitted, 0);
        s->ckptEval = s->ckptCommitted;
        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

JOSIM_API double josim_cosim_tstep(const josim_session_t* s) {
    if (!s || !s->cosimSim) { return 0.0; }
    return s->cosimSim->cosim_step_size();
}

JOSIM_API double josim_cosim_current_time(const josim_session_t* s) {
    if (!s || !s->cosimSim) { return 0.0; }
    return s->committedTime;
}

JOSIM_API int64_t josim_cosim_rollback_count(const josim_session_t* s) {
    if (!s) { return 0; }
    return s->rollbackCount;
}

JOSIM_API int josim_cosim_commit(josim_session_t* s) {
    if (!s || !s->cosimSim) { return -1; }
    if (s->hasEvalState) {
        s->ckptCommitted    = s->ckptEval;
        s->committedStep    = s->evalStep;
        s->committedTime    = s->evalTime;
        s->committedInputs  = s->evalInputs;
        s->committedOutputs = s->evalOutputs;
        for (auto& pe : s->cosimPorts) {
            pe.phiCommitted0 = pe.phiEval0;
            pe.phiCommitted1 = pe.phiEval1;
            pe.phiCommitted2 = pe.phiEval2;
        }
        s->hasEvalState = false;
    }
    return 0;
}

JOSIM_API int josim_cosim_rollback(josim_session_t* s) {
    if (!s || !s->cosimSim) { return -1; }
    s->cosimSim->cosim_restore_ckpt(s->activeMatrix, s->ckptCommitted);
    s->evalStep     = s->committedStep;
    s->evalTime     = s->committedTime;
    s->evalInputs   = s->committedInputs;
    s->hasEvalState = false;
    for (auto& pe : s->cosimPorts) {
        pe.phiEval0 = pe.phiCommitted0;
        pe.phiEval1 = pe.phiCommitted1;
        pe.phiEval2 = pe.phiCommitted2;
    }
    ++s->rollbackCount;
    return 0;
}

JOSIM_API int josim_cosim_step_to(
    josim_session_t* s, double target_time,
    const double* port_inputs, double* port_outputs, double* port_jacobian_diag) {
    if (!s || !s->cosimSim) {
        if (s) { s->lastError = "Co-simulation session not initialized"; }
        return -1;
    }
    try {
        const size_t nP = s->cosimPorts.size();
        const double h  = s->cosimSim->cosim_step_size();

        std::vector<double> trialIn(nP, 0.0);
        if (port_inputs && nP > 0) {
            std::memcpy(trialIn.data(), port_inputs, nP * sizeof(double));
        }

        // Automatic commit / rollback based on target_time progression
        const auto grid_step_for_time = [&](double t) -> int64_t {
            return std::max<int64_t>(s->committedStep + 1, std::llround(t / h));
        };
        if (s->hasEvalState) {
            const int64_t candidateStep = grid_step_for_time(target_time);
            const double  evalStepTime  = static_cast<double>(s->evalStep) * h;
            if (target_time > s->evalTime + 0.25 * h && candidateStep > s->evalStep &&
                target_time >= evalStepTime - 0.1 * h) {
                // SPICE accepted the previous step and advanced past evalStepTime
                josim_cosim_commit(s);
            } else {
                // Either same-time Newton-Raphson re-evaluation, sub-grid step within the same h interval, or LTE rejection
                const bool isRollback = (target_time < s->evalTime - 0.25 * h);
                s->cosimSim->cosim_restore_ckpt(s->activeMatrix, s->ckptCommitted);
                if (isRollback) { ++s->rollbackCount; }
            }
        }

        if (target_time <= s->committedTime + 1e-18 && s->committedStep == 0) {
            // Re-evaluate at t = 0
            s->cosimSim->cosim_restore_ckpt(s->activeMatrix, s->ckptCommitted);
            for (size_t p = 0; p < nP; ++p) {
                apply_port_input_to_source(s, s->cosimPorts[p], trialIn[p]);
            }
            s->cosimSim->cosim_step_one(s->activeMatrix, 0, 0.0);
            for (size_t p = 0; p < nP; ++p) {
                const double phi0    = read_port_node_diff(*s->cosimSim, s->cosimPorts[p]);
                s->evalOutputs[p]    = compute_port_output(*s->cosimSim, s->cosimPorts[p], trialIn[p], phi0, phi0, phi0);
                s->evalJacobian[p]   = compute_single_step_port_jacobian(s, s->cosimPorts[p]);
                s->cosimPorts[p].phiEval0 = phi0;
                s->cosimPorts[p].phiEval1 = phi0;
                s->cosimPorts[p].phiEval2 = phi0;
            }
            s->evalInputs       = trialIn;
            s->committedInputs  = trialIn;
            s->committedOutputs = s->evalOutputs;
            s->cosimSim->cosim_save_ckpt(s->ckptEval, 0);
            if (port_outputs && nP > 0) { std::memcpy(port_outputs, s->evalOutputs.data(), nP * sizeof(double)); }
            if (port_jacobian_diag && nP > 0) {
                std::memcpy(port_jacobian_diag, s->evalJacobian.data(), nP * sizeof(double));
            }
            return 0;
        }

        const int64_t targetStep = grid_step_for_time(target_time);
        const int64_t M          = targetStep - s->committedStep;
        const bool    sameStepNR = (s->hasEvalState && targetStep == s->evalStep);

        run_cosim_substeps(s, targetStep, trialIn, s->evalOutputs, true);
        s->cosimSim->cosim_save_ckpt(s->ckptEval, targetStep);

        std::vector<double> interpOutputs = s->evalOutputs;

        // Compute exact driving-point Jacobian (1-step KLU solve for M == 1, or multi-substep tangent for M > 1)
        if (M == 1) {
            for (size_t p = 0; p < nP; ++p) {
                s->evalJacobian[p] = compute_single_step_port_jacobian(s, s->cosimPorts[p]);
            }
            // Smooth sub-grid interpolation if target_time < targetStep * h (keep s->evalOutputs as exact grid value!)
            const double tCommittedGrid = static_cast<double>(s->committedStep) * h;
            const double alpha          = std::clamp((target_time - tCommittedGrid) / h, 0.0, 1.0);
            if (alpha < 0.999999) {
                for (size_t p = 0; p < nP; ++p) {
                    const double y_c = (p < s->committedOutputs.size()) ? s->committedOutputs[p] : 0.0;
                    const double u_c = (p < s->committedInputs.size()) ? s->committedInputs[p] : 0.0;
                    interpOutputs[p] =
                        (1.0 - alpha) * y_c + alpha * s->evalOutputs[p] +
                        (1.0 - alpha) * s->evalJacobian[p] * (trialIn[p] - u_c);
                }
            }
        } else if (!sameStepNR || port_jacobian_diag != nullptr) {
            if (!sameStepNR) {
                std::vector<double> pertIn  = trialIn;
                std::vector<double> pertOut(nP, 0.0);
                for (size_t p = 0; p < nP; ++p) {
                    if (s->cosimPorts[p].mode == JOSIM_PORT_WAVE_BRANIN) {
                        s->evalJacobian[p] = 1.0 / s->cosimPorts[p].z0;
                        continue;
                    }
                    const double eps = 1e-6 * std::max(1e-3, std::abs(trialIn[p]));
                    pertIn[p]        = trialIn[p] + eps;
                    s->cosimSim->cosim_restore_ckpt(s->activeMatrix, s->ckptCommitted);
                    run_cosim_substeps(s, targetStep, pertIn, pertOut, false);
                    s->evalJacobian[p] = (pertOut[p] - s->evalOutputs[p]) / eps;
                    pertIn[p]          = trialIn[p];
                }
                s->cosimSim->cosim_restore_ckpt(s->activeMatrix, s->ckptEval);
            }
        }

        s->evalStep     = targetStep;
        s->evalTime     = target_time;
        s->evalInputs   = std::move(trialIn);
        s->hasEvalState = true;

        if (port_outputs && nP > 0) {
            std::memcpy(port_outputs, interpOutputs.data(), nP * sizeof(double));
        }
        if (port_jacobian_diag && nP > 0) {
            std::memcpy(port_jacobian_diag, s->evalJacobian.data(), nP * sizeof(double));
        }
        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

JOSIM_API int josim_cosim_run_window(
    josim_session_t* s, double t_stop,
    int64_t n_in_pts, const double* t_in_pts, const double* port_in_flat,
    int64_t* out_n_pts, const double** out_t_pts, const double** out_port_out_flat) {
    if (out_n_pts) { *out_n_pts = 0; }
    if (out_t_pts) { *out_t_pts = nullptr; }
    if (out_port_out_flat) { *out_port_out_flat = nullptr; }
    if (!s || !s->cosimSim) {
        if (s) { s->lastError = "Co-simulation session not initialized"; }
        return -1;
    }
    try {
        josim_cosim_commit(s);
        Simulation&   sim      = *s->cosimSim;
        const size_t  nP       = s->cosimPorts.size();
        const double  h        = sim.cosim_step_size();
        const int64_t stopStep = std::max<int64_t>(s->committedStep, std::llround(t_stop / h));
        const int64_t nSteps   = stopStep - s->committedStep;

        s->windowTimePts.clear();
        s->windowPortOutFlat.clear();
        s->windowTimePts.reserve(static_cast<size_t>(nSteps + 1));
        s->windowPortOutFlat.reserve(static_cast<size_t>(nSteps + 1) * nP);

        auto eval_pwl_port = [&](double t, size_t p) -> double {
            if (n_in_pts <= 0 || !t_in_pts || !port_in_flat) { return s->committedInputs[p]; }
            if (n_in_pts == 1 || t <= t_in_pts[0]) { return port_in_flat[p]; }
            if (t >= t_in_pts[n_in_pts - 1]) { return port_in_flat[static_cast<size_t>(n_in_pts - 1) * nP + p]; }
            const double* it  = std::upper_bound(t_in_pts, t_in_pts + n_in_pts, t);
            const size_t  idx = static_cast<size_t>(std::max<std::ptrdiff_t>(0, (it - t_in_pts) - 1));
            const double  t0  = t_in_pts[idx];
            const double  t1  = t_in_pts[idx + 1];
            const double  u0  = port_in_flat[idx * nP + p];
            const double  u1  = port_in_flat[(idx + 1) * nP + p];
            const double  dt  = t1 - t0;
            return (dt > 1e-30) ? (u0 + ((t - t0) / dt) * (u1 - u0)) : u1;
        };

        // Record start of window (t_committed)
        s->windowTimePts.push_back(s->committedStep * h);
        for (size_t p = 0; p < nP; ++p) {
            const auto&  pe   = s->cosimPorts[p];
            const double u0   = eval_pwl_port(s->committedStep * h, p);
            const double out0 = compute_port_output(
                sim, pe, u0, pe.phiCommitted0, pe.phiCommitted1, pe.phiCommitted2);
            s->windowPortOutFlat.push_back(out0);
        }

        for (int64_t stepIdx = s->committedStep + 1; stepIdx <= stopStep; ++stepIdx) {
            const double stepTime = static_cast<double>(stepIdx) * h;
            for (size_t p = 0; p < nP; ++p) {
                auto&        pe = s->cosimPorts[p];
                const double u  = eval_pwl_port(stepTime, p);
                apply_port_input_to_source(s, pe, u);
                s->committedInputs[p] = u;
                pe.phiCommitted2      = pe.phiCommitted1;
                pe.phiCommitted1      = pe.phiCommitted0;
            }
            sim.cosim_step_one(s->activeMatrix, stepIdx, stepTime);
            sim.cosim_record_step(s->activeMatrix, stepIdx, stepTime);

            s->windowTimePts.push_back(stepTime);
            for (size_t p = 0; p < nP; ++p) {
                auto& pe         = s->cosimPorts[p];
                pe.phiCommitted0 = read_port_node_diff(sim, pe);
                const double out = compute_port_output(
                    sim, pe, s->committedInputs[p], pe.phiCommitted0, pe.phiCommitted1, pe.phiCommitted2);
                s->windowPortOutFlat.push_back(out);
            }
        }

        s->committedStep = stopStep;
        s->committedTime = static_cast<double>(stopStep) * h;
        s->evalStep      = s->committedStep;
        s->evalTime      = s->committedTime;
        s->evalInputs    = s->committedInputs;
        s->hasEvalState  = false;
        sim.cosim_save_ckpt(s->ckptCommitted, stopStep);
        s->ckptEval = s->ckptCommitted;

        if (out_n_pts) { *out_n_pts = static_cast<int64_t>(s->windowTimePts.size()); }
        if (out_t_pts) { *out_t_pts = s->windowTimePts.data(); }
        if (out_port_out_flat) { *out_port_out_flat = s->windowPortOutFlat.data(); }
        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

JOSIM_API int josim_cosim_finalize(josim_session_t* s) {
    if (!s || !s->cosimSim) { return -1; }
    try {
        josim_cosim_commit(s);
        Simulation& sObj = *s->cosimSim;
        sObj.cosim_finish(s->activeMatrix);

        extract_session_events(s, s->activeInput, s->activeMatrix, sObj);

        if (!s->activeInput.assertions.empty()) {
            s->activeOutput.verify_assertions(s->activeInput, s->activeMatrix, sObj);
        }
        if (s->activeInput.argCheckLoops) {
            s->activeOutput.write_loop_audit(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.jitterSpecs.empty()) {
            s->activeOutput.evaluate_jitter(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.measurements.empty()) {
            s->activeOutput.evaluate_measurements(s->activeInput, s->activeMatrix, sObj);
        }
        if (!s->activeInput.argEventsOnly) {
            s->activeOutput.write_output(s->activeInput, s->activeMatrix, sObj);
        }

        s->cleanTraceNames.clear();
        s->cleanTraceNames.reserve(s->activeOutput.traces.size());
        for (const auto& tr : s->activeOutput.traces) {
            s->cleanTraceNames.push_back(unquote(tr.name_));
        }

        s->paramList.clear();
        for (const auto& kv : s->activeInput.parameters) {
            if (!kv.first.subcircuit().has_value() && kv.second.get_value().has_value()) {
                s->paramList.emplace_back(kv.first.name(), *kv.second.get_value());
            }
        }
        std::sort(s->paramList.begin(), s->paramList.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        return 0;
    } catch (const std::exception& e) {
        s->lastError = e.what();
        return -1;
    }
}

} // extern "C"
