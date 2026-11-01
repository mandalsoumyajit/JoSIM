// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/RelevantTrace.hpp"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/Input.hpp"
#include "JoSIM/Matrix.hpp"

#include <algorithm>
#include <filesystem>

using namespace JoSIM;

namespace {

bool trace_glob_match(const std::string& text, const std::string& pat) {
    size_t ti = 0, pi = 0, starIdx = std::string::npos, matchIdx = 0;
    while (ti < text.size()) {
        if (pi < pat.size() && (pat[pi] == '?' || pat[pi] == text[ti])) {
            ++ti;
            ++pi;
        } else if (pi < pat.size() && pat[pi] == '*') {
            starIdx  = pi++;
            matchIdx = ti;
        } else if (starIdx != std::string::npos) {
            pi = starIdx + 1;
            ti = ++matchIdx;
        } else {
            return false;
        }
    }
    while (pi < pat.size() && pat[pi] == '*') { ++pi; }
    return pi == pat.size();
}

} // namespace

void JoSIM::find_relevant_traces(Input& iObj, Matrix& mObj) {
    // Temporary trace to store in known traces
    RelevantTrace          temp;
    int64_t                fIndex = -1;
    std::vector<tokens_t>& c      = iObj.controls;
    // Loop through and handle plot, print and save commands
    for (auto i = 0; i < c.size(); ++i) {
        if ((c.at(i).front() == "PRINT") || (c.at(i).front() == "PLOT") || (c.at(i).front() == "SAVE")) {
            // Mutable version of the control line tokens
            tokens_t t = c.at(i);
            // Ensure that the store command conforms to syntax
            if (t.size() < 2 && t.at(0) != "SAVE") {
                Errors::control_errors(ControlErrors::INVALID_OUTPUT_COMMAND, Misc::vector_to_string(t));
            }
            // Fix any naming issues, '.' should be '|'
            for (auto& l : t) {
                if (l.find('.') != std::string::npos) { std::replace(l.begin(), l.end(), '.', '|'); }
            }
            // Remove any "TRAN" tokens. We don't do that here.
            for (auto j = 0; j < t.size(); ++j) {
                if (t.at(j) == "TRAN") { t.erase(t.begin() + j); }
            }
            // Determine the type of plot
            for (size_t j = 1; j < t.size(); ++j) {
                switch (t.at(j).back()) {
                    case ')':
                        switch (t.at(j).at(0)) {
                            case 'I':
                                t.at(j).erase(0, 2);
                                t.at(j).erase(t.at(j).size() - 1, t.at(j).size());
                                handle_current(t.at(j), mObj, fIndex);
                                break;
                            case 'V':
                                t.at(j).erase(0, 2);
                                t.at(j).erase(t.at(j).size() - 1, t.at(j).size());
                                handle_voltage_or_phase(t.at(j), true, mObj, fIndex);
                                break;
                            case 'P':
                                t.at(j).erase(0, 2);
                                t.at(j).erase(t.at(j).size() - 1, t.at(j).size());
                                handle_voltage_or_phase(t.at(j), false, mObj, fIndex);
                                break;
                        }
                        break;
                    case 'V':
                        if (t.at(j) == "DEVV") {
                            for (size_t k = j + 1; k < t.size(); ++k) {
                                handle_voltage_or_phase(t.at(k), true, mObj, fIndex);
                            }
                        } else if (t.size() < 4) {
                            handle_voltage_or_phase(t.at(j + 1), true, mObj, fIndex);
                        } else {
                            handle_voltage_or_phase(t.at(j + 1) + " " + t.at(j + 2), true, mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'E':
                        for (size_t k = j + 1; k < t.size(); ++k) {
                            handle_voltage_or_phase(t.at(k), false, mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'P':
                        if (t.at(j) == "DEVP") {
                            for (size_t k = j + 1; k < t.size(); ++k) {
                                handle_voltage_or_phase(t.at(k), false, mObj, fIndex);
                            }
                        } else if (t.size() < 4) {
                            handle_voltage_or_phase(t.at(j + 1), false, mObj, fIndex);
                        } else {
                            handle_voltage_or_phase(t.at(j + 1) + " " + t.at(j + 2), false, mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'I':
                        for (size_t k = j + 1; k < t.size(); ++k) {
                            handle_current(t.at(k), mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'H':
                        if (t.at(j).find("#BRANCH") != std::string::npos) {
                            t.at(j) = t.at(j).substr(0, t.at(j).find("#BRANCH"));
                            handle_current(t.at(j), mObj, fIndex);
                            break;
                        } else {
                            Errors::control_errors(ControlErrors::INVALID_OUTPUT_COMMAND,
                                                   t.at(j) + " in line:\n" + Misc::vector_to_string(c.at(i)));
                            break;
                        }
                    case ']':
                        if (t.at(j).at(0) == '@') {
                            switch (t.at(j).at(t.at(j).size() - 2)) {
                                case 'C':
                                    t.at(j).erase(0, 1);
                                    t.at(j).erase(t.at(j).size() - 3, t.at(j).size());
                                    handle_current(t.at(j), mObj, fIndex);
                                    break;
                                case 'V':
                                    t.at(j).erase(0, 1);
                                    t.at(j).erase(t.at(j).size() - 3, t.at(j).size());
                                    handle_voltage_or_phase(t.at(j), true, mObj, fIndex);
                                    break;
                                case 'P':
                                    t.at(j).erase(0, 1);
                                    t.at(j).erase(t.at(j).size() - 3, t.at(j).size());
                                    handle_voltage_or_phase(t.at(j), false, mObj, fIndex);
                                    break;
                            }
                            break;
                        } else {
                            Errors::control_errors(ControlErrors::INVALID_OUTPUT_COMMAND,
                                                   t.at(j) + " in line:\n" + Misc::vector_to_string(c.at(i)));
                            break;
                        }
                    default:
                        Errors::control_errors(ControlErrors::INVALID_OUTPUT_COMMAND,
                                               t.at(j) + " in line:\n" + Misc::vector_to_string(c.at(i)));
                        break;
                }
                if (t.at(1) == "NODE") { break; }
            }
        } else if (c.at(i).front() == "FILE") {
            fIndex++;
        }
    }
    if (iObj.argEventsOnly) {
        mObj.relevantTraces.erase(
                std::remove_if(mObj.relevantTraces.begin(),
                               mObj.relevantTraces.end(),
                               [](const RelevantTrace& rt) { return rt.storageType != StorageType::Phase; }),
                mObj.relevantTraces.end());
        if (mObj.relevantTraces.empty()
            && (iObj.argEvents || iObj.argMargins || iObj.argSens || iObj.argSweep || iObj.verifyEventsFile.has_value()
                || (iObj.assertions.empty() && iObj.measurements.empty() && iObj.jitterSpecs.empty()))) {
            for (const auto& devVar : mObj.components.devices) {
                if (std::holds_alternative<JJ>(devVar)) {
                    const auto&   jjDev = std::get<JJ>(devVar);
                    RelevantTrace temp;
                    temp.device        = true;
                    temp.storageType   = StorageType::Phase;
                    temp.deviceLabel   = "\"P(" + jjDev.netlistInfo.label_ + ")\"";
                    temp.index1        = jjDev.indexInfo.posIndex_;
                    temp.index2        = jjDev.indexInfo.negIndex_;
                    temp.variableIndex = jjDev.variableIndex_;
                    mObj.relevantTraces.emplace_back(temp);
                }
            }
        }
    } else if (iObj.argMargins || iObj.argSens || iObj.argSweep || iObj.argEvents
               || iObj.verifyEventsFile.has_value()) {
        bool hasPhaseTrace = false;
        for (const auto& rt : mObj.relevantTraces) {
            if (rt.storageType == StorageType::Phase) {
                hasPhaseTrace = true;
                break;
            }
        }
        if (!hasPhaseTrace) {
            for (const auto& devVar : mObj.components.devices) {
                if (std::holds_alternative<JJ>(devVar)) {
                    const auto&   jjDev = std::get<JJ>(devVar);
                    RelevantTrace temp;
                    temp.device             = true;
                    temp.storageType        = StorageType::Phase;
                    temp.deviceLabel        = "\"P(" + jjDev.netlistInfo.label_ + ")\"";
                    temp.index1             = jjDev.indexInfo.posIndex_;
                    temp.index2             = jjDev.indexInfo.negIndex_;
                    temp.variableIndex      = jjDev.variableIndex_;
                    temp.autoAddedForEvents = true;
                    mObj.relevantTraces.emplace_back(temp);
                }
            }
        }
    }
    if (!iObj.assertions.empty() || !iObj.jitterSpecs.empty() || !iObj.marginTraces.empty()) {
        auto ensure_assert_phase_trace = [&](std::string rawName) {
            rawName.erase(std::remove(rawName.begin(), rawName.end(), '\"'), rawName.end());
            rawName.erase(std::remove(rawName.begin(), rawName.end(), ' '), rawName.end());
            rawName.erase(std::remove(rawName.begin(), rawName.end(), '\r'), rawName.end());
            if (rawName.empty()) { return; }
            std::transform(rawName.begin(), rawName.end(), rawName.begin(), ::toupper);
            std::replace(rawName.begin(), rawName.end(), '.', '|');
            std::string inner = rawName;
            if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0) && inner.back() == ')') {
                inner = inner.substr(2, inner.size() - 3);
            }
            const std::string expectedLbl = "\"P(" + inner + ")\"";
            for (const auto& rt : mObj.relevantTraces) {
                if (rt.storageType == StorageType::Phase && rt.deviceLabel.has_value()
                    && rt.deviceLabel.value() == expectedLbl) {
                    return;
                }
            }
            const size_t prevSize = mObj.relevantTraces.size();
            handle_voltage_or_phase(inner, false, mObj, -1);
            for (size_t idx = prevSize; idx < mObj.relevantTraces.size();) {
                bool dup = false;
                for (size_t k = 0; k < idx; ++k) {
                    if (mObj.relevantTraces[k].storageType == StorageType::Phase
                        && mObj.relevantTraces[k].deviceLabel == mObj.relevantTraces[idx].deviceLabel) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    mObj.relevantTraces.erase(mObj.relevantTraces.begin() + static_cast<ptrdiff_t>(idx));
                } else {
                    mObj.relevantTraces[idx].autoAddedForEvents = !iObj.argEventsOnly;
                    ++idx;
                }
            }
        };
        for (const auto& as : iObj.assertions) {
            ensure_assert_phase_trace(as.trace);
            if (!as.refTrace.empty()) { ensure_assert_phase_trace(as.refTrace); }
        }
        for (const auto& js : iObj.jitterSpecs) {
            ensure_assert_phase_trace(js.trace);
            if (!js.refTrace.empty()) { ensure_assert_phase_trace(js.refTrace); }
        }
        for (const auto& mt : iObj.marginTraces) {
            ensure_assert_phase_trace(mt);
        }
    }
    if (!iObj.measurements.empty()) {
        auto ensure_meas_trace = [&](std::string rawName, bool forcePhase) {
            rawName.erase(std::remove(rawName.begin(), rawName.end(), '\"'), rawName.end());
            rawName.erase(std::remove(rawName.begin(), rawName.end(), ' '), rawName.end());
            rawName.erase(std::remove(rawName.begin(), rawName.end(), '\r'), rawName.end());
            if (rawName.empty()) { return; }
            std::transform(rawName.begin(), rawName.end(), rawName.begin(), ::toupper);
            std::replace(rawName.begin(), rawName.end(), '.', '|');
            std::string inner  = rawName;
            char        prefix = '\0';
            if (inner.size() > 3 && (inner.rfind("P(", 0) == 0 || inner.rfind("V(", 0) == 0 || inner.rfind("I(", 0) == 0)
                && inner.back() == ')') {
                prefix = inner[0];
                inner  = inner.substr(2, inner.size() - 3);
            }
            const size_t prevSize = mObj.relevantTraces.size();
            if (forcePhase) {
                handle_voltage_or_phase(inner, false, mObj, -1);
            } else if (prefix == 'I') {
                handle_current(inner, mObj, -1);
            } else if (prefix == 'V') {
                handle_voltage_or_phase(inner, true, mObj, -1);
            } else if (prefix == 'P') {
                handle_voltage_or_phase(inner, false, mObj, -1);
            } else {
                handle_voltage_or_phase(inner, iObj.argAnal == AnalysisType::Voltage, mObj, -1);
            }
            for (size_t idx = prevSize; idx < mObj.relevantTraces.size();) {
                bool dup = false;
                for (size_t k = 0; k < idx; ++k) {
                    if (mObj.relevantTraces[k].storageType == mObj.relevantTraces[idx].storageType
                        && mObj.relevantTraces[k].deviceLabel == mObj.relevantTraces[idx].deviceLabel) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    mObj.relevantTraces.erase(mObj.relevantTraces.begin() + static_cast<ptrdiff_t>(idx));
                } else {
                    mObj.relevantTraces[idx].autoAddedForEvents = !iObj.argEventsOnly;
                    ++idx;
                }
            }
        };
        for (const auto& ms : iObj.measurements) {
            const bool isEventKind = (ms.kind == MeasKind::COUNT || ms.kind == MeasKind::FREQ
                                      || ms.kind == MeasKind::VPEAK || ms.kind == MeasKind::FWHM
                                      || ms.kind == MeasKind::DT_PI || ms.kind == MeasKind::AREA);
            if (!ms.trace.empty() && ms.kind != MeasKind::WHEN && ms.kind != MeasKind::TRIG_TARG
                && ms.kind != MeasKind::PARAM) {
                ensure_meas_trace(ms.trace, isEventKind);
            }
            if (ms.trig.active && !ms.trig.isFixedTime) {
                ensure_meas_trace(ms.trig.trace, ms.trig.isSfqEvent);
                if (!ms.trig.rhsTrace.empty()) { ensure_meas_trace(ms.trig.rhsTrace, false); }
            }
            if (ms.targ.active && !ms.targ.isFixedTime) {
                ensure_meas_trace(ms.targ.trace, ms.targ.isSfqEvent);
                if (!ms.targ.rhsTrace.empty()) { ensure_meas_trace(ms.targ.rhsTrace, false); }
            }
        }
    }
    // Store the indices of the identified traces (dropping unused JJ node/variable indices)
    for (auto& i : mObj.relevantTraces) {
        if (i.variableIndex) {
            const bool useVarIdx = (iObj.argAnal == AnalysisType::Phase && i.storageType == StorageType::Voltage)
                                   || (iObj.argAnal == AnalysisType::Voltage && i.storageType == StorageType::Phase);
            if (useVarIdx) {
                i.index1.reset();
                i.index2.reset();
            } else {
                i.variableIndex.reset();
            }
        }
        if (i.index1) { mObj.relevantIndices.emplace_back(i.index1.value()); }
        if (i.index2) { mObj.relevantIndices.emplace_back(i.index2.value()); }
        if (i.variableIndex) { mObj.relevantIndices.emplace_back(i.variableIndex.value()); }
    }
    // Remove any dupicate indices used for storing
    mObj.relevantIndices.erase(uniquify(mObj.relevantIndices.begin(), mObj.relevantIndices.end()),
                               mObj.relevantIndices.end());
}

void JoSIM::handle_current(const std::string& s, Matrix& mObj, int64_t fIndex) {
    if (s.find('*') != std::string::npos || s.find('?') != std::string::npos) {
        size_t matched = 0;
        for (const auto& devVar : mObj.components.devices) {
            const std::string lbl = std::visit(
                    [](const auto& d) -> std::string { return d.netlistInfo.label_; }, devVar);
            const bool hasCur = std::visit(
                    [](const auto& d) -> bool { return d.indexInfo.currentIndex_.has_value(); }, devVar);
            if (hasCur && trace_glob_match(lbl, s)) {
                handle_current(lbl, mObj, fIndex);
                ++matched;
            }
        }
        for (const auto& cs : mObj.components.currentsources) {
            if (trace_glob_match(cs.netlistInfo.label_, s)) {
                handle_current(cs.netlistInfo.label_, mObj, fIndex);
                ++matched;
            }
        }
        if (matched == 0) { Errors::control_errors(ControlErrors::NODECURRENT, s); }
        return;
    }
    RelevantTrace temp;
    temp.fIndex      = fIndex;
    temp.storageType = StorageType::Current;
    if (s.at(0) != 'I') {
        auto it = mObj.components.deviceMap.find(s);
        if (it == mObj.components.deviceMap.end()) {
            const size_t barPos = s.rfind('|');
            if (barPos != std::string::npos) {
                const std::string suffix = s.substr(barPos + 1);
                if (suffix == "G" || suffix == "GATE") {
                    const std::string base = s.substr(0, barPos);
                    it                     = mObj.components.deviceMap.find("L$GATE$" + base);
                    if (it == mObj.components.deviceMap.end()) {
                        it = mObj.components.deviceMap.find("R$GATE$" + base);
                    }
                    if (it == mObj.components.deviceMap.end()) {
                        auto nwIt = mObj.components.deviceMap.find(base);
                        if (nwIt != mObj.components.deviceMap.end()) {
                            if (const auto* nw = std::get_if<Nanowire>(&mObj.components.devices.at(nwIt->second))) {
                                if (!nw->gateDevLabel_.empty()) {
                                    it = mObj.components.deviceMap.find(nw->gateDevLabel_);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (it != mObj.components.deviceMap.end()) {
            temp.deviceLabel = "\"I(" + s + ")\"";
            temp.device      = true;
            std::visit([&](const auto& device) { temp.index1 = device.indexInfo.currentIndex_.value(); },
                       mObj.components.devices.at(it->second));
            mObj.relevantTraces.emplace_back(temp);
        }
        if (!temp.deviceLabel) { Errors::control_errors(ControlErrors::NODECURRENT, s); }
    } else {
        auto it = mObj.components.csMap.find(s);
        if (it != mObj.components.csMap.end()) {
            temp.deviceLabel = "\"I(" + s + ")\"";
            temp.device      = true;
            temp.sourceIndex = mObj.components.currentsources.at(it->second).sourceIndex_;
            mObj.relevantTraces.emplace_back(temp);
        }
    }
}

void JoSIM::handle_voltage_or_phase(const std::string& s, bool voltage, Matrix& mObj, int64_t fIndex) {
    std::vector<std::string> tokens = Misc::tokenize(s, " ,");
    if (!tokens.empty() && (s.find('*') != std::string::npos || s.find('?') != std::string::npos)) {
        size_t matched = 0;
        if (tokens.size() == 1) {
            for (const auto& devVar : mObj.components.devices) {
                const std::string lbl = std::visit(
                        [](const auto& d) -> std::string { return d.netlistInfo.label_; }, devVar);
                if (trace_glob_match(lbl, tokens[0])) {
                    handle_voltage_or_phase(lbl, voltage, mObj, fIndex);
                    ++matched;
                }
            }
            if (matched == 0) {
                std::vector<std::string> nodeNames;
                nodeNames.reserve(mObj.nm.size());
                for (const auto& kv : mObj.nm) {
                    if (trace_glob_match(kv.first, tokens[0])) { nodeNames.push_back(kv.first); }
                }
                std::sort(nodeNames.begin(), nodeNames.end());
                for (const auto& nn : nodeNames) {
                    handle_voltage_or_phase(nn, voltage, mObj, fIndex);
                    ++matched;
                }
            }
        } else if (tokens.size() >= 2) {
            std::vector<std::string> nodeNames;
            nodeNames.reserve(mObj.nm.size());
            for (const auto& kv : mObj.nm) {
                if (trace_glob_match(kv.first, tokens[0])) { nodeNames.push_back(kv.first); }
            }
            std::sort(nodeNames.begin(), nodeNames.end());
            for (const auto& nn : nodeNames) {
                handle_voltage_or_phase(nn + " " + tokens[1], voltage, mObj, fIndex);
                ++matched;
            }
        }
        if (matched == 0) { Errors::control_errors(ControlErrors::UNKNOWN_DEVICE, s); }
        return;
    }
    RelevantTrace temp;
    temp.fIndex = fIndex;
    if (voltage) {
        temp.storageType = StorageType::Voltage;
    } else {
        temp.storageType = StorageType::Phase;
    }
    auto it = mObj.components.deviceMap.find(tokens.at(0));
    if (it != mObj.components.deviceMap.end()) {
        const auto& devVar = mObj.components.devices.at(it->second);
        if (tokens.size() < 1) {
            Errors::control_errors(ControlErrors::MATHOPS, s);
        } else {
            if (s.at(0) != 'B') {
                if (voltage) {
                    temp.deviceLabel = "\"V(" + s + ")\"";
                } else {
                    temp.deviceLabel = "\"P(" + s + ")\"";
                }
                temp.device = true;
                std::visit(
                        [&](const auto& device) {
                            temp.index1 = device.indexInfo.posIndex_;
                            temp.index2 = device.indexInfo.negIndex_;
                        },
                        devVar);
                mObj.relevantTraces.emplace_back(temp);
            } else if (std::holds_alternative<JJ>(devVar)) {
                if (voltage) {
                    temp.deviceLabel = "\"V(" + s + ")\"";
                } else {
                    temp.deviceLabel = "\"P(" + s + ")\"";
                }
                temp.device        = true;
                const auto& jjDev  = std::get<JJ>(devVar);
                temp.index1        = jjDev.indexInfo.posIndex_;
                temp.index2        = jjDev.indexInfo.negIndex_;
                temp.variableIndex = jjDev.variableIndex_;
                mObj.relevantTraces.emplace_back(temp);
            }
        }
    }
    if (!temp.deviceLabel) {
        if (tokens.size() > 1) {
            if (mObj.nm.count(tokens.at(0)) != 0) {
                temp.index1 = mObj.nm.at(tokens.at(0));
                if (mObj.nm.count(tokens.at(1)) != 0) {
                    temp.index2 = mObj.nm.at(tokens.at(1));
                } else {
                    if (tokens.at(1) != "0" && tokens.at(1).find("GND") == std::string::npos) {
                        Errors::control_errors(ControlErrors::UNKNOWN_DEVICE, tokens.at(1));
                    }
                }
                if (voltage) {
                    temp.deviceLabel = "\"V(" + tokens.at(0) + "," + tokens.at(1) + ")\"";
                } else {
                    temp.deviceLabel = "\"P(" + tokens.at(0) + "," + tokens.at(1) + ")\"";
                }
                mObj.relevantTraces.emplace_back(temp);
            } else {
                if (tokens.at(0) != "0" && tokens.at(0).find("GND") == std::string::npos) {
                    Errors::control_errors(ControlErrors::UNKNOWN_DEVICE, tokens.at(0));
                }
            }
        } else {
            if (mObj.nm.count(s) != 0) {
                temp.index1 = mObj.nm.at(s);
                if (voltage) {
                    temp.deviceLabel = "\"V(" + tokens.at(0) + ")\"";
                } else {
                    temp.deviceLabel = "\"P(" + tokens.at(0) + ")\"";
                }
                mObj.relevantTraces.emplace_back(temp);
            } else {
                Errors::control_errors(ControlErrors::UNKNOWN_DEVICE, s);
            }
        }
    }
}
