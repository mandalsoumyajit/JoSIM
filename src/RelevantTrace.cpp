// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/RelevantTrace.hpp"

#include "JoSIM/Input.hpp"
#include "JoSIM/Matrix.hpp"

#include <filesystem>

using namespace JoSIM;

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
                        if (t.size() < 4) {
                            handle_voltage_or_phase(t.at(j + 1), true, mObj, fIndex);
                        } else {
                            handle_voltage_or_phase(t.at(j + 1) + " " + t.at(j + 2), true, mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'E':
                        handle_voltage_or_phase(t.at(j + 1), false, mObj, fIndex);
                        j = t.size();
                        break;
                    case 'P':
                        if (t.size() < 4) {
                            handle_voltage_or_phase(t.at(j + 1), false, mObj, fIndex);
                        } else {
                            handle_voltage_or_phase(t.at(j + 1) + " " + t.at(j + 2), false, mObj, fIndex);
                        }
                        j = t.size();
                        break;
                    case 'I':
                        handle_current(t.at(j + 1), mObj, fIndex);
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
    // Store the indices of the identified traces
    for (const auto& i : mObj.relevantTraces) {
        if (i.index1) { mObj.relevantIndices.emplace_back(i.index1.value()); }
        if (i.index2) { mObj.relevantIndices.emplace_back(i.index2.value()); }
        if (i.variableIndex) { mObj.relevantIndices.emplace_back(i.variableIndex.value()); }
    }
    // Remove any dupicate indices used for storing
    mObj.relevantIndices.erase(uniquify(mObj.relevantIndices.begin(), mObj.relevantIndices.end()),
                               mObj.relevantIndices.end());
}

void JoSIM::handle_current(const std::string& s, Matrix& mObj, int64_t fIndex) {
    RelevantTrace temp;
    temp.fIndex      = fIndex;
    temp.storageType = StorageType::Current;
    if (s.at(0) != 'I') {
        auto it = mObj.components.deviceMap.find(s);
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
    RelevantTrace            temp;
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
