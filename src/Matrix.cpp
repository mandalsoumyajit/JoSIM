// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Matrix.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Function.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Noise.hpp"
#include "JoSIM/ProgressBar.hpp"
#include "JoSIM/Rng.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_set>

using namespace JoSIM;

void Matrix::create_matrix(Input& iObj) {
    if (!iObj.ptlLumpedDone) {
        iObj.ptlLumpedDone = true;
        if (iObj.argLumpPtl && iObj.transSim.tstep() > 0.0) {
            const double h0    = iObj.transSim.tstep();
            const double minTd = std::max(iObj.argMinPtl, iObj.argTxInterp ? (h0 * (1.0 - 1e-12)) : (0.5 * h0));
            std::unordered_set<std::string> printedTokens;
            for (const auto& ctrl : iObj.controls) {
                if (!ctrl.empty() && (ctrl.front() == "PRINT" || ctrl.front() == "PLOT" || ctrl.front() == "SAVE")) {
                    for (size_t k = 1; k < ctrl.size(); ++k) {
                        printedTokens.insert(ctrl[k]);
                        const auto p1 = ctrl[k].find('(');
                        const auto p2 = ctrl[k].find(')');
                        if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1 + 1) {
                            printedTokens.insert(ctrl[k].substr(p1 + 1, p2 - p1 - 1));
                        }
                    }
                }
            }
            auto is_gnd_node = [](const std::string& n) {
                return n == "0" || n.find("GND") != std::string::npos;
            };
            std::vector<std::pair<tokens_t, string_o>> extraCaps;
            int64_t                                    lumpedCount = 0;
            for (auto& line : iObj.netlist.expNetlist) {
                if (line.first.empty() || line.first.front().at(0) != 'T' || line.first.size() < 6) { continue; }
                expand_inline_parameters(line, iObj.parameters);
                const std::string& tLabel = line.first.at(0);
                if (printedTokens.count(tLabel) != 0) { continue; }
                const std::string& n1Pos = line.first.at(1);
                const std::string& n1Neg = line.first.at(2);
                const std::string& n2Pos = line.first.at(3);
                const std::string& n2Neg = line.first.at(4);
                const bool         sameRef = (is_gnd_node(n1Neg) && is_gnd_node(n2Neg)) || (n1Neg == n2Neg);
                if (!sameRef || n1Pos == n2Pos) { continue; }

                double rawTd = 0.0;
                double z0    = 0.0;
                bool   hasTd = false;
                bool   hasZ0 = false;
                for (size_t k = 5; k < line.first.size(); ++k) {
                    const std::string& tok = line.first.at(k);
                    if (tok.size() >= 4 && ::memcmp(tok.c_str(), "Z0=", 3) == 0) {
                        z0    = parse_param(tok.substr(3), iObj.parameters, line.second);
                        hasZ0 = true;
                    } else if (tok.size() >= 4 && ::memcmp(tok.c_str(), "TD=", 3) == 0) {
                        rawTd = parse_param(tok.substr(3), iObj.parameters, line.second);
                        hasTd = true;
                    }
                }
                if (hasTd && hasZ0 && z0 > 0.0 && rawTd > 0.0 && rawTd < minTd) {
                    const double lEq   = z0 * rawTd;
                    const double cPort = rawTd / (2.0 * z0);
                    const std::string origLabel = tLabel;
                    const std::string p1Pos     = n1Pos;
                    const std::string p1Neg     = n1Neg;
                    const std::string p2Pos     = n2Pos;
                    const std::string p2Neg     = n2Neg;
                    line.first = tokens_t{"L$ptl$" + origLabel, p1Pos, p2Pos, Misc::precise_to_string(lEq), "SPREAD=0"};
                    if (cPort > 0.0) {
                        if (p1Pos != p1Neg) {
                            extraCaps.emplace_back(
                                    tokens_t{"C$ptl1$" + origLabel, p1Pos, p1Neg, Misc::precise_to_string(cPort), "SPREAD=0"},
                                    line.second);
                        }
                        if (p2Pos != p2Neg) {
                            extraCaps.emplace_back(
                                    tokens_t{"C$ptl2$" + origLabel, p2Pos, p2Neg, Misc::precise_to_string(cPort), "SPREAD=0"},
                                    line.second);
                        }
                    }
                    ++lumpedCount;
                }
            }
            if (!extraCaps.empty()) {
                iObj.netlist.expNetlist.insert(iObj.netlist.expNetlist.end(), extraCaps.begin(), extraCaps.end());
            }
            if (lumpedCount > 0 && iObj.argVerb >= 1) {
                std::cout << "JoSIM: Replaced " << lumpedCount << " sub-timestep transmission line(s) (TD < " << minTd
                          << " s) with lumped L-C pi equivalents\n";
            }
        }
    }
    iObj.netlist.rawExpNetlist = iObj.netlist.expNetlist;
    while (needsTR_) {
        // Rewind RNG
        Rng::rewind();
        // Do matrix setup
        setup(iObj);
        // Create the components
        create_components(iObj);
        if (needsTR_) { reduce_step(iObj); }
    }
    // Handle mutual inductances
    handle_mutual_inductance(iObj);
    // Create the compressed storage row format required for simulation
    create_csr();
}

void Matrix::setup(Input& iObj) {
    needsTR_ = false;
    spread.get_spreads(iObj);
    Noise::determine_global_temperature(iObj);
    Noise::determine_noise_effective_bandwidth(iObj);
    // Create a node counter variable
    int64_t nodeCounter = 0;
    // Variables to store node configs since they are already identified here
    nodeConfig.resize(iObj.netlist.expNetlist.size(), NodeConfig::GND);
    // Only available when 4 node components are detected
    nodeConfig2.resize(iObj.netlist.expNetlist.size(), NodeConfig::GND);
    int64_t cc = 0;
    for (auto& i : iObj.netlist.expNetlist) {
        if (i.first.at(0).at(0) != 'K') {
            // Expand all inline parameters
            expand_inline_parameters(i, iObj.parameters);
            // All devices should atleast have 4 parts
            if (i.first.size() < 4) {
                // Complain if it doesnt
                Errors::invalid_component_errors(ComponentErrors::INVALID_COMPONENT_DECLARATION,
                                                 Misc::vector_to_string(i.first));
            }
            // Create a node map that maps the node names to numbers
            if (i.first.at(1).find("GND") == std::string::npos && i.first.at(1) != "0") {
                // Add the first node to the map if not ground
                if (nm.count(i.first.at(1)) == 0) { nm[i.first.at(1)] = nodeCounter++; }
                nodeConfig.at(cc) = NodeConfig::POSGND;
            }
            if (i.first.at(2).find("GND") == std::string::npos && i.first.at(2) != "0") {
                // Add the second node to the map if not ground
                if (nm.count(i.first.at(2)) == 0) { nm[i.first.at(2)] = nodeCounter++; }
                if (nodeConfig.at(cc) == NodeConfig::POSGND) {
                    nodeConfig.at(cc) = NodeConfig::POSNEG;
                } else {
                    nodeConfig.at(cc) = NodeConfig::GNDNEG;
                }
            }
            // If the device is as 4 node device
            if (std::string("EFGHT").find(i.first.front().at(0)) != std::string::npos) {
                // Ensure the device has at least 6 parts
                if (i.first.size() < 6) {
                    // Complain if it isnt
                    Errors::invalid_component_errors(ComponentErrors::INVALID_COMPONENT_DECLARATION,
                                                     Misc::vector_to_string(i.first));
                }
                if (i.first.at(3).find("GND") == std::string::npos && i.first.at(3) != "0") {
                    // Add the third node to the map if not ground
                    if (nm.count(i.first.at(3)) == 0) { nm[i.first.at(3)] = nodeCounter++; }
                    nodeConfig2.at(cc) = NodeConfig::POSGND;
                }
                if (i.first.at(4).find("GND") == std::string::npos && i.first.at(4) != "0") {
                    // Add the fourth node to the map if not ground
                    if (nm.count(i.first.at(4)) == 0) { nm[i.first.at(4)] = nodeCounter++; }
                    if (nodeConfig2.at(cc) == NodeConfig::POSGND) {
                        nodeConfig2.at(cc) = NodeConfig::POSNEG;
                    } else {
                        nodeConfig2.at(cc) = NodeConfig::GNDNEG;
                    }
                }
            }
        }
        ++cc;
    }
    // Resize the node connection vector to match the size of the node map
    nc.resize(nm.size());
    // Set the index to the first branch current to the size of the node map
    branchIndex = nm.size();
}

void Matrix::create_components(Input& iObj) {
    int64_t     cc = 0;
    ProgressBar bar;
    if (!iObj.argMin) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Creating Matrix");
        bar.set_total((float) iObj.netlist.expNetlist.size());
    }
    // Map from (source function specification + subckt) to index in sourcegen for deterministic deduplication
    std::unordered_map<std::string, size_t> sourceDedup;
    auto get_or_create_source = [&](const std::pair<tokens_t, string_o>& line) -> size_t {
        std::string fullLine = Misc::vector_to_string(line.first);
        if (fullLine.find("NOISE") == std::string::npos && line.first.size() > 3) {
            std::string key;
            for (size_t k = 3; k < line.first.size(); ++k) {
                key.append(line.first[k]);
                key.push_back(' ');
            }
            if (line.second) {
                key.push_back('|');
                key.append(line.second.value());
            }
            auto it = sourceDedup.find(key);
            if (it != sourceDedup.end()) { return it->second; }
            const size_t newIdx = sourcegen.size();
            sourcegen.emplace_back();
            sourcegen.back().parse_function(fullLine, iObj, line.second);
            sourceDedup.emplace( std::move(key), newIdx );
            return newIdx;
        }
        const size_t newIdx = sourcegen.size();
        sourcegen.emplace_back();
        sourcegen.back().parse_function(fullLine, iObj, line.second);
        return newIdx;
    };

    // Loop through all the components in the netlist
    for (auto it = iObj.netlist.expNetlist.begin(); it != iObj.netlist.expNetlist.end(); it++) {
        const auto& i = *it;
        // If not minimal printing
        if (!iObj.argMin) {
            // Report progress
            bar.update((float) cc);
        }
        // First character of first token indicates device type
        switch (i.first.front().at(0)) {
                // Inductors (most common so do them first)
            case 'L':
                // Create an inductor and add it to the component list
                components.devices.emplace_back(Inductor(i, nodeConfig.at(cc), nm, lm, nc, iObj, spread, branchIndex));
                // Store this inductor's component list index for reference
                components.inductorIndices.emplace_back(components.devices.size() - 1);
                break;
                // Josephson junction (JJ)
            case 'B':
                // Create a JJ and add it to the component list
                components.devices.emplace_back(JJ(i, nodeConfig.at(cc), nm, lm, nc, iObj, spread, branchIndex));
                // Store this JJ's component list index for reference
                components.junctionIndices.emplace_back(components.devices.size() - 1);
                break;
                // Resistors
            case 'R':
                // Create a resistor and add it to the component list
                components.devices.emplace_back(Resistor(i, nodeConfig.at(cc), nm, lm, nc, iObj, spread, branchIndex));
                // Store this resistor's component list index for reference
                components.resistorIndices.emplace_back(components.devices.size() - 1);
                break;
                // Current Source
            case 'I': {
                const size_t srcIdx = get_or_create_source(i);
                components.currentsources.emplace_back(CurrentSource(i, nodeConfig.at(cc), nm, lm, srcIdx));
                components.csMap.emplace(components.currentsources.back().netlistInfo.label_,
                                         components.currentsources.size() - 1);
                break;
            }
                // Capacitors
            case 'C':
                // Create a capacitor and add it to the component list
                components.devices.emplace_back(Capacitor(i, nodeConfig.at(cc), nm, lm, nc, iObj, spread, branchIndex));
                // Store this capacitor's component list index for reference
                components.capacitorIndices.emplace_back(components.devices.size() - 1);
                break;
                // Voltage/Phase Source
            case 'P': {
                const size_t srcIdx = get_or_create_source(i);
                components.devices.emplace_back(PhaseSource(i, nodeConfig.at(cc), nm, lm, nc, branchIndex, srcIdx));
                components.psIndices.emplace_back(components.devices.size() - 1);
                break;
            }
            case 'V': {
                const size_t srcIdx = get_or_create_source(i);
                components.devices.emplace_back(VoltageSource(i, nodeConfig.at(cc), nm, lm, nc, branchIndex, srcIdx));
                components.vsIndices.emplace_back(components.devices.size() - 1);
                break;
            }
                // Mutual inductance
            case 'K':
                // Add line to the mutual inductance list for later inspection
                components.mutualinductances.emplace_back(i);
                break;
                // Transmission lines (4 node device)
            case 'T':
                // Create a transmission line and add it to the component list
                components.devices.emplace_back(TransmissionLine(i,
                                                                 nodeConfig.at(cc),
                                                                 nodeConfig2.at(cc),
                                                                 nm,
                                                                 lm,
                                                                 nc,
                                                                 iObj.parameters,
                                                                 iObj.argAnal,
                                                                 iObj.transSim.tstep(),
                                                                 branchIndex,
                                                                 iObj.argTxInterp));
                if (std::get<TransmissionLine>(components.devices.back()).timestepDelay_ < 1) {
                    needsTR_ = true;
                    return;
                }
                // Store the transmission line component list index for reference
                components.txIndices.emplace_back(components.devices.size() - 1);
                break;
                // Voltage controlled current source (4 node device)
            case 'G':
                // Create a vccs and add it to the component list
                components.devices.emplace_back(VCCS(i,
                                                     nodeConfig.at(cc),
                                                     nodeConfig2.at(cc),
                                                     nm,
                                                     lm,
                                                     nc,
                                                     iObj.parameters,
                                                     branchIndex,
                                                     iObj.argAnal,
                                                     iObj.transSim.tstep()));
                // Store the vccs component list index for reference
                components.vccsIndices.emplace_back(components.devices.size() - 1);
                break;
                // Current controlled current source (4 node device)
            case 'F':
                // Create a cccs and add it to the component list
                components.devices.emplace_back(
                        CCCS(i, nodeConfig.at(cc), nodeConfig2.at(cc), nm, lm, nc, iObj.parameters, branchIndex));
                // Store the cccs component list index for reference
                components.cccsIndices.emplace_back(components.devices.size() - 1);
                break;
                // Voltage controlled voltage source (4 node device)
            case 'E':
                // Create a vcvs and add it to the component list
                components.devices.emplace_back(
                        VCVS(i, nodeConfig.at(cc), nodeConfig2.at(cc), nm, lm, nc, iObj.parameters, branchIndex));
                // Store the vcvs component list index for reference
                components.vcvsIndices.emplace_back(components.devices.size() - 1);
                break;
                // Current controlled voltage source (4 node device)
            case 'H':
                // Create a ccvs and add it to the component list
                components.devices.emplace_back(CCVS(i,
                                                     nodeConfig.at(cc),
                                                     nodeConfig2.at(cc),
                                                     nm,
                                                     lm,
                                                     nc,
                                                     iObj.parameters,
                                                     branchIndex,
                                                     iObj.argAnal,
                                                     iObj.transSim.tstep()));
                // Store the ccvs component list index for reference
                components.ccvsIndices.emplace_back(components.devices.size() - 1);
                break;
                // This is an error. This means that the component was unidentified.
            default: Errors::invalid_component_errors(ComponentErrors::UNKNOWN_DEVICE_TYPE, i.first.front());
        }
        // Increment the component counter
        ++cc;
    }
    components.deviceMap.reserve(components.devices.size());
    for (int64_t devIdx = 0; devIdx < static_cast<int64_t>(components.devices.size()); ++devIdx) {
        std::visit([&](const auto& device) { components.deviceMap.emplace(device.netlistInfo.label_, devIdx); },
                   components.devices[devIdx]);
    }
    if (!iObj.argMin) {
        bar.complete();
        std::cout << "\n";
    }
}

void Matrix::handle_mutual_inductance(Input& iObj) {
    int64_t     cc       = 0;
    int64_t     skippedK = 0;
    ProgressBar bar2;
    if (!iObj.argMin && components.mutualinductances.size() != 0) {
        bar2.create_thread();
        bar2.set_bar_width(30);
        bar2.fill_bar_progress_with("O");
        bar2.fill_bar_remainder_with(" ");
        bar2.set_status_text("Adding Mutual Inductances");
        bar2.set_total((float) iObj.netlist.expNetlist.size());
    }
    // Loop through all identified mutual inductances
    for (const auto& s : components.mutualinductances) {
        // If not minimal printing
        if (!iObj.argMin && components.mutualinductances.size() != 0) {
            // Report progress
            bar2.update((float) cc);
        }
        // If the mutual inductance line has less than 4 tokens
        if (s.first.size() < 4) {
            // Complain
            Errors::invalid_component_errors(ComponentErrors::MUT_ERROR, Misc::vector_to_string(s.first));
        }
        // Variables to store the indices of the relevant inductors
        int_o ind1Index, ind2Index;
        auto  it1 = components.deviceMap.find(s.first.at(1));
        if (it1 != components.deviceMap.end()) { ind1Index = it1->second; }
        auto it2 = components.deviceMap.find(s.first.at(2));
        if (it2 != components.deviceMap.end()) { ind2Index = it2->second; }
        // Complain if any of these inductors do not exist
        if (!ind1Index) { Errors::invalid_component_errors(ComponentErrors::MISSING_INDUCTOR, s.first.at(1)); }
        if (!ind2Index) { Errors::invalid_component_errors(ComponentErrors::MISSING_INDUCTOR, s.first.at(2)); }
        // Fetch the relevant inductors in the component list given the indices
        auto&  ind1   = std::get<Inductor>(components.devices.at(ind1Index.value()));
        auto&  ind2   = std::get<Inductor>(components.devices.at(ind2Index.value()));
        // Calculate the coupling factor
        double cf     = parse_param(s.first.at(3), iObj.parameters, s.second);
        if (iObj.argKCutoff > 0.0 && std::abs(cf) < iObj.argKCutoff) {
            ++skippedK;
            ++cc;
            continue;
        }
        // Calculate the mutual inductance for this pair
        double mutual = cf * std::sqrt(ind1.netlistInfo.value_ * ind2.netlistInfo.value_);
        if (mutual != 0) {
            // Add the mutual inductance to each inductor respectively
            ind1.add_mutualInductance(
                    mutual, iObj.argAnal, iObj.transSim.tstep(), ind2.indexInfo.currentIndex_.value());
            ind2.add_mutualInductance(
                    mutual, iObj.argAnal, iObj.transSim.tstep(), ind1.indexInfo.currentIndex_.value());
            // Add this mutual inductance to each inductor's matrix stamp
            ind1.set_mutualInductance(std::make_pair(ind2Index.value(), mutual));
            ind2.set_mutualInductance(std::make_pair(ind1Index.value(), mutual));
        }
        ++cc;
    }
    // Led the user know the matrix creation is complete
    if (!iObj.argMin && components.mutualinductances.size() != 0) {
        bar2.complete();
        std::cout << "\n";
    }
    if (skippedK > 0 && !iObj.argMin && iObj.argVerb >= 1) {
        std::cout << "JoSIM: Pruned " << skippedK << " of " << components.mutualinductances.size()
                  << " mutual inductance(s) with |k| < " << iObj.argKCutoff << "\n";
    }
}

void Matrix::reduce_step(Input& iObj) {
    iObj.transSim.tstep(iObj.transSim.tstep() / 2);
    nodeConfig.clear();
    nodeConfig2.clear();
    Components newComponents;
    components = newComponents;
    sourcegen.clear();
    sourcegen.shrink_to_fit();
    nm.clear();
    nodeconnections newNC;
    nc = newNC;
    lm.clear();
    branchIndex = 0;
    nz.clear();
    ci.clear();
    rp.clear();
    relevantIndices.clear();
    relevantTraces.clear();
}

void Matrix::create_csr() {
    // Create the row pointer vector first so rp.back() gives the exact NNZ count
    create_rp();
    // Create the non zero vector
    create_nz();
    // Create the column index vector
    create_ci();
}

void Matrix::create_nz() {
    // Non zero vector should be empty, but clear it to make sure
    nz.clear();
    if (!rp.empty() && rp.back() > 0) { nz.reserve(static_cast<size_t>(rp.back())); }
    // Loop through all the node connections
    for (const auto& it : nc) {
        // Loop through each connection
        for (const auto& ti : it) {
            // Store the non zero values of the connection
            nz.emplace_back(ti.first);
        }
    }
    // Loop through all the devices
    for (auto& i : components.devices) {
        std::visit(
                [&](auto& device) {
                    const auto& nonZeros = device.matrixInfo.nonZeros_;
                    nz.insert(nz.end(), nonZeros.begin(), nonZeros.end());
                    if constexpr (std::is_same_v<std::decay_t<decltype(device)>, JJ>) {
                        device.nzBackIndex_ = static_cast<int64_t>(nz.size()) - 1;
                    }
                },
                i);
    }
}

void Matrix::create_ci() {
    // Clear the column index vector as safety precaution
    ci.clear();
    if (!rp.empty() && rp.back() > 0) { ci.reserve(static_cast<size_t>(rp.back())); }
    // Loop through all the node connections
    for (const auto& it : nc) {
        // Loop through each connection
        for (const auto& ti : it) {
            // Add the column index for the connection to the column index vector
            ci.emplace_back(ti.second);
        }
    }
    // Loop through all the devices
    for (auto& i : components.devices) {
        std::visit(
                [&](const auto& device) {
                    const auto& columnIndices = device.matrixInfo.columnIndex_;
                    ci.insert(ci.end(), columnIndices.begin(), columnIndices.end());
                },
                i);
    }
}

void Matrix::create_rp() {
    // Clear the row pointer vector
    rp.clear();
    rp.reserve(nc.size() + components.devices.size() * 2 + 1);
    // First entry should always be zero
    rp.emplace_back(0);
    // Add the number of node connections in each row to the row pointer vector
    for (const auto& it : nc) { rp.emplace_back(rp.back() + it.size()); }
    // Loop through all the devices
    for (auto& i : components.devices) {
        std::visit(
                [&](const auto& device) {
                    const auto& rowPointer = device.matrixInfo.rowPointer_;
                    for (const auto& ti : rowPointer) { rp.emplace_back(rp.back() + ti); }
                },
                i);
    }
}
