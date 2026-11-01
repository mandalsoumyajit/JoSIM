// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Netlist.hpp"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/Components.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/ProgressBar.hpp"

#include <algorithm>
#include <cstring> // for strcmp
#include <thread>

using namespace JoSIM;

namespace {

tokens_t normalize_subckt_call_tokens(const tokens_t& raw) {
    if (raw.empty()) { return {}; }
    tokens_t balanced;
    balanced.push_back(raw.front());
    for (size_t i = 1; i < raw.size(); ++i) {
        if (raw[i] == "PARAMS:" || raw[i] == "PARAM:") { continue; }
        std::string cur   = raw[i];
        int         depth = 0;
        for (char c : cur) {
            if (c == '(' || c == '{' || c == '[') {
                ++depth;
            } else if (c == ')' || c == '}' || c == ']') {
                --depth;
            }
        }
        while (depth > 0 && i + 1 < raw.size()) {
            ++i;
            cur += raw[i];
            for (char c : raw[i]) {
                if (c == '(' || c == '{' || c == '[') {
                    ++depth;
                } else if (c == ')' || c == '}' || c == ']') {
                    --depth;
                }
            }
        }
        balanced.push_back(std::move(cur));
    }

    tokens_t merged;
    merged.push_back(balanced.front());
    for (size_t i = 1; i < balanced.size(); ++i) {
        if (balanced[i] == "=" && merged.size() > 1 && i + 1 < balanced.size()) {
            merged.back() += "=" + balanced[i + 1];
            ++i;
        } else if (merged.size() > 1 && merged.back().back() == '=') {
            merged.back() += balanced[i];
        } else if (!balanced[i].empty() && balanced[i].front() == '=' && merged.size() > 1) {
            merged.back() += balanced[i];
        } else {
            merged.push_back(balanced[i]);
        }
    }
    return merged;
}

std::string strip_outer_braces_str(std::string s) {
    while (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
        int  depth       = 0;
        bool closesAtEnd = true;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '{') {
                ++depth;
            } else if (s[i] == '}') {
                --depth;
                if (depth == 0) {
                    closesAtEnd = false;
                    break;
                }
            }
        }
        if (closesAtEnd) {
            s = s.substr(1, s.size() - 2);
        } else {
            break;
        }
    }
    return s;
}

} // namespace

void Netlist::id_io_subc_label(const tokens_t&                                    lineTokens,
                               tokens_t&                                          io,
                               s_map&                                             params,
                               std::string&                                       subcktName,
                               std::string&                                       label,
                               const std::unordered_map<std::string, Subcircuit>& subcircuits) {
    const tokens_t normTokens = normalize_subckt_call_tokens(lineTokens);
    label                     = normTokens.front();
    bool found                = false;
    for (size_t i = 1; i < normTokens.size(); ++i) {
        if (normTokens.at(i).find('=') != std::string::npos) {
            const size_t eqPos = normTokens.at(i).find('=');
            std::string  param = normTokens.at(i).substr(0, eqPos);
            std::string  value = normTokens.at(i).substr(eqPos + 1);
            if (!param.empty() && !value.empty()) {
                params[param] = value;
            }
        } else if (subcircuits.count(normTokens.at(i)) != 0) {
            found      = true;
            subcktName = normTokens.at(i);
        } else {
            io.emplace_back(normTokens.at(i));
        }
    }
    if (!found) { Errors::input_errors(InputErrors::UNKNOWN_SUBCKT, Misc::vector_to_string(lineTokens)); }
}

bool Netlist::rename_io_nodes(std::string& node, const tokens_t& subIO, const tokens_t& parentIO) {
    // Loop through the subcircuit IO in a single pass
    for (size_t l = 0; l < subIO.size(); ++l) {
        // If the nodes match
        if (node == subIO[l]) {
            // Replace the node with the IO node
            node = parentIO[l];
            // Return true
            return true;
        }
    }
    // Return false
    return false;
}

void Netlist::expand_io(Subcircuit&        subc,
                        s_map&             params,
                        const tokens_t&    io,
                        const std::string& label,
                        const std::string& subcktName,
                        const string_o&    callerScope) {
    // Sanity check
    if (io.size() != subc.io.size()) { Errors::input_errors(InputErrors::IO_MISMATCH, label); }

    bool createInstScope = false;
    if (parametersPtr != nullptr && !subcktName.empty()) {
        if (!params.empty()) {
            createInstScope = true;
        } else if (callerScope.has_value()) {
            for (const auto& kv : *parametersPtr) {
                if (kv.first.subcircuit() == callerScope) {
                    createInstScope = true;
                    break;
                }
            }
        }
    }

    if (createInstScope) {
        hasInstScopes = true;
        std::vector<std::string> origScopes;
        for (const auto& line : subc.lines) {
            const std::string sc = line.second.value_or(subcktName);
            if (std::find(origScopes.begin(), origScopes.end(), sc) == origScopes.end()) {
                origScopes.push_back(sc);
            }
        }
        std::vector<std::pair<ParameterName, Parameter>> existingParams(parametersPtr->begin(), parametersPtr->end());
        for (const auto& origScope : origScopes) {
            const std::string newScope = origScope + "|" + label;
            if (callerScope.has_value()) {
                for (const auto& kv : existingParams) {
                    if (kv.first.subcircuit() == callerScope) {
                        Parameter p = kv.second;
                        p.reset_value();
                        (*parametersPtr)[ParameterName(kv.first.name(), newScope)] = p;
                    }
                }
            }
            for (const auto& kv : existingParams) {
                if (kv.first.subcircuit() && kv.first.subcircuit().value() == subcktName) {
                    Parameter p = kv.second;
                    p.reset_value();
                    (*parametersPtr)[ParameterName(kv.first.name(), newScope)] = p;
                }
            }
            if (origScope != subcktName) {
                for (const auto& kv : existingParams) {
                    if (kv.first.subcircuit() && kv.first.subcircuit().value() == origScope) {
                        Parameter p = kv.second;
                        p.reset_value();
                        (*parametersPtr)[ParameterName(kv.first.name(), newScope)] = p;
                    }
                }
            }
            for (const auto& pkv : params) {
                const std::string cleanExpr = strip_outer_braces_str(pkv.second);
                Parameter         p;
                if (!callerScope.has_value()) {
                    const double v = parse_param(cleanExpr, *parametersPtr, std::nullopt, false);
                    if (!std::isnan(v)) {
                        p.set_expression(Misc::precise_to_string(v));
                        p.set_value(v);
                    } else {
                        p.set_expression(cleanExpr);
                    }
                } else {
                    p.set_expression(cleanExpr);
                }
                (*parametersPtr)[ParameterName(pkv.first, newScope)] = p;
            }
        }
    }

    // Loop through the identified subcircuit
    for (auto k = 0; k < subc.lines.size(); ++k) {
        // Set shorthand for long variable name
        tokens_t& tokens = subc.lines.at(k).first;
        if (createInstScope) {
            const std::string origScope = subc.lines.at(k).second.value_or(subcktName);
            subc.lines.at(k).second     = origScope + "|" + label;
        }
        // Check for value parameterization
        if (!params.empty() && params.count(tokens.front()) != 0) { insert_parameter(tokens, params); }
        // Append the label of the parent to the subcircuit label in-place
        tokens.at(0).reserve(tokens.at(0).size() + 1 + label.size());
        tokens.at(0).push_back('|');
        tokens.at(0).append(label);
        // Determine amount of nodes to process
        const char devType   = tokens.at(0).at(0);
        int64_t    nodeCount = (devType == 'E' || devType == 'F' || devType == 'G' || devType == 'H' || devType == 'T')
                                       ? 4
                                       : 2;
        if (devType == 'N') {
            nodeCount = 2;
            for (size_t m = 3; m < tokens.size() && nodeCount < 4; ++m) {
                if (Nanowire::is_node_token(tokens.at(m), *this, subc.lines.at(k).second)) {
                    ++nodeCount;
                } else {
                    break;
                }
            }
            for (size_t m = 1; m < tokens.size(); ++m) {
                for (const char* pfx : {"GATE=", "GATE_IND=", "GATEDEV="}) {
                    const size_t pfxLen = std::strlen(pfx);
                    if (tokens.at(m).compare(0, pfxLen, pfx) == 0 && tokens.at(m).size() > pfxLen) {
                        tokens.at(m).push_back('|');
                        tokens.at(m).append(label);
                    }
                }
            }
        }
        // Loop through all the nodes changing where necessary
        for (int64_t n = 1; n < nodeCount + 1; ++n) {
            // Ensure nodes don't identify as ground
            if (tokens.at(n) != "0" && tokens.at(n) != "GND") {
                // Ensure the node is not part of the IO
                if (!rename_io_nodes(tokens.at(n), subc.io, io)) {
                    // Append the label to the node number in-place to make it unique
                    tokens.at(n).reserve(tokens.at(n).size() + 1 + label.size());
                    tokens.at(n).push_back('|');
                    tokens.at(n).append(label);
                }
            }
        }
    }
}

void Netlist::insert_parameter(tokens_t& t, s_map& params) {
    std::string twonode  = "RCLK";
    std::string fournode = "EFHGT";
    if (twonode.find(t.front().front()) != std::string::npos) {
        t.at(3) = params.at(t.front());
    } else if (fournode.find(t.front().front()) != std::string::npos) {
        if (t.front().front() == 'T') {
            for (auto& i : t) {
                if (i.find("TD=") != std::string::npos) { i = "TD=" + params.at(t.front()); }
            }
        } else {
            t.at(5) = params.at(t.front());
        }
    } else if (t.front().front() == 'B' || t.front().front() == 'N') {
        for (auto& i : t) {
            if (i.find("AREA=") != std::string::npos) {
                i = "AREA=" + params.at(t.front());
            } else if (i.find("IC=") != std::string::npos) {
                i = "IC=" + params.at(t.front());
            } else if (i.find("ISW=") != std::string::npos) {
                i = "ISW=" + params.at(t.front());
            }
        }
    }
}

void Netlist::expand_subcircuits() {
    // std::vector<std::pair<tokens_t, string_o>> moddedLines;
    std::string subcktName, label;
    // Loop through subcircuits, line by line
    for (const auto& i : subcircuits) {
        for (const auto& j : i.second.lines) {
            // If a line is found that starts with an 'X'
            if (j.first.front().at(0) == 'X') {
                // This subcircuit contains a subcircuit
                subcircuits.at(i.first).containsSubckt = true;
                // Increase the expansion counter
                subcircuits.at(i.first).subcktCounter++;
                // Increase the nest depth
                nestedSubcktCount++;
            }
        }
    }
    ProgressBar bar;
    if (!argMin) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Expanding Subcircuits");
        bar.set_total((float) nestedSubcktCount);
    }
    // If not minimal printing
    int64_t cc = 0;
    // While we are nested (depth not zero)
    while (nestedSubcktCount != 0) {
        // If not minimal printing
        if (!argMin) {
            // Report progress
            bar.update((float) cc);
        }
        // Loop through subcircuits
        for (const auto& i : subcircuits) {
            if (sanityCheck && sanityCheckSubckts.count(i.first) != 0) {
                for (auto node : i.second.io) { increment_subcircuit_node_count(i.first, node); }
            }
            Subcircuit& subcircuit = subcircuits.at(i.first);
            if (!subcircuit.containsSubckt) { continue; }
            std::vector<std::pair<tokens_t, string_o>> newLines;
            newLines.reserve(subcircuit.lines.size() * 2);
            for (size_t j = 0; j < subcircuit.lines.size(); ++j) {
                tokens_t& subcCurrentLine = subcircuit.lines[j].first;
                // If the line denotes a subcircuit
                if (subcCurrentLine.front().at(0) == 'X') {
                    // Variable to store io and parameters
                    tokens_t io;
                    s_map    params;
                    id_io_subc_label(subcCurrentLine, io, params, subcktName, label, subcircuits);
                    if (sanityCheck && sanityCheckSubckts.count(i.first) != 0) {
                        for (auto node : io) { increment_subcircuit_node_count(i.first, node); }
                    }
                    const Subcircuit& targetSubc = subcircuits.at(subcktName);
                    // If not nested
                    if (!targetSubc.containsSubckt) {
                        Subcircuit subc = targetSubc;
                        // Expand the IO nodes of the subcircuit for this instance
                        expand_io(subc, params, io, label, subcktName, i.first);
                        // Append the expanded lines in O(1) amortized time
                        newLines.insert(newLines.end(),
                                        std::make_move_iterator(subc.lines.begin()),
                                        std::make_move_iterator(subc.lines.end()));
                        // Reduce the total amount of subcircuits in this subcircuit by 1
                        --subcircuit.subcktCounter;
                        ++cc;
                        // Reduce the total nested subcircuit count by 1
                        --nestedSubcktCount;
                        // If this subcircuit subcircuit counter becomes zero
                        if (subcircuit.subcktCounter == 0) {
                            // This subcircuit no longer contains subcircuits
                            subcircuit.containsSubckt = false;
                        }
                    } else {
                        newLines.push_back(std::move(subcircuit.lines[j]));
                    }
                } else {
                    newLines.push_back(std::move(subcircuit.lines[j]));
                }
            }
            subcircuit.lines = std::move(newLines);
        }
    }
    // Let the user know subcircuit expansion is complete
    if (!argMin) {
        bar.complete();
        std::cout << "\n";
    }
}

void Netlist::expand_maindesign() {
    // std::vector<std::pair<std::string, std::string>> moddedLines;
    ProgressBar bar;
    if (!argMin) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Expanding Main Circuit");
        bar.set_total((float) maindesign.size());
    }
    size_t totalExpLines = 0;
    for (const auto& line : maindesign) {
        if (!line.empty() && line.front().at(0) == 'X') {
            for (size_t k = 1; k < line.size(); ++k) {
                auto it = subcircuits.find(line[k]);
                if (it != subcircuits.end()) {
                    totalExpLines += it->second.lines.size();
                    break;
                }
            }
        } else {
            ++totalExpLines;
        }
    }
    expNetlist.reserve(totalExpLines);
    // Loop through the identified main design, line by line
    for (auto i = 0; i < maindesign.size(); ++i) {
        std::string subcktName, label;
        // If not minimal printing
        if (!argMin) {
            // Report progress
            bar.update((float) i);
        }
        // If the line denotes a subcircuit
        if (maindesign.at(i).front().at(0) == 'X') {
            // Variable to store io and parameters
            tokens_t io;
            s_map    params;
            id_io_subc_label(maindesign.at(i), io, params, subcktName, label, subcircuits);
            if (sanityCheck) {
                for (auto node : io) { increment_maindesign_node_count(node); }
            }
            // Copy of subcircuit for this instance
            Subcircuit subc = subcircuits.at(subcktName);
            // Expand the appropriate IO nodes of the subcircuit for this instance
            expand_io(subc, params, io, label, subcktName, std::nullopt);
            // Add the expanded subcircuit lines to the expanded netlist
            expNetlist.insert(expNetlist.end(),
                              std::make_move_iterator(subc.lines.begin()),
                              std::make_move_iterator(subc.lines.end()));
            // If the line is not a subcircuit
        } else {
            // Add the line tokens to the expanded netlist
            expNetlist.push_back(std::make_pair(maindesign.at(i), std::nullopt));
        }
    }
    if (hasInstScopes && parametersPtr != nullptr && !parametersPtr->empty()) {
        parse_parameters(*parametersPtr);
    }
    // Let the user know main design expansion is complete
    if (!argMin) {
        bar.complete();
        std::cout << "\n";
    }
    subcktTotal = subcircuits.size();
    subcircuits.clear();
}

void Netlist::increment_maindesign_node_count(std::string node) {
    if (node == "0" || node == "GND") { return; }
    if (mainNodeCounts.count(node) == 0) {
        mainNodeCounts[node] = 1;
    } else {
        mainNodeCounts.at(node)++;
    }
}

void Netlist::increment_subcircuit_node_count(std::string subcktName, std::string node) {
    if (node == "0" || node == "GND") { return; }
    if (subcktNodeCounts.count(subcktName) == 0) { subcktNodeCounts[subcktName]; }
    if (subcktNodeCounts.at(subcktName).count(node) == 0) {
        subcktNodeCounts.at(subcktName)[node] = 1;
    } else {
        subcktNodeCounts.at(subcktName).at(node)++;
    }
}

void sanity_check_node(std::string node, int32_t count, bool& is_clean) {
    if (count < 2) {
        is_clean = false;
        std::cout << "  - Node '" << node << "' looks either undefined or unused." << std::endl;
    } else if (count > 2) {
        is_clean = false;
        std::cout << "  - Node '" << node << "' looks overused (" << count << ")." << std::endl;
    }
}

void Netlist::sanity_check_maindesign() {
    std::cout << "main design:" << std::endl;
    bool is_clean = true;
    for (auto kv : mainNodeCounts) { sanity_check_node(kv.first, kv.second, is_clean); }
    if (is_clean) {
        std::cout << "  Everything looks fine! (Checked " << mainNodeCounts.size() << " nodes.)" << std::endl;
    }
}

void Netlist::sanity_check_subcircuits() {
    for (auto subcktName : sanityCheckSubckts) {
        if (subcktNodeCounts.count(subcktName) == 0) {
            std::cout << "Subcircuit '" << subcktName << "' does not exist. Skip." << std::endl;
            continue;
        }
        std::cout << "'" << subcktName << "':" << std::endl;
        bool is_clean = true;
        for (auto kv : subcktNodeCounts.at(subcktName)) { sanity_check_node(kv.first, kv.second, is_clean); }
        if (is_clean) {
            std::cout << "  Everything looks fine! (Checked " << subcktNodeCounts.at(subcktName).size() << " nodes.)"
                      << std::endl;
        }
    }
}

void Netlist::sanity_check(const Components& components) {
    if (!sanityCheck) { return; }

    // Update node counts in main design
    for (const auto& device : components.devices) {
        std::visit(
                [&](const auto& dev) {
                    // Only consider nodes in maindesign (top)
                    if (dev.netlistInfo.label_.find('|') == std::string::npos) {
                        for (const auto& node : dev.nodes) { increment_maindesign_node_count(node); }
                    }
                },
                device);
    }
    for (const auto& source : components.currentsources) {
        // If device belongs to a subcircuit
        if (source.netlistInfo.label_.find("|") != std::string::npos) { continue; }
        // Only consider nodes in maindesign (top)
        for (const auto& node : source.nodes) { increment_maindesign_node_count(node); }
    }

    std::cout << "\nSanity check results:" << std::endl;
    sanity_check_maindesign();
    sanity_check_subcircuits();
    std::cout << std::endl;
}
