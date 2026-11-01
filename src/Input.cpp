// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Input.hpp"

#include "JoSIM/ProgressBar.hpp"
#include "JoSIM/Rng.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <thread>

using namespace JoSIM;

std::vector<tokens_t> Input::read_input(LineInput& input, string_o fileName) {
    // When the standard input is selected, "fileName" will be in the nullopt
    // state.
    if (fileName != std::nullopt) {
        if (std::filesystem::path(fileName.value()).has_parent_path()) {
            fileParentPath = std::filesystem::path(fileName.value()).parent_path().string();
        }
    }
    // Variable to store the read line
    std::string           line;
    // Variable to store all the read lines
    std::vector<tokens_t> fileLines;
    // Variable to store the tokenized line
    tokens_t              tokens;
    // Do this until the end of file is reached.
    while (input.next()) {
        // Create a mutable line
        line = input.line();
        // Remove all Carriage Return characters
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        // Remove all the leading, trailing and extra white spaces
        Misc::collapse_spaces(line);
        if (!line.empty() && line.at(0) != '*' && line.at(0) != '#') {
            // Tokenize the line
            tokens = Misc::tokenize(line);
            // Uppercase just the first token for ".INCLUDE" and ".FILE" check
            std::transform(tokens.begin(), tokens.begin() + 1, tokens.begin(), [](std::string& c) -> std::string {
                std::transform(c.begin(), c.end(), c.begin(), toupper);
                return c;
            });
            // If the line contains a "INCLUDE" statement
            if (tokens.at(0) == ".INCLUDE" || tokens.at(0) == "INCLUDE") {
                // Variable to store path to file to include
                std::string includeFile;
                // If reading from a file
                if (fileName) {
                    // Sanity check to prevent cyclic includes
                    if (std::filesystem::path(fileName.value()).c_str()
                        == std::filesystem::path(fileName.value()).parent_path().append(tokens.at(1)).c_str()) {
                        Errors::input_errors(InputErrors::CYCLIC_INCLUDE, fileName);
                    }
                    includeFile = std::filesystem::path(fileName.value()).parent_path().append(tokens.at(1)).string();
                } else {
                    includeFile = std::filesystem::current_path().append(tokens.at(1)).string();
                }
                // Create a new LineInput variable for the included file
                FileInput             file(includeFile);
                // Attempt to read the contents of the included file
                std::vector<tokens_t> tempInclude = Input::read_input(file, includeFile);
                // Insert the lines from the included file in place
                fileLines.insert(fileLines.end(), tempInclude.begin(), tempInclude.end());
                // If the line contains a "FILE" statement
            } else if (tokens.at(0) == ".FILE" || tokens.at(0) == "FILE") {
                // Ensure there is a second token
                if (tokens.size() < 2) {
                    Errors::control_errors(ControlErrors::INVALID_FILE_COMMAND, line);
                } else {
                    // Sanity check, if parent path of output file is empty then
                    // change path to input file path, otherwise file is written
                    // in executable location
                    auto path = std::filesystem::path(tokens.at(1));
                    if (!path.has_parent_path() && fileParentPath) {
                        path = std::filesystem::path(fileParentPath.value()).append(tokens.at(1));
                    }
                    output_files.emplace_back(OutputFile(path.string()));
                }
                fileLines.emplace_back(tokens);
                // Preserve file path case for state checkpoint and event verification commands
            } else if (tokens.at(0) == ".SAVESTATE" || tokens.at(0) == "SAVESTATE" || tokens.at(0) == ".SAVEBIAS"
                       || tokens.at(0) == "SAVEBIAS" || tokens.at(0) == ".LOADSTATE" || tokens.at(0) == "LOADSTATE"
                       || tokens.at(0) == ".LOADBIAS" || tokens.at(0) == "LOADBIAS"
                       || tokens.at(0) == ".VERIFY_EVENTS" || tokens.at(0) == "VERIFY_EVENTS"
                       || tokens.at(0) == ".VERIFYEVENTS" || tokens.at(0) == "VERIFYEVENTS") {
                if (tokens.size() > 2) {
                    std::transform(tokens.begin() + 2, tokens.end(), tokens.begin() + 2, [](std::string& c) -> std::string {
                        std::transform(c.begin(), c.end(), c.begin(), toupper);
                        return c;
                    });
                }
                fileLines.emplace_back(tokens);
                // If the line contains a "END" statement
            } else if (tokens.at(0) == ".END" || tokens.at(0) == "END") {
                break;
                // If the line does not contain an "INCLUDE", "FILE" or "END" statement
            } else {
                // Transform the entire line to upper case (preserving file paths after FILE=, MARGINFILE=, VERIFYEVENTS=, EVENTS=, EVENTSONLY=, OUT=, JITTERFILE=)
                const bool isIvOrSquid = (tokens.at(0) == ".IV" || tokens.at(0) == "IV" || tokens.at(0) == ".SQUID"
                                          || tokens.at(0) == "SQUID");
                std::transform(tokens.begin() + 1, tokens.end(), tokens.begin() + 1, [isIvOrSquid](std::string& c) -> std::string {
                    const size_t eqPos = c.find('=');
                    if (eqPos != std::string::npos) {
                        std::string key = c.substr(0, eqPos);
                        std::transform(key.begin(), key.end(), key.begin(), toupper);
                        if (key == "FILE" || key == "OUT" || key == "MARGINFILE" || key == "MARGINSFILE"
                            || key == "MEASFILE" || key == "MEASUREFILE" || key == "LOOPFILE" || key == "LOOPSFILE"
                            || key == "JITTERFILE" || key == "VERIFYEVENTS" || key == "VERIFY_EVENTS"
                            || key == "EVENTS" || key == "EVENTSONLY") {
                            return key + "=" + c.substr(eqPos + 1);
                        }
                    } else if (isIvOrSquid) {
                        std::string cl = c;
                        std::transform(cl.begin(), cl.end(), cl.begin(), ::tolower);
                        if (cl.find('/') != std::string::npos || cl.find('\\') != std::string::npos
                            || (cl.size() > 4
                                && (cl.substr(cl.size() - 4) == ".csv" || cl.substr(cl.size() - 4) == ".dat"
                                    || cl.substr(cl.size() - 4) == ".txt"))
                            || (cl.size() > 5 && cl.substr(cl.size() - 5) == ".json")) {
                            return c;
                        }
                    }
                    std::transform(c.begin(), c.end(), c.begin(), toupper);
                    return c;
                });
                // If the line starts with a '+' append it to the previous line.
                if (tokens.at(0).at(0) == '+') {
                    // Remove the '+'
                    tokens.at(0) = tokens.at(0).substr(1);
                    if (tokens.at(0).empty()) {
                        fileLines.back().insert(fileLines.back().end(), tokens.begin() + 1, tokens.end());
                    } else {
                        fileLines.back().insert(fileLines.back().end(), tokens.begin(), tokens.end());
                    }
                    // Add the line to the read in lines variable
                } else {
                    fileLines.emplace_back(tokens);
                }
            }
        }
    }
    // If the file was read in but has no contents report an error
    if (fileLines.empty()) {
        Errors::input_errors(InputErrors::EMPTY_FILE, fileName);
        // Return the read in file lines.
    } else {
        return fileLines;
    }
    // Return to satisfy return type. This should never be reached.
    return {};
}

void Input::parse_input(string_o fileName) {
    std::vector<tokens_t> fileLines;
    if (fileName) {
        FileInput input(fileName.value());
        fileLines = Input::read_input(input, fileName);
    } else {
        ConsoleInput input;
        fileLines = Input::read_input(input);
    }
    parse_tokens_lines(fileLines);
}

void Input::parse_input_string(const std::string& content, string_o baseDir) {
    if (baseDir.has_value() && !baseDir->empty()) { fileParentPath = baseDir; }
    StringInput           input(content);
    std::vector<tokens_t> fileLines = Input::read_input(input, std::nullopt);
    parse_tokens_lines(fileLines);
}

void Input::parse_tokens_lines(std::vector<tokens_t>& fileLines) {
    std::thread printingThread;
    string_o    subckt;
    bool        control = false;
    ProgressBar bar;
    if (!argMin) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Parsing Input");
        bar.set_total((float) fileLines.size());
    }
    // If not minimal printing
    netlist.argMin        = argMin;
    netlist.parametersPtr = &parameters;
    // Loop through all the tokenized input
    for (auto i = 0; i < fileLines.size(); ++i) {
        // If not minimal printing
        if (!argMin) {
            // Report progress
            bar.update((float) i);
        }
        // Determine if the line is a control (subcircuit, analysis, print, etc.)
        if (fileLines.at(i).front().at(0) == '.') {
            // If the line is the begining of a subcircuit
            if (::strcmp(fileLines.at(i).front().c_str(), ".SUBCKT") == 0) {
                // Indicate that subcircuits exist within this netlist
                netlist.containsSubckt = true;
                // Subcircuit is required to atleast have a name
                if (fileLines.at(i).size() > 1) {
                    // Subcircuit is required to have at least some form of IO
                    if (fileLines.at(i).size() > 2) {
                        // Set the subcircuit name the second token
                        subckt = fileLines.at(i).at(1);
                        if (netlist.subcircuits.count(subckt.value()) != 0) {
                            Errors::input_errors(InputErrors::DUPLICATE_SUBCIRCUIT, subckt.value());
                            netlist.subcircuits[subckt.value()].io.clear();
                            netlist.subcircuits[subckt.value()].lines.clear();
                        }
                        // Separate IO nodes from default parameter assignments (PARAMS: key=val ...)
                        tokens_t subParamTokens = {".PARAM"};
                        bool     inSubParams    = false;
                        for (size_t j = 2; j < fileLines.at(i).size(); ++j) {
                            const std::string& tok = fileLines.at(i).at(j);
                            if (tok == "PARAMS:" || tok == "PARAM:") {
                                inSubParams = true;
                                continue;
                            }
                            if (inSubParams || tok.find('=') != std::string::npos
                                || (j + 1 < fileLines.at(i).size() && fileLines.at(i).at(j + 1).front() == '=')) {
                                inSubParams = true;
                                subParamTokens.push_back(tok);
                            } else {
                                netlist.subcircuits[subckt.value()].io.push_back(tok);
                            }
                        }
                        if (netlist.subcircuits[subckt.value()].io.empty()) {
                            Errors::input_errors(InputErrors::MISSING_SUBCKT_IO);
                        }
                        if (subParamTokens.size() > 1) {
                            create_parameter(subParamTokens, parameters, subckt);
                        }
                        // Complain if no IO is found
                    } else {
                        Errors::input_errors(InputErrors::MISSING_SUBCKT_IO);
                    }
                    // Complain if no subcircuit name is found
                } else {
                    Errors::input_errors(InputErrors::MISSING_SUBCKT_NAME);
                }
                // If parameter, create a parameter
            } else if (::strcmp(fileLines.at(i).front().c_str(), ".PARAM") == 0) {
                create_parameter(fileLines.at(i), parameters, subckt);
                // If control, set flag as start of controls
            } else if (::strcmp(fileLines.at(i).front().c_str(), ".CONTROL") == 0) {
                control = true;
                // End subcircuit, set flag
            } else if (::strcmp(fileLines.at(i).front().c_str(), ".ENDS") == 0) {
                subckt = std::nullopt;
                // End control section, set flag
            } else if (::strcmp(fileLines.at(i).front().c_str(), ".ENDC") == 0) {
                control = false;
                // If model, add model to models list
            } else if (::strcmp(fileLines.at(i).front().c_str(), ".MODEL") == 0) {
                netlist.models[std::make_pair(fileLines.at(i).at(1), subckt)] = fileLines.at(i);
                // If neither of these, normal control, add to controls list
            } else {
                // Controls cannot exist within a subcircuit
                if (!subckt) {
                    // Remove the preceding '.' from the first token
                    fileLines.at(i).front() = fileLines.at(i).front().substr(1);
                    // Store the control
                    controls.push_back(fileLines.at(i));
                    // Complain that controls exist in subcircuit
                } else {
                    Errors::input_errors(InputErrors::SUBCKT_CONTROLS, subckt.value());
                }
            }
            // If control section flag set
        } else if (control) {
            // If parameter, add to unparsed parameter list
            if (::strcmp(fileLines.at(i).front().c_str(), "PARAM") == 0) {
                create_parameter(fileLines.at(i), parameters, subckt);
                // If model, add to models list
            } else if (::strcmp(fileLines.at(i).front().c_str(), "MODEL") == 0) {
                netlist.models[std::make_pair(fileLines.at(i).at(1), subckt)] = fileLines.at(i);
                // If neither, add to controls list
            } else {
                if (!subckt) {
                    controls.push_back(fileLines.at(i));
                } else {
                    Errors::input_errors(InputErrors::SUBCKT_CONTROLS, subckt.value());
                }
            }
            // If not a control, normal device line
        } else {
            // If subcircuit flag, add line to relevant subcircuit
            if (subckt) {
                netlist.subcircuits[subckt.value()].lines.push_back(std::make_pair(fileLines.at(i), subckt));
                // If not, add line to main design
            } else {
                netlist.maindesign.push_back(fileLines.at(i));
            }
        }
    }
    // Append CLI --iv and --squid specs to controls
    auto append_cli_ctrl = [&](const std::string& cmd, const std::vector<std::string>& specs) {
        for (const auto& s : specs) {
            tokens_t t = Misc::tokenize(s);
            if (t.empty()) { continue; }
            for (auto& tok : t) {
                const size_t eqPos = tok.find('=');
                if (eqPos != std::string::npos) {
                    std::string key = tok.substr(0, eqPos);
                    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
                    if (key == "FILE" || key == "OUT") {
                        tok = key + "=" + tok.substr(eqPos + 1);
                        continue;
                    }
                } else {
                    std::string cl = tok;
                    std::transform(cl.begin(), cl.end(), cl.begin(), ::tolower);
                    if (cl.find('/') != std::string::npos || cl.find('\\') != std::string::npos
                        || (cl.size() > 4 && (cl.substr(cl.size() - 4) == ".csv" || cl.substr(cl.size() - 4) == ".dat"))
                        || (cl.size() > 5 && cl.substr(cl.size() - 5) == ".json")) {
                        continue;
                    }
                }
                std::transform(tok.begin(), tok.end(), tok.begin(), ::toupper);
            }
            if (t.front() != cmd && t.front() != "." + cmd) { t.insert(t.begin(), cmd); }
            else if (t.front().front() == '.') { t.front() = t.front().substr(1); }
            controls.push_back(std::move(t));
        }
    };
    append_cli_ctrl("IV", cliIvSpecs);
    append_cli_ctrl("SQUID", cliSquidSpecs);

    // Sanity check all conrols for syntax and functionality
    syntax_check_controls(controls);
    if (!parameters.empty()) { parse_parameters(parameters); }
    parse_options_and_state_controls();
    // Seed deterministic RNG streams (noise + spread) once per parsed run.
    if (auto seed = find_seed_option(); seed.has_value()) {
        Rng::start_run(*seed);
    } else {
        Rng::start_run_auto();
    }
    // Let the user know the input reading is complete
    if (!argMin) {
        bar.complete();
        std::cout << "\n";
    }
    if (!argMin && argVerb) { std::cout << "RNG seed: " << Rng::base_seed() << "\n"; }
    // If main is empty, complain
    if (netlist.maindesign.empty()) { Errors::input_errors(InputErrors::MISSING_MAIN); }
}

void Input::syntax_check_controls(std::vector<tokens_t>& controls) {
    // This will simply check controls, complaining if any are not allowed
    std::vector<std::string> v = {"PRINT",
                                  "TRAN",
                                  "SAVE",
                                  "PLOT",
                                  "END",
                                  "TEMP",
                                  "NEB",
                                  "SPREAD",
                                  "SWEEP",
                                  "MARGINS",
                                  "MARGIN",
                                  "VERIFY_EVENTS",
                                  "VERIFYEVENTS",
                                  "ASSERT",
                                  "VEC",
                                  "MEAS",
                                  "MEASURE",
                                  "IC",
                                  "CHECK_LOOPS",
                                  "CHECKLOOPS",
                                  "JITTER",
                                  "BER",
                                  "FILE",
                                  "IV",
                                  "SQUID",
                                  "SENS",
                                  "SENSITIVITY",
                                  "OPTION",
                                  "SAVESTATE",
                                  "LOADSTATE",
                                  "SAVEBIAS",
                                  "LOADBIAS"};
    for (auto i : controls) {
        if (std::find(v.begin(), v.end(), i.at(0)) == v.end()) {
            Errors::input_errors(InputErrors::UNKNOWN_CONTROL, i.at(0));
        }
    }
}

uint32_t Input::parse_event_metrics(const std::string& spec, uint32_t currentMask) {
    uint32_t    mask = currentMask;
    std::string s    = spec;
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t next = s.find_first_of(",+;:", pos);
        if (next == std::string::npos) { next = s.size(); }
        std::string item = s.substr(pos, next - pos);
        Misc::ltrim(item);
        Misc::rtrim(item);
        if (!item.empty()) {
            if (item == "0" || item == "FALSE" || item == "NONE") {
                mask = EVENT_METRIC_NONE;
            } else if (item == "1" || item == "TRUE" || item == "PULSE" || item == "DEFAULT") {
                mask |= EVENT_METRIC_PULSE;
            } else if (item == "ALL" || item == "FULL") {
                mask |= EVENT_METRIC_ALL;
            } else if (item == "T_PEAK" || item == "TPEAK") {
                mask |= EVENT_METRIC_T_PEAK;
            } else if (item == "V_PEAK" || item == "VPEAK" || item == "PEAK" || item == "HEIGHT") {
                mask |= EVENT_METRIC_V_PEAK;
            } else if (item == "FWHM" || item == "WIDTH") {
                mask |= EVENT_METRIC_FWHM;
            } else if (item == "DT_PI" || item == "DTPI") {
                mask |= EVENT_METRIC_DT_PI;
            } else if (item == "AREA" || item == "FLUX") {
                mask |= EVENT_METRIC_AREA;
            } else if (item == "TIME=PEAK" || item == "TIME=T_PEAK" || item == "TIME=TPEAK" || item == "TIME=V_PEAK"
                       || item == "TIME=VPEAK" || item == "TIMEPEAK") {
                mask |= EVENT_METRIC_TIME_PEAK;
            } else if (item == "TIME=CROSS" || item == "TIME=TCROSS" || item == "TIMECROSS") {
                mask &= ~static_cast<uint32_t>(EVENT_METRIC_TIME_PEAK);
            }
        }
        pos = next + 1;
    }
    return mask;
}

void Input::parse_sweep_tokens(const tokens_t& c) {
    if (c.size() < 3) { return; }
    tokens_t t = c;
    for (auto& s : t) {
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    }

    SweepDim dim;
    dim.target     = t.at(1);
    size_t modeIdx = 2;
    while (modeIdx < t.size()
           && ((!dim.target.empty() && dim.target.back() == ',') || (!t.at(modeIdx).empty() && t.at(modeIdx).front() == ','))) {
        dim.target += t.at(modeIdx);
        ++modeIdx;
    }
    if (modeIdx >= t.size()) { return; }

    const bool isMultiOrGroup
            = (!dim.target.empty() && (dim.target.front() == '@' || dim.target.find(',') != std::string::npos));
    size_t  startIdx  = modeIdx;
    int64_t explicitN = 0;
    if (t.at(modeIdx) == "MARGIN" || t.at(modeIdx) == "REL" || t.at(modeIdx) == "SCALE" || t.at(modeIdx) == "MULT") {
        dim.mode       = SweepMode::Margin;
        dim.isRelative = true;
        startIdx       = modeIdx + 1;
    } else if (t.at(modeIdx) == "LIST") {
        dim.mode       = SweepMode::List;
        dim.isRelative = isMultiOrGroup;
        startIdx       = modeIdx + 1;
    } else if (t.at(modeIdx) == "LIN") {
        dim.mode       = SweepMode::Linear;
        dim.isRelative = isMultiOrGroup;
        startIdx       = modeIdx + 1;
    } else {
        dim.mode       = SweepMode::Linear;
        dim.isRelative = isMultiOrGroup;
        startIdx       = modeIdx;
    }

    std::vector<double> numArgs;
    for (size_t k = startIdx; k < t.size(); ++k) {
        const std::string& tok = t.at(k);
        if (tok == "REL" || tok == "MARGIN" || tok == "SCALE" || tok == "MULT") {
            dim.isRelative = true;
            continue;
        }
        if (tok == "ABS") {
            dim.isRelative = false;
            continue;
        }
        if (tok.rfind("N=", 0) == 0) {
            explicitN = std::max<int64_t>(2, static_cast<int64_t>(std::llround(parse_param(tok.substr(2), parameters))));
            continue;
        }
        if (tok.rfind("COUNT=", 0) == 0) {
            explicitN = std::max<int64_t>(2, static_cast<int64_t>(std::llround(parse_param(tok.substr(6), parameters))));
            continue;
        }
        if (tok.rfind("POINTS=", 0) == 0) {
            explicitN = std::max<int64_t>(2, static_cast<int64_t>(std::llround(parse_param(tok.substr(7), parameters))));
            continue;
        }
        if (tok.rfind("TOL=", 0) == 0 || tok.rfind("MARGINTOL=", 0) == 0) {
            const double tv = parse_param(tok.substr(tok.find('=') + 1), parameters);
            if (std::isfinite(tv) && tv >= 0.0) { argMarginTol = tv; }
            continue;
        }
        double v = parse_param(tok, parameters);
        if (std::isfinite(v)) { numArgs.push_back(v); }
    }

    auto push_linspace = [&](double a, double b, int64_t nPts) {
        nPts = std::max<int64_t>(2, nPts);
        dim.values.reserve(static_cast<size_t>(nPts));
        for (int64_t idx = 0; idx < nPts; ++idx) {
            dim.values.push_back(a + (b - a) * static_cast<double>(idx) / static_cast<double>(nPts - 1));
        }
    };
    auto push_steps = [&](double a, double b, double st) {
        if (std::abs(st) <= 0.0) {
            dim.values.push_back(a);
            return;
        }
        if ((b >= a && st < 0.0) || (b < a && st > 0.0)) { st = -st; }
        const double tol = 1e-9 * std::abs(st);
        for (int64_t idx = 0; idx < 100000; ++idx) {
            const double v = a + static_cast<double>(idx) * st;
            if ((st > 0.0 && v > b + tol) || (st < 0.0 && v < b - tol)) { break; }
            dim.values.push_back(v);
        }
    };

    if (dim.mode == SweepMode::List) {
        dim.values = std::move(numArgs);
    } else if (dim.mode == SweepMode::Margin) {
        if (numArgs.size() == 1) {
            const double  d    = std::abs(numArgs[0]);
            const int64_t nPts = (explicitN > 0) ? explicitN : (argMcRuns > 1 ? argMcRuns : 11);
            push_linspace(1.0 - d, 1.0 + d, nPts);
        } else if (numArgs.size() == 2) {
            if (explicitN == 0 && numArgs[1] >= 2.0 && std::floor(numArgs[1]) == numArgs[1]
                && std::abs(numArgs[0]) < 1.0) {
                const double  d    = std::abs(numArgs[0]);
                const int64_t nPts = static_cast<int64_t>(std::llround(numArgs[1]));
                push_linspace(1.0 - d, 1.0 + d, nPts);
            } else {
                double minVal = numArgs[0];
                double maxVal = numArgs[1];
                if (minVal < 0.0 && maxVal > 0.0 && std::abs(minVal) < 1.0 && maxVal < 1.0) {
                    minVal += 1.0;
                    maxVal += 1.0;
                }
                const int64_t nPts = (explicitN > 0) ? explicitN : (argMcRuns > 1 ? argMcRuns : 11);
                push_linspace(minVal, maxVal, nPts);
            }
        } else if (numArgs.size() >= 3) {
            double minVal = numArgs[0];
            double maxVal = numArgs[1];
            if (minVal < 0.0 && maxVal > 0.0 && std::abs(minVal) < 1.0 && maxVal < 1.0) {
                minVal += 1.0;
                maxVal += 1.0;
            }
            const double third = numArgs[2];
            if (explicitN > 0) {
                push_linspace(minVal, maxVal, explicitN);
            } else if (third >= 2.0 && std::floor(third) == third) {
                push_linspace(minVal, maxVal, static_cast<int64_t>(std::llround(third)));
            } else {
                push_steps(minVal, maxVal, third);
            }
        }
    } else if (t.at(2) == "LIN") {
        if (numArgs.size() >= 3) {
            const double a = numArgs[0], b = numArgs[1], cVal = numArgs[2];
            if (a >= 2.0 && std::floor(a) == a && !(cVal >= 2.0 && std::floor(cVal) == cVal && a < b)) {
                push_linspace(b, cVal, static_cast<int64_t>(std::llround(a)));
            } else {
                push_linspace(a, b, static_cast<int64_t>(std::llround(cVal)));
            }
        } else if (numArgs.size() == 2) {
            const int64_t nPts = (explicitN > 0) ? explicitN : (argMcRuns > 1 ? argMcRuns : 11);
            push_linspace(numArgs[0], numArgs[1], nPts);
        }
    } else {
        if (numArgs.size() >= 3) {
            const double a = numArgs[0], b = numArgs[1], st = numArgs[2];
            if (explicitN > 0) {
                push_linspace(a, b, explicitN);
            } else if (std::abs(st) > std::abs(b - a) && st >= 2.0 && std::floor(st) == st) {
                push_linspace(a, b, static_cast<int64_t>(std::llround(st)));
            } else {
                push_steps(a, b, st);
            }
        } else if (numArgs.size() == 2) {
            const int64_t nPts = (explicitN > 0) ? explicitN : (argMcRuns > 1 ? argMcRuns : 11);
            push_linspace(numArgs[0], numArgs[1], nPts);
        } else if (numArgs.size() == 1) {
            dim.values.push_back(numArgs[0]);
        }
    }

    if (dim.values.empty()) { return; }
    for (auto& existing : sweeps) {
        if (existing.target == dim.target) {
            existing = std::move(dim);
            argSweep = true;
            return;
        }
    }
    if (sweeps.size() < 2) {
        sweeps.push_back(std::move(dim));
        argSweep = true;
    }
}

void Input::parse_sweep_spec(const std::string& spec) {
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        char& ch = cleaned[idx];
        if (ch == ':' || ch == ';') {
            ch = ' ';
        } else if (ch == ',') {
            size_t next = idx + 1;
            while (next < cleaned.size() && cleaned[next] == ' ') { ++next; }
            bool isSep = false;
            if (next < cleaned.size()) {
                const char nc = cleaned[next];
                if ((nc >= '0' && nc <= '9') || nc == '+' || nc == '-' || nc == '.') {
                    isSep = true;
                } else {
                    std::string rem = cleaned.substr(next, 7);
                    std::transform(rem.begin(), rem.end(), rem.begin(), ::toupper);
                    if (rem.rfind("LIST", 0) == 0 || rem.rfind("LIN", 0) == 0 || rem.rfind("MARGIN", 0) == 0
                        || rem.rfind("REL", 0) == 0 || rem.rfind("SCALE", 0) == 0 || rem.rfind("MULT", 0) == 0
                        || rem.rfind("ABS", 0) == 0 || rem.rfind("N=", 0) == 0 || rem.rfind("COUNT=", 0) == 0
                        || rem.rfind("POINTS=", 0) == 0) {
                        isSep = true;
                    }
                }
            }
            if (isSep) { ch = ' '; }
        }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if ( firstUpper != "SWEEP") {
        toks.insert(toks.begin(), "SWEEP");
    }
    parse_sweep_tokens(toks);
}

void Input::parse_margin_tokens(const tokens_t& c) {
    argMargins = true;
    MarginSpec spec;
    std::vector<double> numArgs;
    bool hasRange = false;
    bool hasMin   = false;
    bool hasMax   = false;
    bool hasN     = false;

    auto resolve_path = [&](const std::string& raw) -> std::string {
        auto p = std::filesystem::path(raw);
        if (!p.has_parent_path() && fileParentPath) {
            p = std::filesystem::path(fileParentPath.value()).append(raw);
        }
        return p.string();
    };

    for (size_t k = 1; k < c.size(); ++k) {
        std::string tok = c.at(k);
        if (tok.empty()) { continue; }
        const size_t eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            std::string key = tok.substr(0, eqPos);
            std::string val = tok.substr(eqPos + 1);
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);
            if (key == "FILE" || key == "OUT" || key == "MARGINFILE" || key == "MARGINSFILE") {
                if (!val.empty()) { marginFile = resolve_path(val); }
                continue;
            }
            if (key == "VERIFY" || key == "VERIFYEVENTS" || key == "VERIFY_EVENTS") {
                if (!val.empty()) { verifyEventsFile = resolve_path(val); }
                continue;
            }
            std::transform(val.begin(), val.end(), val.begin(), ::toupper);
            if (key == "RANGE" || key == "DELTA" || key == "SPAN") {
                const double d = std::abs(parse_param(val, parameters));
                if (std::isfinite(d) && d > 0.0) {
                    spec.minVal = std::max(0.0, 1.0 - d);
                    spec.maxVal = 1.0 + d;
                    hasRange    = true;
                }
                continue;
            }
            if (key == "MIN" || key == "START" || key == "LO" || key == "LOW") {
                const double v = parse_param(val, parameters);
                if (std::isfinite(v)) {
                    spec.minVal = v;
                    hasMin      = true;
                }
                continue;
            }
            if (key == "MAX" || key == "STOP" || key == "HI" || key == "HIGH") {
                const double v = parse_param(val, parameters);
                if (std::isfinite(v)) {
                    spec.maxVal = v;
                    hasMax      = true;
                }
                continue;
            }
            if (key == "STEP" || key == "D" || key == "DX") {
                const double v = std::abs(parse_param(val, parameters));
                if (std::isfinite(v) && v > 0.0) {
                    spec.stepVal = v;
                    spec.hasStep = true;
                }
                continue;
            }
            if (key == "N" || key == "NPTS" || key == "PTS" || key == "POINTS" || key == "COUNT") {
                const int64_t n = std::max<int64_t>(3, static_cast<int64_t>(std::llround(parse_param(val, parameters))));
                spec.numPts     = n;
                hasN            = true;
                continue;
            }
            if (key == "TOL" || key == "MARGINTOL") {
                const double tv = std::abs(parse_param(val, parameters));
                if (std::isfinite(tv) && tv >= 0.0) {
                    spec.tol     = tv;
                    argMarginTol = tv;
                }
                continue;
            }
        }

        std::string tokUpper = tok;
        std::transform(tokUpper.begin(), tokUpper.end(), tokUpper.begin(), ::toupper);
        if (tokUpper == "MARGIN" || tokUpper == "MARGINS" || tokUpper == "REL" || tokUpper == "SCALE") { continue; }

        // Check if tok is a pure numeric literal (e.g. 0.4, 21, 1.5, 50P)
        const char fc = tokUpper.front();
        if ((fc >= '0' && fc <= '9') || fc == '+' || fc == '-' || fc == '.') {
            const double v = parse_param(tokUpper, parameters);
            if (std::isfinite(v)) {
                numArgs.push_back(v);
                continue;
            }
        }

        // Otherwise split by ',' into target names
        size_t pos = 0;
        while (pos < tokUpper.size()) {
            size_t next = tokUpper.find(',', pos);
            if (next == std::string::npos) { next = tokUpper.size(); }
            if (next > pos) {
                spec.targets.push_back(tokUpper.substr(pos, next - pos));
            }
            pos = next + 1;
        }
    }

    if (!hasRange && !hasMin && !hasMax && !numArgs.empty()) {
        if (numArgs.size() == 1) {
            const double d = std::abs(numArgs[0]);
            if (d > 0.0 && d < 1.0) {
                spec.minVal = std::max(0.0, 1.0 - d);
                spec.maxVal = 1.0 + d;
            } else if (d >= 3.0 && std::floor(d) == d) {
                spec.numPts = static_cast<int64_t>(std::llround(d));
            }
        } else if (numArgs.size() == 2) {
            if (!hasN && numArgs[1] >= 3.0 && std::floor(numArgs[1]) == numArgs[1] && std::abs(numArgs[0]) < 1.0) {
                const double d = std::abs(numArgs[0]);
                spec.minVal    = std::max(0.0, 1.0 - d);
                spec.maxVal    = 1.0 + d;
                spec.numPts    = static_cast<int64_t>(std::llround(numArgs[1]));
            } else {
                double a = numArgs[0], b = numArgs[1];
                if (a < 0.0 && b > 0.0 && std::abs(a) < 1.0 && b < 1.0) {
                    a += 1.0;
                    b += 1.0;
                }
                spec.minVal = a;
                spec.maxVal = b;
            }
        } else if (numArgs.size() >= 3) {
            double a = numArgs[0], b = numArgs[1], cVal = numArgs[2];
            if (a < 0.0 && b > 0.0 && std::abs(a) < 1.0 && b < 1.0) {
                a += 1.0;
                b += 1.0;
            }
            spec.minVal = a;
            spec.maxVal = b;
            if (!hasN && cVal >= 3.0 && std::floor(cVal) == cVal) {
                spec.numPts = static_cast<int64_t>(std::llround(cVal));
            } else if (cVal > 0.0) {
                spec.stepVal = cVal;
                spec.hasStep = true;
            }
            if (numArgs.size() >= 4 && numArgs[3] > 0.0 && numArgs[3] < 1e-6) {
                spec.tol     = numArgs[3];
                argMarginTol = numArgs[3];
            }
        }
    }

    if (spec.minVal < 0.0 && spec.maxVal > 0.0 && std::abs(spec.minVal) < 1.0 && spec.maxVal < 1.0) {
        spec.minVal += 1.0;
        spec.maxVal += 1.0;
    }
    if (spec.targets.empty()) { spec.targets.push_back("@DEFAULT"); }
    marginSpecs.push_back(std::move(spec));
}

void Input::parse_margin_spec(const std::string& spec) {
    argMargins = true;
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        if (cleaned[idx] == ':' || cleaned[idx] == ';') { cleaned[idx] = ' '; }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "MARGINS" && firstUpper != "MARGIN") {
        toks.insert(toks.begin(), "MARGINS");
    }
    parse_margin_tokens(toks);
}

void Input::parse_assert_tokens(const tokens_t& c) {
    if (c.size() < 2) { return; }

    auto decode_assert_bits = [](std::string s) -> std::string {
        for (const char* pfx : {"PATTERN=", "PAT=", "BITS=", "VEC="}) {
            if (s.rfind(pfx, 0) == 0) {
                s = s.substr(std::strlen(pfx));
                break;
            }
        }
        std::string out;
        if (s.rfind("0X", 0) == 0 || s.rfind("HEX:", 0) == 0) {
            const size_t start = (s.rfind("0X", 0) == 0) ? 2 : 4;
            for (size_t i = start; i < s.size(); ++i) {
                const char ch = s[i];
                if (ch == '_') { continue; }
                int nib = -1;
                if (ch >= '0' && ch <= '9') {
                    nib = ch - '0';
                } else if (ch >= 'A' && ch <= 'F') {
                    nib = 10 + (ch - 'A');
                } else {
                    return "";
                }
                out.push_back(((nib >> 3) & 1) ? '1' : '0');
                out.push_back(((nib >> 2) & 1) ? '1' : '0');
                out.push_back(((nib >> 1) & 1) ? '1' : '0');
                out.push_back((nib & 1) ? '1' : '0');
            }
            return out;
        }
        if (s.size() >= 2 && s[0] == 'B') { s = s.substr(1); }
        for (char ch : s) {
            if (ch == '_') {
                continue;
            } else if (ch == '1' || ch == '+' || ch == 'H') {
                out.push_back('1');
            } else if (ch == '0' || ch == 'L') {
                out.push_back('0');
            } else if (ch == '-' || ch == 'N') {
                out.push_back('-');
            } else if (ch == 'X' || ch == '?' || ch == '.') {
                out.push_back('X');
            } else {
                return "";
            }
        }
        return out;
    };

    AssertSpec as;
    as.rawLine = Misc::vector_to_string(c);
    size_t idx = 1;
    if (c[idx].rfind("TRACE=", 0) == 0) {
        as.trace = c[idx].substr(6);
        ++idx;
    } else {
        as.trace = c[idx++];
    }
    std::transform(as.trace.begin(), as.trace.end(), as.trace.begin(), ::toupper);

    bool     hasExplicitKind = false;
    tokens_t posToks;
    for (; idx < c.size(); ++idx) {
        std::string tok = c[idx];
        std::transform(tok.begin(), tok.end(), tok.begin(), ::toupper);
        if (tok == "AFTER" && idx + 1 < c.size()) {
            as.kind         = AssertKind::AFTER;
            as.refTrace     = c[++idx];
            std::transform(as.refTrace.begin(), as.refTrace.end(), as.refTrace.begin(), ::toupper);
            hasExplicitKind = true;
            continue;
        }
        if ((tok == "COUNT" || tok == "SFQ" || tok == "NPULSES" || tok == "PULSES") && idx + 1 < c.size()
            && c[idx + 1].find('=') == std::string::npos) {
            as.kind         = AssertKind::COUNT;
            as.expCount     = static_cast<int64_t>(std::llround(parse_param(c[++idx], parameters)));
            hasExplicitKind = true;
            continue;
        }
        const size_t eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            const std::string key = tok.substr(0, eqPos);
            const std::string val = tok.substr(eqPos + 1);
            if (key == "BITS" || key == "PAT" || key == "PATTERN" || key == "VEC") {
                as.kind         = AssertKind::BITSTREAM;
                as.bits         = decode_assert_bits(val);
                hasExplicitKind = true;
            } else if (key == "COUNT" || key == "SFQ" || key == "NPULSES" || key == "PULSES") {
                as.kind         = AssertKind::COUNT;
                as.expCount     = static_cast<int64_t>(std::llround(parse_param(val, parameters)));
                hasExplicitKind = true;
            } else if (key == "TIMES" || key == "AT") {
                as.kind         = AssertKind::TIMES;
                hasExplicitKind = true;
                size_t p        = 0;
                while (p < val.size()) {
                    size_t next = val.find_first_of(",;:", p);
                    if (next == std::string::npos) { next = val.size(); }
                    if (next > p) {
                        const double tv = parse_param(val.substr(p, next - p), parameters);
                        if (std::isfinite(tv)) { as.expTimes.push_back(tv); }
                    }
                    p = next + 1;
                }
            } else if (key == "AFTER" || key == "REF") {
                as.kind         = AssertKind::AFTER;
                as.refTrace     = val;
                hasExplicitKind = true;
            } else if (key == "TD" || key == "TSTART" || key == "START") {
                as.td = parse_param(val, parameters);
            } else if (key == "PER" || key == "TBIT" || key == "PERIOD") {
                as.per = parse_param(val, parameters);
            } else if (key == "WINDOW" || key == "W" || key == "PW") {
                as.window = parse_param(val, parameters);
            } else if (key == "TOL") {
                as.tol = std::abs(parse_param(val, parameters));
            } else if (key == "DELAY" || key == "DT") {
                as.delay = parse_param(val, parameters);
            } else if (key == "FROM") {
                as.tFrom = parse_param(val, parameters);
            } else if (key == "TO" || key == "TSTOP") {
                as.tTo = parse_param(val, parameters);
            }
        } else {
            posToks.push_back(tok);
        }
    }

    if (!posToks.empty()) {
        if (!hasExplicitKind) {
            if (posToks.size() == 1 && as.per <= 0.0) {
                const std::string& p0 = posToks[0];
                if (p0.rfind("0X", 0) == 0 || p0.rfind("B", 0) == 0 || p0.find('_') != std::string::npos) {
                    as.kind = AssertKind::BITSTREAM;
                    as.bits = decode_assert_bits(p0);
                } else {
                    as.kind     = AssertKind::COUNT;
                    as.expCount = static_cast<int64_t>(std::llround(parse_param(p0, parameters)));
                }
            } else {
                as.kind = AssertKind::BITSTREAM;
                as.bits = decode_assert_bits(posToks[0]);
                if (posToks.size() >= 2 && as.td <= 0.0) { as.td = parse_param(posToks[1], parameters); }
                if (posToks.size() >= 3 && as.per <= 0.0) { as.per = parse_param(posToks[2], parameters); }
                if (posToks.size() >= 4 && as.tol < 0.0) { as.tol = std::abs(parse_param(posToks[3], parameters)); }
            }
        } else if (as.kind == AssertKind::BITSTREAM) {
            size_t pIdx = 0;
            if (as.bits.empty() && pIdx < posToks.size()) { as.bits = decode_assert_bits(posToks[pIdx++]); }
            if (as.td <= 0.0 && pIdx < posToks.size()) { as.td = parse_param(posToks[pIdx++], parameters); }
            if (as.per <= 0.0 && pIdx < posToks.size()) { as.per = parse_param(posToks[pIdx++], parameters); }
            if (as.tol < 0.0 && pIdx < posToks.size()) { as.tol = std::abs(parse_param(posToks[pIdx++], parameters)); }
        }
    }

    if (as.kind == AssertKind::BITSTREAM && as.per <= 0.0) {
        const double tstop = (transSim.tstop() > 0.0) ? transSim.tstop() : 100e-12;
        const size_t nB    = std::max<size_t>(1, as.bits.size());
        as.per             = std::max(1e-12, (tstop - as.td) / static_cast<double>(nB));
    }
    if (as.window <= 0.0) { as.window = as.per; }

    assertions.push_back(std::move(as));
}

void Input::parse_assert_spec(const std::string& spec) {
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        if (cleaned[idx] == ':' || cleaned[idx] == ';') { cleaned[idx] = ' '; }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if ( firstUpper != "ASSERT" && firstUpper != "VEC") {
        toks.insert(toks.begin(), "ASSERT");
    }
    parse_assert_tokens(toks);
}

void Input::parse_meas_tokens(const tokens_t& c) {
    if (c.size() < 3) { return; }

    auto strip_quotes_or_braces = [](std::string s) -> std::string {
        Misc::ltrim(s);
        Misc::rtrim(s);
        while (s.size() >= 2
               && ((s.front() == '\'' && s.back() == '\'') || (s.front() == '"' && s.back() == '"')
                   || (s.front() == '{' && s.back() == '}'))) {
            s = s.substr(1, s.size() - 2);
            Misc::ltrim(s);
            Misc::rtrim(s);
        }
        return s;
    };

    tokens_t grouped;
    for (size_t i = 0; i < c.size(); ++i) {
        std::string cur = c[i];
        auto        is_unclosed = [](const std::string& str) -> bool {
            int  brace = 0, paren = 0;
            bool inSq = false, inDq = false;
            for (char ch : str) {
                if (ch == '\'' && !inDq) {
                    inSq = !inSq;
                } else if (ch == '"' && !inSq) {
                    inDq = !inDq;
                } else if (!inSq && !inDq) {
                    if (ch == '{') {
                        ++brace;
                    } else if (ch == '}') {
                        --brace;
                    } else if (ch == '(') {
                        ++paren;
                    } else if (ch == ')') {
                        --paren;
                    }
                }
            }
            return inSq || inDq || brace > 0 || paren > 0;
        };
        while (is_unclosed(cur) && i + 1 < c.size()) {
            cur += c[++i];
        }
        grouped.push_back(std::move(cur));
    }

    tokens_t toks;
    for (size_t i = 0; i < grouped.size(); ++i) {
        if (grouped[i] == "=" && !toks.empty() && i + 1 < grouped.size()) {
            toks.back() += "=" + grouped[i + 1];
            ++i;
        } else if (!grouped[i].empty() && grouped[i].back() == '=' && i + 1 < grouped.size()) {
            toks.push_back(grouped[i] + grouped[i + 1]);
            ++i;
        } else if (!grouped[i].empty() && grouped[i].front() == '=' && !toks.empty()) {
            toks.back() += grouped[i];
        } else {
            toks.push_back(grouped[i]);
        }
    }

    if (toks.size() < 3) { return; }
    MeasSpec ms;
    ms.rawLine = Misc::vector_to_string(c);

    size_t      idx = 1;
    std::string t1  = toks[idx];
    std::transform(t1.begin(), t1.end(), t1.begin(), ::toupper);
    if (t1 == "TRAN" || t1 == "TRANSIENT" || t1 == "DC") {
        ++idx;
    }
    if (idx >= toks.size()) { return; }
    ms.name = toks[idx++];
    std::transform(ms.name.begin(), ms.name.end(), ms.name.begin(), ::toupper);
    if (idx >= toks.size()) { return; }

    auto parse_trigger_slice = [&](size_t startIdx, size_t endIdx) -> MeasTrigger {
        MeasTrigger tr;
        tr.active            = true;
        bool hasVal          = false;
        bool hasExplicitCnt  = false;
        bool hasExplicitSfq  = false;
        for (size_t k = startIdx; k < endIdx && k < toks.size(); ++k) {
            std::string tok = toks[k];
            std::transform(tok.begin(), tok.end(), tok.begin(), ::toupper);
            const size_t eqPos = tok.find('=');
            if (eqPos != std::string::npos) {
                const std::string key = strip_quotes_or_braces(tok.substr(0, eqPos));
                const std::string val = strip_quotes_or_braces(tok.substr(eqPos + 1));
                if (key == "VAL" || key == "VALUE") {
                    tr.val = parse_param(val, parameters);
                    hasVal = true;
                } else if (key == "TD" || key == "TSTART" || key == "FROM") {
                    tr.td = parse_param(val, parameters);
                } else if (key == "AT") {
                    tr.isFixedTime = true;
                    tr.fixedTime   = parse_param(val, parameters);
                } else if (key == "RISE") {
                    tr.dir         = MeasCrossDir::RISE;
                    hasExplicitCnt = true;
                    tr.count       = (val == "LAST") ? -1 : std::max<int64_t>(1, std::llround(parse_param(val, parameters)));
                } else if (key == "FALL") {
                    tr.dir         = MeasCrossDir::FALL;
                    hasExplicitCnt = true;
                    tr.count       = (val == "LAST") ? -1 : std::max<int64_t>(1, std::llround(parse_param(val, parameters)));
                } else if (key == "CROSS") {
                    tr.dir         = MeasCrossDir::CROSS;
                    hasExplicitCnt = true;
                    tr.count       = (val == "LAST") ? -1 : std::max<int64_t>(1, std::llround(parse_param(val, parameters)));
                } else if (key == "SFQ" || key == "PULSE" || key == "EVENT") {
                    tr.isSfqEvent  = true;
                    hasExplicitSfq = true;
                    tr.sfqIndex    = (val == "LAST") ? -1 : std::max<int64_t>(1, std::llround(parse_param(val, parameters)));
                } else if (key == "POL" || key == "POLARITY") {
                    if (val == "+" || val == "1" || val == "POS" || val == "RISE") {
                        tr.dir = MeasCrossDir::RISE;
                    } else if (val == "-" || val == "-1" || val == "NEG" || val == "FALL") {
                        tr.dir = MeasCrossDir::FALL;
                    } else {
                        tr.dir = MeasCrossDir::CROSS;
                    }
                } else if (key == "TRACE") {
                    tr.trace = val;
                } else {
                    tr.trace = key;
                    if (val.rfind("V(", 0) == 0 || val.rfind("P(", 0) == 0 || val.rfind("I(", 0) == 0) {
                        tr.rhsTrace = val;
                    } else {
                        tr.val = parse_param(val, parameters);
                        hasVal = true;
                    }
                }
            } else {
                if (tok == "SFQ" || tok == "PULSE" || tok == "EVENT") {
                    tr.isSfqEvent = true;
                } else if (tok == "LAST") {
                    tr.count       = -1;
                    tr.sfqIndex    = -1;
                    hasExplicitCnt = true;
                } else if (tok == "RISE") {
                    tr.dir = MeasCrossDir::RISE;
                } else if (tok == "FALL") {
                    tr.dir = MeasCrossDir::FALL;
                } else if (tok == "CROSS") {
                    tr.dir = MeasCrossDir::CROSS;
                } else if (tr.trace.empty()) {
                    tr.trace = strip_quotes_or_braces(tok);
                }
            }
        }
        if (tr.isSfqEvent && hasExplicitCnt && !hasExplicitSfq) { tr.sfqIndex = tr.count; }
        if (!hasVal && tr.rhsTrace.empty() && !tr.isFixedTime && !tr.isSfqEvent && !tr.trace.empty()) {
            tr.isSfqEvent = true;
            tr.sfqIndex   = tr.count;
        }
        return tr;
    };

    std::string kw = toks[idx];
    std::transform(kw.begin(), kw.end(), kw.begin(), ::toupper);

    auto parse_goal_token = [&](const std::string& rawTok) -> bool {
        std::string tk = rawTok;
        std::transform(tk.begin(), tk.end(), tk.begin(), ::toupper);
        const size_t eqPos = tk.find('=');
        if (eqPos == std::string::npos) { return false; }
        const std::string key = strip_quotes_or_braces(tk.substr(0, eqPos));
        const std::string val = strip_quotes_or_braces(tk.substr(eqPos + 1));
        if (key == "MIN") {
            ms.hasMinGoal = true;
            ms.minGoal    = parse_param(val, parameters);
            return true;
        } else if (key == "MAX") {
            ms.hasMaxGoal = true;
            ms.maxGoal    = parse_param(val, parameters);
            return true;
        } else if (key == "VAL" || key == "GOAL" || key == "EXPECT") {
            ms.hasValGoal = true;
            ms.valGoal    = parse_param(val, parameters);
            return true;
        } else if (key == "TOL") {
            ms.goalTol = std::abs(parse_param(val, parameters));
            return true;
        }
        return false;
    };

    if (kw.rfind("PARAM=", 0) == 0 || kw.rfind("EXPR=", 0) == 0) {
        ms.kind = MeasKind::PARAM;
        ms.expr = strip_quotes_or_braces(kw.substr(kw.find('=') + 1));
        for (size_t k = idx + 1; k < toks.size(); ++k) {
            if (parse_goal_token(toks[k])) { continue; }
            std::string extra = toks[k];
            std::transform(extra.begin(), extra.end(), extra.begin(), ::toupper);
            ms.expr += strip_quotes_or_braces(extra);
        }
        measurements.push_back(std::move(ms));
        return;
    }
    if (kw == "PARAM" || kw == "EXPR") {
        ms.kind = MeasKind::PARAM;
        std::string combined;
        for (size_t k = idx + 1; k < toks.size(); ++k) {
            if (parse_goal_token(toks[k])) { continue; }
            std::string extra = toks[k];
            std::transform(extra.begin(), extra.end(), extra.begin(), ::toupper);
            combined += strip_quotes_or_braces(extra);
        }
        ms.expr = strip_quotes_or_braces(combined);
        measurements.push_back(std::move(ms));
        return;
    }

    if (kw == "TRIG" || kw == "TRIGGER") {
        ms.kind        = MeasKind::TRIG_TARG;
        size_t targPos = toks.size();
        for (size_t k = idx + 1; k < toks.size(); ++k) {
            std::string tk = toks[k];
            std::transform(tk.begin(), tk.end(), tk.begin(), ::toupper);
            if (tk == "TARG" || tk == "TARGET") {
                targPos = k;
                break;
            }
        }
        ms.trig = parse_trigger_slice(idx + 1, targPos);
        if (targPos < toks.size()) {
            ms.targ = parse_trigger_slice(targPos + 1, toks.size());
        }
        measurements.push_back(std::move(ms));
        return;
    }

    if (kw == "WHEN") {
        ms.kind  = MeasKind::WHEN;
        ms.trig  = parse_trigger_slice(idx + 1, toks.size());
        ms.trace = ms.trig.trace;
        measurements.push_back(std::move(ms));
        return;
    }

    if (kw == "FIND" || kw == "DERIV" || kw == "DERIVATIVE") {
        const bool isDeriv = (kw != "FIND");
        if (idx + 1 >= toks.size()) { return; }
        ms.trace = strip_quotes_or_braces(toks[idx + 1]);
        std::transform(ms.trace.begin(), ms.trace.end(), ms.trace.begin(), ::toupper);
        size_t whenPos = toks.size();
        for (size_t k = idx + 2; k < toks.size(); ++k) {
            std::string tk = toks[k];
            std::transform(tk.begin(), tk.end(), tk.begin(), ::toupper);
            if (tk == "WHEN") {
                whenPos = k;
                break;
            }
        }
        if (whenPos < toks.size()) {
            ms.kind = isDeriv ? MeasKind::DERIV : MeasKind::FIND_WHEN;
            ms.trig = parse_trigger_slice(whenPos + 1, toks.size());
        } else {
            ms.kind = isDeriv ? MeasKind::DERIV : MeasKind::FIND_AT;
            for (size_t k = idx + 2; k < toks.size(); ++k) {
                std::string tk = toks[k];
                std::transform(tk.begin(), tk.end(), tk.begin(), ::toupper);
                if (tk.rfind("AT=", 0) == 0) {
                    ms.atTime = parse_param(strip_quotes_or_braces(tk.substr(3)), parameters);
                }
            }
        }
        measurements.push_back(std::move(ms));
        return;
    }

    if (kw == "AVG" || kw == "MEAN") {
        ms.kind = MeasKind::AVG;
    } else if (kw == "RMS") {
        ms.kind = MeasKind::RMS;
    } else if (kw == "MIN") {
        ms.kind = MeasKind::MIN;
    } else if (kw == "MAX") {
        ms.kind = MeasKind::MAX;
    } else if (kw == "MIN_AT" || kw == "MINAT") {
        ms.kind = MeasKind::MIN_AT;
    } else if (kw == "MAX_AT" || kw == "MAXAT") {
        ms.kind = MeasKind::MAX_AT;
    } else if (kw == "PP" || kw == "PK2PK") {
        ms.kind = MeasKind::PP;
    } else if (kw == "INTEG" || kw == "INTEGRAL") {
        ms.kind = MeasKind::INTEG;
    } else if (kw == "COUNT" || kw == "NPULSES" || kw == "PULSES") {
        ms.kind = MeasKind::COUNT;
    } else if (kw == "FREQ" || kw == "FREQUENCY") {
        ms.kind = MeasKind::FREQ;
    } else if (kw == "VPEAK" || kw == "V_PEAK") {
        ms.kind = MeasKind::VPEAK;
    } else if (kw == "FWHM") {
        ms.kind = MeasKind::FWHM;
    } else if (kw == "DT_PI" || kw == "DTPI") {
        ms.kind = MeasKind::DT_PI;
    } else if (kw == "AREA" || kw == "FLUX") {
        ms.kind = MeasKind::AREA;
    } else {
        return;
    }

    if (idx + 1 >= toks.size()) { return; }
    ms.trace = strip_quotes_or_braces(toks[idx + 1]);
    std::transform(ms.trace.begin(), ms.trace.end(), ms.trace.begin(), ::toupper);

    for (size_t k = idx + 2; k < toks.size(); ++k) {
        std::string tk = toks[k];
        std::transform(tk.begin(), tk.end(), tk.begin(), ::toupper);
        if (tk == "TRIG" || tk == "TRIGGER") {
            size_t targPos = toks.size();
            for (size_t m = k + 1; m < toks.size(); ++m) {
                std::string tm = toks[m];
                std::transform(tm.begin(), tm.end(), tm.begin(), ::toupper);
                if (tm == "TARG" || tm == "TARGET") {
                    targPos = m;
                    break;
                }
            }
            ms.trig = parse_trigger_slice(k + 1, targPos);
            if (targPos < toks.size()) {
                ms.targ = parse_trigger_slice(targPos + 1, toks.size());
            }
            break;
        }
        const size_t eqPos = tk.find('=');
        if (eqPos != std::string::npos) {
            if (parse_goal_token(tk)) { continue; }
            const std::string key = strip_quotes_or_braces(tk.substr(0, eqPos));
            const std::string val = strip_quotes_or_braces(tk.substr(eqPos + 1));
            if (key == "FROM" || key == "TSTART" || key == "TD") {
                ms.hasFrom = true;
                ms.tFrom   = parse_param(val, parameters);
            } else if (key == "TO" || key == "TSTOP") {
                ms.hasTo = true;
                ms.tTo   = parse_param(val, parameters);
            } else if (key == "SFQ" || key == "PULSE" || key == "EVENT") {
                ms.hasSfqIndex = true;
                ms.sfqIndex    = (val == "LAST") ? -1 : std::max<int64_t>(1, std::llround(parse_param(val, parameters)));
            } else if (key == "POL" || key == "POLARITY") {
                if (val == "+" || val == "1" || val == "POS" || val == "RISE") {
                    ms.trig.dir = MeasCrossDir::RISE;
                } else if (val == "-" || val == "-1" || val == "NEG" || val == "FALL") {
                    ms.trig.dir = MeasCrossDir::FALL;
                }
            }
        } else if (tk == "LAST") {
            ms.hasSfqIndex = true;
            ms.sfqIndex    = -1;
        }
    }

    measurements.push_back(std::move(ms));
}

void Input::parse_meas_spec(const std::string& spec) {
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    bool        inSq = false, inDq = false;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        if (cleaned[idx] == '\'' && !inDq) {
            inSq = !inSq;
        } else if (cleaned[idx] == '"' && !inSq) {
            inDq = !inDq;
        } else if (!inSq && !inDq && (cleaned[idx] == ':' || cleaned[idx] == ';')) {
            cleaned[idx] = ' ';
        }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "MEAS" && firstUpper != "MEASURE") {
        toks.insert(toks.begin(), "MEAS");
    }
    parse_meas_tokens(toks);
}

void Input::parse_ic_tokens(const tokens_t& c) {
    if (c.size() < 2) { return; }
    auto resolve_val = [&](std::string s) -> double {
        while (s.size() >= 2 && ((s.front() == '\'' && s.back() == '\'') || (s.front() == '"' && s.back() == '"')
                                 || (s.front() == '{' && s.back() == '}'))) {
            s = s.substr(1, s.size() - 2);
        }
        return parse_param(s, parameters);
    };

    tokens_t grouped;
    for (size_t i = 1; i < c.size(); ++i) {
        std::string cur   = c[i];
        int         depth = 0;
        for (char ch : cur) {
            if (ch == '(' || ch == '{') {
                ++depth;
            } else if (ch == ')' || ch == '}') {
                --depth;
            }
        }
        while (depth > 0 && i + 1 < c.size()) {
            ++i;
            cur += c[i];
            for (char ch : c[i]) {
                if (ch == '(' || ch == '{') {
                    ++depth;
                } else if (ch == ')' || ch == '}') {
                    --depth;
                }
            }
        }
        grouped.push_back(std::move(cur));
    }

    tokens_t toks;
    for (size_t i = 0; i < grouped.size(); ++i) {
        if (grouped[i] == "=" && !toks.empty() && i + 1 < grouped.size()) {
            toks.back() += "=" + grouped[i + 1];
            ++i;
        } else if (!toks.empty() && toks.back().back() == '=') {
            toks.back() += grouped[i];
        } else if (!grouped[i].empty() && grouped[i].front() == '=' && !toks.empty()) {
            toks.back() += grouped[i];
        } else {
            toks.push_back(grouped[i]);
        }
    }

    for (const auto& tok : toks) {
        const size_t eqPos = tok.find('=');
        if (eqPos == std::string::npos || eqPos == 0 || eqPos + 1 >= tok.size()) { continue; }
        std::string lhs = tok.substr(0, eqPos);
        std::string rhs = tok.substr(eqPos + 1);
        std::transform(lhs.begin(), lhs.end(), lhs.begin(), ::toupper);
        const double val = resolve_val(rhs);
        if (!std::isfinite(val)) { continue; }

        IcSpec spec;
        spec.value = val;
        if (lhs.rfind("V(", 0) == 0 && lhs.back() == ')') {
            std::string inner = lhs.substr(2, lhs.size() - 3);
            const size_t commaPos = inner.find(',');
            if (commaPos != std::string::npos) {
                std::string n2 = inner.substr(commaPos + 1);
                inner          = inner.substr(0, commaPos);
                if (!n2.empty() && n2 != "0" && n2 != "GND") {
                    IcSpec negSpec;
                    negSpec.kind   = IcTargetKind::NODE_VOLTAGE;
                    negSpec.target = n2;
                    negSpec.value  = 0.0;
                    initialConditions.push_back(std::move(negSpec));
                }
            }
            spec.kind   = (!inner.empty() && (inner.front() == 'B' || inner.front() == 'C'))
                                  ? IcTargetKind::DEV_VOLTAGE
                                  : IcTargetKind::NODE_VOLTAGE;
            spec.target = inner;
        } else if ((lhs.rfind("P(", 0) == 0 || lhs.rfind("PHI(", 0) == 0) && lhs.back() == ')') {
            const size_t openPos  = lhs.find('(');
            std::string  inner    = lhs.substr(openPos + 1, lhs.size() - openPos - 2);
            const size_t commaPos = inner.find(',');
            if (commaPos != std::string::npos) { inner = inner.substr(0, commaPos); }
            spec.kind   = (!inner.empty() && (inner.front() == 'B' || inner.front() == 'C'))
                                  ? IcTargetKind::DEV_PHASE
                                  : IcTargetKind::NODE_PHASE;
            spec.target = inner;
        } else if (lhs.rfind("I(", 0) == 0 && lhs.back() == ')') {
            spec.kind   = IcTargetKind::BRANCH_CURRENT;
            spec.target = lhs.substr(2, lhs.size() - 3);
        } else {
            spec.kind   = (argAnal == AnalysisType::Phase) ? IcTargetKind::NODE_PHASE : IcTargetKind::NODE_VOLTAGE;
            spec.target = lhs;
        }
        if (!spec.target.empty()) {
            initialConditions.push_back(std::move(spec));
        }
    }
}

void Input::parse_ic_spec(const std::string& spec) {
    tokens_t toks = Misc::tokenize(spec);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "IC" && firstUpper != ".IC") {
        toks.insert(toks.begin(), "IC");
    }
    parse_ic_tokens(toks);
}

void Input::parse_check_loops_tokens(const tokens_t& c) {
    argCheckLoops = true;
    auto resolve_path = [&](const std::string& raw) -> std::string {
        auto p = std::filesystem::path(raw);
        if (!p.has_parent_path() && fileParentPath) {
            p = std::filesystem::path(fileParentPath.value()).append(raw);
        }
        return p.string();
    };
    for (size_t i = 1; i < c.size(); ++i) {
        std::string tok = c[i];
        std::string up  = tok;
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        if (up == "STRICT" || up == "STRICT=1" || up == "STRICT=TRUE") {
            argCheckLoopsStrict = true;
        } else if (up == "WARN" || up == "WARN_ONLY" || up == "WARNONLY" || up == "STRICT=0"
                   || up == "STRICT=FALSE") {
            argCheckLoopsStrict = false;
        } else if (up.rfind("FILE=", 0) == 0 || up.rfind("LOOPFILE=", 0) == 0 || up.rfind("LOOPSFILE=", 0) == 0) {
            loopFile = resolve_path(tok.substr(tok.find('=') + 1));
        }
    }
}

void Input::parse_jitter_tokens(const tokens_t& c) {
    auto decode_bits = [](std::string raw) -> std::string {
        raw.erase(std::remove(raw.begin(), raw.end(), '\"'), raw.end());
        raw.erase(std::remove(raw.begin(), raw.end(), '\''), raw.end());
        raw.erase(std::remove(raw.begin(), raw.end(), '_'), raw.end());
        raw.erase(std::remove(raw.begin(), raw.end(), ' '), raw.end());
        if (raw.empty()) { return ""; }
        std::string up = raw;
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        if (up.rfind("0B", 0) == 0) {
            up = up.substr(2);
        } else if (!up.empty() && up.front() == 'B' && up.size() > 1
                   && up.find_first_not_of("01X", 1) == std::string::npos) {
            up = up.substr(1);
        } else if (up.rfind("0X", 0) == 0) {
            std::string hex = up.substr(2);
            std::string bits;
            bits.reserve(hex.size() * 4);
            for (char hc : hex) {
                int v = -1;
                if (hc >= '0' && hc <= '9') {
                    v = hc - '0';
                } else if (hc >= 'A' && hc <= 'F') {
                    v = 10 + (hc - 'A');
                }
                if (v < 0) { return ""; }
                for (int b = 3; b >= 0; --b) { bits.push_back(((v >> b) & 1) ? '1' : '0'); }
            }
            return bits;
        }
        std::string out;
        for (char ch : up) {
            if (ch == '0' || ch == '1' || ch == 'X') { out.push_back(ch); }
        }
        return out;
    };

    tokens_t toks;
    for (size_t i = 1; i < c.size(); ++i) {
        if (c[i] == "=" && !toks.empty() && i + 1 < c.size()) {
            toks.back() += "=" + c[i + 1];
            ++i;
        } else if (!c[i].empty() && c[i].back() == '=' && i + 1 < c.size()) {
            toks.push_back(c[i] + c[i + 1]);
            ++i;
        } else if (!c[i].empty() && c[i].front() == '=' && !toks.empty()) {
            toks.back() += c[i];
        } else {
            toks.push_back(c[i]);
        }
    }

    JitterSpec js;
    tokens_t   posToks;
    for (size_t idx = 0; idx < toks.size(); ++idx) {
        std::string tok = toks[idx];
        std::string up  = tok;
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        if (up == "TRAN" || up == "TRANSIENT") { continue; }
        if ((up == "AFTER" || up == "REF" || up == "TRIG") && idx + 1 < toks.size()
            && toks[idx + 1].find('=') == std::string::npos) {
            js.mode     = JitterMode::REF_TRACE;
            js.refTrace = toks[++idx];
            std::transform(js.refTrace.begin(), js.refTrace.end(), js.refTrace.begin(), ::toupper);
            continue;
        }
        if (up == "STRICT") {
            js.strict = true;
            continue;
        }
        if (up == "SPARSE") {
            js.sparse = true;
            continue;
        }
        const size_t eqPos = up.find('=');
        if (eqPos != std::string::npos) {
            const std::string key    = up.substr(0, eqPos);
            const std::string val    = up.substr(eqPos + 1);
            const std::string rawVal = tok.substr(eqPos + 1);
            if (key == "NAME" || key == "LABEL") {
                js.name = val;
            } else if (key == "TRACE" || key == "TARG" || key == "TARGET") {
                js.trace = val;
            } else if (key == "REF" || key == "AFTER" || key == "TRIG") {
                js.mode     = JitterMode::REF_TRACE;
                js.refTrace = val;
            } else if (key == "PER" || key == "CLK" || key == "PERIOD" || key == "TBIT") {
                js.mode = JitterMode::CLOCK_GRID;
                js.per  = std::abs(parse_param(val, parameters));
            } else if (key == "TD" || key == "TSTART" || key == "START") {
                js.td    = parse_param(val, parameters);
                js.hasTd = true;
            } else if (key == "BITS" || key == "PAT" || key == "PATTERN" || key == "VEC") {
                js.mode = JitterMode::CLOCK_GRID;
                js.bits = decode_bits(val);
            } else if (key == "WINDOW" || key == "W" || key == "PW") {
                js.window = std::abs(parse_param(val, parameters));
            } else if (key == "DELAY" || key == "OFFSET" || key == "NOM_DELAY") {
                js.nomDelay    = parse_param(val, parameters);
                js.hasNomDelay = true;
            } else if (key == "MAX_DELAY" || key == "MAXDELAY" || key == "TOL") {
                js.maxDelay = std::abs(parse_param(val, parameters));
            } else if (key == "FROM") {
                js.tFrom   = parse_param(val, parameters);
                js.hasFrom = true;
            } else if (key == "TO" || key == "TSTOP") {
                js.tTo   = parse_param(val, parameters);
                js.hasTo = true;
            } else if (key == "COUNT" || key == "NPULSES" || key == "PULSES" || key == "SFQ") {
                js.expCount = static_cast<int64_t>(std::llround(parse_param(val, parameters)));
            } else if (key == "SPARSE") {
                js.sparse = (val == "1" || val == "TRUE");
            } else if (key == "STRICT") {
                js.strict = (val == "1" || val == "TRUE");
            } else if (key == "MAX_RMS" || key == "MAXRMS" || key == "MAX_JITTER" || key == "MAXJITTER") {
                js.maxRms    = std::abs(parse_param(val, parameters));
                js.hasMaxRms = true;
            } else if (key == "MAX_PP" || key == "MAXPP") {
                js.maxPp    = std::abs(parse_param(val, parameters));
                js.hasMaxPp = true;
            } else if (key == "MAX_SKEW" || key == "MAXSKEW" || key == "MAX_LATENCY") {
                js.maxSkew    = std::abs(parse_param(val, parameters));
                js.hasMaxSkew = true;
            } else if (key == "MAX_BER" || key == "MAXBER") {
                js.maxBer    = std::abs(parse_param(val, parameters));
                js.hasMaxBer = true;
            } else if (key == "MAX_ERRORS" || key == "MAXERRORS" || key == "MAX_ERR" || key == "MAXERR") {
                js.maxErrs    = static_cast<int64_t>(std::llround(parse_param(val, parameters)));
                js.hasMaxErrs = true;
            } else if (key == "FILE" || key == "JITTERFILE") {
                auto p = std::filesystem::path(rawVal);
                if (!p.has_parent_path() && fileParentPath) {
                    p = std::filesystem::path(fileParentPath.value()).append(rawVal);
                }
                jitterFile = p.string();
            }
        } else {
            posToks.push_back(up);
        }
    }

    if (js.trace.empty()) {
        if (posToks.empty()) {
            js.trace = "B*";
        } else if (posToks.size() == 1) {
            js.trace = posToks[0];
        } else if (posToks.size() >= 2) {
            if (js.name.empty()) { js.name = posToks[0]; }
            js.trace = posToks[1];
            if (posToks.size() >= 3 && js.refTrace.empty()) {
                js.refTrace = posToks[2];
                js.mode     = JitterMode::REF_TRACE;
            }
        }
    }
    if (js.mode == JitterMode::CLOCK_GRID && js.per <= 0.0 && !js.bits.empty()) {
        const double tstop = (transSim.tstop() > 0.0) ? transSim.tstop() : 100e-12;
        const size_t nB    = std::max<size_t>(1, js.bits.size());
        js.per             = std::max(1e-12, (tstop - js.td) / static_cast<double>(nB));
    }
    if (js.window <= 0.0 && js.per > 0.0) { js.window = js.per; }

    jitterSpecs.push_back(std::move(js));
}

void Input::parse_jitter_spec(const std::string& spec) {
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        if (cleaned[idx] == ':' || cleaned[idx] == ';') { cleaned[idx] = ' '; }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "JITTER" && firstUpper != ".JITTER" && firstUpper != "BER" && firstUpper != ".BER") {
        toks.insert(toks.begin(), "JITTER");
    }
    parse_jitter_tokens(toks);
}

void Input::parse_sens_tokens(const tokens_t& c) {
    argSens = true;
    SensSpec spec;
    std::vector<double> numArgs;

    auto resolve_path = [&](const std::string& raw) -> std::string {
        auto p = std::filesystem::path(raw);
        if (!p.has_parent_path() && fileParentPath) {
            p = std::filesystem::path(fileParentPath.value()).append(raw);
        }
        return p.string();
    };

    for (size_t k = 1; k < c.size(); ++k) {
        std::string tok = c.at(k);
        if (tok.empty()) { continue; }
        const size_t eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            std::string key    = tok.substr(0, eqPos);
            std::string rawVal = tok.substr(eqPos + 1);
            std::string val    = rawVal;
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);
            if (key == "FILE" || key == "OUT" || key == "SENSFILE" || key == "SENS_FILE"
                || key == "SENSITIVITYFILE") {
                if (!rawVal.empty()) { sensFile = resolve_path(rawVal); }
                continue;
            }
            std::transform(val.begin(), val.end(), val.begin(), ::toupper);
            if (key == "DELTA" || key == "DX" || key == "PERTURB" || key == "PERT") {
                if (!val.empty() && val.back() == '%') { val.pop_back(); }
                double d = std::abs(parse_param(val, parameters));
                if (std::isfinite(d) && d > 0.0) {
                    if (d > 0.5 && d <= 50.0) { d *= 0.01; }
                    spec.delta = std::clamp(d, 1e-4, 0.5);
                }
                continue;
            }
            if (key == "METRIC" || key == "TARGET" || key == "MEAS" || key == "MEASURE") {
                spec.metric = val;
                continue;
            }
            if (key == "GOAL" || key == "VAL" || key == "TARGET_VAL" || key == "TARGETVAL") {
                const double g = parse_param(val, parameters);
                if (std::isfinite(g)) {
                    spec.goal    = g;
                    spec.hasGoal = true;
                }
                continue;
            }
            if (key == "CENTER" || key == "MARGINS" || key == "MARGIN" || key == "OPTIMIZE") {
                spec.center = (val != "0" && val != "FALSE" && val != "NO");
                continue;
            }
            if (key == "RANGE" || key == "SPAN") {
                if (!val.empty() && val.back() == '%') { val.pop_back(); }
                double d = std::abs(parse_param(val, parameters));
                if (std::isfinite(d) && d > 0.0) {
                    if (d > 1.0 && d <= 90.0) { d *= 0.01; }
                    spec.minVal = std::max(0.01, 1.0 - d);
                    spec.maxVal = 1.0 + d;
                    spec.center = true;
                }
                continue;
            }
            if (key == "MIN" || key == "START" || key == "LO" || key == "LOW") {
                const double v = parse_param(val, parameters);
                if (std::isfinite(v)) {
                    spec.minVal = v;
                    spec.center = true;
                }
                continue;
            }
            if (key == "MAX" || key == "STOP" || key == "HI" || key == "HIGH") {
                const double v = parse_param(val, parameters);
                if (std::isfinite(v)) {
                    spec.maxVal = v;
                    spec.center = true;
                }
                continue;
            }
            if (key == "STEP" || key == "D") {
                const double v = std::abs(parse_param(val, parameters));
                if (std::isfinite(v) && v > 0.0) {
                    spec.stepVal = v;
                    spec.hasStep = true;
                    spec.center  = true;
                }
                continue;
            }
            if (key == "N" || key == "NPTS" || key == "PTS" || key == "POINTS" || key == "COUNT") {
                const int64_t n = std::max<int64_t>(3, static_cast<int64_t>(std::llround(parse_param(val, parameters))));
                spec.numPts     = n;
                spec.hasStep    = false;
                spec.center     = true;
                continue;
            }
            if (key == "TOL" || key == "MARGINTOL" || key == "MARGIN_TOL") {
                const double v = std::abs(parse_param(val, parameters));
                if (std::isfinite(v) && v >= 0.0) {
                    spec.tol     = v;
                    argMarginTol = v;
                }
                continue;
            }
        }

        std::string tokUpper = tok;
        std::transform(tokUpper.begin(), tokUpper.end(), tokUpper.begin(), ::toupper);
        if (tokUpper == "SENS" || tokUpper == ".SENS" || tokUpper == "SENSITIVITY" || tokUpper == ".SENSITIVITY"
            || tokUpper == "REL" || tokUpper == "SCALE" || tokUpper == "MULT" || tokUpper == "LIN") {
            continue;
        }
        if (tokUpper == "CENTER" || tokUpper == "MARGINS" || tokUpper == "MARGIN" || tokUpper == "OPTIMIZE"
            || tokUpper == "FULL") {
            spec.center = true;
            continue;
        }
        if (tokUpper == "DEVICES" || tokUpper == "ALL_DEVS" || tokUpper == "CELL") {
            spec.targets.push_back("@DEVICES");
            continue;
        }
        if (tokUpper == "SUBCKTS" || tokUpper == "CELLS") {
            spec.targets.push_back("@SUBCKTS");
            continue;
        }
        if (tokUpper == "CATS" || tokUpper == "CATEGORIES" || tokUpper == "DEFAULT") {
            spec.targets.push_back("@DEFAULT");
            continue;
        }

        std::string numTok = tokUpper;
        bool        hadPct = false;
        if (!numTok.empty() && numTok.back() == '%') {
            numTok.pop_back();
            hadPct = true;
        }
        const char fc = numTok.empty() ? '\0' : numTok.front();
        if ((fc >= '0' && fc <= '9') || fc == '+' || fc == '-' || fc == '.') {
            double v = parse_param(numTok, parameters);
            if (std::isfinite(v)) {
                if (hadPct) { v *= 0.01; }
                numArgs.push_back(v);
                continue;
            }
        }

        size_t pos = 0;
        while (pos < tokUpper.size()) {
            size_t next = tokUpper.find(',', pos);
            if (next == std::string::npos) { next = tokUpper.size(); }
            if (next > pos) {
                std::string item = tokUpper.substr(pos, next - pos);
                if (item == "DEVICES" || item == "ALL_DEVS" || item == "CELL") {
                    item = "@DEVICES";
                } else if (item == "SUBCKTS" || item == "CELLS") {
                    item = "@SUBCKTS";
                } else if (item == "CATS" || item == "CATEGORIES" || item == "DEFAULT") {
                    item = "@DEFAULT";
                }
                spec.targets.push_back(item);
            }
            pos = next + 1;
        }
    }

    if (numArgs.size() == 1) {
        double d = std::abs(numArgs[0]);
        if (d > 0.5 && d <= 50.0) { d *= 0.01; }
        if (d > 0.0 && d <= 0.5) { spec.delta = d; }
    } else if (numArgs.size() == 2) {
        spec.minVal = numArgs[0];
        spec.maxVal = numArgs[1];
        spec.center = true;
    } else if (numArgs.size() >= 3) {
        spec.minVal  = numArgs[0];
        spec.maxVal  = numArgs[1];
        spec.stepVal = std::abs(numArgs[2]);
        spec.hasStep = (spec.stepVal > 0.0 && spec.stepVal < 1.0);
        if (!spec.hasStep && numArgs[2] >= 3.0) {
            spec.numPts = static_cast<int64_t>(std::llround(numArgs[2]));
        }
        spec.center = true;
    }

    if (spec.targets.empty()) { spec.targets.push_back("@DEVICES"); }
    sensSpecs.push_back(std::move(spec));
}

void Input::parse_sens_spec(const std::string& spec) {
    if (spec.empty()) { return; }
    std::string cleaned = spec;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        char& ch = cleaned[idx];
        if (ch == ':' || ch == ';') { ch = ' '; }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "SENS" && firstUpper != ".SENS" && firstUpper != "SENSITIVITY"
        && firstUpper != ".SENSITIVITY") {
        toks.insert(toks.begin(), "SENS");
    }
    parse_sens_tokens(toks);
}


void Input::parse_options_and_state_controls() {
    auto resolve_path = [&](const std::string& raw) -> std::string {
        auto p = std::filesystem::path(raw);
        if (!p.has_parent_path() && fileParentPath) {
            p = std::filesystem::path(fileParentPath.value()).append(raw);
        }
        return p.string();
    };
    for (const auto& c : controls) {
        if (c.empty()) { continue; }
        if (c.front() == "SAVESTATE" || c.front() == "SAVEBIAS") {
            if (c.size() >= 2) {
                std::optional<double> saveTime = std::nullopt;
                if (c.size() >= 3) { saveTime = parse_param(c.at(2), parameters); }
                saveStateFile = std::make_pair(resolve_path(c.at(1)), saveTime);
            }
        } else if (c.front() == "LOADSTATE" || c.front() == "LOADBIAS") {
            if (c.size() >= 2) { loadStateFile = resolve_path(c.at(1)); }
        } else if (c.front() == "SWEEP") {
            parse_sweep_tokens(c);
        } else if (c.front() == "MARGINS" || c.front() == "MARGIN") {
            parse_margin_tokens(c);
        } else if (c.front() == "VERIFY_EVENTS" || c.front() == "VERIFYEVENTS") {
            if (c.size() >= 2) { verifyEventsFile = resolve_path(c.at(1)); }
            if (c.size() >= 3) {
                std::string tStr = c.at(2);
                std::transform(tStr.begin(), tStr.end(), tStr.begin(), ::toupper);
                if (tStr.rfind("TOL=", 0) == 0) { tStr = tStr.substr(4); }
                const double tv = parse_param(tStr, parameters);
                if (std::isfinite(tv) && tv >= 0.0) { argMarginTol = tv; }
            }
        } else if (c.front() == "ASSERT" || c.front() == "VEC") {
            parse_assert_tokens(c);
        } else if (c.front() == "MEAS" || c.front() == "MEASURE") {
            parse_meas_tokens(c);
        } else if (c.front() == "IC") {
            parse_ic_tokens(c);
        } else if (c.front() == "CHECK_LOOPS" || c.front() == "CHECKLOOPS") {
            parse_check_loops_tokens(c);
        } else if (c.front() == "JITTER" || c.front() == "BER") {
            parse_jitter_tokens(c);
        } else if (c.front() == "SENS" || c.front() == "SENSITIVITY") {
            parse_sens_tokens(c);
        } else if (c.front() == "OPTION") {
            for (size_t k = 1; k < c.size(); ++k) {
                const std::string& tok = c.at(k);
                if (tok == "FIXEDSTEP" || tok == "FIXEDSTEP=1" || tok == "FIXEDSTEP=TRUE") {
                    argFixedStep = true;
                } else if (tok == "FIXEDSTEP=0" || tok == "FIXEDSTEP=FALSE") {
                    argFixedStep = false;
                } else if (tok == "WARNSTEP" || tok == "WARNSTEP=1" || tok == "WARNSTEP=TRUE") {
                    argWarnStep = true;
                } else if (tok == "TXINTERP" || tok == "TXINTERP=1" || tok == "FRACTX" || tok == "FRACTX=1") {
                    argTxInterp = true;
                } else if (tok == "TXINTERP=0" || tok == "FRACTX=0") {
                    argTxInterp = false;
                } else if (tok == "CORRECTOR") {
                    argCorrector = 1;
                } else if (tok.rfind("CORRECTOR=", 0) == 0) {
                    argCorrector = static_cast<int64_t>(parse_param(tok.substr(10), parameters));
                } else if (tok.rfind("CORRECTORTOL=", 0) == 0) {
                    argCorrectorTol = parse_param(tok.substr(13), parameters);
                } else if (tok.rfind("KCUTOFF=", 0) == 0) {
                    argKCutoff = std::abs(parse_param(tok.substr(8), parameters));
                } else if (tok == "GPU" || tok == "GPU=1" || tok == "GPU=TRUE") {
                    argGpu     = true;
                    argGpuMode = 1;
                    argPar     = true;
                } else if (tok == "GPU=2") {
                    argGpu     = true;
                    argGpuMode = 2;
                    argPar     = true;
                } else if (tok == "GPU=3" || tok == "GPUMONO" || tok == "GPUMONO=1" || tok == "GPUMONO=TRUE"
                           || tok == "CUSOLVERRF" || tok == "CUSOLVERRF=1") {
                    argGpu     = true;
                    argGpuMode = 3;
                    argPar     = true;
                } else if (tok == "GPU=4" || tok == "GPUNODAL" || tok == "GPUNODAL=1" || tok == "GPUNODAL=TRUE") {
                    argGpu     = true;
                    argGpuMode = 4;
                    argPar     = true;
                    argNodal   = true;
                } else if (tok == "GPU=0" || tok == "GPU=FALSE") {
                    argGpu     = false;
                    argGpuMode = 0;
                } else if (tok == "NODAL" || tok == "NODAL=TRUE" || tok == "NODAL=AUTO") {
                    argNodal      = true;
                    argNodalLevel = 0;
                } else if (tok == "NODAL=1") {
                    argNodal      = true;
                    argNodalLevel = 1;
                } else if (tok == "NODAL=2") {
                    argNodal      = true;
                    argNodalLevel = 2;
                } else if (tok == "NODAL=0" || tok == "NODAL=FALSE") {
                    argNodal      = false;
                    argNodalLevel = 0;
                } else if (tok == "BYPASS" || tok == "BYPASS=1" || tok == "BYPASS=TRUE" || tok == "LATENCY"
                           || tok == "LATENCY=1") {
                    argBypass = true;
                } else if (tok == "BYPASS=0" || tok == "BYPASS=FALSE" || tok == "LATENCY=0") {
                    argBypass = false;
                } else if (tok.rfind("BYPASSTOL=", 0) == 0) {
                    argBypass    = true;
                    argBypassTol = std::abs(parse_param(tok.substr(10), parameters));
                } else if (tok.rfind("LATENCY=", 0) == 0) {
                    argBypass    = true;
                    argBypassTol = std::abs(parse_param(tok.substr(8), parameters));
                } else if (tok.rfind("MC=", 0) == 0) {
                    argMcRuns = std::max<int64_t>(1, static_cast<int64_t>(parse_param(tok.substr(3), parameters)));
                } else if (tok.rfind("BATCH=", 0) == 0) {
                    argMcRuns = std::max<int64_t>(1, static_cast<int64_t>(parse_param(tok.substr(6), parameters)));
                } else if (tok.rfind("RUNS=", 0) == 0) {
                    argMcRuns = std::max<int64_t>(1, static_cast<int64_t>(parse_param(tok.substr(5), parameters)));
                } else if (tok == "MCALL" || tok == "MCALL=1" || tok == "MCALL=TRUE" || tok == "SWEEPALL"
                           || tok == "SWEEPALL=1" || tok == "SWEEPALL=TRUE") {
                    argMcAll = true;
                } else if (tok == "MCALL=0" || tok == "MCALL=FALSE" || tok == "SWEEPALL=0" || tok == "SWEEPALL=FALSE") {
                    argMcAll = false;
                } else if (tok.rfind("SWEEP=", 0) == 0) {
                    parse_sweep_spec(tok.substr(6));
                } else if (tok == "MARGINS" || tok == "MARGIN" || tok == "MARGINS=1" || tok == "MARGINS=TRUE") {
                    argMargins = true;
                } else if (tok == "MARGINS=0" || tok == "MARGINS=FALSE" || tok == "NOMARGINS") {
                    argMargins = false;
                } else if (tok.rfind("MARGINS=", 0) == 0) {
                    parse_margin_spec(tok.substr(8));
                } else if (tok.rfind("MARGIN=", 0) == 0) {
                    parse_margin_spec(tok.substr(7));
                } else if (tok.rfind("MARGINTOL=", 0) == 0 || tok.rfind("MARGIN_TOL=", 0) == 0) {
                    const double tv = std::abs(parse_param(tok.substr(tok.find('=') + 1), parameters));
                    if (std::isfinite(tv) && tv >= 0.0) { argMarginTol = tv; }
                } else if (tok.rfind("MARGINFILE=", 0) == 0 || tok.rfind("MARGINSFILE=", 0) == 0) {
                    argMargins = true;
                    marginFile = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok.rfind("VERIFYEVENTS=", 0) == 0 || tok.rfind("VERIFY_EVENTS=", 0) == 0) {
                    verifyEventsFile = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok.rfind("ASSERT=", 0) == 0) {
                    parse_assert_spec(tok.substr(7));
                } else if (tok.rfind("MEASFILE=", 0) == 0 || tok.rfind("MEASUREFILE=", 0) == 0) {
                    measFile = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok.rfind("MEAS=", 0) == 0 || tok.rfind("MEASURE=", 0) == 0) {
                    parse_meas_spec(tok.substr(tok.find('=') + 1));
                } else if (tok.rfind("IC=", 0) == 0) {
                    parse_ic_spec(tok.substr(3));
                } else if (tok == "CHECKLOOPS" || tok == "CHECK_LOOPS" || tok == "CHECKLOOPS=1"
                           || tok == "CHECK_LOOPS=1" || tok == "CHECKLOOPS=TRUE" || tok == "CHECK_LOOPS=TRUE") {
                    argCheckLoops = true;
                } else if (tok == "CHECKLOOPS=STRICT" || tok == "CHECK_LOOPS=STRICT") {
                    argCheckLoops       = true;
                    argCheckLoopsStrict = true;
                } else if (tok == "CHECKLOOPS=0" || tok == "CHECK_LOOPS=0" || tok == "CHECKLOOPS=FALSE"
                           || tok == "CHECK_LOOPS=FALSE" || tok == "NOCHECKLOOPS") {
                    argCheckLoops = false;
                } else if (tok.rfind("LOOPFILE=", 0) == 0 || tok.rfind("LOOPSFILE=", 0) == 0) {
                    argCheckLoops = true;
                    loopFile      = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok == "JITTER" || tok == "JITTER=1" || tok == "JITTER=TRUE") {
                    parse_jitter_spec("B*");
                } else if (tok.rfind("JITTER=", 0) == 0) {
                    parse_jitter_spec(tok.substr(7));
                } else if (tok.rfind("JITTERFILE=", 0) == 0 || tok.rfind("JITTER_FILE=", 0) == 0) {
                    jitterFile = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok == "SENS" || tok == "SENSITIVITY" || tok == "SENS=1" || tok == "SENS=TRUE") {
                    argSens = true;
                } else if (tok == "SENS=0" || tok == "SENS=FALSE" || tok == "NOSENS") {
                    argSens = false;
                } else if (tok.rfind("SENS=", 0) == 0) {
                    parse_sens_spec(tok.substr(5));
                } else if (tok.rfind("SENSITIVITY=", 0) == 0) {
                    parse_sens_spec(tok.substr(12));
                } else if (tok.rfind("SENSFILE=", 0) == 0 || tok.rfind("SENS_FILE=", 0) == 0
                           || tok.rfind("SENSITIVITYFILE=", 0) == 0) {
                    argSens  = true;
                    sensFile = resolve_path(tok.substr(tok.find('=') + 1));
                } else if (tok == "LUMPPTL" || tok == "LUMPPTL=1" || tok == "LUMPPTL=TRUE") {
                    argLumpPtl = true;
                } else if (tok == "LUMPPTL=0" || tok == "LUMPPTL=FALSE" || tok == "NOLUMPPTL") {
                    argLumpPtl = false;
                } else if (tok.rfind("MINPTL=", 0) == 0) {
                    argLumpPtl = true;
                    argMinPtl  = std::abs(parse_param(tok.substr(7), parameters));
                } else if (tok == "EVENTS" || tok == "EVENTS=1" || tok == "EVENTS=TRUE") {
                    argEvents = true;
                } else if (tok == "EVENTS=0" || tok == "EVENTS=FALSE" || tok == "NOEVENTS") {
                    argEvents     = false;
                    argEventsOnly = false;
                } else if (tok.rfind("EVENTS=", 0) == 0) {
                    argEvents              = true;
                    const std::string val  = tok.substr(7);
                    const uint32_t    mTry = parse_event_metrics(val, EVENT_METRIC_NONE);
                    if (mTry != EVENT_METRIC_NONE && val.find('.') == std::string::npos
                        && val.find('/') == std::string::npos && val.find('\\') == std::string::npos) {
                        argEventMetrics = parse_event_metrics(val, argEventMetrics);
                    } else {
                        eventsFile = resolve_path(val);
                    }
                } else if (tok == "EVENTSONLY" || tok == "EVENTSONLY=1" || tok == "EVENTSONLY=TRUE") {
                    argEvents     = true;
                    argEventsOnly = true;
                } else if (tok == "EVENTSONLY=0" || tok == "EVENTSONLY=FALSE") {
                    argEventsOnly = false;
                } else if (tok.rfind("EVENTSONLY=", 0) == 0) {
                    argEvents              = true;
                    argEventsOnly          = true;
                    const std::string val  = tok.substr(11);
                    const uint32_t    mTry = parse_event_metrics(val, EVENT_METRIC_NONE);
                    if (mTry != EVENT_METRIC_NONE && val.find('.') == std::string::npos
                        && val.find('/') == std::string::npos && val.find('\\') == std::string::npos) {
                        argEventMetrics = parse_event_metrics(val, argEventMetrics);
                    } else {
                        eventsFile = resolve_path(val);
                    }
                } else if (tok == "EVENTMETRICS" || tok == "EVENTPULSE" || tok == "EVENTPULSE=1"
                           || tok == "EVENTPULSE=TRUE") {
                    argEvents       = true;
                    argEventMetrics = parse_event_metrics("PULSE", argEventMetrics);
                } else if (tok == "EVENTMETRICS=0" || tok == "EVENTMETRICS=FALSE" || tok == "EVENTMETRICS=NONE"
                           || tok == "EVENTPULSE=0" || tok == "EVENTPULSE=FALSE" || tok == "NOEVENTMETRICS") {
                    argEventMetrics = EVENT_METRIC_NONE;
                } else if (tok.rfind("EVENTMETRICS=", 0) == 0) {
                    argEvents       = true;
                    argEventMetrics = parse_event_metrics(tok.substr(13), argEventMetrics);
                } else if (tok.rfind("EVENTPULSE=", 0) == 0) {
                    argEvents       = true;
                    argEventMetrics = parse_event_metrics(tok.substr(11), argEventMetrics);
                } else if (tok == "ROLLBACK" || tok == "ROLLBACK=1" || tok == "ROLLBACK=TRUE"
                           || tok == "ROLLBACK=WINDOW") {
                    argRollback = 1;
                } else if (tok == "ROLLBACK=2" || tok == "ROLLBACK=STAY") {
                    argRollback = 2;
                } else if (tok == "NOROLLBACK" || tok == "ROLLBACK=0" || tok == "ROLLBACK=FALSE") {
                    argRollback = 0;
                }
            }
        }
    }
    for (const auto& s : cliSweeps) {
        parse_sweep_spec(s);
    }
    for (const auto& ms : cliMarginSpecs) {
        parse_margin_spec(ms);
    }
    for (const auto& as : cliAssertions) {
        parse_assert_spec(as);
    }
    for (const auto& ms : cliMeasurements) {
        parse_meas_spec(ms);
    }
    for (const auto& ic : cliInitialConditions) {
        parse_ic_spec(ic);
    }
    for (const auto& js : cliJitterSpecs) {
        parse_jitter_spec(js);
    }
    for (const auto& ss : cliSensSpecs) {
        parse_sens_spec(ss);
    }
    bool hasWaveformMeas = false;
    for (const auto& ms : measurements) {
        if (ms.kind == MeasKind::VPEAK || ms.kind == MeasKind::FWHM || ms.kind == MeasKind::DT_PI
            || ms.kind == MeasKind::AREA) {
            argEventMetrics |= EVENT_METRIC_ALL;
        }
        if (ms.needs_waveform()) { hasWaveformMeas = true; }
    }
    if (argMargins && marginSpecs.empty()) {
        MarginSpec defaultSpec;
        defaultSpec.targets = {"@DEFAULT"};
        marginSpecs.push_back(std::move(defaultSpec));
    }
    if (argSens && sensSpecs.empty()) {
        SensSpec defaultSens;
        defaultSens.targets = {"@DEVICES"};
        sensSpecs.push_back(std::move(defaultSens));
    }
    if (!sweeps.empty() && !argMargins && !argSens) {
        int64_t totalSweepRuns = 1;
        for (const auto& sd : sweeps) {
            totalSweepRuns *= static_cast<int64_t>(sd.values.size());
        }
        argMcRuns = std::max<int64_t>(1, totalSweepRuns);
        argSweep  = true;
    }
    bool hasFileControl = false;
    for (const auto& c : controls) {
        if (!c.empty() && c.front() == "FILE") {
            hasFileControl = true;
            break;
        }
    }
    if ((argMargins || argSens || verifyEventsFile.has_value() || !assertions.empty() || argCheckLoops
         || !jitterSpecs.empty() || (!measurements.empty() && !hasWaveformMeas))
        && !cli_output_file.has_value() && !hasFileControl && !argGpu && argAnal == AnalysisType::Phase
        && !hasWaveformMeas) {
        argEventsOnly = true;
        if ((argMargins || argSens) && !argExplicitNodal) {
            argNodal = true;
#ifdef _OPENMP
            argPar   = true;
#endif
        }
    }
    if (argEvents) {
        if (!cli_output_file.has_value() && !hasFileControl && !hasWaveformMeas) { argEventsOnly = true; }
        if (!eventsFile.has_value()) {
            if (cli_output_file.has_value()) {
                if (argEventsOnly) {
                    eventsFile = cli_output_file->name();
                } else {
                    std::filesystem::path p(cli_output_file->name());
                    std::string           ext      = p.extension().string();
                    std::string           extUpper = ext;
                    std::transform(extUpper.begin(), extUpper.end(), extUpper.begin(), ::toupper);
                    const std::string evExt = (extUpper == ".CSV" || extUpper == ".DAT") ? ext : ".npy";
                    p.replace_extension(".events" + evExt);
                    eventsFile = p.string();
                }
            } else {
                eventsFile = std::filesystem::current_path().append("events.npy").string();
            }
        }
    }
}

std::optional<uint64_t> Input::find_seed_option() const {
    for (const auto& c : controls) {
        if (c.empty()) { continue; }
        if (c.front() != "OPTION") { continue; }

        for (size_t k = 1; k < c.size(); ++k) {
            std::string tok = c.at(k);

            // SEED=123
            if (tok.rfind("SEED=", 0) == 0) {
                tok = tok.substr(5);
            }
            // SEED 123
            else if (tok == "SEED" && (k + 1) < c.size()) {
                tok = c.at(++k);
            } else {
                continue;
            }

            // Allow AUTO to mean "random seed"
            if (tok == "AUTO") { return std::nullopt; }

            // First try integer literal:
            uint64_t u{};
            auto [p, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), u);
            if (ec == std::errc{} && p == tok.data() + tok.size()) { return u; }

            // Fallback: expression -> double (but then RANGE CHECK!)
            double dv = parse_param(tok, parameters);
            if (!std::isfinite(dv) || dv < 0.0 || dv > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
                Errors::input_errors(InputErrors::UNKNOWN_CONTROL, "OPTION SEED");
            }

            // If you want only integral seeds:
            double iv = std::floor(dv);
            if (iv != dv) { Errors::input_errors(InputErrors::UNKNOWN_CONTROL, "OPTION SEED"); }

            return static_cast<uint64_t>(iv);
        }
    }
    return std::nullopt;
}
