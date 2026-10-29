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
                // Preserve file path case for state checkpoint commands
            } else if (tokens.at(0) == ".SAVESTATE" || tokens.at(0) == "SAVESTATE" || tokens.at(0) == ".SAVEBIAS"
                       || tokens.at(0) == "SAVEBIAS" || tokens.at(0) == ".LOADSTATE" || tokens.at(0) == "LOADSTATE"
                       || tokens.at(0) == ".LOADBIAS" || tokens.at(0) == "LOADBIAS") {
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
                // Transform the entire line to upper case
                std::transform(tokens.begin() + 1, tokens.end(), tokens.begin() + 1, [](std::string& c) -> std::string {
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
    // srand(time(NULL));
    //  Create a seperate thread that will be used for printing creation progress
    std::thread           printingThread;
    // Variable to store the tokenized input
    std::vector<tokens_t> fileLines;
    // Create a optional string to store subcircuit names
    string_o              subckt;
    // Create boolean to determine if in a control block
    bool                  control = false;
    // If a file name is provided then input is from file
    if (fileName) {
        // Create a stream input from file
        FileInput input(fileName.value());
        // Use stream input to read lines and return tokenized input
        fileLines = Input::read_input(input, fileName);
        // If no filename then input from standard input
    } else {
        // Create standard input stream
        ConsoleInput input;
        // Use stream input to read lines and return tokenized input
        fileLines = Input::read_input(input);
    }
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
    netlist.argMin = argMin;
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
                        // Populate the a subcircuit at given name with its IO nodes
                        for (auto j = 2; j < fileLines.at(i).size(); ++j) {
                            netlist.subcircuits[subckt.value()].io.push_back(fileLines.at(i).at(j));
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
                                  "FILE",
                                  "IV",
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
    SweepDim dim;
    dim.target = c.at(1);
    std::transform(dim.target.begin(), dim.target.end(), dim.target.begin(), ::toupper);

    tokens_t t = c;
    for (auto& s : t) {
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    }

    size_t startIdx  = 2;
    int64_t explicitN = 0;
    if (t.at(2) == "MARGIN" || t.at(2) == "REL" || t.at(2) == "SCALE" || t.at(2) == "MULT") {
        dim.mode       = SweepMode::Margin;
        dim.isRelative = true;
        startIdx       = 3;
    } else if (t.at(2) == "LIST") {
        dim.mode       = SweepMode::List;
        dim.isRelative = (!dim.target.empty() && dim.target.front() == '@');
        startIdx       = 3;
    } else if (t.at(2) == "LIN") {
        dim.mode       = SweepMode::Linear;
        dim.isRelative = (!dim.target.empty() && dim.target.front() == '@');
        startIdx       = 3;
    } else {
        dim.mode       = SweepMode::Linear;
        dim.isRelative = (!dim.target.empty() && dim.target.front() == '@');
        startIdx       = 2;
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
    for (char& ch : cleaned) {
        if (ch == ':' || ch == ',' || ch == ';') { ch = ' '; }
    }
    tokens_t toks = Misc::tokenize(cleaned);
    if (toks.empty()) { return; }
    std::string firstUpper = toks.front();
    std::transform(firstUpper.begin(), firstUpper.end(), firstUpper.begin(), ::toupper);
    if (firstUpper != "SWEEP") {
        toks.insert(toks.begin(), "SWEEP");
    }
    parse_sweep_tokens(toks);
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
                } else if (tok == "NODAL" || tok == "NODAL=1" || tok == "NODAL=TRUE") {
                    argNodal = true;
                } else if (tok == "NODAL=0" || tok == "NODAL=FALSE") {
                    argNodal = false;
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
    if (!sweeps.empty()) {
        int64_t totalSweepRuns = 1;
        for (const auto& sd : sweeps) {
            totalSweepRuns *= static_cast<int64_t>(sd.values.size());
        }
        argMcRuns = std::max<int64_t>(1, totalSweepRuns);
        argSweep  = true;
    }
    if (argEvents) {
        bool hasFileControl = false;
        for (const auto& c : controls) {
            if (!c.empty() && c.front() == "FILE") {
                hasFileControl = true;
                break;
            }
        }
        if (!cli_output_file.has_value() && !hasFileControl) { argEventsOnly = true; }
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
