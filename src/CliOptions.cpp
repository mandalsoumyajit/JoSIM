// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/CliOptions.hpp"

#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/TypeDefines.hpp"

#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

using namespace JoSIM;

tokens_t CliOptions::argv_to_tokens(const int64_t& argc, const char** argv) {
    // Tokens variable of size argc to store arguments
    tokens_t tokens(argc - 1);
    // If argument count is less or equal to one
    if (argc <= 1) {
        // Complain that josim was run without arguments
        Errors::cli_errors(CLIErrors::TOO_FEW_ARGUMENTS);
    }
    // Loop through arguments, starting at 1 (0 is josim-cli)
    for (int64_t i = 1; i < argc; ++i) {
        // Add the arguments at the appropriate position
        tokens.at(i - 1) = std::string(argv[i]);
    }
    // Return the tokens
    return tokens;
}

vector_pair_t<char_o, string_o> CliOptions::argument_pairs(const tokens_t& tokens) {
    // Variable to store the argument pairs
    vector_pair_t<char_o, string_o> ap;
    // Loop through the argument tokens
    for (auto t : tokens) {
        if (t.empty()) { continue; }
        // If the string is a switch, remove leading '-' chars and split on the first '=' if present
        if (t.at(0) == '-') {
            size_t      eqPos = t.find('=');
            std::string swStr = (eqPos != std::string::npos) ? t.substr(0, eqPos) : t;
            string_o    val   = (eqPos != std::string::npos) ? string_o(t.substr(eqPos + 1)) : std::nullopt;
            swStr.erase(std::remove(swStr.begin(), swStr.end(), '-'), swStr.end());
            if (swStr.empty()) { Errors::cli_errors(CLIErrors::UNKNOWN_SWITCH, t); }
            std::string swLower = swStr;
            std::transform(swLower.begin(), swLower.end(), swLower.begin(), ::tolower);
            char swChar = swStr.at(0);
            if (swLower.rfind("eventmetric", 0) == 0 || swLower.rfind("eventpulse", 0) == 0) {
                swChar = 'E';
            } else if (swLower.rfind("sweep", 0) == 0) {
                swChar = 'S';
            }
            ap.emplace_back(std::make_pair(swChar, val));
        } else {
            if (!ap.empty() && !ap.back().second) {
                if (t.at(0) == '=' && t.size() > 1) {
                    ap.back().second = t.substr(1);
                } else {
                    ap.back().second = t;
                }
            } else {
                ap.emplace_back(std::make_pair(std::nullopt, t));
            }
        }
    }
    // Sanity check: if no positional input file was found and the last switch swallowed the input filename
    bool hasInput = false;
    for (const auto& kv : ap) {
        if (!kv.first || kv.first.value() == 'i') {
            hasInput = true;
            break;
        }
    }
    if (!hasInput && !ap.empty() && ap.back().first && ap.back().second) {
        const char sw = ap.back().first.value();
        if (sw == 'p' || sw == 'g' || sw == 'n' || sw == 'f' || sw == 'w' || sw == 'm' || sw == 'e' || sw == 'E'
            || ap.size() == 1) {
            ap.emplace_back(std::make_pair(std::nullopt, ap.back().second));
            ap.at(ap.size() - 2).second = std::nullopt;
        }
    }
    return ap;
}

CliOptions CliOptions::parse(int64_t argc, const char** argv) {
    // Variable where all the CLI options will be stored
    CliOptions                      out;
    bool                            explicit_stdin_input = false; // Only used to suppress warning
    // Parse and generate argument pairs from tokens
    vector_pair_t<char_o, string_o> ap                   = out.argument_pairs(out.argv_to_tokens(argc, argv));
    // Loop through the argument pairs
    for (auto i : ap) {
        // If the argument is not accompanied by any switch
        if (!i.first) {
            // This argument is the file name
            out.cir_file_name = i.second;
            auto path         = std::filesystem::path(i.second.value());
            // Else if the switch is not empty
        } else {
            // Use the char in a case statment
            switch (i.first.value()) {
                    // Set analysis type
                case 'a':
                    if (!i.second) {
                        Errors::cli_errors(CLIErrors::NO_ANALYSIS);
                    } else if (i.second.value() == "0") {
                        out.analysis_type = AnalysisType::Voltage;
                    } else if (i.second.value() == "1") {
                        out.analysis_type = AnalysisType::Phase;
                    } else {
                        Errors::cli_errors(CLIErrors::INVALID_ANALYSIS);
                    }
                    break;
                    // Batched Monte Carlo / .spread trials
                case 'b':
                    try {
                        out.batchRuns = std::max<int64_t>(1, std::stoll(i.second.value_or("1")));
                    } catch (const std::invalid_argument&) {
                        out.batchRuns = 1;
                    }
                    break;
                    // Enable predictor-corrector iterations for JJs
                case 'c':
                    try {
                        out.corrector = std::stoll(i.second.value_or("1"));
                    } catch (const std::invalid_argument&) {
                        out.corrector = 1;
                    }
                    break;
                    // Extract SFQ switching events
                case 'e':
                    if (!i.second) {
                        out.events = true;
                    } else if (i.second.value() == "0" || i.second.value() == "false" || i.second.value() == "FALSE") {
                        out.events = false;
                    } else if (i.second.value() == "1" || i.second.value() == "true" || i.second.value() == "TRUE") {
                        out.events = true;
                    } else {
                        out.events      = true;
                        out.events_file = i.second.value();
                    }
                    break;
                    // Select SFQ pulse metrics (t_peak, v_peak, fwhm, dt_pi, area, time=peak)
                case 'E':
                    out.events        = true;
                    out.event_metrics = i.second.value_or("pulse");
                    break;
                    // Disable automatic step halving (fixed step)
                case 'f':
                    out.fixedStep = (i.second.value_or("1") != "0");
                    break;
                    // Enable GPU acceleration (and parallel partitioned mode)
                case 'g':
                    if (!i.second) {
                        out.gpu      = true;
                        out.gpuMode  = 1;
                        out.parallel = true;
                    } else {
                        try {
                            const int64_t gval = std::stoll(i.second.value());
                            out.gpu            = (gval != 0);
                            out.gpuMode        = gval;
                            if (out.gpu) { out.parallel = true; }
                            if (gval == 4) { out.nodal = true; }
                        } catch (const std::invalid_argument&) {
                            out.gpu      = true;
                            out.gpuMode  = 1;
                            out.parallel = true;
                        }
                    }
                    break;
                    // Enable Schur-complement nodal admittance solver
                case 'n':
                    out.nodal = (i.second.value_or("1") != "0");
                    break;
                    // Show help menu
                case 'h':
                    display_help();
                    exit(0);
                    // Set input to standard input
                case 'i':
                    out.cir_file_name    = std::nullopt;
                    explicit_stdin_input = true;
                    break;
                    // Enable minimal reporting
                case 'm': try { out.minimal = std::stoi(i.second.value_or("1"));
                    } catch (const std::invalid_argument&) {
                        Errors::cli_errors(CLIErrors::INVALID_MINIMAL, i.second.value());
                    }
                    break;
                    // Set output file
                case 'o':
                    // If no file name is specified but output is specified
                    if (!i.second) {
                        // Store the output in a file called output.csv at cwd
                        out.output_file = OutputFile(std::filesystem::current_path().append("output.csv").string());
                        // If a file name was given
                    } else {
                        out.output_file = OutputFile(i.second.value());
                    }
                    break;
                    // Enable parallel processing (EXPERIMENTAL)
                case 'p':
#ifdef _OPENMP
                    out.parallel = true;
                    if (!out.minimal) { std::cout << "Parallelization is ENABLED" << std::endl; }
#else
                    if (!out.minimal) { std::cout << "Parallelization is DISABLED" << std::endl; }
#endif
                    break;
                    // Sanity check input circuit netlist
                case 's':
                    out.sanityCheck = true;
                    if (i.second) {
                        std::transform(
                                i.second.value().begin(), i.second.value().end(), i.second.value().begin(), toupper);
                        out.sanityCheckSubckts.insert(i.second.value());
                    }
                    break;
                    // Deterministic parameter / margin sweep
                case 'S':
                    if (i.second && !i.second.value().empty()) {
                        out.sweeps.push_back(i.second.value());
                    }
                    break;
                    // Enable verbose mode
                case 'V': try { out.verbose = std::stoi(i.second.value_or("1"));
                    } catch (const std::invalid_argument&) {
                        Errors::verbosity_errors(VerbosityErrors::INVALID_VERBOSITY_LEVEL, i.second.value());
                    }
                    break;
                    // Show version info
                case 'v':
                    exit(0);
                    break;
                    // Unknown switch was specified
                default: Errors::cli_errors(CLIErrors::UNKNOWN_SWITCH, std::string(1, i.first.value()));
            }
        }
    }
    // Do a few sanity checks
    // If the input was not specified
    if (!out.cir_file_name && !(out.minimal && explicit_stdin_input)) {
        // Complain and assume standard input
        // Suppress warning if -i AND -m options were given
        Errors::cli_errors(CLIErrors::NO_INPUT);
    }
    // If an output file and an input file name were specified
    if (out.output_file && out.cir_file_name) {
        // But they match each other
        if (out.output_file.value().name() == out.cir_file_name.value()) {
            // Complain and exit since continuing will overwrite input file
            Errors::cli_errors(CLIErrors::INPUT_SAME_OUTPUT);
        }
    }
    // If minimal flag is not specified
    if (!out.minimal) {
        // Print to screen the input and output locations
        if (out.cir_file_name) {
            std::cout << "Input: " << out.cir_file_name.value() << std::endl;
        } else {
            std::cout << "Input: "
                      << "standard input" << std::endl;
        }
        std::cout << std::endl;
        if (out.output_file) {
            std::cout << "Output: " << out.output_file.value().name() << std::endl;
            std::cout << std::endl;
        }
    }

    return out;
}

void CliOptions::display_help() {
    // Format the help interface
    // Preamble
    std::cout << "JoSIM help interface\n";
    std::cout << "====================\n";
    // Analysis type
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-a" << std::setw(3) << std::left << "|"
              << "Specifies the analysis type." << std::endl;
    std::cout << std::setw(16) << std::left << "--analysis=" << std::setw(3) << std::left << "|"
              << "0 for Voltage analysis." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "1 for Phase analysis (Default)." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Batched Monte Carlo
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-b" << std::setw(3) << std::left << "|"
              << "Executes N batched Monte Carlo (.spread) or parameter sweep (.sweep) trials." << std::endl;
    std::cout << std::setw(16) << std::left << "--batch=" << std::setw(3) << std::left << "|"
              << "Combine with -p (OpenMP CPU), -n (Nodal CPU), or -g (CUDA GPU), e.g. -p -b 256." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Corrector iterations
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-c" << std::setw(3) << std::left << "|"
              << "Enables implicit predictor-corrector iterations for JJs." << std::endl;
    std::cout << std::setw(16) << std::left << "--corrector=" << std::setw(3) << std::left << "|"
              << "Specify max iterations per step (default 2 if -c is given without value)." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // SFQ switching events
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-e" << std::setw(3) << std::left << "|"
              << "Extracts 2*pi SFQ phase-slip switching events to a compact .npy or .csv file." << std::endl;
    std::cout << std::setw(16) << std::left << "--events=" << std::setw(3) << std::left << "|"
              << "When used without -o, runs in zero-allocation events-only mode." << std::endl;
    std::cout << std::setw(16) << std::left << "-E" << std::setw(3) << std::left << "|"
              << "Selects additional SFQ pulse metrics: pulse (default: t_peak,v_peak,fwhm,dt_pi)," << std::endl;
    std::cout << std::setw(16) << std::left << "--event-metrics=" << std::setw(3) << std::left << "|"
              << "all (+area), or comma-separated list including time=peak / time=cross." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Fixed timestep
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-f" << std::setw(3) << std::left << "|"
              << "Disables automatic timestep halving when JJ phase jumps exceed 0.4*pi." << std::endl;
    std::cout << std::setw(16) << std::left << "--fixed-step" << std::setw(3) << std::left << "|" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // GPU acceleration
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-g" << std::setw(3) << std::left << "|"
              << "Enables NVIDIA CUDA GPU solver (requires -DUSING_CUDA=ON)." << std::endl;
    std::cout << std::setw(16) << std::left << "--gpu=" << std::setw(3) << std::left << "|"
              << "1 (Default): Auto-dispatch (Windowed PTL / cuSolverRF / Hybrid)." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "2: Hybrid Device-GPU + Host-KLU solver." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "3: Fully GPU-Resident cuSolverRF Monolithic MNA solver." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "4: Fully GPU-Resident cuSolverRF Nodal Admittance solver." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Help menu
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-h" << std::setw(3) << std::left << "|"
              << "Displays this help menu" << std::endl;
    std::cout << std::setw(16) << std::left << "--help" << std::setw(3) << std::left << "|"
              << " " << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Input from standard in
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-i" << std::setw(3) << std::left << "|"
              << "Input circuit netlist from standard input." << std::endl;
    std::cout << std::setw(16) << std::left << "--input" << std::setw(3) << std::left << "|"
              << "This command ignores any additional input file specified." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "Reads from standard input until the "
              << ".end command or EOF character is specified." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Minimal output
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-m" << std::setw(3) << std::left << "|"
              << "Disables most output." << std::endl;
    std::cout << std::setw(16) << std::left << "--minimal" << std::setw(3) << std::left << "|"
              << " " << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Nodal admittance solver
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-n" << std::setw(3) << std::left << "|"
              << "Enables Schur-Complement Nodal Admittance solver." << std::endl;
    std::cout << std::setw(16) << std::left << "--nodal" << std::setw(3) << std::left << "|"
              << "Eliminates internal branch rows (JJ/R/C/L/T/V/P) to shrink the matrix" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "by 2x-6x (1.5x-4.4x faster on monolithic circuits; runs on 1 thread" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "by default, or combine with -p as '-p -n')." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Output
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-o" << std::setw(3) << std::left << "|"
              << "Specify output file for simulation results." << std::endl;
    std::cout << std::setw(16) << std::left << "--output=" << std::setw(3) << std::left << "|"
              << "Supported extensions: .csv, .dat, .raw, .bin, .npy, .rawb" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "Use .npy or .bin for 20x-37x faster I/O on many-trace decks." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Parallel Processing
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-p" << std::setw(3) << std::left << "|"
              << "Enables OpenMP multi-core parallelization (requires -DUSING_OPENMP=ON)." << std::endl;
    std::cout << std::setw(16) << std::left << "--parallel" << std::setw(3) << std::left << "|"
              << "Activates multi-step windowed PTL block-parallel solving (P >= 4)," << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "parallel device evaluation on large meshes, and batched Monte Carlo (-b)." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Sanity check switch
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-s" << std::setw(3) << std::left << "|"
              << "(EXPERIMENTAL) Sanity check circuit netlist (main design). "
              << "Subcircuits can also be checked if specified." << std::endl;
    std::cout << std::setw(16) << std::left << "--sanitycheck" << std::setw(3) << std::left << "|"
              << "Multiple subcircuits can be specified by providing CLI option "
              << "multiple times. (e.g. -s SUBCKT1 -s SUBCKT2)" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|"
              << "Checks every superconductor circuit cell node connects to exactly "
              << "one other node, assuming the design denotes high-level connections." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Parameter / margin sweep
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-S" << std::setw(3) << std::left << "|"
              << "Specifies a deterministic parameter or margin sweep (1D or 2D Cartesian)." << std::endl;
    std::cout << std::setw(16) << std::left << "--sweep=" << std::setw(3) << std::left << "|"
              << "Examples: -S @BIAS:MARGIN:0.3:21, -S @BIAS:0.7:1.3:0.05, -S Scaling:0.8:1.2:9." << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Verbose switch
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-V" << std::setw(3) << std::left << "|"
              << "Runs JoSIM in verbose mode." << std::endl;
    std::cout << std::setw(16) << std::left << "--Verbose=" << std::setw(3) << std::left << "|"
              << "Defaults to minimal(1), can be medium(2) or maximum(3)" << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;
    // Version info
    // ---------------------------------------------------------------------------
    std::cout << std::setw(16) << std::left << "-v" << std::setw(3) << std::left << "|"
              << "Displays the JoSIM version info only." << std::endl;
    std::cout << std::setw(16) << std::left << "--version" << std::setw(3) << std::left << "|"
              << " " << std::endl;
    std::cout << std::setw(16) << std::left << "  " << std::setw(3) << std::left << "|" << std::endl;

    // ---------------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "Example commands:\n";
    std::cout << "  Single-thread nodal (general/monolithic): josim-cli -n -o out.npy test.cir\n";
    std::cout << "  Multi-core OpenMP + nodal fallback:       josim-cli -p -n -o out.npy test.cir\n";
    std::cout << "  CUDA GPU auto-dispatch:                   josim-cli -g -o out.npy test.cir\n";
    std::cout << "  Batched Monte Carlo (256 trials, OpenMP): josim-cli -p -b 256 -o mc.npy test.cir\n";
    std::cout << std::endl;
    exit(0);
}

void CliOptions::version_info() {
    std::cout << std::endl;
    std::cout << "JoSIM: Josephson Junction Superconductive SPICE Circuit Simulator" << std::endl;
    std::cout << "Copyright (C) 2020 by Johannes Delport (jdelport@sun.ac.za)" << std::endl;
    std::cout << "v" << VERSION << "." << GIT_COMMIT_HASH << " compiled on " << __DATE__ << " at " << __TIME__
              << std::endl;
#ifndef NDEBUG
    std::cout << "(Debug)" << std::endl;
#endif
    std::cout << std::endl;
}
