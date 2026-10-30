// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_INPUT_HPP
#define JOSIM_INPUT_HPP

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/CliOptions.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/FileOutputType.hpp"
#include "JoSIM/LineInput.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Netlist.hpp"
#include "JoSIM/ParameterName.hpp"
#include "JoSIM/Transient.hpp"
#include "JoSIM/TypeDefines.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace JoSIM {

enum EventMetricFlags : uint32_t {
    EVENT_METRIC_NONE      = 0u,
    EVENT_METRIC_T_PEAK    = 1u << 0,
    EVENT_METRIC_V_PEAK    = 1u << 1,
    EVENT_METRIC_FWHM      = 1u << 2,
    EVENT_METRIC_DT_PI     = 1u << 3,
    EVENT_METRIC_AREA      = 1u << 4,
    EVENT_METRIC_TIME_PEAK = 1u << 5,
    EVENT_METRIC_PULSE     = EVENT_METRIC_T_PEAK | EVENT_METRIC_V_PEAK | EVENT_METRIC_FWHM | EVENT_METRIC_DT_PI,
    EVENT_METRIC_ALL       = EVENT_METRIC_PULSE | EVENT_METRIC_AREA
};

enum class SweepMode {
    Linear,
    Margin,
    List
};

struct SweepDim {
    std::string         target;
    SweepMode           mode       = SweepMode::Linear;
    bool                isRelative = false;
    std::vector<double> values;
};

class Input {
  public:
    Netlist                                      netlist;
    Transient                                    transSim;
    std::optional<double>                        globalTemp;
    double                                       neB = 1E12;
    std::vector<tokens_t>                        fileLines, controls;
    std::vector<std::string>                     relevantX;
    std::unordered_map<ParameterName, Parameter> parameters;
    std::optional<OutputFile>                    cli_output_file;
    std::vector<OutputFile>                      output_files;
    std::optional<std::string>                   fileParentPath;
    AnalysisType                                                 argAnal;
    int64_t                                                      argVerb;
    bool                                                         argMin;
    bool                                                         argPar          = false;
    bool                                                         argGpu          = false;
    int64_t                                                      argGpuMode      = 1;
    bool                                                         argNodal        = false;
    int64_t                                                      argNodalLevel   = 0;
    bool                                                         argBypass       = false;
    double                                                       argBypassTol    = 1e-6;
    bool                                                         argFixedStep    = false;
    bool                                                         argWarnStep     = false;
    int64_t                                                      argRollback     = 1;
    bool                                                         argTxInterp     = false;
    int64_t                                                      argCorrector    = 0;
    double                                                       argCorrectorTol = 1e-3;
    double                                                       argKCutoff      = 0.0;
    int64_t                                                      argMcRuns       = 1;
    bool                                                         argMcAll        = false;
    bool                                                         argSweep        = false;
    std::vector<SweepDim>                                        sweeps;
    std::vector<std::string>                                     cliSweeps;
    bool                                                         argLumpPtl      = true;
    double                                                       argMinPtl       = 0.0;
    bool                                                         ptlLumpedDone   = false;
    bool                                                         argEvents       = false;
    bool                                                         argEventsOnly   = false;
    uint32_t                                                     argEventMetrics = EVENT_METRIC_NONE;
    std::optional<std::string>                                   eventsFile;
    std::optional<std::pair<std::string, std::optional<double>>> saveStateFile;
    std::optional<std::string>                                   loadStateFile;

    Input(AnalysisType at = AnalysisType::Phase, int64_t verb = 0, bool min = false, bool par = false)
        : argAnal(at), argVerb(verb), argMin(min), argPar(par) {};

    Input(CliOptions& cli_options) {
        argAnal                    = cli_options.analysis_type;
        argVerb                    = cli_options.verbose;
        argMin                     = cli_options.minimal;
        argPar                     = cli_options.parallel;
        argGpu                     = cli_options.gpu;
        argGpuMode                 = cli_options.gpuMode;
        argNodal                   = cli_options.nodal;
        argFixedStep               = cli_options.fixedStep;
        argEvents                  = cli_options.events;
        eventsFile                 = cli_options.events_file;
        if (cli_options.event_metrics.has_value()) {
            argEventMetrics = parse_event_metrics(cli_options.event_metrics.value(), argEventMetrics);
        }
        cliSweeps                  = cli_options.sweeps;
        argCorrector               = cli_options.corrector;
        argMcRuns                  = cli_options.batchRuns;
        cli_output_file            = cli_options.output_file;
        netlist.sanityCheck        = cli_options.sanityCheck;
        netlist.sanityCheckSubckts = cli_options.sanityCheckSubckts;
    }

    static uint32_t       parse_event_metrics(const std::string& spec, uint32_t currentMask = EVENT_METRIC_NONE);
    void                  parse_sweep_tokens(const tokens_t& c);
    void                  parse_sweep_spec(const std::string& spec);
    std::vector<tokens_t> read_input(LineInput& input, string_o fileName = std::nullopt);
    void                  parse_input(string_o fileName = std::nullopt);
    void                  syntax_check_controls(std::vector<tokens_t>& controls);
    void                  parse_options_and_state_controls();

  private:
    std::optional<uint64_t> find_seed_option() const;
};
} // namespace JoSIM

#endif // JOSIM_INPUT_HPP
