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

struct MarginSpec {
    std::vector<std::string> targets;
    double                   minVal  = 0.5;
    double                   maxVal  = 1.5;
    int64_t                  numPts  = 21;
    bool                     hasStep = false;
    double                   stepVal = 0.05;
    std::optional<double>    tol;
};

struct MarginTarget {
    std::string          target;
    std::string          category;
    std::string          displayName;
    double               nominalValue  = 1.0;
    std::string          unit;
    std::vector<double>  values;
    std::vector<int64_t> trialIndices;
    size_t               nomPointIdx   = 0;
    size_t               minusPointIdx = 0;
    size_t               plusPointIdx  = 0;
    double               delta         = 0.05;
    bool                 isDeviceOnly  = false;
};

struct SensSpec {
    std::vector<std::string> targets;
    double                   delta      = 0.05;
    std::string              metric;
    bool                     hasGoal    = false;
    double                   goal       = 0.0;
    bool                     center     = false;
    double                   minVal     = 0.5;
    double                   maxVal     = 1.5;
    int64_t                  numPts     = 21;
    bool                     hasStep    = false;
    double                   stepVal    = 0.05;
    std::optional<double>    tol;
};

enum class AssertKind {
    BITSTREAM = 0,
    COUNT     = 1,
    TIMES     = 2,
    AFTER     = 3
};

struct AssertSpec {
    AssertKind          kind       = AssertKind::BITSTREAM;
    std::string         rawLine;
    std::string         trace;
    std::string         refTrace;
    std::string         bits;
    int64_t             expCount   = 0;
    std::vector<double> expTimes;
    double              td         = 0.0;
    double              per        = 0.0;
    double              window     = 0.0;
    double              tol        = -1.0;
    double              delay      = 0.0;
    double              tFrom      = -1.0;
    double              tTo        = -1.0;
};

enum class MeasKind {
    TRIG_TARG = 0,
    AVG       = 1,
    RMS       = 2,
    MIN       = 3,
    MAX       = 4,
    MIN_AT    = 5,
    MAX_AT    = 6,
    PP        = 7,
    INTEG     = 8,
    FIND_AT   = 9,
    FIND_WHEN = 10,
    WHEN      = 11,
    DERIV     = 12,
    PARAM     = 13,
    COUNT     = 14,
    FREQ      = 15,
    VPEAK     = 16,
    FWHM      = 17,
    DT_PI     = 18,
    AREA      = 19
};

enum class MeasCrossDir {
    CROSS = 0,
    RISE  = 1,
    FALL  = 2
};

struct MeasTrigger {
    bool         active      = false;
    std::string  trace;
    std::string  rhsTrace;
    bool         isSfqEvent  = false;
    int64_t      sfqIndex    = 1;
    double       val         = 0.0;
    double       td          = 0.0;
    MeasCrossDir dir         = MeasCrossDir::CROSS;
    int64_t      count       = 1;
    bool         isFixedTime = false;
    double       fixedTime   = 0.0;

    bool needs_waveform() const {
        return active && !isSfqEvent && !isFixedTime;
    }
};

struct MeasSpec {
    MeasKind     kind        = MeasKind::TRIG_TARG;
    std::string  name;
    std::string  rawLine;
    std::string  trace;
    std::string  expr;
    MeasTrigger  trig;
    MeasTrigger  targ;
    bool         hasFrom     = false;
    double       tFrom       = 0.0;
    bool         hasTo       = false;
    double       tTo         = 0.0;
    double       atTime      = 0.0;
    bool         hasSfqIndex = false;
    int64_t      sfqIndex    = 1;
    bool         hasMinGoal  = false;
    double       minGoal     = 0.0;
    bool         hasMaxGoal  = false;
    double       maxGoal     = 0.0;
    bool         hasValGoal  = false;
    double       valGoal     = 0.0;
    double       goalTol     = 1e-9;

    bool needs_waveform() const {
        if (kind == MeasKind::PARAM) { return false; }
        if (kind == MeasKind::COUNT || kind == MeasKind::FREQ || kind == MeasKind::VPEAK
            || kind == MeasKind::FWHM || kind == MeasKind::DT_PI || kind == MeasKind::AREA) {
            return trig.needs_waveform() || targ.needs_waveform();
        }
        if (kind == MeasKind::TRIG_TARG) {
            return trig.needs_waveform() || targ.needs_waveform();
        }
        if (kind == MeasKind::WHEN) {
            return trig.needs_waveform();
        }
        return true;
    }
};

enum class IcTargetKind {
    NODE_VOLTAGE,
    NODE_PHASE,
    BRANCH_CURRENT,
    DEV_PHASE,
    DEV_VOLTAGE
};

struct IcSpec {
    IcTargetKind kind  = IcTargetKind::NODE_VOLTAGE;
    std::string  target;
    double       value = 0.0;
};

enum class JitterMode {
    ENSEMBLE,
    REF_TRACE,
    CLOCK_GRID
};

struct JitterSpec {
    JitterMode  mode        = JitterMode::ENSEMBLE;
    std::string name;
    std::string trace;
    std::string refTrace;
    std::string bits;
    double      td          = 0.0;
    bool        hasTd       = false;
    double      per         = 0.0;
    double      window      = 0.0;
    double      nomDelay    = 0.0;
    bool        hasNomDelay = false;
    double      maxDelay    = 0.0;
    bool        hasFrom     = false;
    double      tFrom       = 0.0;
    bool        hasTo       = false;
    double      tTo         = 0.0;
    int64_t     expCount    = -1;
    bool        sparse      = false;
    bool        strict      = false;
    bool        hasMaxRms   = false;
    double      maxRms      = 0.0;
    bool        hasMaxPp    = false;
    double      maxPp       = 0.0;
    bool        hasMaxSkew  = false;
    double      maxSkew     = 0.0;
    bool        hasMaxBer   = false;
    double      maxBer      = 0.0;
    bool        hasMaxErrs  = false;
    int64_t     maxErrs     = 0;
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
    bool                                                         argMargins      = false;
    bool                                                         argExplicitNodal = false;
    double                                                       argMarginTol    = 50e-12;
    std::optional<std::string>                                   marginFile;
    std::optional<std::string>                                   verifyEventsFile;
    std::vector<MarginSpec>                                      marginSpecs;
    std::vector<MarginTarget>                                    marginTargets;
    std::vector<std::string>                                     cliMarginSpecs;
    std::vector<AssertSpec>                                      assertions;
    std::vector<std::string>                                     cliAssertions;
    std::vector<MeasSpec>                                        measurements;
    std::vector<std::string>                                     cliMeasurements;
    std::optional<std::string>                                   measFile;
    std::vector<IcSpec>                                          initialConditions;
    std::vector<std::string>                                     cliInitialConditions;
    bool                                                         argCheckLoops       = false;
    bool                                                         argCheckLoopsStrict = false;
    std::optional<std::string>                                   loopFile;
    std::vector<JitterSpec>                                      jitterSpecs;
    std::vector<std::string>                                     cliJitterSpecs;
    std::optional<std::string>                                   jitterFile;
    std::vector<std::string>                                     cliIvSpecs;
    std::vector<std::string>                                     cliSquidSpecs;
    bool                                                         argSens         = false;
    std::vector<SensSpec>                                        sensSpecs;
    std::vector<std::string>                                     cliSensSpecs;
    std::optional<std::string>                                   sensFile;
    bool                                                         argLumpPtl           = true;
    double                                                       argMinPtl            = 0.0;
    bool                                                         ptlLumpedDone        = false;
    bool                                                         nanowireGateExpanded = false;
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
        argExplicitNodal           = cli_options.explicitNodal;
        argFixedStep               = cli_options.fixedStep;
        argEvents                  = cli_options.events;
        eventsFile                 = cli_options.events_file;
        if (cli_options.event_metrics.has_value()) {
            argEventMetrics = parse_event_metrics(cli_options.event_metrics.value(), argEventMetrics);
        }
        cliSweeps                  = cli_options.sweeps;
        argMargins                 = cli_options.margins;
        cliMarginSpecs             = cli_options.margin_specs;
        if (cli_options.margin_tol.has_value()) { argMarginTol = cli_options.margin_tol.value(); }
        marginFile                 = cli_options.margin_file;
        verifyEventsFile           = cli_options.verify_events_file;
        cliAssertions              = cli_options.assertions;
        cliMeasurements            = cli_options.measurements;
        measFile                   = cli_options.meas_file;
        cliInitialConditions       = cli_options.initial_conditions;
        argCheckLoops              = cli_options.check_loops;
        loopFile                   = cli_options.loop_file;
        cliJitterSpecs             = cli_options.jitter_specs;
        jitterFile                 = cli_options.jitter_file;
        cliIvSpecs                 = cli_options.iv_specs;
        cliSquidSpecs              = cli_options.squid_specs;
        argSens                    = cli_options.sens;
        cliSensSpecs               = cli_options.sens_specs;
        sensFile                   = cli_options.sens_file;
        argCorrector               = cli_options.corrector;
        argMcRuns                  = cli_options.batchRuns;
        cli_output_file            = cli_options.output_file;
        netlist.sanityCheck        = cli_options.sanityCheck;
        netlist.sanityCheckSubckts = cli_options.sanityCheckSubckts;
    }

    static uint32_t       parse_event_metrics(const std::string& spec, uint32_t currentMask = EVENT_METRIC_NONE);
    void                  parse_sweep_tokens(const tokens_t& c);
    void                  parse_sweep_spec(const std::string& spec);
    void                  parse_margin_tokens(const tokens_t& c);
    void                  parse_margin_spec(const std::string& spec);
    void                  parse_assert_tokens(const tokens_t& c);
    void                  parse_assert_spec(const std::string& spec);
    void                  parse_meas_tokens(const tokens_t& c);
    void                  parse_meas_spec(const std::string& spec);
    void                  parse_ic_tokens(const tokens_t& c);
    void                  parse_ic_spec(const std::string& spec);
    void                  parse_check_loops_tokens(const tokens_t& c);
    void                  parse_jitter_tokens(const tokens_t& c);
    void                  parse_jitter_spec(const std::string& spec);
    void                  parse_sens_tokens(const tokens_t& c);
    void                  parse_sens_spec(const std::string& spec);
    std::vector<tokens_t> read_input(LineInput& input, string_o fileName = std::nullopt);
    void                  parse_input(string_o fileName = std::nullopt);
    void                  parse_input_string(const std::string& content, string_o baseDir = std::nullopt);
    void                  parse_tokens_lines(std::vector<tokens_t>& fileLines);
    void                  syntax_check_controls(std::vector<tokens_t>& controls);
    void                  parse_options_and_state_controls();

  private:
    std::optional<uint64_t> find_seed_option() const;
};
} // namespace JoSIM

#endif // JOSIM_INPUT_HPP
