// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_CLIOPTIONS_HPP
#define JOSIM_CLIOPTIONS_HPP

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/FileOutputType.hpp"
#include "JoSIM/TypeDefines.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace JoSIM {

struct CliOptions {
  private:
    tokens_t                        argv_to_tokens(const int64_t& argc, const char** argv);
    vector_pair_t<char_o, string_o> argument_pairs(const tokens_t& tokens);

  public:
    string_o                        cir_file_name;
    std::optional<OutputFile>       output_file;

    AnalysisType                    analysis_type = AnalysisType::Phase;

    int64_t                         verbose       = 0;
    bool                            minimal       = false;
    bool                            parallel      = false;
    bool                            gpu           = false;
    int64_t                         gpuMode       = 1;
    bool                            nodal         = false;
    bool                            fixedStep     = false;
    bool                            events        = false;
    std::optional<std::string>      events_file;
    std::optional<std::string>      event_metrics;
    std::vector<std::string>        sweeps;
    bool                            explicitNodal = false;
    bool                            margins       = false;
    std::vector<std::string>        margin_specs;
    std::optional<double>           margin_tol;
    std::optional<std::string>      margin_file;
    std::vector<std::string>        margin_traces;
    std::optional<std::string>      verify_events_file;
    std::vector<std::string>        assertions;
    std::vector<std::string>        measurements;
    std::optional<std::string>      meas_file;
    std::vector<std::string>        initial_conditions;
    bool                            check_loops   = false;
    std::optional<std::string>      loop_file;
    std::vector<std::string>        jitter_specs;
    std::optional<std::string>      jitter_file;
    std::vector<std::string>        iv_specs;
    std::vector<std::string>        squid_specs;
    bool                            sens          = false;
    std::vector<std::string>        sens_specs;
    std::optional<std::string>      sens_file;
    int64_t                         corrector     = 0;
    int64_t                         batchRuns     = 1;
    bool                            sanityCheck   = false;
    std::unordered_set<std::string> sanityCheckSubckts;

    // helper functions
    static CliOptions               parse(int64_t argc, const char** argv);
    static void                     display_help();
    static void                     version_info();
};

} // namespace JoSIM

#endif // JOSIM_CLIOPTIONS_HPP
