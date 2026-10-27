// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_SIMULATION_HPP
#define JOSIM_SIMULATION_HPP

#include "JoSIM/Errors.hpp"
#include "JoSIM/LUSolve.hpp"
#include "JoSIM/Matrix.hpp"
#include "JoSIM/Misc.hpp"

#include <cassert>
#include <suitesparse/klu.h>

namespace JoSIM {

#define TRANSIENT      0
#define DC             1
#define AC             2
#define PHASE          3
#define NONE_SPECIFIED 4

class Results {
  public:
    std::vector<std::optional<std::vector<double>>> xVector;
    std::vector<double>                             timeAxis;
    std::vector<int64_t>                            printIndices;
};

struct KluPartition {
    int64_t              n = 0;
    std::vector<int64_t> globalRows;
    std::vector<int64_t> rp;
    std::vector<int64_t> ci;
    std::vector<double>  nz;
    std::vector<double>  xLocal;
    klu_l_symbolic*      symbolic = nullptr;
    klu_l_numeric*       numeric  = nullptr;
    klu_l_common         common{};
    bool                 needsLU = false;
};

class Simulation {
  private:
    bool                SLU = false;
    std::vector<double> x_, b_;
    int64_t             simSize_;
    JoSIM::AnalysisType atyp_;
    bool                minOut_;
    bool                parallel_                = false;
    bool                fixedStep_               = false;
    bool                warnStep_                = false;
    int64_t             verb_                    = 0;
    int64_t             correctorIter_           = 0;
    double              correctorTol_            = 1e-3;
    int64_t             stepHalvings_            = 0;
    std::string         offendingJJ_;
    std::optional<std::pair<std::string, std::optional<double>>> saveStateFile_;
    std::optional<std::string>                                   loadStateFile_;
    bool                hasJJThermalNoise_       = false;
    bool                hasResistorThermalNoise_ = false;
    bool                needsLU_;
    bool                needsTR_ = true;
    bool                startup_;
    double              stepSize_, prstep_, prstart_;

    // In-loop print windowing & decimation schedule
    bool                 useDecimation_ = false;
    std::vector<uint8_t> keepStep_;

    // Contiguous homogeneous device storage for fast hot-loop iteration
    std::vector<JJ>               jjs_;
    std::vector<Inductor>         inductors_;
    std::vector<Resistor>         resistors_;
    std::vector<Capacitor>        capacitors_;
    std::vector<TransmissionLine> txlines_;
    std::vector<VoltageSource>    vsources_;
    std::vector<PhaseSource>      psources_;
    std::vector<CCVS>             ccvss_;
    std::vector<VCCS>             vccss_;
#ifdef SLU
    LUSolve lu;
#else
    int64_t                   simOK_;
    klu_l_symbolic*           Symbolic_      = nullptr;
    klu_l_common              Common_{};
    klu_l_numeric*            Numeric_       = nullptr;
    bool                      usePartitions_ = false;
    std::vector<KluPartition> partitions_;
    std::vector<int64_t>      jjPartIdx_;
    std::vector<int64_t>      jjPartNzIdx_;
#endif

    void setup(Input& iObj, Matrix& mObj);
    void setup_partitions(Matrix& mObj);
    void factorize_lu(Matrix& mObj);
    void refactor_lu(Matrix& mObj);
    void solve_lu(Matrix& mObj);
    void free_lu();
    void trans_sim(Matrix& mObj);
    void setup_b(Matrix& mObj, int64_t i, double step, double factor = 1);
    void reduce_step(Input& iObj, Matrix& mObj);
    void corrector_jj(Matrix& mObj);
    void save_state(const std::string& filename, const Matrix& mObj, int64_t currentStep = -1) const;
    void load_state(const std::string& filename, Matrix& mObj);

    void handle_cs(Matrix& mObj, double& step, const int64_t& i);
    void handle_resistors(Matrix& mObj, double& step);
    void handle_inductors(Matrix& mObj, double factor = 1);
    void handle_capacitors(Matrix& mObj);
    void handle_jj(Matrix& mObj, int64_t& i, double& step, double factor = 1);
    void handle_vs(Matrix& mObj, const int64_t& i, double& step, double factor = 1);
    void handle_ps(Matrix& mObj, const int64_t& i, double& step, double factor = 1);
    void handle_ccvs(Matrix& mObj);
    void handle_vccs(Matrix& mObj);
    void handle_tx(Matrix& mObj, const int64_t& i, double& step, double factor = 1);

  public:
    Results results;

    Simulation(Input& iObj, Matrix& mObj);
};
} // namespace JoSIM
#endif // JOSIM_SIMULATION_HPP
