// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_IV_HPP
#define JOSIM_IV_HPP

#include "Input.hpp"
#include "Matrix.hpp"
#include "Model.hpp"
#include "Simulation.hpp"

#include <string>
#include <utility>
#include <vector>

namespace JoSIM {

struct IvSummary {
    std::string         name;
    std::string         target;
    std::string         mode;
    double              icPos       = 0.0;
    double              icNeg       = 0.0;
    double              irPos       = 0.0;
    double              irNeg       = 0.0;
    double              rNormal     = 0.0;
    double              rfAmp       = 0.0;
    double              rfFreq      = 0.0;
    double              shapiroV1   = 0.0;
    std::vector<double> shapiroDi; // step widths for n = 0, 1, 2, 3
    int64_t             numPoints   = 0;
};

struct SquidPoint {
    double phi  = 0.0;
    double ib   = 0.0;
    double vdc  = 0.0;
    double dvdphi = 0.0;
    double ic   = 0.0;
};

struct SquidSummary {
    std::string name;
    std::string biasSrc;
    std::string fluxTarget;
    bool        is2D        = false;
    double      ibOpt       = 0.0;
    double      vMin        = 0.0;
    double      vMax        = 0.0;
    double      dV          = 0.0;
    double      dvDphiMax   = 0.0;
    double      phiOpt      = 0.0;
    double      icMax       = 0.0;
    double      icMin       = 0.0;
    double      icModDepth  = 0.0;
    int64_t     numFlux     = 0;
    int64_t     numCurrent  = 0;
};

class IV {
  public:
    IV(Input& iObj);
    void                                   setup_iv(const tokens_t& i, Input& iObj);
    void                                   setup_squid(const tokens_t& i, Input& iObj);
    std::vector<std::pair<double, double>> generate_iv(double maxC, Input ivInp);
    std::pair<double, double>              do_simulate(Input& ivInp, Matrix& ivMat);
    void write_iv(std::vector<std::pair<double, double>>& iv_data, const std::string& output_path);
};

} // namespace JoSIM

#endif // JOSIM_IV_HPP
