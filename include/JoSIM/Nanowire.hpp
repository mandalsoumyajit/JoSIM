// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_NANOWIRE_HPP
#define JOSIM_NANOWIRE_HPP

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/BasicComponent.hpp"
#include "JoSIM/Function.hpp"
#include "JoSIM/Input.hpp"
#include "JoSIM/Parameters.hpp"
#include "JoSIM/Spread.hpp"

#include <optional>
#include <string>
#include <unordered_set>

namespace JoSIM {

/*
  Nlabel drain source [gate [gate_neg]] [model] [IC=val] [IG=val] [IR=val] [RN=val] [L=val] [LG=val] [TAU=val] ...

  Electrothermal Superconducting Nanowire / Cryotron (nTron / hTron / SNSPD) channel branch:
  Series kinetic inductance L_ch and dynamic hotspot resistance R_ch(t) = R_n * w(t), w(t) in [0, 1].

  (VOLTAGE MODE)
  (V+ - V-) - ( (3*L_ch)/(2*h) + R_ch(t) ) * I_ch,n = -(2*L_ch/h)*I_ch,n-1 + (L_ch/(2*h))*I_ch,n-2

  (PHASE MODE)
  (phi+ - phi-) - ( L_ch/sigma + (2*h/(3*sigma))*R_ch(t) ) * I_ch,n = (4/3)*phi_R,n-1 - (1/3)*phi_R,n-2
*/

class Nanowire : public BasicComponent {
  private:
    JoSIM::AnalysisType at_;

  public:
    double      h_              = 0.25e-12;
    double      nominalArea_    = 1.0;
    double      area_           = 1.0;
    double      spread_         = 1.0;
    double      nominalIc0_     = 100e-6;  // Zero-gate switching current (default 100 uA)
    double      ic0_            = 100e-6;
    double      nominalIgSw_    = 20e-6;   // Gate critical / choke current (default 20 uA)
    double      igSw_           = 20e-6;
    double      retrapRatio_    = 0.2;     // Retrapping ratio Ir / Ic0 when Ir not explicitly set
    bool        hasExplicitIr_  = false;
    double      nominalIr_      = 20e-6;   // Retrapping current (default 0.2 * Ic0)
    double      ir_             = 20e-6;
    double      nominalRn_      = 500.0;   // Normal-state hotspot resistance (default 500 Ohm)
    double      rn_             = 500.0;
    double      nominalLch_     = 20e-12;  // Channel kinetic inductance (default 20 pH)
    double      lch_            = 20e-12;
    double      lg_             = 5e-12;   // Gate inductance for 3/4-terminal cryotron (default 5 pH)
    double      rg_             = 0.0;     // Gate resistance for resistive-heater cryotron (default 0 Ohm)
    double      alpha_          = 1.0;     // Gate suppression exponent (default 1.0)
    double      tauTh_          = 15e-12;  // Thermal cooling/recovery time constant (default 15 ps)
    double      tauHs_          = 2e-12;   // Hotspot nucleation/growth time constant (default 2 ps)
    std::string gateDevLabel_;             // Controlling gate branch or source label
    int64_t     gateCurIndex_   = -1;      // MNA row index of gate branch current
    int64_t     gateSourceIndex_= -1;      // Sourcegen index if gate is an ideal CurrentSource
    int64_t     nzBackIndex_    = -1;      // CSR nonzero index of diagonal entry

    // Dynamic electrothermal state
    int64_t     state_          = 0;       // 0 = superconducting/cooling, 1 = active hotspot/normal
    double      w_              = 0.0;     // Normalized hotspot fraction in [0, 1]
    double      rCh_            = 0.0;     // Currently stamped hotspot resistance R_n * w
    double      phiRn1_         = 0.0;     // Hotspot resistor phase drop at step n-1 (Phase mode)
    double      phiRn2_         = 0.0;     // Hotspot resistor phase drop at step n-2 (Phase mode)
    double      prevRhs_        = 0.0;     // Previous RHS term (4/3 phiRn1 - 1/3 phiRn2)
    double      In2_            = 0.0;     // Channel current at step n-2 (Voltage mode)
    double      In3_            = 0.0;     // Channel current at step n-3 (Voltage mode)
    double      In4_            = 0.0;     // Channel current at step n-4 (Voltage mode)

    Nanowire(const std::pair<tokens_t, string_o>& s,
             const NodeConfig&                    ncon,
             const nodemap&                       nm,
             std::unordered_set<std::string>&     lm,
             nodeconnections&                     nc,
             Input&                               iObj,
             Spread&                              spread,
             int64_t&                             bi);

    double      diag_entry(double h) const noexcept;
    double      switching_current(double iGate) const noexcept;
    bool        step_hotspot(double iCh, double iGate, double h) noexcept;
    void        apply_area_scale(double scale, double h) noexcept;
    void        update_timestep(const double& factor) override;

    static bool is_node_token(const std::string& tok, const Netlist& nl, const string_o& subckt);
    static void extract_gate_params(const std::pair<tokens_t, string_o>& s,
                                    const Input&                         iObj,
                                    double&                              outLg,
                                    double&                              outRg,
                                    double&                              outSpread);
};

} // namespace JoSIM
#endif // JOSIM_NANOWIRE_HPP
