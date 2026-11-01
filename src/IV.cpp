// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/IV.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Parameters.hpp"
#include "JoSIM/RelevantTrace.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace JoSIM;

namespace {

std::string fmt_sci(double val) {
    std::ostringstream oss;
    oss << std::uppercase << std::scientific << std::setprecision(8) << val;
    return oss.str();
}

std::string strip_trace_wrapper(std::string s) {
    Misc::ltrim(s);
    Misc::rtrim(s);
    if (s.size() >= 3 && s.back() == ')') {
        const size_t lp = s.find('(');
        if (lp != std::string::npos) {
            s = s.substr(lp + 1, s.size() - lp - 2);
            Misc::ltrim(s);
            Misc::rtrim(s);
        }
    }
    return s;
}

void resolve_voltage_nodes(const std::string&     vnodeSpec,
                           const Matrix&          mObj,
                           const CurrentSource*   defaultCs,
                           std::optional<int64_t>& posIdx,
                           std::optional<int64_t>& negIdx) {
    posIdx.reset();
    negIdx.reset();
    auto lookup_node = [&](std::string n) -> std::optional<int64_t> {
        Misc::ltrim(n);
        Misc::rtrim(n);
        if (n.empty() || n == "0" || n == "GND") { return std::nullopt; }
        auto it = mObj.nm.find(n);
        if (it != mObj.nm.end()) { return it->second; }
        return std::nullopt;
    };

    if (!vnodeSpec.empty()) {
        std::string spec = strip_trace_wrapper(vnodeSpec);
        const size_t comma = spec.find(',');
        if (comma != std::string::npos) {
            posIdx = lookup_node(spec.substr(0, comma));
            negIdx = lookup_node(spec.substr(comma + 1));
            return;
        }
        auto devIt = mObj.components.deviceMap.find(spec);
        if (devIt != mObj.components.deviceMap.end()) {
            const auto& dev = mObj.components.devices.at(static_cast<size_t>(devIt->second));
            std::visit(
                [&](const auto& d) {
                    posIdx = d.indexInfo.posIndex_;
                    negIdx = d.indexInfo.negIndex_;
                },
                dev);
            return;
        }
        posIdx = lookup_node(spec);
        return;
    }

    if (defaultCs != nullptr) {
        // In SPICE, Isrc n1 n2 injects positive current into n2 (negIndex_) and out of n1 (posIndex_)
        posIdx = defaultCs->indexInfo.negIndex_;
        negIdx = defaultCs->indexInfo.posIndex_;
        return;
    }

    if (!mObj.nm.empty()) { posIdx = 0; }
}

void configure_relevant_port(Matrix&                 mObj,
                             AnalysisType            atyp,
                             std::optional<int64_t>  posIdx,
                             std::optional<int64_t>  negIdx) {
    mObj.relevantTraces.clear();
    mObj.relevantIndices.clear();
    RelevantTrace rt;
    rt.device      = false;
    rt.storageType = (atyp == AnalysisType::Phase) ? StorageType::Phase : StorageType::Voltage;
    rt.index1      = posIdx;
    rt.index2      = negIdx;
    mObj.relevantTraces.push_back(rt);
    if (posIdx.has_value()) { mObj.relevantIndices.push_back(*posIdx); }
    if (negIdx.has_value() && (!posIdx.has_value() || *negIdx != *posIdx)) {
        mObj.relevantIndices.push_back(*negIdx);
    }
}

double compute_steady_voltage(const Simulation&       sim,
                              AnalysisType            atyp,
                              std::optional<int64_t>  posIdx,
                              std::optional<int64_t>  negIdx,
                              double                  dt,
                              double                  avgStartFrac,
                              double                  rfFreq,
                              bool                    classicBoxcar) {
    const std::vector<double>* vPos = nullptr;
    const std::vector<double>* vNeg = nullptr;
    if (posIdx.has_value() && *posIdx >= 0 && static_cast<size_t>(*posIdx) < sim.results.xVector.size()
        && sim.results.xVector[*posIdx].has_value()) {
        vPos = &(*sim.results.xVector[*posIdx]);
    }
    if (negIdx.has_value() && *negIdx >= 0 && static_cast<size_t>(*negIdx) < sim.results.xVector.size()
        && sim.results.xVector[*negIdx].has_value()) {
        vNeg = &(*sim.results.xVector[*negIdx]);
    }
    const size_t N = vPos ? vPos->size() : (vNeg ? vNeg->size() : 0);
    if (N < 4) { return 0.0; }

    auto sample_at = [&](size_t k) -> double {
        return (vPos ? (*vPos)[k] : 0.0) - (vNeg ? (*vNeg)[k] : 0.0);
    };

    if (classicBoxcar && atyp == AnalysisType::Voltage) {
        double sum  = 0.0;
        size_t half = N / 2;
        if (half == 0) { return 0.0; }
        for (size_t k = half; k < N; ++k) { sum += sample_at(k); }
        return sum / static_cast<double>(half);
    }

    size_t k1 = static_cast<size_t>(std::clamp(avgStartFrac, 0.05, 0.90) * static_cast<double>(N));
    size_t k2 = N - 1;
    if (k2 <= k1 + 2) { k1 = N / 2; }

    // If microwave RF drive is active, snap the averaging window to an exact integer number of RF cycles
    if (rfFreq > 0.0 && dt > 0.0) {
        const double tRf      = 1.0 / rfFreq;
        const double maxSpanT = static_cast<double>(k2 - k1) * dt;
        if (maxSpanT >= tRf) {
            const int64_t nCycles = std::max<int64_t>(1, static_cast<int64_t>(std::floor(maxSpanT / tRf)));
            const size_t  spanK   = static_cast<size_t>(std::llround(static_cast<double>(nCycles) * tRf / dt));
            if (spanK >= 2 && spanK < k2) { k1 = k2 - spanK; }
        }
    }

    if (atyp == AnalysisType::Phase) {
        if (rfFreq > 0.0 && k2 > k1) {
            // Over an exact integer number of RF cycles, the net phase change gives the exact locked Shapiro voltage
            return Constants::SIGMA * (sample_at(k2) - sample_at(k1)) / (static_cast<double>(k2 - k1) * dt);
        }
        const size_t M = k2 - k1;
        if (M == 0) { return 0.0; }
        double wSum  = 0.0;
        double wvSum = 0.0;
        for (size_t m = 0; m < M; ++m) {
            const double w  = 0.5 * (1.0 - std::cos(2.0 * Constants::PI * static_cast<double>(m) / static_cast<double>(M)));
            const double vk = Constants::SIGMA * (sample_at(k1 + m + 1) - sample_at(k1 + m)) / dt;
            wvSum += w * vk;
            wSum  += w;
        }
        return (wSum > 0.0) ? (wvSum / wSum) : 0.0;
    } else {
        const size_t M = k2 - k1;
        if (M == 0) { return sample_at(k2); }
        if (rfFreq > 0.0) {
            double sum = 0.5 * (sample_at(k1) + sample_at(k2));
            for (size_t k = k1 + 1; k < k2; ++k) { sum += sample_at(k); }
            return sum / static_cast<double>(M);
        }
        double wSum  = 0.0;
        double wvSum = 0.0;
        for (size_t m = 0; m <= M; ++m) {
            const double w = 0.5 * (1.0 - std::cos(2.0 * Constants::PI * static_cast<double>(m) / static_cast<double>(M)));
            wvSum += w * sample_at(k1 + m);
            wSum  += w;
        }
        return (wSum > 0.0) ? (wvSum / wSum) : 0.0;
    }
}

void set_exported_param(param_map& params, const std::string& baseName, const std::string& id, double val) {
    Parameter p;
    p.set_expression(fmt_sci(val));
    p.set_value(val);
    params[{baseName, std::nullopt}] = p;
    if (!id.empty()) { params[{baseName + "_" + id, std::nullopt}] = p; }
}

IvSummary analyze_iv_curve(const std::vector<std::pair<double, double>>& iv_data,
                           const std::string&                            name,
                           const std::string&                            target,
                           const std::string&                            mode,
                           double                                        vTh,
                           double                                        dI,
                           double                                        rfAmp,
                           double                                        rfFreq,
                           param_map&                                    params) {
    IvSummary s;
    s.name      = name;
    s.target    = target;
    s.mode      = mode;
    s.rfAmp     = rfAmp;
    s.rfFreq    = rfFreq;
    s.numPoints = static_cast<int64_t>(iv_data.size());
    if (iv_data.empty()) { return s; }

    bool foundIcPos = false, foundIrPos = false, foundIcNeg = false, foundIrNeg = false;
    double iMaxPos = 0.0, vAtMaxPos = 0.0;
    double iHigh80 = 0.0, vAtHigh80 = 0.0;

    for (size_t k = 0; k < iv_data.size(); ++k) {
        const double v = iv_data[k].first;
        const double i = iv_data[k].second;
        if (i > iMaxPos) {
            iMaxPos   = i;
            vAtMaxPos = v;
        }
        if (k == 0) { continue; }
        const double vPrev = iv_data[k - 1].first;
        const double iPrev = iv_data[k - 1].second;
        const double di    = i - iPrev;

        // Positive ascending branch: switching current Ic+
        if (!foundIcPos && i > 0.0 && di > 0.0 && std::abs(vPrev) <= vTh && std::abs(v) > vTh) {
            s.icPos    = 0.5 * (iPrev + i);
            foundIcPos = true;
        }
        // Positive descending branch: retrapping current Ir+
        if (!foundIrPos && i >= 0.0 && di < 0.0 && std::abs(vPrev) > vTh && std::abs(v) <= vTh) {
            s.irPos    = 0.5 * (iPrev + i);
            foundIrPos = true;
        }
        // Negative ascending branch: switching current |Ic-|
        if (!foundIcNeg && i < 0.0 && di < 0.0 && std::abs(vPrev) <= vTh && std::abs(v) > vTh) {
            s.icNeg    = 0.5 * std::abs(iPrev + i);
            foundIcNeg = true;
        }
        // Negative descending branch: retrapping current |Ir-|
        if (!foundIrNeg && i <= 0.0 && di > 0.0 && std::abs(vPrev) > vTh && std::abs(v) <= vTh) {
            s.irNeg    = 0.5 * std::abs(iPrev + i);
            foundIrNeg = true;
        }
    }

    if (!foundIrPos) { s.irPos = s.icPos; }
    if (!foundIcNeg) { s.icNeg = s.icPos; }
    if (!foundIrNeg) { s.irNeg = s.irPos; }

    for (const auto& pt : iv_data) {
        if (pt.second >= 0.75 * iMaxPos && pt.second <= 0.85 * iMaxPos && pt.first > 0.0) {
            iHigh80   = pt.second;
            vAtHigh80 = pt.first;
            break;
        }
    }
    if (iMaxPos > 0.0) {
        if (rfFreq <= 0.0 && s.irPos >= 0.85 * s.icPos && s.icPos > 0.0 && iMaxPos > 1.05 * s.icPos) {
            // Overdamped RSJ relation: V(I) = R_N * sqrt(I^2 - Ic^2)
            s.rNormal = std::abs(vAtMaxPos) / std::sqrt(iMaxPos * iMaxPos - s.icPos * s.icPos);
        } else if (iHigh80 > 0.0 && iMaxPos > iHigh80 * 1.05 && vAtMaxPos > vAtHigh80 * 1.08) {
            s.rNormal = (vAtMaxPos - vAtHigh80) / (iMaxPos - iHigh80);
        } else {
            s.rNormal = std::abs(vAtMaxPos) / iMaxPos;
        }
    }

    // Shapiro step extraction when microwave RF drive is active
    s.shapiroDi.assign(4, 0.0);
    if (rfFreq > 0.0) {
        s.shapiroV1       = Constants::PHI_ZERO * rfFreq;
        const double vTol = std::max(0.08 * s.shapiroV1, 1.5e-6);
        for (int n = 0; n <= 3; ++n) {
            const double vTarget = static_cast<double>(n) * s.shapiroV1;
            int64_t      cnt     = 0;
            double       iMinStep = 1e300, iMaxStep = -1e300;
            for (size_t k = 0; k < iv_data.size(); ++k) {
                const double v = iv_data[k].first;
                const double i = iv_data[k].second;
                if (k > 0 && (i - iv_data[k - 1].second) < 0.0) { continue; } // ascending branch only
                if (i < -1e-12) { continue; }
                if (std::abs(v - vTarget) <= vTol) {
                    ++cnt;
                    iMinStep = std::min(iMinStep, i);
                    iMaxStep = std::max(iMaxStep, i);
                }
            }
            if (cnt > 0) {
                s.shapiroDi[static_cast<size_t>(n)] = (cnt > 1) ? (iMaxStep - iMinStep + 0.5 * dI) : dI;
            }
        }
        if (s.icPos == 0.0 && s.shapiroDi[0] > 0.0) { s.icPos = s.shapiroDi[0]; }
    }

    set_exported_param(params, "IV_IC_POS", name, s.icPos);
    set_exported_param(params, "IV_IC_NEG", name, s.icNeg);
    set_exported_param(params, "IV_IR_POS", name, s.irPos);
    set_exported_param(params, "IV_IR_NEG", name, s.irNeg);
    set_exported_param(params, "IV_RN", name, s.rNormal);
    if (rfFreq > 0.0) {
        set_exported_param(params, "SHAPIRO_V1", name, s.shapiroV1);
        set_exported_param(params, "SHAPIRO_DI0", name, s.shapiroDi[0]);
        set_exported_param(params, "SHAPIRO_DI1", name, s.shapiroDi[1]);
        set_exported_param(params, "SHAPIRO_DI2", name, s.shapiroDi[2]);
        set_exported_param(params, "SHAPIRO_DI3", name, s.shapiroDi[3]);
    }
    return s;
}

void print_iv_summary(const IvSummary& s, bool minOut) {
    if (minOut) { return; }
    std::cout << "\n========================================================================================================\n";
    std::cout << "JoSIM I-V" << (s.rfFreq > 0.0 ? " & Shapiro-Step" : "") << " Characterization Report (" << s.target
              << ", " << s.mode << ", " << s.numPoints << " points)\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Ic+ (switch) : " << (s.icPos * 1e6) << " uA    |  Ir+ (retrap) : " << (s.irPos * 1e6) << " uA\n";
    std::cout << "  |Ic-|        : " << (s.icNeg * 1e6) << " uA    |  |Ir-|        : " << (s.irNeg * 1e6)
              << " uA    |  R_N (asymp) : " << std::setprecision(3) << s.rNormal << " Ohm\n";
    if (s.rfFreq > 0.0 && s.shapiroDi.size() >= 4) {
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2) << "  RF Drive     : f_RF = " << (s.rfFreq * 1e-9)
                  << " GHz, I_RF = " << (s.rfAmp * 1e6) << " uA  =>  V_1 = Phi_0 * f_RF = " << std::setprecision(3)
                  << (s.shapiroV1 * 1e6) << " uV\n";
        std::cout << std::setprecision(2) << "  Shapiro Steps: dI_0 = " << (s.shapiroDi[0] * 1e6)
                  << " uA,  dI_1 = " << (s.shapiroDi[1] * 1e6) << " uA,  dI_2 = " << (s.shapiroDi[2] * 1e6)
                  << " uA,  dI_3 = " << (s.shapiroDi[3] * 1e6) << " uA\n";
    }
    std::cout << "========================================================================================================\n";
}

} // namespace

IV::IV(Input& iObj) {
    for (const auto& i : iObj.controls) {
        if (i.empty()) { continue; }
        if (i.front() == "IV" || i.front() == ".IV") {
            setup_iv(i, iObj);
        } else if (i.front() == "SQUID" || i.front() == ".SQUID") {
            setup_squid(i, iObj);
        }
    }
}

void IV::setup_iv(const tokens_t& i, Input& iObj) {
    if (i.size() < 2) { Errors::control_errors(ControlErrors::INVALID_IV_COMMAND, Misc::vector_to_string(i)); }

    std::vector<std::string> pos;
    std::string              nameId, modelSpec, srcSpec, vnodeSpec, outFile;
    std::optional<double>    maxC_opt, minC_opt, step_opt, area_opt, rshunt_opt;
    std::optional<double>    tstop_opt, tstep_opt, tavg_opt, vth_opt, rfAmp_opt, rfFreq_opt;
    std::optional<bool>      hyst_opt;
    bool                     hasKeywords = false;

    for (size_t idx = 1; idx < i.size(); ++idx) {
        const std::string& tok   = i[idx];
        const size_t       eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            hasKeywords     = true;
            std::string key = tok.substr(0, eqPos);
            std::string val = tok.substr(eqPos + 1);
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);
            if (key == "NAME" || key == "ID") {
                nameId = val;
            } else if (key == "MODEL") {
                modelSpec = val;
            } else if (key == "SRC" || key == "ISRC" || key == "BIAS") {
                srcSpec = val;
            } else if (key == "VNODE" || key == "TRACE" || key == "PORT") {
                vnodeSpec = val;
            } else if (key == "OUT" || key == "FILE") {
                outFile = val;
            } else if (key == "MAX" || key == "IMAX") {
                maxC_opt = parse_param(val, iObj.parameters);
            } else if (key == "MIN" || key == "IMIN") {
                minC_opt = parse_param(val, iObj.parameters);
            } else if (key == "STEP" || key == "ISTEP" || key == "INC") {
                step_opt = std::abs(parse_param(val, iObj.parameters));
            } else if (key == "AREA") {
                area_opt = parse_param(val, iObj.parameters);
            } else if (key == "RSHUNT" || key == "RS") {
                rshunt_opt = parse_param(val, iObj.parameters);
            } else if (key == "HYST" || key == "HYSTERESIS") {
                hyst_opt = (val != "0" && val != "FALSE" && val != "false");
            } else if (key == "TSTOP") {
                tstop_opt = parse_param(val, iObj.parameters);
            } else if (key == "TSTEP") {
                tstep_opt = parse_param(val, iObj.parameters);
            } else if (key == "TAVG" || key == "TSTART") {
                tavg_opt = parse_param(val, iObj.parameters);
            } else if (key == "VTH" || key == "VTHRESH") {
                vth_opt = std::abs(parse_param(val, iObj.parameters));
            } else if (key == "RF_AMP" || key == "AC_AMP" || key == "IRF") {
                rfAmp_opt = parse_param(val, iObj.parameters);
            } else if (key == "RF_FREQ" || key == "FREQ" || key == "FRF") {
                rfFreq_opt = parse_param(val, iObj.parameters);
            }
        } else if (tok == "HYST") {
            hasKeywords = true;
            hyst_opt    = true;
        } else if (tok == "NOHYST") {
            hasKeywords = true;
            hyst_opt    = false;
        } else {
            pos.push_back(tok);
        }
    }

    std::string target = !modelSpec.empty() ? modelSpec : (!pos.empty() ? pos[0] : "CIRCUIT");
    if (!maxC_opt.has_value() && pos.size() >= 2) { maxC_opt = parse_param(pos[1], iObj.parameters); }
    if (outFile.empty() && pos.size() >= 3) { outFile = pos[2]; }

    if (!maxC_opt.has_value() || *maxC_opt <= 0.0) {
        Errors::control_errors(ControlErrors::INVALID_IV_COMMAND, Misc::vector_to_string(i));
        return;
    }
    const double maxC = *maxC_opt;

    // Determine whether target is a JJ model or circuit-level source
    tokens_t tModel = Misc::tokenize(target, ".|");
    std::string modelName = tModel.empty() ? target : tModel.at(0);
    string_o subc;
    if (tModel.size() > 1) { subc = tModel.at(1); }
    const bool isModel = (iObj.netlist.models.count(std::make_pair(modelName, subc)) != 0);

    if (!isModel && target != "CIRCUIT") {
        // Check if target matches a current source in the expanded netlist
        bool foundCs = false;
        for (const auto& line : iObj.netlist.expNetlist) {
            if (!line.first.empty() && line.first.front() == target) {
                foundCs = true;
                if (srcSpec.empty()) { srcSpec = target; }
                break;
            }
        }
        if (!foundCs && srcSpec.empty()) {
            Errors::control_errors(ControlErrors::IV_MODEL_NOT_FOUND, modelName + ((subc) ? "|" + subc.value() : ""));
            return;
        }
    }

    // Fast path for classic 3-argument model .iv (100% compatible with legacy .iv <model> <maxC> <outfile>)
    if (isModel && !hasKeywords) {
        if (outFile.empty()) {
            Errors::control_errors(ControlErrors::INVALID_IV_COMMAND, Misc::vector_to_string(i));
            return;
        }
        Input ivInp = iObj;
        ivInp.controls.clear();
        ivInp.argMargins    = false;
        ivInp.argSens       = false;
        ivInp.argCheckLoops = false;
        ivInp.assertions.clear();
        ivInp.measurements.clear();
        ivInp.jitterSpecs.clear();
        tokens_t jj              = {"B01", "1", "0", modelName, "AREA=1"};
        tokens_t ib              = {"IB01", "0", "1", "PWL(0", "0", "10P", "0", "50P", "2.5U)"};
        ivInp.netlist.expNetlist = {std::make_pair(jj, subc), std::make_pair(ib, std::nullopt)};
        ivInp.transSim.tstep(0.05E-12);
        ivInp.transSim.tstop(500E-12);
        ivInp.argAnal = JoSIM::AnalysisType::Voltage;
        ivInp.argMin  = true;
        auto iv_data  = generate_iv(maxC, ivInp);

        auto path = std::filesystem::path(outFile);
        if (!path.has_parent_path() && iObj.fileParentPath) {
            path = std::filesystem::path(iObj.fileParentPath.value()).append(outFile);
        }
        write_iv(iv_data, path.string());
        const double dI = (maxC < 25e-6) ? (maxC / 100.0) : 2.5e-6;
        analyze_iv_curve(iv_data, nameId, modelName, "MODEL", 5.0e-6, dI, 0.0, 0.0, iObj.parameters);
        return;
    }

    // Enhanced Model-Level or Circuit-Level / Shapiro-Step .iv
    const double dI     = step_opt.value_or((maxC < 50e-6) ? (maxC / 50.0) : 2.5e-6);
    const double rfAmp  = rfAmp_opt.value_or(0.0);
    const double rfFreq = rfFreq_opt.value_or(0.0);
    const double vTh    = vth_opt.value_or(5.0e-6);
    const bool   hyst   = hyst_opt.value_or(!minC_opt.has_value() && rfFreq <= 0.0);

    Input baseInp = iObj;
    baseInp.controls.clear();
    baseInp.argMargins    = false;
    baseInp.argSens       = false;
    baseInp.argCheckLoops = false;
    baseInp.argEventsOnly = false;
    baseInp.assertions.clear();
    baseInp.measurements.clear();
    baseInp.jitterSpecs.clear();
    baseInp.argMin  = true;
    baseInp.argPar  = false;
    baseInp.argGpu  = false;
    baseInp.argMcRuns = 1;

    const double tstep = tstep_opt.value_or(isModel ? 0.05e-12 : std::min(0.1e-12, iObj.transSim.tstep()));
    const double tstop = tstop_opt.value_or(isModel ? 500e-12 : std::max(250e-12, iObj.transSim.tstop()));
    const double tAvg  = tavg_opt.value_or(0.45 * tstop);
    baseInp.transSim.tstep(tstep);
    baseInp.transSim.tstop(tstop);
    baseInp.transSim.prstart(0.0);
    baseInp.transSim.prstep(tstep);

    const std::string tRamp1Str = fmt_sci(0.05 * tstop);
    const std::string tRamp2Str = fmt_sci(0.15 * tstop);
    std::string       sweptCsLabel;

    if (isModel) {
        baseInp.argAnal          = (rfFreq > 0.0) ? AnalysisType::Phase : AnalysisType::Voltage;
        const double area        = area_opt.value_or(1.0);
        tokens_t     jj          = {"B01", "1", "0", modelName, "AREA=" + fmt_sci(area)};
        tokens_t     ib          = {"IB01", "0", "1", "PWL(0", "0", tRamp1Str, "0", tRamp2Str, "1.000123U)"};
        baseInp.netlist.expNetlist = {std::make_pair(jj, subc), std::make_pair(ib, std::nullopt)};
        if (rshunt_opt.has_value() && *rshunt_opt > 0.0) {
            tokens_t rsh = {"RSH01", "1", "0", fmt_sci(*rshunt_opt)};
            baseInp.netlist.expNetlist.emplace_back(rsh, std::nullopt);
        }
        if (rfAmp != 0.0 && rfFreq > 0.0) {
            tokens_t irf = {"IRF01", "0", "1", "SIN(0", fmt_sci(rfAmp), fmt_sci(rfFreq) + ")"};
            baseInp.netlist.expNetlist.emplace_back(irf, std::nullopt);
        }
        sweptCsLabel = "IB01";
        if (vnodeSpec.empty()) { vnodeSpec = "1"; }
    } else {
        baseInp.argAnal = AnalysisType::Phase;
        std::string n1 = "0", n2 = "1";
        bool        matched = false;
        for (auto& line : baseInp.netlist.expNetlist) {
            if (line.first.empty() || line.first.front().empty() || line.first.front()[0] != 'I') { continue; }
            if (srcSpec.empty() || line.first.front() == srcSpec) {
                sweptCsLabel = line.first.front();
                if (line.first.size() >= 3) {
                    n1 = line.first[1];
                    n2 = line.first[2];
                }
                line.first = {sweptCsLabel, n1, n2, "PWL(0", "0", tRamp1Str, "0", tRamp2Str, "1.000123U)"};
                matched    = true;
                break;
            }
        }
        if (!matched) {
            Errors::control_errors(ControlErrors::INVALID_IV_COMMAND, Misc::vector_to_string(i));
            return;
        }
        if (rfAmp != 0.0 && rfFreq > 0.0) {
            tokens_t irf = {"IRF_IV_01", n1, n2, "SIN(0", fmt_sci(rfAmp), fmt_sci(rfFreq) + ")"};
            baseInp.netlist.expNetlist.emplace_back(irf, std::nullopt);
        }
    }

    // Build sweep points (prePulse, targetCurr)
    std::vector<std::pair<double, double>> points;
    if (hyst) {
        double c = 0.0;
        while (c <= maxC + 1e-15) {
            points.emplace_back(0.0, c);
            c += dI;
        }
        c -= dI;
        while (c >= -1e-15) {
            points.emplace_back(maxC, c);
            c -= dI;
        }
        c = 0.0;
        while (c >= -maxC - 1e-15) {
            points.emplace_back(0.0, c);
            c -= dI;
        }
        c += dI;
        while (c <= 1e-15) {
            points.emplace_back(-maxC, c);
            c += dI;
        }
    } else {
        const double minC = minC_opt.value_or(0.0);
        for (double c = minC; c <= maxC + 1e-15 * std::max(1.0, std::abs(maxC)); c += dI) {
            points.emplace_back(c, c);
        }
    }

    std::vector<std::pair<double, double>> iv_data(points.size());
    const int64_t                          nPts        = static_cast<int64_t>(points.size());
    const double                           avgStartFrac = std::clamp(tAvg / tstop, 0.1, 0.85);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Input  locInp = baseInp;
        Matrix locMat;
        locMat.create_matrix(locInp);
        const auto baseNz = locMat.nz;

        const CurrentSource* sweptCs = nullptr;
        auto                 csIt    = locMat.components.csMap.find(sweptCsLabel);
        if (csIt != locMat.components.csMap.end()) {
            sweptCs = &locMat.components.currentsources.at(static_cast<size_t>(csIt->second));
        } else if (!locMat.components.currentsources.empty()) {
            sweptCs = &locMat.components.currentsources.front();
        }
        const int64_t srcIdx = sweptCs ? sweptCs->sourceIndex_ : 0;

        std::optional<int64_t> posIdx, negIdx;
        resolve_voltage_nodes(vnodeSpec, locMat, sweptCs, posIdx, negIdx);
        configure_relevant_port(locMat, locInp.argAnal, posIdx, negIdx);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
        for (int64_t idx = 0; idx < nPts; ++idx) {
            locMat.nz = baseNz;
            locInp.transSim.tstep(tstep);
            locMat.sourcegen.at(static_cast<size_t>(srcIdx)).ampValues({0.0, points[idx].first, points[idx].second});
            Simulation sim(locInp, locMat);
            const double vAvg = compute_steady_voltage(
                sim, locInp.argAnal, posIdx, negIdx, tstep, avgStartFrac, rfFreq, false);
            iv_data[static_cast<size_t>(idx)] = std::make_pair(vAvg, points[idx].second);
        }
    }

    if (!outFile.empty()) {
        auto path = std::filesystem::path(outFile);
        if (!path.has_parent_path() && iObj.fileParentPath) {
            path = std::filesystem::path(iObj.fileParentPath.value()).append(outFile);
        }
        write_iv(iv_data, path.string());
    }

    const std::string modeDesc = isModel ? (rfFreq > 0.0 ? "MODEL+SHAPIRO" : "MODEL")
                                         : (rfFreq > 0.0 ? "CIRCUIT+SHAPIRO" : "CIRCUIT");
    IvSummary summary = analyze_iv_curve(
        iv_data, nameId, isModel ? modelName : sweptCsLabel, modeDesc, vTh, dI, rfAmp, rfFreq, iObj.parameters);
    print_iv_summary(summary, iObj.argMin);
}

void IV::setup_squid(const tokens_t& i, Input& iObj) {
    if (i.size() < 2) { Errors::control_errors(ControlErrors::INVALID_IV_COMMAND, Misc::vector_to_string(i)); }

    std::vector<std::string> pos;
    std::string              nameId, srcSpec, fluxTarget, vnodeSpec, outFile;
    std::optional<double>    ib_opt, imin_opt, imax_opt, istep_opt;
    std::optional<double>    fmin_opt, fmax_opt, fstep_opt;
    std::optional<double>    tstop_opt, tstep_opt, tavg_opt, vth_opt;

    for (size_t idx = 1; idx < i.size(); ++idx) {
        const std::string& tok   = i[idx];
        const size_t       eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            std::string key = tok.substr(0, eqPos);
            std::string val = tok.substr(eqPos + 1);
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);
            if (key == "NAME" || key == "ID") {
                nameId = val;
            } else if (key == "SRC" || key == "ISRC" || key == "BIAS") {
                srcSpec = val;
            } else if (key == "L" || key == "IND" || key == "INDUCTOR" || key == "FSRC" || key == "FLUX_SRC") {
                fluxTarget = val;
            } else if (key == "VNODE" || key == "TRACE" || key == "PORT") {
                vnodeSpec = val;
            } else if (key == "OUT" || key == "FILE") {
                outFile = val;
            } else if (key == "IB" || key == "IBIAS") {
                ib_opt = parse_param(val, iObj.parameters);
            } else if (key == "IMIN") {
                imin_opt = parse_param(val, iObj.parameters);
            } else if (key == "IMAX" || key == "MAX") {
                imax_opt = parse_param(val, iObj.parameters);
            } else if (key == "ISTEP" || key == "STEP") {
                istep_opt = std::abs(parse_param(val, iObj.parameters));
            } else if (key == "FMIN" || key == "PHIMIN") {
                fmin_opt = parse_param(val, iObj.parameters);
            } else if (key == "FMAX" || key == "PHIMAX") {
                fmax_opt = parse_param(val, iObj.parameters);
            } else if (key == "FSTEP" || key == "PHISTEP") {
                fstep_opt = std::abs(parse_param(val, iObj.parameters));
            } else if (key == "TSTOP") {
                tstop_opt = parse_param(val, iObj.parameters);
            } else if (key == "TSTEP") {
                tstep_opt = parse_param(val, iObj.parameters);
            } else if (key == "TAVG" || key == "TSTART") {
                tavg_opt = parse_param(val, iObj.parameters);
            } else if (key == "VTH" || key == "VTHRESH") {
                vth_opt = std::abs(parse_param(val, iObj.parameters));
            }
        } else {
            pos.push_back(tok);
        }
    }

    if (srcSpec.empty() && !pos.empty()) { srcSpec = pos[0]; }
    if (fluxTarget.empty() && pos.size() >= 2) { fluxTarget = pos[1]; }
    if (outFile.empty() && pos.size() >= 3) { outFile = pos[2]; }

    Input baseInp = iObj;
    baseInp.controls.clear();
    baseInp.argMargins    = false;
    baseInp.argSens       = false;
    baseInp.argCheckLoops = false;
    baseInp.argEventsOnly = false;
    baseInp.assertions.clear();
    baseInp.measurements.clear();
    baseInp.jitterSpecs.clear();
    baseInp.argAnal   = AnalysisType::Phase;
    baseInp.argMin    = true;
    baseInp.argPar    = false;
    baseInp.argGpu    = false;
    baseInp.argMcRuns = 1;

    const double tstep = tstep_opt.value_or(std::min(0.1e-12, iObj.transSim.tstep()));
    const double tstop = tstop_opt.value_or(std::max(250e-12, iObj.transSim.tstop()));
    const double tAvg  = tavg_opt.value_or(0.45 * tstop);
    const double vTh   = vth_opt.value_or(5.0e-6);
    baseInp.transSim.tstep(tstep);
    baseInp.transSim.tstop(tstop);
    baseInp.transSim.prstart(0.0);
    baseInp.transSim.prstep(tstep);

    const std::string tRamp1Str = fmt_sci(0.02 * tstop);
    const std::string tRamp2Str = fmt_sci(0.15 * tstop);

    // Find bias current source in expanded netlist
    std::string sweptCsLabel;
    double      nominalIb = 200e-6;
    for (auto& line : baseInp.netlist.expNetlist) {
        if (line.first.empty() || line.first.front().empty() || line.first.front()[0] != 'I') { continue; }
        if (srcSpec.empty() || line.first.front() == srcSpec) {
            sweptCsLabel = line.first.front();
            // Extract nominal final amplitude from existing source if possible
            if (!line.first.empty()) {
                std::string lastTok = line.first.back();
                while (!lastTok.empty() && lastTok.back() == ')') { lastTok.pop_back(); }
                const double cand = parse_param(lastTok, iObj.parameters);
                if (std::isfinite(cand) && std::abs(cand) > 0.0) { nominalIb = std::abs(cand); }
            }
            std::string n1 = (line.first.size() >= 3) ? line.first[1] : "0";
            std::string n2 = (line.first.size() >= 3) ? line.first[2] : "1";
            line.first     = {sweptCsLabel, n1, n2, "PWL(0", "0", tRamp1Str, "0", tRamp2Str, "1.000123U)"};
            break;
        }
    }

    // Check if fluxTarget is a current source (I*) or inductor(s) (L*)
    bool        fluxIsCurrentSource = (!fluxTarget.empty() && fluxTarget[0] == 'I');
    std::string fluxCsLabel;
    if (fluxIsCurrentSource) {
        for (auto& line : baseInp.netlist.expNetlist) {
            if (!line.first.empty() && line.first.front() == fluxTarget) {
                fluxCsLabel    = fluxTarget;
                std::string n1 = (line.first.size() >= 3) ? line.first[1] : "0";
                std::string n2 = (line.first.size() >= 3) ? line.first[2] : "1";
                line.first     = {fluxCsLabel, n1, n2, "PWL(0", "0", tRamp1Str, "2.000456U", tRamp2Str, "2.000456U)"};
                break;
            }
        }
    }

    // Build flux grid and current grid
    const double fMin  = fmin_opt.value_or(0.0);
    const double fMax  = fmax_opt.value_or(1.0);
    const double fStep = fstep_opt.value_or(std::max(1e-6, (fMax - fMin) / 20.0));
    std::vector<double> fluxGrid;
    for (double f = fMin; f <= fMax + 1e-9 * std::max(1.0, std::abs(fStep)); f += fStep) {
        fluxGrid.push_back(f);
    }
    if (fluxGrid.empty()) { fluxGrid.push_back(fMin); }

    const bool          is2D = imax_opt.has_value();
    std::vector<double> currGrid;
    if (is2D) {
        const double iMax  = *imax_opt;
        const double iMin  = imin_opt.value_or(0.0);
        const double iStep = istep_opt.value_or(std::max(1e-9, (iMax - iMin) / 20.0));
        for (double c = iMin; c <= iMax + 1e-9 * std::max(1.0, std::abs(iStep)); c += iStep) {
            currGrid.push_back(c);
        }
    } else {
        currGrid.push_back(ib_opt.value_or(nominalIb));
    }

    const size_t  nFlux        = fluxGrid.size();
    const size_t  nCurr        = currGrid.size();
    const int64_t totalPts     = static_cast<int64_t>(nFlux * nCurr);
    const double  avgStartFrac = std::clamp(tAvg / tstop, 0.1, 0.85);
    std::vector<double> vSurface(static_cast<size_t>(totalPts), 0.0);

    // Parse inductor target(s) if fluxTarget is not a current source
    std::vector<std::pair<std::string, double>> indTargets;
    if (!fluxIsCurrentSource && !fluxTarget.empty()) {
        tokens_t parts = Misc::tokenize(fluxTarget, ",");
        for (size_t k = 0; k < parts.size(); ++k) {
            std::string lbl = parts[k];
            double      w   = 1.0;
            if (k + 1 < parts.size()) {
                char*  endPtr = nullptr;
                double candW  = std::strtod(parts[k + 1].c_str(), &endPtr);
                if (endPtr != parts[k + 1].c_str() && *endPtr == '\0') {
                    w = candW;
                    ++k;
                }
            }
            indTargets.emplace_back(lbl, w);
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Input  locInp = baseInp;
        Matrix locMat;
        locMat.create_matrix(locInp);
        const auto baseNz = locMat.nz;

        const CurrentSource* sweptCs = nullptr;
        auto                 csIt    = locMat.components.csMap.find(sweptCsLabel);
        if (csIt != locMat.components.csMap.end()) {
            sweptCs = &locMat.components.currentsources.at(static_cast<size_t>(csIt->second));
        } else if (!locMat.components.currentsources.empty()) {
            sweptCs = &locMat.components.currentsources.front();
        }
        const int64_t biasSrcIdx = sweptCs ? sweptCs->sourceIndex_ : 0;

        int64_t fluxSrcIdx = -1;
        if (fluxIsCurrentSource) {
            auto fcsIt = locMat.components.csMap.find(fluxCsLabel);
            if (fcsIt != locMat.components.csMap.end()) {
                fluxSrcIdx = locMat.components.currentsources.at(static_cast<size_t>(fcsIt->second)).sourceIndex_;
            }
        }

        std::vector<std::pair<int64_t, std::pair<double, double>>> indDevMap; // (devIdx, (basePhiOffset, weight))
        if (!fluxIsCurrentSource) {
            if (!indTargets.empty()) {
                for (const auto& tw : indTargets) {
                    auto dIt = locMat.components.deviceMap.find(tw.first);
                    if (dIt != locMat.components.deviceMap.end()) {
                        const int64_t dIdx = dIt->second;
                        if (std::holds_alternative<Inductor>(locMat.components.devices[static_cast<size_t>(dIdx)])) {
                            const double basePhi =
                                std::get<Inductor>(locMat.components.devices[static_cast<size_t>(dIdx)]).phiOffset_;
                            indDevMap.push_back({dIdx, {basePhi, tw.second}});
                        }
                    }
                }
            }
            if (indDevMap.empty() && !locMat.components.inductorIndices.empty()) {
                const int64_t dIdx    = locMat.components.inductorIndices.front();
                const double  basePhi = std::get<Inductor>(locMat.components.devices[static_cast<size_t>(dIdx)]).phiOffset_;
                indDevMap.push_back({dIdx, {basePhi, 1.0}});
            }
        }

        std::optional<int64_t> posIdx, negIdx;
        resolve_voltage_nodes(vnodeSpec, locMat, sweptCs, posIdx, negIdx);
        configure_relevant_port(locMat, locInp.argAnal, posIdx, negIdx);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 2)
#endif
        for (int64_t p = 0; p < totalPts; ++p) {
            const size_t fIdx = static_cast<size_t>(p) / nCurr;
            const size_t cIdx = static_cast<size_t>(p) % nCurr;
            const double phi  = fluxGrid[fIdx];
            const double ib   = currGrid[cIdx];

            locMat.nz = baseNz;
            locInp.transSim.tstep(tstep);
            locMat.sourcegen.at(static_cast<size_t>(biasSrcIdx)).ampValues({0.0, ib, ib});
            if (fluxIsCurrentSource && fluxSrcIdx >= 0) {
                locMat.sourcegen.at(static_cast<size_t>(fluxSrcIdx)).ampValues({0.0, phi, phi});
            } else {
                for (const auto& entry : indDevMap) {
                    auto& ind      = std::get<Inductor>(locMat.components.devices[static_cast<size_t>(entry.first)]);
                    ind.phiOffset_ = entry.second.first + 2.0 * Constants::PI * entry.second.second * phi;
                }
            }

            Simulation sim(locInp, locMat);
            vSurface[static_cast<size_t>(p)] =
                compute_steady_voltage(sim, locInp.argAnal, posIdx, negIdx, tstep, avgStartFrac, 0.0, false);
        }
    }

    // Post-process 1D V(Phi) and 2D Ic(Phi) metrics
    SquidSummary sq;
    sq.name       = nameId;
    sq.biasSrc    = sweptCsLabel;
    sq.fluxTarget = !fluxTarget.empty() ? fluxTarget : "L_AUTO";
    sq.is2D       = is2D;
    sq.numFlux    = static_cast<int64_t>(nFlux);
    sq.numCurrent = static_cast<int64_t>(nCurr);

    std::vector<double> icVsPhi(nFlux, 0.0);
    if (is2D) {
        sq.icMax = -1e300;
        sq.icMin = 1e300;
        for (size_t fIdx = 0; fIdx < nFlux; ++fIdx) {
            double icPhi = currGrid.back();
            for (size_t cIdx = 0; cIdx < nCurr; ++cIdx) {
                const double v = std::abs(vSurface[fIdx * nCurr + cIdx]);
                if (v > vTh) {
                    icPhi = (cIdx > 0) ? (0.5 * (currGrid[cIdx - 1] + currGrid[cIdx])) : currGrid[cIdx];
                    break;
                }
            }
            icVsPhi[fIdx] = icPhi;
            sq.icMax      = std::max(sq.icMax, icPhi);
            sq.icMin      = std::min(sq.icMin, icPhi);
        }
        if (sq.icMax > 0.0) { sq.icModDepth = (sq.icMax - sq.icMin) / sq.icMax; }
    }

    // Select optimal bias slice (or the single 1D slice) that maximizes voltage modulation dV(Phi)
    size_t bestCIdx = 0;
    double bestDv   = -1.0;
    for (size_t cIdx = 0; cIdx < nCurr; ++cIdx) {
        if (ib_opt.has_value() && is2D && std::abs(currGrid[cIdx] - *ib_opt) < 0.5 * istep_opt.value_or(1e-6)) {
            bestCIdx = cIdx;
            break;
        }
        double vMinC = 1e300, vMaxC = -1e300;
        for (size_t fIdx = 0; fIdx < nFlux; ++fIdx) {
            const double v = std::abs(vSurface[fIdx * nCurr + cIdx]);
            vMinC          = std::min(vMinC, v);
            vMaxC          = std::max(vMaxC, v);
        }
        if (vMaxC - vMinC > bestDv) {
            bestDv   = vMaxC - vMinC;
            bestCIdx = cIdx;
        }
    }

    sq.ibOpt     = currGrid[bestCIdx];
    sq.vMin      = 1e300;
    sq.vMax      = -1e300;
    sq.dvDphiMax = 0.0;
    sq.phiOpt    = fluxGrid.front();

    std::vector<double> dvDphiSlice(nFlux, 0.0);
    for (size_t fIdx = 0; fIdx < nFlux; ++fIdx) {
        const double v = std::abs(vSurface[fIdx * nCurr + bestCIdx]);
        sq.vMin        = std::min(sq.vMin, v);
        sq.vMax        = std::max(sq.vMax, v);
        double slope   = 0.0;
        if (nFlux >= 2) {
            if (fIdx == 0) {
                slope = (vSurface[1 * nCurr + bestCIdx] - vSurface[0 * nCurr + bestCIdx]) / (fluxGrid[1] - fluxGrid[0]);
            } else if (fIdx + 1 == nFlux) {
                slope = (vSurface[fIdx * nCurr + bestCIdx] - vSurface[(fIdx - 1) * nCurr + bestCIdx])
                        / (fluxGrid[fIdx] - fluxGrid[fIdx - 1]);
            } else {
                slope = (vSurface[(fIdx + 1) * nCurr + bestCIdx] - vSurface[(fIdx - 1) * nCurr + bestCIdx])
                        / (fluxGrid[fIdx + 1] - fluxGrid[fIdx - 1]);
            }
        }
        dvDphiSlice[fIdx] = slope;
        if (std::abs(slope) > sq.dvDphiMax) {
            sq.dvDphiMax = std::abs(slope);
            sq.phiOpt    = fluxGrid[fIdx];
        }
    }
    sq.dV = sq.vMax - sq.vMin;

    // Export SQUID metrics into .param scope for downstream .meas expressions
    set_exported_param(iObj.parameters, "SQUID_VMIN", nameId, sq.vMin);
    set_exported_param(iObj.parameters, "SQUID_VMAX", nameId, sq.vMax);
    set_exported_param(iObj.parameters, "SQUID_DV", nameId, sq.dV);
    set_exported_param(iObj.parameters, "SQUID_DV_DPHI_MAX", nameId, sq.dvDphiMax);
    set_exported_param(iObj.parameters, "SQUID_PHI_OPT", nameId, sq.phiOpt);
    set_exported_param(iObj.parameters, "SQUID_IB_OPT", nameId, sq.ibOpt);
    if (is2D) {
        set_exported_param(iObj.parameters, "SQUID_IC_MAX", nameId, sq.icMax);
        set_exported_param(iObj.parameters, "SQUID_IC_MIN", nameId, sq.icMin);
        set_exported_param(iObj.parameters, "SQUID_IC_MOD", nameId, sq.icModDepth);
    }

    if (!outFile.empty()) {
        auto path = std::filesystem::path(outFile);
        if (!path.has_parent_path() && iObj.fileParentPath) {
            path = std::filesystem::path(iObj.fileParentPath.value()).append(outFile);
        }
        std::ofstream ofs(path.string());
        if (ofs.is_open()) {
            ofs << std::scientific << std::setprecision(6);
            if (is2D) {
                ofs << "Phi,I_bias,V_dc,Ic_at_Phi\n";
                for (size_t fIdx = 0; fIdx < nFlux; ++fIdx) {
                    for (size_t cIdx = 0; cIdx < nCurr; ++cIdx) {
                        ofs << fluxGrid[fIdx] << "," << currGrid[cIdx] << "," << vSurface[fIdx * nCurr + cIdx] << ","
                            << icVsPhi[fIdx] << "\n";
                    }
                }
            } else {
                ofs << "Phi,I_bias,V_dc,dV_dPhi\n";
                for (size_t fIdx = 0; fIdx < nFlux; ++fIdx) {
                    ofs << fluxGrid[fIdx] << "," << currGrid[0] << "," << vSurface[fIdx] << "," << dvDphiSlice[fIdx]
                        << "\n";
                }
            }
        }
    }

    if (!iObj.argMin) {
        std::cout << "\n========================================================================================================\n";
        std::cout << "JoSIM SQUID Characterization Report (" << sq.biasSrc << " x " << sq.fluxTarget << ", "
                  << (is2D ? "2D I_b x Phi" : "1D V(Phi)") << ", " << nFlux << "x" << nCurr << " pts)\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Optimal Bias : I_b = " << (sq.ibOpt * 1e6) << " uA,  Phi_opt = " << std::setprecision(3)
                  << sq.phiOpt << " Phi0\n";
        std::cout << "  V(Phi) Swing : V_min = " << (sq.vMin * 1e6) << " uV,  V_max = " << (sq.vMax * 1e6)
                  << " uV,  dV = " << (sq.dV * 1e6) << " uV,  |dV/dPhi|_max = " << (sq.dvDphiMax * 1e6) << " uV/Phi0\n";
        if (is2D) {
            std::cout << std::setprecision(2) << "  Ic(Phi) Lobe : Ic_max = " << (sq.icMax * 1e6)
                      << " uA,  Ic_min = " << (sq.icMin * 1e6)
                      << " uA,  Modulation = " << (sq.icModDepth * 100.0) << "%\n";
        }
        std::cout << "========================================================================================================\n";
    }
}

std::vector<std::pair<double, double>> IV::generate_iv(double maxC, Input ivInp) {
    const double                           currentIncrement = (maxC < 25e-6) ? (maxC / 100.0) : 25E-7;
    double                                 currentCurr      = 0.0;
    std::vector<std::pair<double, double>> points;
    while (currentCurr <= maxC) {
        currentCurr += currentIncrement;
        points.emplace_back(0.0, currentCurr);
    }
    while (currentCurr >= 0) {
        currentCurr -= currentIncrement;
        points.emplace_back(maxC, currentCurr);
    }
    while (currentCurr >= -maxC) {
        currentCurr -= currentIncrement;
        points.emplace_back(0.0, currentCurr);
    }
    while (currentCurr <= 0) {
        currentCurr += currentIncrement;
        points.emplace_back(-maxC, currentCurr);
    }

    std::vector<std::pair<double, double>> iv_data(points.size());
    const int64_t                          nPts = static_cast<int64_t>(points.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Input locInp  = ivInp;
        locInp.argPar = false;
        locInp.argGpu = false;
        locInp.argMin = true;
        Matrix locMat;
        locMat.create_matrix(locInp);
        const auto baseNz = locMat.nz;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (int64_t idx = 0; idx < nPts; ++idx) {
            locMat.nz = baseNz;
            locInp.transSim.tstep(0.05E-12);
            locMat.sourcegen.back().ampValues({0.0, points[idx].first, points[idx].second});
            iv_data[static_cast<size_t>(idx)] = do_simulate(locInp, locMat);
        }
    }
    return iv_data;
}

std::pair<double, double> IV::do_simulate(Input& ivInp, Matrix& ivMat) {
    // Simulate the iv curve
    Simulation ivSim(ivInp, ivMat);
    // Add the results to the stack
    double&    current  = ivSim.results.xVector.back().value().back();
    auto&      voltVect = ivSim.results.xVector.front().value();
    double     voltage  = 0;
    for (auto i = voltVect.size() / 2; i < voltVect.size(); ++i) { voltage += voltVect.at(i); }
    voltage = voltage / (voltVect.size() / 2);
    return std::make_pair(voltage, current);
}

void IV::write_iv(std::vector<std::pair<double, double>>& iv_data, const std::string& output_path) {
    auto          path = std::filesystem::path(output_path);
    std::ofstream outfile(path.string());
    outfile << std::setprecision(6);
    outfile << std::scientific;
    if (outfile.is_open()) {
        outfile << "V,I\n";
        for (auto i : iv_data) { outfile << i.first << "," << i.second << "\n"; }
        outfile.close();
    } else {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, path.string());
    }
}
