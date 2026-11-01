// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Nanowire.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

using namespace JoSIM;

namespace {

bool is_nanowire_model_keyword(const std::string& s) {
    return (s == "NTRON" || s == "HTRON" || s == "SNSPD" || s == "NANOWIRE" || s == "CRYOTRON" || s == "CRYO");
}

const tokens_t* find_nanowire_model(const std::string& tok, const Netlist& nl, const string_o& subckt) {
    auto it = nl.models.find(std::make_pair(tok, subckt));
    if (it != nl.models.end()) { return &it->second; }
    if (subckt.has_value()) {
        const size_t pipePos = subckt->find('|');
        if (pipePos != std::string::npos) {
            const string_o baseSub = subckt->substr(0, pipePos);
            auto           itBase  = nl.models.find(std::make_pair(tok, baseSub));
            if (itBase != nl.models.end()) { return &itBase->second; }
        }
    }
    auto itGlob = nl.models.find(std::make_pair(tok, std::nullopt));
    if (itGlob != nl.models.end()) { return &itGlob->second; }
    return nullptr;
}

tokens_t tokenize_model_params(const tokens_t& modelLine) {
    if (modelLine.size() < 3) { return {}; }
    tokens_t raw(modelLine.begin() + 2, modelLine.end());
    std::string joined = Misc::vector_to_string(raw);
    const size_t openParen = joined.find('(');
    if (openParen != std::string::npos) {
        joined = joined.substr(openParen + 1);
    } else {
        for (const char* kw : {"CRYOTRON", "NANOWIRE", "NTRON", "HTRON", "SNSPD", "CRYO"}) {
            const std::string kws(kw);
            if (joined.rfind(kws, 0) == 0) {
                joined = joined.substr(kws.size());
                break;
            }
        }
    }
    tokens_t tokens = Misc::tokenize(joined, "=(), ");
    int      parOpen = -1;
    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        if (parOpen != -1) {
            tokens.at(parOpen) += "," + tokens.at(i);
            tokens.erase(tokens.begin() + i);
            --i;
            if (!tokens.at(parOpen).empty() && tokens.at(parOpen).back() == '}') {
                tokens.at(parOpen).pop_back();
                parOpen = -1;
            }
        } else if (!tokens.at(i).empty() && tokens.at(i).front() == '{') {
            parOpen      = i;
            tokens.at(i) = tokens.at(i).substr(1);
            if (!tokens.at(i).empty() && tokens.at(i).back() == '}') {
                tokens.at(i).pop_back();
                parOpen = -1;
            }
        }
    }
    return tokens;
}

} // namespace

bool Nanowire::is_node_token(const std::string& tok, const Netlist& nl, const string_o& subckt) {
    if (tok.empty()) { return false; }
    if (tok.find('=') != std::string::npos || tok.find('(') != std::string::npos || tok.find(')') != std::string::npos
        || tok.find('{') != std::string::npos || tok.find('}') != std::string::npos) {
        return false;
    }
    if (is_nanowire_model_keyword(tok)) { return false; }
    if (find_nanowire_model(tok, nl, subckt) != nullptr) { return false; }

    const char c0 = tok.front();
    if (std::isdigit(static_cast<unsigned char>(c0)) || c0 == '.' || c0 == '-' || c0 == '+') {
        const size_t pipePos = tok.find('|');
        const size_t endIdx  = (pipePos != std::string::npos) ? pipePos : tok.size();
        if (endIdx == 0) { return false; }
        for (size_t k = 0; k < endIdx; ++k) {
            if (!std::isdigit(static_cast<unsigned char>(tok[k]))) {
                return false; // Numeric value with decimal/exponent/SI suffix (e.g. 100U), not an integer node ID
            }
        }
        return true;
    }
    return true;
}

void Nanowire::extract_gate_params(const std::pair<tokens_t, string_o>& s,
                                   const Input&                         iObj,
                                   double&                              outLg,
                                   double&                              outRg,
                                   double&                              outSpread) {
    outLg     = 5e-12;
    outRg     = 0.0;
    outSpread = 1.0;

    for (size_t idx = 3; idx < s.first.size(); ++idx) {
        const std::string& tok = s.first.at(idx);
        if (tok.find('=') != std::string::npos) { continue; }
        if (const tokens_t* mLine = find_nanowire_model(tok, iObj.netlist, s.second)) {
            const tokens_t mToks = tokenize_model_params(*mLine);
            for (size_t k = 0; k + 1 < mToks.size(); k += 2) {
                const std::string& key = mToks[k];
                const double       val = parse_param(mToks[k + 1], iObj.parameters, s.second);
                if (!std::isfinite(val)) { continue; }
                if (key == "LG" || key == "LGATE") {
                    outLg = std::max(0.0, val);
                } else if (key == "RG" || key == "RGATE") {
                    outRg = std::max(0.0, val);
                }
            }
        }
    }

    for (size_t idx = 3; idx < s.first.size(); ++idx) {
        const std::string& tok   = s.first.at(idx);
        const size_t       eqPos = tok.find('=');
        if (eqPos == std::string::npos) { continue; }
        const std::string key = tok.substr(0, eqPos);
        const std::string val = tok.substr(eqPos + 1);
        if (key == "LG" || key == "LGATE") {
            outLg = std::max(0.0, parse_param(val, iObj.parameters, s.second));
        } else if (key == "RG" || key == "RGATE") {
            outRg = std::max(0.0, parse_param(val, iObj.parameters, s.second));
        } else if (key == "SPREAD") {
            outSpread = parse_param(val, iObj.parameters, s.second);
        }
    }
}

Nanowire::Nanowire(const std::pair<tokens_t, string_o>& s,
                   const NodeConfig&                    ncon,
                   const nodemap&                       nm,
                   std::unordered_set<std::string>&     lm,
                   nodeconnections&                     nc,
                   Input&                               iObj,
                   Spread&                              spread,
                   int64_t&                             bi) {
    at_ = iObj.argAnal;
    h_  = iObj.transSim.tstep();

    if (lm.count(s.first.at(0)) != 0) {
        Errors::invalid_component_errors(ComponentErrors::DUPLICATE_LABEL, s.first.at(0));
    }
    netlistInfo.label_ = s.first.at(0);
    lm.emplace(s.first.at(0));

    auto apply_param = [&](const std::string& key, const std::string& valStr) {
        if (key == "GATE" || key == "GATE_IND" || key == "GATEDEV") {
            gateDevLabel_ = valStr;
            return;
        }
        const double val = parse_param(valStr, iObj.parameters, s.second);
        if (!std::isfinite(val)) { return; }
        if (key == "IC" || key == "ISW" || key == "ICRIT" || key == "IC0") {
            nominalIc0_ = std::max(1e-12, std::abs(val));
        } else if (key == "IG" || key == "IGSW" || key == "IGCRIT" || key == "IG0") {
            nominalIgSw_ = std::max(0.0, std::abs(val));
        } else if (key == "IR" || key == "IRETRAP" || key == "IHOLD") {
            nominalIr_     = std::max(0.0, std::abs(val));
            hasExplicitIr_ = true;
        } else if (key == "RETRAP") {
            retrapRatio_ = std::clamp(val, 0.0, 1.0);
        } else if (key == "RN" || key == "RHS" || key == "RMAX" || key == "RNORM") {
            nominalRn_ = std::max(1e-6, val);
        } else if (key == "L" || key == "LCH" || key == "LK" || key == "LKIN") {
            nominalLch_ = std::max(1e-16, val);
        } else if (key == "LG" || key == "LGATE") {
            lg_ = std::max(0.0, val);
        } else if (key == "RG" || key == "RGATE") {
            rg_ = std::max(0.0, val);
        } else if (key == "ALPHA" || key == "EXP" || key == "GAMMA") {
            alpha_ = std::max(0.1, val);
        } else if (key == "TAU" || key == "TAU_TH" || key == "TAUTH" || key == "TFALL" || key == "TREC") {
            tauTh_ = std::max(1e-15, val);
        } else if (key == "TAU_HS" || key == "TAUHS" || key == "TRISE" || key == "TNUC") {
            tauHs_ = std::max(1e-15, val);
        } else if (key == "AREA" || key == "MULT" || key == "W") {
            nominalArea_ = std::max(1e-6, val);
        } else if (key == "SPREAD") {
            spread_ = val;
        }
    };

    // 1. Apply any .model definition referenced on the device line
    for (size_t idx = 3; idx < s.first.size(); ++idx) {
        const std::string& tok = s.first.at(idx);
        if (tok.find('=') != std::string::npos) { continue; }
        if (const tokens_t* mLine = find_nanowire_model(tok, iObj.netlist, s.second)) {
            const tokens_t mToks = tokenize_model_params(*mLine);
            for (size_t k = 0; k + 1 < mToks.size(); k += 2) {
                apply_param(mToks[k], mToks[k + 1]);
            }
        }
    }

    // 2. Apply inline parameter overrides (or positional IC at token 3)
    for (size_t idx = 3; idx < s.first.size(); ++idx) {
        const std::string& tok   = s.first.at(idx);
        const size_t       eqPos = tok.find('=');
        if (eqPos != std::string::npos) {
            apply_param(tok.substr(0, eqPos), tok.substr(eqPos + 1));
        } else if (idx == 3 && find_nanowire_model(tok, iObj.netlist, s.second) == nullptr
                   && !is_nanowire_model_keyword(tok)) {
            const double v = parse_param(tok, iObj.parameters, s.second, false);
            if (std::isfinite(v) && v > 0.0) { nominalIc0_ = v; }
        }
    }

    if (!hasExplicitIr_) {
        nominalIr_ = retrapRatio_ * nominalIc0_;
    }

    area_ = spread.spread_value(nominalArea_, Spread::JJ, spread_);
    apply_area_scale(1.0, h_);

    indexInfo.nodeConfig_   = ncon;
    indexInfo.currentIndex_ = bi++;
    set_node_indices(tokens_t(s.first.begin() + 1, s.first.begin() + 3), nm, nc);
    set_matrix_info();
    matrixInfo.nonZeros_.emplace_back(diag_entry(h_));
}

void Nanowire::apply_area_scale(double scale, double h) noexcept {
    const double effArea = std::max(1e-12, area_ * scale);
    ic0_                 = nominalIc0_ * effArea;
    igSw_                = nominalIgSw_ * effArea;
    ir_                  = (hasExplicitIr_ ? nominalIr_ : (retrapRatio_ * nominalIc0_)) * effArea;
    rn_                  = nominalRn_ / effArea;
    lch_                 = std::max(1e-16, nominalLch_ / effArea);
    netlistInfo.value_   = ic0_;
    rCh_                 = rn_ * w_;
    h_                   = h;
    if (!matrixInfo.nonZeros_.empty()) {
        matrixInfo.nonZeros_.back() = diag_entry(h_);
    }
}

double Nanowire::diag_entry(double h) const noexcept {
    if (at_ == AnalysisType::Voltage) {
        return -((1.5 * lch_ / h) + rCh_);
    }
    return -((lch_ / Constants::SIGMA) + ((2.0 * h) / (3.0 * Constants::SIGMA)) * rCh_);
}

double Nanowire::switching_current(double iGate) const noexcept {
    if (igSw_ <= 0.0 || gateDevLabel_.empty()) { return ic0_; }
    const double ratio = std::abs(iGate) / igSw_;
    if (ratio >= 1.0) { return 0.0; }
    if (ratio <= 0.0) { return ic0_; }
    const double supp = (alpha_ == 1.0) ? (1.0 - ratio) : (1.0 - std::pow(ratio, alpha_));
    return ic0_ * std::max(0.0, supp);
}

bool Nanowire::step_hotspot(double iCh, double iGate, double h) noexcept {
    const double absIch = std::abs(iCh);
    const double iSw    = switching_current(iGate);
    const bool   overSw = (absIch > iSw && absIch > 1e-9);

    if (state_ == 0) {
        if (overSw) {
            state_ = 1;
        }
    } else {
        if (!overSw && absIch <= ir_) {
            state_ = 0;
        }
    }

    double wNew = w_;
    if (state_ == 1) {
        const double overDrive = (overSw && ic0_ > 0.0) ? ((absIch - iSw) / (0.25 * ic0_)) : 1.0;
        const double tauEff    = tauHs_ / std::max(1.0, overDrive);
        wNew                   = 1.0 - (1.0 - w_) * std::exp(-h / std::max(1e-15, tauEff));
        if (wNew > 0.9999) { wNew = 1.0; }
    } else if (w_ > 0.0) {
        wNew = w_ * std::exp(-h / std::max(1e-15, tauTh_));
        if (wNew < 1e-4) { wNew = 0.0; }
    }

    w_                  = wNew;
    const double rChNew = rn_ * w_;
    if (std::abs(rChNew - rCh_) > 1e-5 * rn_ || (rChNew == 0.0 && rCh_ != 0.0) || (rChNew != 0.0 && rCh_ == 0.0)) {
        rCh_                        = rChNew;
        matrixInfo.nonZeros_.back() = diag_entry(h);
        return true;
    }
    return false;
}

void Nanowire::update_timestep(const double& factor) {
    h_ *= factor;
    matrixInfo.nonZeros_.back() = diag_entry(h_);
}
