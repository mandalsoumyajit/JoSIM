// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Function.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Misc.hpp"
#include "JoSIM/Parameters.hpp"
#include "spline.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace JoSIM;

namespace {

bool decode_bit_pattern(const std::string& rawIn, std::vector<double>& outBits) {
    outBits.clear();
    std::string s = rawIn;
    for (const char* pfx : {"PATTERN=", "PAT=", "BITS=", "VEC="}) {
        if (s.rfind(pfx, 0) == 0) {
            s = s.substr(std::strlen(pfx));
            break;
        }
    }
    if (s.empty()) { return false; }

    if (s.rfind("0X", 0) == 0 || s.rfind("HEX:", 0) == 0) {
        const size_t start = (s.rfind("0X", 0) == 0) ? 2 : 4;
        if (start >= s.size()) { return false; }
        for (size_t i = start; i < s.size(); ++i) {
            const char c = s[i];
            if (c == '_') { continue; }
            int nib = -1;
            if (c >= '0' && c <= '9') {
                nib = c - '0';
            } else if (c >= 'A' && c <= 'F') {
                nib = 10 + (c - 'A');
            } else {
                return false;
            }
            outBits.push_back(((nib >> 3) & 1) ? 1.0 : 0.0);
            outBits.push_back(((nib >> 2) & 1) ? 1.0 : 0.0);
            outBits.push_back(((nib >> 1) & 1) ? 1.0 : 0.0);
            outBits.push_back((nib & 1) ? 1.0 : 0.0);
        }
        return !outBits.empty();
    }

    if (s.size() >= 2 && s[0] == 'B') { s = s.substr(1); }
    for (char c : s) {
        if (c == '_') {
            continue;
        } else if (c == '1' || c == '+' || c == 'H') {
            outBits.push_back(1.0);
        } else if (c == '0' || c == 'L') {
            outBits.push_back(0.0);
        } else if (c == '-' || c == 'N') {
            outBits.push_back(-1.0);
        } else {
            outBits.clear();
            return false;
        }
    }
    return !outBits.empty();
}

bool is_explicit_pattern_token(const std::string& tok) {
    if (tok.rfind("0X", 0) == 0 || tok.rfind("HEX:", 0) == 0 || tok.rfind("PAT=", 0) == 0
        || tok.rfind("BITS=", 0) == 0 || tok.rfind("PATTERN=", 0) == 0) {
        std::vector<double> tmp;
        return decode_bit_pattern(tok, tmp);
    }
    if (tok.size() >= 2 && tok[0] == 'B') {
        std::vector<double> tmp;
        return decode_bit_pattern(tok, tmp);
    }
    if (tok.size() >= 2 && tok.find('_') != std::string::npos) {
        std::vector<double> tmp;
        return decode_bit_pattern(tok, tmp);
    }
    return false;
}

bool is_raw_bitstring_token(const std::string& tok) {
    if (tok.empty()) { return false; }
    bool hasBitChar = false;
    for (size_t i = 0; i < tok.size(); ++i) {
        const char c = tok[i];
        if (c == '0' || c == '1') {
            hasBitChar = true;
        } else if (c == '_' || c == '-' || c == 'N' || c == '+') {
            // valid ternary / separator
        } else {
            return false;
        }
    }
    return hasBitChar;
}

} // namespace

void Function::parse_function(const std::string& str, const Input& iObj, const string_o& subckt) {
    const auto openParen  = str.find('(');
    const auto closeParen = str.rfind(')');
    std::string fHead;
    if (openParen != std::string::npos) {
        size_t endKw = openParen;
        while (endKw > 0 && std::isspace(static_cast<unsigned char>(str[endKw - 1]))) { --endKw; }
        size_t startKw = endKw;
        while (startKw > 0 && !std::isspace(static_cast<unsigned char>(str[startKw - 1]))) { --startKw; }
        fHead = str.substr(startKw, endKw - startKw);
    }
    char devPrefix = 'I';
    for (char c : str) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            devPrefix = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            break;
        }
    }

    const auto  first = (openParen != std::string::npos) ? (openParen + 1) : 0;
    const auto  last  = (closeParen != std::string::npos && closeParen >= first) ? closeParen : str.size();
    std::string p     = str.substr(first, last - first);
    tokens_t    t     = Misc::tokenize(p, " ,");

    /* PWL(0 V0 T1 V1 T2 V2 ... TN VN) */
    if (fHead.find("PWL") != std::string::npos || (fHead.empty() && str.find("PWL") != std::string::npos)) {
        fType_ = FunctionType::PWL;
        parse_pwl(t, iObj, subckt);
    }
    /* PULSE(A1 A2 TD[0.0] TR[TSTEP] TF[TSTEP] PW[TSTOP] PER[TSTOP])*/
    else if (fHead.find("PULSE") != std::string::npos || (fHead.empty() && str.find("PULSE") != std::string::npos)) {
        fType_ = FunctionType::PULSE;
        parse_pulse(t, iObj, subckt);
    }
    /* SIN(VO VA FREQ[1/TSTOP] TD[0.0] THETA[0.0]) */
    else if (fHead.find("SIN") != std::string::npos || (fHead.empty() && str.find("SIN") != std::string::npos)) {
        fType_ = FunctionType::SINUSOID;
        parse_sin(t, iObj, subckt);
    }
    /* CUS(WaveFile.dat TS[TSTEP] SF[1.0] IM[1] TD[0.0] PER[0]) */
    else if (fHead.find("CUS") != std::string::npos || (fHead.empty() && str.find("CUS") != std::string::npos)) {
        fType_ = FunctionType::CUS;
        parse_cus(t, iObj, subckt);
    }
    /* NOISE(VA TD TSTEP) */
    else if (fHead.find("NOISE") != std::string::npos || (fHead.empty() && str.find("NOISE") != std::string::npos)) {
        fType_ = FunctionType::NOISE;
        parse_noise(t, iObj, subckt);
    }
    /* PWS(0 0 T1 V1 T2 V2 .. TN VN) */
    else if (fHead.find("PWS") != std::string::npos || (fHead.empty() && str.find("PWS") != std::string::npos)) {
        fType_ = FunctionType::PWS;
        parse_pwl(t, iObj, subckt);
    }
    /* EXP(V1 V2 TD1[0.0] TAU1[TSTEP] TD2[TD1+TSTEP] TAU2[TSTEP]) */
    else if (fHead.find("EXP") != std::string::npos || (fHead.empty() && str.find("EXP") != std::string::npos)) {
        fType_ = FunctionType::EXP;
        parse_exp(t, iObj, subckt);
    }
    /* PAT(V0 V1 TD TR TF [PW] PER PATTERN [RZ|NRZ] [R=repeat]) */
    else if (fHead.find("PAT") != std::string::npos) {
        fType_ = FunctionType::PAT;
        parse_pat(t, iObj, subckt);
    }
    /* SFQ([TD [PER [FWHM [PATTERN_OR_COUNT]]]] [TIMES=t1,t2,...] [AMP=val] [PHI0=mult]) */
    else if (fHead.find("SFQ") != std::string::npos) {
        fType_ = FunctionType::SFQ;
        parse_sfq(t, iObj, subckt, devPrefix);
    }
    /* DC VALUE */
    else if (str.find("DC") != std::string::npos) {
        if (!std::isnan(parse_param(Misc::tokenize(str).back(), iObj.parameters, subckt))) {
            fType_ = FunctionType::DC;
            parse_dc(t, iObj, subckt);
        }
    }
    /* Only a value was provided? Assume user is trying to make a DC source */
    else if (!std::isnan(parse_param(Misc::tokenize(str).back(), iObj.parameters, subckt))) {
        fType_ = FunctionType::DC;
        parse_dc(t, iObj, subckt);
    } else {
        // Assume DC
    }
}

void Function::parse_pwl(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* PWL(0 V0 T1 V1 T2 V2 ... TN VN) */

    std::vector<double> timesteps;
    std::vector<double> values;

    // Must have pairs
    if (t.size() < 2 || (t.size() % 2) != 0) {
        Errors::function_errors(FunctionErrors::TOO_FEW_VALUES,
                                "Expected time/value pairs, got " + std::to_string(t.size()) + " tokens");
        timeValues_.clear();
        ampValues_.clear();
        return;
    }

    // Parse first pair using parse_param (NOT stod)
    double t0 = parse_param(t.at(0), iObj.parameters, s);
    double v0 = parse_param(t.at(1), iObj.parameters, s);

    // If first time isn't 0, prepend (0, v0) instead of (0,0)
    if (t0 != 0.0) {
        Errors::function_errors(FunctionErrors::INITIAL_VALUES, t.at(0) + " & " + t.at(1));
        timesteps.push_back(0.0);
        values.push_back(v0);
    }

    // Add the actual first point
    timesteps.push_back(t0);
    values.push_back(v0);

    // Parse remaining pairs
    for (size_t i = 2; i < t.size(); i += 2) {
        double ti = parse_param(t.at(i), iObj.parameters, s);
        double vi = parse_param(t.at(i + 1), iObj.parameters, s);

        // Enforce strictly increasing time
        if (ti <= timesteps.back()) {
            Errors::function_errors(FunctionErrors::TOO_FEW_TIMESTEPS,
                                    "Time values must be strictly increasing. Got " + std::to_string(ti) + " after "
                                            + std::to_string(timesteps.back()));
            // You can choose to return/throw; returning prevents undefined behavior later
            timeValues_.clear();
            ampValues_.clear();
            return;
        }

        timesteps.push_back(ti);
        values.push_back(vi);
    }

    // Final sanity (should always match now)
    if (timesteps.size() != values.size()) {
        Errors::function_errors(FunctionErrors::TOO_FEW_VALUES,
                                std::to_string(timesteps.size()) + " timesteps & " + std::to_string(values.size())
                                        + " values");
    }

    timeValues_ = std::move(timesteps);
    ampValues_  = std::move(values);
}

void Function::parse_pulse(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* PULSE(A1 A2 TD[0.0] TR[TSTEP] TF[TSTEP] PW[TSTOP] PER[TSTOP])*/
    if (t.size() < 2) { Errors::function_errors(FunctionErrors::PULSE_TOO_FEW_ARGUMENTS, std::to_string(t.size())); }
    // Step and stop time shorthand
    const double& tstop = iObj.transSim.tstop();
    const double& tstep = iObj.transSim.tstep();
    // Set the 2 amplitudes
    ampValues_.emplace_back(parse_param(t.at(0), iObj.parameters, s));
    ampValues_.emplace_back(parse_param(t.at(1), iObj.parameters, s));
    // Set default values for optional arguments
    timeValues_.emplace_back(0.0);
    timeValues_.emplace_back(tstep);
    timeValues_.emplace_back(tstep);
    timeValues_.emplace_back(tstop);
    timeValues_.emplace_back(tstop);
    // Test if values exist
    if (t.size() >= 3) { timeValues_.at(0) = parse_param(t.at(2), iObj.parameters, s); }
    if (t.size() >= 4) { timeValues_.at(1) = parse_param(t.at(3), iObj.parameters, s); }
    if (t.size() >= 5) { timeValues_.at(2) = parse_param(t.at(4), iObj.parameters, s); }
    if (t.size() >= 6) { timeValues_.at(3) = parse_param(t.at(5), iObj.parameters, s); }
    if (t.size() >= 7) { timeValues_.at(4) = parse_param(t.at(6), iObj.parameters, s); }
    if (iObj.argVerb) {
        // If pulse width is 0.0 then there is no pulse, then useless so warn
        if (timeValues_.at(3) == 0.0) { Errors::function_errors(FunctionErrors::PULSE_WIDTH_ZERO, t.at(5)); }
        // If repeat time is 0.0 then impossible, so warn
        if (timeValues_.at(4) == 0.0) { Errors::function_errors(FunctionErrors::PULSE_REPEAT, t.at(6)); }
    }
    // Store the stop time
    timeValues_.emplace_back(tstop);
}

void Function::parse_sin(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* SIN(VO VA FREQ[1/TSTOP] TD[0.0] THETA[0.0] PHASE[0.0deg] NCYCLES[0]) */

    if (t.size() < 2) {
        Errors::function_errors(FunctionErrors::SIN_TOO_FEW_ARGUMENTS, std::to_string(t.size()));
        return;
    }

    const double tstop = iObj.transSim.tstop();

    ampValues_.clear();
    timeValues_.clear();

    // VO, VA
    ampValues_.emplace_back(parse_param(t.at(0), iObj.parameters, s)); // VO
    ampValues_.emplace_back(parse_param(t.at(1), iObj.parameters, s)); // VA

    if (iObj.argVerb && ampValues_.at(1) == 0.0) { Errors::function_errors(FunctionErrors::SIN_VA_ZERO, t.at(1)); }

    // timeValues_ layout:
    // [0]=FREQ (Hz), [1]=TD (s), [2]=THETA (1/s), [3]=PHASE (rad), [4]=NCYCLES
    timeValues_.assign(5, 0.0);

    timeValues_.at(0) = (tstop > 0.0) ? (1.0 / tstop) : 0.0; // default freq
    timeValues_.at(1) = 0.0;                                 // TD
    timeValues_.at(2) = 0.0;                                 // THETA
    timeValues_.at(3) = 0.0;                                 // PHASE (rad)
    timeValues_.at(4) = 0.0;                                 // NCYCLES

    if (t.size() >= 3) { timeValues_.at(0) = parse_param(t.at(2), iObj.parameters, s); }
    if (t.size() >= 4) { timeValues_.at(1) = parse_param(t.at(3), iObj.parameters, s); }
    if (t.size() >= 5) { timeValues_.at(2) = parse_param(t.at(4), iObj.parameters, s); }

    // PHASE in degrees -> radians (SPICE convention)
    if (t.size() >= 6) {
        const double phase_deg = parse_param(t.at(5), iObj.parameters, s);
        timeValues_.at(3)      = phase_deg * (Constants::PI / 180.0);
    }

    // NCYCLES (0 or negative => infinite)
    if (t.size() >= 7) { timeValues_.at(4) = parse_param(t.at(6), iObj.parameters, s); }

    // Basic sanity (avoid div-by-zero when using NCYCLES)
    if (iObj.argVerb && timeValues_.at(4) > 0.0 && timeValues_.at(0) <= 0.0) {
        Errors::function_errors(FunctionErrors::SIN_TOO_FEW_ARGUMENTS, "NCYCLES requires positive FREQ");
    }
}

void Function::parse_cus(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* CUS(WaveFile.dat TS[TSTEP] SF[1.0] IM[1] TD[0.0] PER[0]) */
    if (t.size() < 2) { Errors::function_errors(FunctionErrors::CUS_TOO_FEW_ARGUMENTS, std::to_string(t.size())); }
    // First argument is required and is the wave file name
    std::filesystem::path WFline(t.at(0));
    if (!WFline.has_parent_path()) {
        WFline = std::filesystem::path(iObj.fileParentPath.value());
        WFline.append(t.at(0));
    }
    if (!std::filesystem::exists(WFline)) {
        auto pp   = WFline.parent_path();
        auto stem = WFline.filename().string();
        std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
        WFline = std::filesystem::path(pp.append(stem));
        if (!std::filesystem::exists(WFline)) { Errors::function_errors(FunctionErrors::CUS_WF_NOT_FOUND, t.at(0)); }
    }
    std::vector<std::string> WF;
    const double&            tstep = iObj.transSim.tstep();
    // Default values for the arguments
    miscValues_.emplace_back(tstep);
    miscValues_.emplace_back(1.0);
    miscValues_.emplace_back(1);
    miscValues_.emplace_back(0.0);
    miscValues_.emplace_back(0);
    // Test if optional values exist
    if (t.size() >= 2) { miscValues_.at(0) = parse_param(t.at(1), iObj.parameters, s); }
    if (t.size() >= 3) { miscValues_.at(1) = parse_param(t.at(2), iObj.parameters, s); }
    if (t.size() >= 4) { miscValues_.at(2) = parse_param(t.at(3), iObj.parameters, s); }
    if (t.size() >= 5) { miscValues_.at(3) = parse_param(t.at(4), iObj.parameters, s); }
    if (t.size() >= 6) { miscValues_.at(4) = parse_param(t.at(5), iObj.parameters, s); }

    if (iObj.argVerb) {
        if (miscValues_.at(1) == 0.0) { Errors::function_errors(FunctionErrors::CUS_SF_ZERO, t.at(1)); }
    }
    if (miscValues_.at(2) > 2 || miscValues_.at(2) < 0) {
        Errors::function_errors(FunctionErrors::CUS_UNKNOWN_IM, Misc::vector_to_string(t));
    }
    std::ifstream wffile(std::filesystem::absolute(WFline));
    std::string   line;
    if (wffile.good()) {
        getline(wffile, line);
    } else {
        Errors::function_errors(FunctionErrors::CUS_WF_NOT_FOUND, line);
    }
    wffile.close();
    WF = Misc::tokenize(line, " ,;");
    for (auto i = 0; i < WF.size(); ++i) { ampValues_.emplace_back(Misc::modifier(WF.at(i)) * miscValues_.at(1)); }
    if (miscValues_.at(0) < tstep) { miscValues_.at(0) = tstep; }
    for (int i = 0; i < ampValues_.size(); ++i) { timeValues_.emplace_back(miscValues_.at(3) + i * miscValues_.at(0)); }
    if (static_cast<int>(std::round(miscValues_.at(2))) == 2 && timeValues_.size() >= 3) {
        cusSpline_ = std::make_shared<tk::spline>(timeValues_, ampValues_);
    }
    // Store the stop time
    miscValues_.emplace_back(iObj.transSim.tstop());
}

void Function::parse_noise(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* NOISE(VA TD[0.0] TSTEP[TSTEP]) */
    if (t.size() < 2) { Errors::function_errors(FunctionErrors::NOISE_TOO_FEW_ARGUMENTS, std::to_string(t.size())); }
    const double& tstep = iObj.transSim.tstep();
    // Set amplitudes
    ampValues_.emplace_back(parse_param(t.at(0), iObj.parameters, s));
    ampValues_.emplace_back(0.0);
    // Default values for the arguments
    timeValues_.emplace_back(0.0);
    timeValues_.emplace_back(tstep);
    // Test if optional values exist
    if (t.size() >= 2) { timeValues_.at(0) = parse_param(t.at(1), iObj.parameters, s); }
    if (t.size() >= 3) { timeValues_.at(1) = parse_param(t.at(2), iObj.parameters, s); }
    miscValues_.emplace_back(0.0);
    miscValues_.emplace_back(ampValues_.at(0) * Misc::grand() / sqrt(2.0 * timeValues_.back()));
    miscValues_.emplace_back(timeValues_.at(0));
    miscValues_.emplace_back(timeValues_.at(0) + timeValues_.at(1));
    if (iObj.argVerb) {
        if (ampValues_.at(0) == 0.0) { Errors::function_errors(FunctionErrors::NOISE_VA_ZERO, t.at(1)); }
    }
}

void Function::parse_dc(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* DC VALUE */
    // Set amplitudes
    ampValues_.emplace_back(parse_param(t.back(), iObj.parameters, s));
}

void Function::parse_exp(const tokens_t& t, const Input& iObj, const string_o& s) {
    /* EXP(V1 V2 TD1[0.0] TAU1[TSTEP] TD2[TD1+TSTEP] TAU2[TSTEP]) */
    if (t.size() < 2) { Errors::function_errors(FunctionErrors::EXP_TOO_FEW_ARGUMENTS, std::to_string(t.size())); }
    const double& tstep = iObj.transSim.tstep();
    // Set amplitudes
    ampValues_.emplace_back(parse_param(t.at(0), iObj.parameters, s));
    ampValues_.emplace_back(parse_param(t.at(1), iObj.parameters, s));
    // Default values for the arguments
    timeValues_.emplace_back(0.0);
    timeValues_.emplace_back(tstep);
    timeValues_.emplace_back(tstep);
    timeValues_.emplace_back(tstep);
    // Test if optional values exist
    if (t.size() >= 3) { timeValues_.at(0) = parse_param(t.at(2), iObj.parameters, s); }
    if (t.size() >= 4) { timeValues_.at(1) = parse_param(t.at(3), iObj.parameters, s); }
    if (t.size() >= 5) { timeValues_.at(2) = parse_param(t.at(4), iObj.parameters, s); }
    if (t.size() >= 6) { timeValues_.at(3) = parse_param(t.at(5), iObj.parameters, s); }
}

void Function::parse_pat(const tokens_t& t, const Input& iObj, const string_o& s) {
    /*
       Supported forms:
         1. PAT(V0 V1 TD TR TF PW PER <PATTERN> [RZ|NRZ] [R=repeat])
         2. PAT(V0 V1 TD TR TF PER <PATTERN> [RZ|NRZ] [RB [R]])
         3. PAT(V0 V1 TD PER <PATTERN> [RZ|NRZ] [PW=val])
       <PATTERN> supports binary/ternary ('1', '0', '-', 'N', '_'), 'B1011...', or hex '0xA5' / 'HEX:A5'.
    */
    ampValues_.clear();
    timeValues_.clear();
    miscValues_.clear();

    const double tstep = (iObj.transSim.tstep() > 0.0) ? iObj.transSim.tstep() : 1e-12;
    double       v0    = 0.0;
    double       v1    = 0.0;
    double       td    = 0.0;
    double       tr    = tstep;
    double       tf    = tstep;
    double       pw    = -1.0;
    double       per   = 20.0 * tstep;
    int          rzOpt = -1; // -1 = auto, 0 = NRZ, 1 = RZ
    double       rep   = 0.0;
    double       rb    = 0.0;

    bool                hasKwV0 = false, hasKwV1 = false, hasKwTd = false, hasKwTr = false, hasKwTf = false;
    bool                hasKwPw = false, hasKwPer = false;
    std::vector<double> bits;
    tokens_t            posToks;

    for (const auto& tok : t) {
        if (tok == "RZ" || tok == "RZ=1" || tok == "RZ=TRUE" || tok == "NRZ=0" || tok == "NRZ=FALSE") {
            rzOpt = 1;
        } else if (tok == "NRZ" || tok == "NRZ=1" || tok == "NRZ=TRUE" || tok == "RZ=0" || tok == "RZ=FALSE") {
            rzOpt = 0;
        } else if (tok == "REPEAT" || tok == "INF" || tok == "R=INF" || tok == "REPEAT=INF") {
            rep = -1.0;
        } else if (tok.rfind("PAT=", 0) == 0 || tok.rfind("BITS=", 0) == 0 || tok.rfind("PATTERN=", 0) == 0
                   || tok.rfind("VEC=", 0) == 0) {
            decode_bit_pattern(tok, bits);
        } else if (tok.rfind("V0=", 0) == 0 || tok.rfind("VLO=", 0) == 0 || tok.rfind("ALOW=", 0) == 0) {
            v0      = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
            hasKwV0 = true;
        } else if (tok.rfind("V1=", 0) == 0 || tok.rfind("VHI=", 0) == 0 || tok.rfind("AHIGH=", 0) == 0) {
            v1      = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
            hasKwV1 = true;
        } else if (tok.rfind("TD=", 0) == 0) {
            td      = parse_param(tok.substr(3), iObj.parameters, s);
            hasKwTd = true;
        } else if (tok.rfind("TR=", 0) == 0) {
            tr      = parse_param(tok.substr(3), iObj.parameters, s);
            hasKwTr = true;
        } else if (tok.rfind("TF=", 0) == 0) {
            tf      = parse_param(tok.substr(3), iObj.parameters, s);
            hasKwTf = true;
        } else if (tok.rfind("PW=", 0) == 0) {
            pw      = parse_param(tok.substr(3), iObj.parameters, s);
            hasKwPw = true;
        } else if (tok.rfind("PER=", 0) == 0 || tok.rfind("TBIT=", 0) == 0 || tok.rfind("TSAMPLE=", 0) == 0) {
            per      = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
            hasKwPer = true;
        } else if (tok.rfind("R=", 0) == 0 || tok.rfind("REPEAT=", 0) == 0) {
            const std::string rv = tok.substr(tok.find('=') + 1);
            rep                  = (rv == "INF" || rv == "-1") ? -1.0 : parse_param(rv, iObj.parameters, s);
        } else if (tok.rfind("RB=", 0) == 0) {
            rb = std::max(0.0, parse_param(tok.substr(3), iObj.parameters, s));
        } else {
            posToks.push_back(tok);
        }
    }

    size_t patPosIdx = SIZE_MAX;
    if (bits.empty()) {
        for (size_t i = 0; i < posToks.size(); ++i) {
            if (is_explicit_pattern_token(posToks[i])) {
                decode_bit_pattern(posToks[i], bits);
                patPosIdx = i;
                break;
            }
        }
    }
    if (bits.empty() && !posToks.empty()) {
        for (size_t idx = posToks.size(); idx > 0; --idx) {
            const size_t i = idx - 1;
            if (i >= 2 && is_raw_bitstring_token(posToks[i])) {
                decode_bit_pattern(posToks[i], bits);
                patPosIdx = i;
                break;
            }
        }
    }

    tokens_t numToks;
    for (size_t i = 0; i < posToks.size(); ++i) {
        if (i == patPosIdx) {
            if (i + 1 < posToks.size()) { rb = std::max(0.0, parse_param(posToks[i + 1], iObj.parameters, s)); }
            if (i + 2 < posToks.size()) { rep = parse_param(posToks[i + 2], iObj.parameters, s); }
            break;
        }
        numToks.push_back(posToks[i]);
    }

    if (numToks.size() >= 1 && !hasKwV0) { v0 = parse_param(numToks[0], iObj.parameters, s); }
    if (numToks.size() >= 2 && !hasKwV1) { v1 = parse_param(numToks[1], iObj.parameters, s); }
    if (numToks.size() >= 3 && !hasKwTd) { td = parse_param(numToks[2], iObj.parameters, s); }

    if (numToks.size() >= 7) {
        if (!hasKwTr) { tr = parse_param(numToks[3], iObj.parameters, s); }
        if (!hasKwTf) { tf = parse_param(numToks[4], iObj.parameters, s); }
        if (!hasKwPw) { pw = parse_param(numToks[5], iObj.parameters, s); }
        if (!hasKwPer) { per = parse_param(numToks[6], iObj.parameters, s); }
    } else if (numToks.size() == 6) {
        if (!hasKwTr) { tr = parse_param(numToks[3], iObj.parameters, s); }
        if (!hasKwTf) { tf = parse_param(numToks[4], iObj.parameters, s); }
        if (!hasKwPer) { per = parse_param(numToks[5], iObj.parameters, s); }
    } else if (numToks.size() == 5) {
        if (!hasKwTr) { tr = parse_param(numToks[3], iObj.parameters, s); }
        if (!hasKwTf) { tf = tr; }
        if (!hasKwPer) { per = parse_param(numToks[4], iObj.parameters, s); }
    } else if (numToks.size() == 4) {
        if (!hasKwPer) { per = parse_param(numToks[3], iObj.parameters, s); }
    }

    bool isNrz = true;
    if (rzOpt == 1) {
        isNrz = false;
    } else if (rzOpt == 0) {
        isNrz = true;
    } else if (pw > 0.0 && pw < per) {
        isNrz = false;
    }

    if (!isNrz && pw <= 0.0) {
        pw = std::max(tstep, 0.5 * per - tr - tf);
    } else if (pw <= 0.0) {
        pw = per;
    }

    if (bits.empty()) {
        Errors::function_errors(FunctionErrors::PULSE_TOO_FEW_ARGUMENTS, "PAT requires a bit pattern");
        bits.push_back(0.0);
    }

    ampValues_  = {v0, v1};
    timeValues_ = {td, std::max(0.0, tr), std::max(0.0, tf), std::max(0.0, pw), std::max(tstep, per), isNrz ? 1.0 : 0.0, rep, rb};
    miscValues_ = std::move(bits);
}

void Function::parse_sfq(const tokens_t& t, const Input& iObj, const string_o& s, char devPrefix) {
    /*
       SFQ([TD [PER [FWHM [PATTERN_OR_COUNT]]]])
       Keyword overrides: TD=, PER=, FWHM=/W=, AMP=/VA=/I=, PHI0=/NPHI0=, PAT=/BITS=, COUNT=/N=, TIMES=/AT=, R=/REPEAT=, MODE=V|P|I
    */
    ampValues_.clear();
    timeValues_.clear();
    miscValues_.clear();

    const double tstop    = (iObj.transSim.tstop() > 0.0) ? iObj.transSim.tstop() : 100e-12;
    double       td       = 5e-12;
    double       per      = 20e-12;
    double       fwhm     = 2e-12;
    double       phi0Mult = 1.0;
    double       userAmp  = -1.0;
    int64_t      count    = -1;
    double       rep      = 0.0;
    char         modeChar = devPrefix;

    bool                hasKwTd = false, hasKwPer = false, hasKwFwhm = false;
    std::vector<double> bits;
    std::vector<double> explicitTimes;
    std::vector<double> explicitSigns;
    tokens_t            posToks;

    auto parse_times_list = [&](const std::string& listStr) {
        size_t pos = 0;
        while (pos < listStr.size()) {
            size_t next = listStr.find_first_of(",;:", pos);
            if (next == std::string::npos) { next = listStr.size(); }
            if (next > pos) {
                std::string item = listStr.substr(pos, next - pos);
                double      sign = 1.0;
                if (!item.empty() && item[0] == '-') {
                    sign = -1.0;
                    item = item.substr(1);
                } else if (!item.empty() && item[0] == '+') {
                    item = item.substr(1);
                }
                const double tv = parse_param(item, iObj.parameters, s);
                if (std::isfinite(tv) && tv >= 0.0) {
                    explicitTimes.push_back(tv);
                    explicitSigns.push_back(sign);
                }
            }
            pos = next + 1;
        }
    };

    for (const auto& tok : t) {
        if (tok.rfind("TD=", 0) == 0) {
            td      = parse_param(tok.substr(3), iObj.parameters, s);
            hasKwTd = true;
        } else if (tok.rfind("PER=", 0) == 0 || tok.rfind("TBIT=", 0) == 0) {
            per      = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
            hasKwPer = true;
        } else if (tok.rfind("FWHM=", 0) == 0 || tok.rfind("W=", 0) == 0 || tok.rfind("PW=", 0) == 0
                   || tok.rfind("WIDTH=", 0) == 0) {
            fwhm      = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
            hasKwFwhm = true;
        } else if (tok.rfind("AMP=", 0) == 0 || tok.rfind("VA=", 0) == 0 || tok.rfind("I=", 0) == 0
                   || tok.rfind("IPEAK=", 0) == 0 || tok.rfind("VPEAK=", 0) == 0) {
            userAmp = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
        } else if (tok.rfind("PHI0=", 0) == 0 || tok.rfind("NPHI0=", 0) == 0 || tok.rfind("FLUX=", 0) == 0) {
            phi0Mult = parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s);
        } else if (tok.rfind("COUNT=", 0) == 0 || tok.rfind("N=", 0) == 0 || tok.rfind("NPULSES=", 0) == 0) {
            count = static_cast<int64_t>(std::round(parse_param(tok.substr(tok.find('=') + 1), iObj.parameters, s)));
        } else if (tok.rfind("PAT=", 0) == 0 || tok.rfind("BITS=", 0) == 0 || tok.rfind("PATTERN=", 0) == 0) {
            decode_bit_pattern(tok, bits);
        } else if (tok.rfind("TIMES=", 0) == 0 || tok.rfind("AT=", 0) == 0) {
            parse_times_list(tok.substr(tok.find('=') + 1));
        } else if (tok.rfind("R=", 0) == 0 || tok.rfind("REPEAT=", 0) == 0) {
            const std::string rv = tok.substr(tok.find('=') + 1);
            rep                  = (rv == "INF" || rv == "-1") ? -1.0 : parse_param(rv, iObj.parameters, s);
        } else if (tok.rfind("MODE=", 0) == 0) {
            const std::string mv = tok.substr(5);
            if (!mv.empty()) { modeChar = mv[0]; }
        } else {
            posToks.push_back(tok);
        }
    }

    size_t patPosIdx = SIZE_MAX;
    if (bits.empty()) {
        for (size_t i = 0; i < posToks.size(); ++i) {
            if (is_explicit_pattern_token(posToks[i])) {
                decode_bit_pattern(posToks[i], bits);
                patPosIdx = i;
                break;
            }
        }
    }

    tokens_t numToks;
    for (size_t i = 0; i < posToks.size(); ++i) {
        if (i == patPosIdx) { continue; }
        numToks.push_back(posToks[i]);
    }

    if (bits.empty() && numToks.size() >= 4 && is_raw_bitstring_token(numToks.back()) && numToks.back().size() >= 2) {
        decode_bit_pattern(numToks.back(), bits);
        numToks.pop_back();
    } else if (bits.empty() && count < 0 && numToks.size() >= 4) {
        count = static_cast<int64_t>(std::round(parse_param(numToks.back(), iObj.parameters, s)));
        numToks.pop_back();
    }

    if (numToks.size() >= 1 && !hasKwTd) { td = parse_param(numToks[0], iObj.parameters, s); }
    if (numToks.size() >= 2 && !hasKwPer) { per = parse_param(numToks[1], iObj.parameters, s); }
    if (numToks.size() >= 3 && !hasKwFwhm) { fwhm = parse_param(numToks[2], iObj.parameters, s); }

    if (numToks.size() == 1 && !hasKwPer && bits.empty() && count < 0 && explicitTimes.empty()) {
        count = 1;
    }

    fwhm                  = std::max(1e-15, std::abs(fwhm));
    const bool isPhaseSrc = (modeChar == 'P');
    double     peakAmp    = 0.0;
    if (userAmp >= 0.0) {
        peakAmp = userAmp * phi0Mult;
    } else if (isPhaseSrc) {
        peakAmp = 2.0 * Constants::PI * phi0Mult;
    } else if (modeChar == 'I') {
        peakAmp = 500e-6 * phi0Mult;
    } else {
        peakAmp = (Constants::PHI_ZERO / fwhm) * phi0Mult;
    }

    std::vector<std::pair<double, double>> events;
    if (!explicitTimes.empty()) {
        for (size_t i = 0; i < explicitTimes.size(); ++i) {
            events.emplace_back(explicitTimes[i], explicitSigns[i]);
        }
    } else if (!bits.empty()) {
        const int64_t nBits  = static_cast<int64_t>(bits.size());
        const int64_t passes = (rep < 0.0) ? std::max<int64_t>(1, static_cast<int64_t>(std::ceil((tstop - td) / std::max(1e-15, per * nBits))) + 1)
                                           : (1 + std::max<int64_t>(0, static_cast<int64_t>(std::round(rep))));
        for (int64_t p = 0; p < passes; ++p) {
            for (int64_t k = 0; k < nBits; ++k) {
                const double b = bits[static_cast<size_t>(k)];
                if (std::abs(b) > 0.0) {
                    const double tk = td + static_cast<double>(p * nBits + k) * per;
                    if (tk <= tstop + 4.0 * fwhm) { events.emplace_back(tk, b); }
                }
            }
        }
    } else {
        if (per <= 0.0 || count == 1) {
            events.emplace_back(td, 1.0);
        } else {
            const int64_t maxCount = (count > 0)
                                             ? count
                                             : (static_cast<int64_t>(std::ceil(std::max(0.0, tstop - td) / per)) + 2);
            for (int64_t k = 0; k < maxCount; ++k) {
                const double tk = td + static_cast<double>(k) * per;
                if (count < 0 && tk > tstop + 4.0 * fwhm) { break; }
                events.emplace_back(tk, 1.0);
            }
        }
    }

    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    ampValues_ = {peakAmp};
    timeValues_.reserve(2 + events.size());
    timeValues_.push_back(fwhm);
    timeValues_.push_back(isPhaseSrc ? 1.0 : 0.0);
    miscValues_.reserve(events.size());
    for (const auto& ev : events) {
        timeValues_.push_back(ev.first);
        miscValues_.push_back(ev.second);
    }
}

double Function::return_pwl(double& x) {
    // If x larger than the last timestep then assume last amplitude value
    if (x >= timeValues_.back()) {
        return ampValues_.back();
        // Else check within which range x falls
    } else {
        const size_t n = timeValues_.size() - 1;
        size_t       i = (lastIdx_ < n && x >= timeValues_[lastIdx_]) ? lastIdx_ : 0;
        for (; i < n; ++i) {
            if (x >= timeValues_[i] && x < timeValues_[i + 1]) {
                lastIdx_   = i;
                double& y2 = ampValues_[i + 1];
                double& y1 = ampValues_[i];
                double& x2 = timeValues_[i + 1];
                double& x1 = timeValues_[i];
                // Calculate function value and return it
                return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1);
            }
        }
    }
    return 0.0;
}

double Function::return_pulse(double& x) {
    // Shorthand for the relevant time values
    double& td    = timeValues_.at(0);
    double& tr    = timeValues_.at(1);
    double& tf    = timeValues_.at(2);
    double& pw    = timeValues_.at(3);
    double& per   = timeValues_.at(4);
    double& alow  = ampValues_.at(0);
    double& ahigh = ampValues_.at(1);
    auto    val   = alow;
    if (x < td) { return (val); }
    int64_t index = static_cast<int64_t>((x - td) / per);
    auto    time  = x - td - index * per;
    if (time < tr) {
        val = alow + (ahigh - alow) * time / tr;
    } else if (time < tr + pw) {
        val = ahigh;
    } else if (time < tr + pw + tf) {
        val = ahigh + (alow - ahigh) * (time - tr - pw) / tf;
    } else {
        val = alow;
    }

    return (val);
}

double Function::return_sin(double& x) {
    const double vo      = ampValues_.at(0);
    const double va      = ampValues_.at(1);

    const double freq    = timeValues_.at(0);
    const double td      = timeValues_.at(1);
    const double theta   = timeValues_.at(2);
    const double phase   = timeValues_.at(3); // radians
    const double ncycles = timeValues_.at(4);

    // Before delay: DC offset only
    if (x < td) { return vo; }

    const double t = x - td;

    // Stop after N cycles (if requested)
    if (ncycles > 0.0 && freq > 0.0) {
        const double t_end = ncycles / freq;
        if (t >= t_end) {
            return vo; // "stops" at offset
        }
    }

    // Damped sine: VO + VA * exp(-theta*t) * sin(2*pi*freq*t + phase)
    return vo + va * std::sin(2.0 * Constants::PI * freq * t + phase) * std::exp(-theta * t);
}

double Function::return_cus(double& x) {
    auto& ts  = miscValues_.at(0);
    int   im  = static_cast<int>(std::round(miscValues_.at(2)));
    auto& td  = miscValues_.at(3);
    bool  per = static_cast<bool>(std::round(miscValues_.at(4)));
    if (x < td) {
        return 0.0;
    } else {
        double region = ampValues_.size() * ts;
        double norm_x = (x - td);
        if (per) { norm_x = (x - td) - (std::floor(x / region) * region); }
        if (norm_x > region) { return 0.0; }
        int idx = std::min(static_cast<int>(norm_x / ts), static_cast<int>(ampValues_.size() - 1));
        if (im == 0) {
            return ampValues_.at(idx);
        } else if (im == 1) {
            double& y2 = ampValues_.at(idx);
            double  y1 = 0.0;
            if (idx != 0) { y1 = ampValues_.at(idx - 1); }
            return y1 + ((y2 - y1) / ts) * (norm_x - (idx * ts));
        } else if (im == 2) {
            if (!cusSpline_) { cusSpline_ = std::make_shared<tk::spline>(timeValues_, ampValues_); }
            return (*static_cast<const tk::spline*>(cusSpline_.get()))(norm_x);
        }
    }
    return 0.0;
}

double Function::return_noise(double& x) {
    if (x < timeValues_.front()) {
        return ampValues_.at(1);
    } else {
        if (x >= miscValues_.at(3)) {
            miscValues_.at(0) = miscValues_.at(1);
            miscValues_.at(1) = ampValues_.at(0) * Misc::grand() / sqrt(2.0 * timeValues_.back());
            miscValues_.at(2) = miscValues_.at(3);
            if (x == miscValues_.at(3)) { miscValues_.at(3) = miscValues_.at(2) + timeValues_.back(); }
            if (x > miscValues_.at(3)) { miscValues_.at(3) = x; }
        }
        double& y2 = miscValues_.at(1);
        double& y1 = miscValues_.at(0);
        double& x2 = miscValues_.at(3);
        double& x1 = miscValues_.at(2);
        // Calculate function value and return it
        return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1);
    }
}

double Function::return_pws(double& x) {
    if (timeValues_.empty() || ampValues_.empty()) { return 0.0; }

    // Clamp before first point (optional but usually correct)
    if (x <= timeValues_.front()) { return ampValues_.front(); }

    // Hold last value after last time (your original behavior)
    if (x >= timeValues_.back()) { return ampValues_.back(); }

    // Find the segment [x1, x2)
    const size_t n = timeValues_.size() - 1;
    size_t       i = (lastIdx_ < n && x >= timeValues_[lastIdx_]) ? lastIdx_ : 0;
    for (; i < n; ++i) {
        double x1 = timeValues_[i];
        double x2 = timeValues_[i + 1];

        if (x >= x1 && x < x2) {
            lastIdx_  = i;
            double y1 = ampValues_[i];
            double y2 = ampValues_[i + 1];

            double dx = x2 - x1;
            if (dx <= 0.0) {
                return y2; // should never happen if parse validated
            }

            double u = (x - x1) / dx; // 0..1

            // sinusoidal spline (half-cosine)
            return y1 + (y2 - y1) * 0.5 * (1.0 - std::cos(Constants::PI * u));
        }
    }

    // Should be unreachable if x is within range
    return ampValues_.back();
}

double Function::return_dc() { return ampValues_.back(); }

double Function::return_exp(double& x) {
    // Shorthand for time values
    double& a1   = ampValues_.at(0);
    double& a2   = ampValues_.at(1);
    double& td1  = timeValues_.at(0);
    double& tau1 = timeValues_.at(1);
    double& td2  = timeValues_.at(2);
    double& tau2 = timeValues_.at(3);
    if (x < td1) {
        return ampValues_.at(0);
    } else if (x >= td1 && x < td2) {
        return a1 + (a2 - a1) * (1 - exp(-(x - td1) / tau1));
    } else if (x >= td2) {
        return a1 + (a2 - a1) * (1 - exp(-(x - td1) / tau1)) + (a1 - a2) * (1 - exp(-(x - td2) / tau2));
    }
    return 0.0;
}

double Function::return_pat(double& x) {
    if (ampValues_.size() < 2 || timeValues_.size() < 8 || miscValues_.empty()) { return 0.0; }
    const double  v0      = ampValues_[0];
    const double  v1      = ampValues_[1];
    const double  td      = timeValues_[0];
    const double  tr      = timeValues_[1];
    const double  tf      = timeValues_[2];
    const double  pw      = timeValues_[3];
    const double  per     = timeValues_[4];
    const bool    isNrz   = (timeValues_[5] > 0.5);
    const double  rep     = timeValues_[6];
    const int64_t rb      = std::clamp<int64_t>(static_cast<int64_t>(std::round(timeValues_[7])),
                                           0,
                                           static_cast<int64_t>(miscValues_.size()) - 1);
    const int64_t effBits = static_cast<int64_t>(miscValues_.size()) - rb;

    if (x < td || per <= 0.0 || effBits <= 0) { return v0; }

    auto get_bit = [&](int64_t k) -> double {
        if (k < 0) { return 0.0; }
        if (k < effBits) { return miscValues_[static_cast<size_t>(rb + k)]; }
        if (rep < 0.0) { return miscValues_[static_cast<size_t>(rb + (k % effBits))]; }
        const int64_t totalPasses = 1 + std::max<int64_t>(0, static_cast<int64_t>(std::round(rep)));
        if (k < effBits * totalPasses) { return miscValues_[static_cast<size_t>(rb + (k % effBits))]; }
        return 0.0;
    };

    const int64_t k      = static_cast<int64_t>(std::floor((x - td) / per));
    const double  localT = (x - td) - static_cast<double>(k) * per;
    const double  bitK   = get_bit(k);

    if (!isNrz) {
        if (std::abs(bitK) <= 0.0) { return v0; }
        const double vPeak = (bitK > 0.0) ? v1 : (v0 - (v1 - v0));
        if (localT < tr) {
            return (tr > 0.0) ? (v0 + (vPeak - v0) * (localT / tr)) : vPeak;
        } else if (localT < tr + pw) {
            return vPeak;
        } else if (localT < tr + pw + tf) {
            return (tf > 0.0) ? (vPeak + (v0 - vPeak) * ((localT - tr - pw) / tf)) : v0;
        }
        return v0;
    } else {
        auto level_of = [&](double b) -> double {
            return (b > 0.0) ? v1 : ((b < 0.0) ? (v0 - (v1 - v0)) : v0);
        };
        const double curVal  = level_of(bitK);
        const double prevVal = (k > 0) ? level_of(get_bit(k - 1)) : v0;
        if (std::abs(curVal - prevVal) <= 1e-30) { return curVal; }
        const double tTrans = (curVal > prevVal) ? tr : tf;
        if (tTrans > 0.0 && localT < tTrans) {
            return prevVal + (curVal - prevVal) * (localT / tTrans);
        }
        return curVal;
    }
}

double Function::return_sfq(double& x) {
    if (ampValues_.empty() || timeValues_.size() < 2 || miscValues_.empty()) { return 0.0; }
    const double peakAmp = ampValues_[0];
    const double fwhm    = timeValues_[0];
    const bool   isPhase = (timeValues_[1] > 0.5);
    const double dur     = 2.0 * fwhm;
    const size_t nPulses = miscValues_.size();

    if (!isPhase) {
        double       sum   = 0.0;
        const double tMin  = x - dur;
        const auto   itBeg = std::lower_bound(timeValues_.begin() + 2, timeValues_.end(), tMin);
        for (auto it = itBeg; it != timeValues_.end(); ++it) {
            const double t0 = *it;
            if (t0 > x) { break; }
            const double tau = x - t0;
            if (tau >= 0.0 && tau <= dur) {
                const size_t idx = static_cast<size_t>(std::distance(timeValues_.begin() + 2, it));
                const double s   = std::sin(Constants::PI * tau / dur);
                sum             += miscValues_[idx] * peakAmp * (s * s);
            }
        }
        return sum;
    } else {
        double sum = 0.0;
        for (size_t i = 0; i < nPulses; ++i) {
            const double t0 = timeValues_[2 + i];
            if (x <= t0) { break; }
            const double tau = x - t0;
            if (tau >= dur) {
                sum += miscValues_[i] * peakAmp;
            } else {
                const double u = tau / dur;
                sum += miscValues_[i] * peakAmp * (u - std::sin(2.0 * Constants::PI * u) / (2.0 * Constants::PI));
            }
        }
        return sum;
    }
}

double Function::value(double x) {
    switch (fType_) {
        case FunctionType::PWL: return return_pwl(x);
        case FunctionType::PULSE: return return_pulse(x);
        case FunctionType::SINUSOID: return return_sin(x);
        case FunctionType::CUS: return return_cus(x);
        case FunctionType::NOISE: return return_noise(x);
        case FunctionType::PWS: return return_pws(x);
        case FunctionType::DC: return return_dc();
        case FunctionType::EXP: return return_exp(x);
        case FunctionType::PAT: return return_pat(x);
        case FunctionType::SFQ: return return_sfq(x);
        default: return 0.0;
    }
}

void Function::ampValues(std::vector<double> values) { ampValues_ = values; }
