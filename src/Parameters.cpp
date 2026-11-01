// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Parameters.hpp"

#include "JoSIM/Constants.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/Misc.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>

using namespace JoSIM;

namespace {

bool has_top_level_equals(const std::string& tok) {
    int depth = 0;
    for (char c : tok) {
        if (c == '(' || c == '{' || c == '[') {
            ++depth;
        } else if (c == ')' || c == '}' || c == ']') {
            depth = std::max(0, depth - 1);
        } else if (c == '=' && depth == 0) {
            return true;
        }
    }
    return false;
}

std::string strip_outer_braces(std::string s) {
    while (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
        int  depth      = 0;
        bool closesAtEnd = true;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '{') {
                ++depth;
            } else if (s[i] == '}') {
                --depth;
                if (depth == 0) {
                    closesAtEnd = false;
                    break;
                }
            }
        }
        if (closesAtEnd) {
            s = s.substr(1, s.size() - 2);
        } else {
            break;
        }
    }
    return s;
}

const std::vector<std::string> funcs1 = {"COS",   "SIN",   "TAN",   "ACOS",  "ASIN",  "ATAN",  "COSH",  "SINH",
                                         "TANH",  "ACOSH", "ASINH", "ATANH", "EXP",   "EXP2",  "LOG",   "LN",
                                         "LOG10", "LOG2",  "SQRT",  "CBRT",  "ABS",   "FABS",  "SGN",   "SIGN",
                                         "FLOOR", "CEIL",  "ROUND", "NINT",  "TRUNC", "INT",   "FRAC",  "STEP",
                                         "U",     "URAMP", "ERF",   "ERFC",  "DB",    "NEG"};

const std::vector<std::string> funcs2 = {"MIN", "MAX", "ATAN2", "POW", "PWR", "PWRS", "HYPOT", "MOD", "FMOD"};

const std::vector<std::string> funcs3 = {"IF", "TERNARY", "LIMIT", "CLAMP"};

bool is_func1(const std::string& s) { return std::find(funcs1.begin(), funcs1.end(), s) != funcs1.end(); }
bool is_func2(const std::string& s) { return std::find(funcs2.begin(), funcs2.end(), s) != funcs2.end(); }
bool is_func3(const std::string& s) { return std::find(funcs3.begin(), funcs3.end(), s) != funcs3.end(); }
bool is_known_func(const std::string& s) { return is_func1(s) || is_func2(s) || is_func3(s); }

const Parameter* lookup_scoped_param(const std::string& name, const param_map& params, const string_o& subc) {
    if (subc.has_value()) {
        std::string curScope = subc.value();
        while (true) {
            auto it = params.find(ParameterName(name, curScope));
            if (it != params.end()) { return &it->second; }
            const size_t barPos = curScope.rfind('|');
            if (barPos == std::string::npos) { break; }
            curScope = curScope.substr(0, barPos);
        }
    }
    auto itGlobal = params.find(ParameterName(name, std::nullopt));
    if (itGlobal != params.end()) { return &itGlobal->second; }
    return nullptr;
}

double eval_func3(const std::string& op, double v0, double v1, double v2) {
    if (op == "IF" || op == "TERNARY") {
        return (std::abs(v0) > 0.0) ? v1 : v2;
    } else if (op == "LIMIT" || op == "CLAMP") {
        const double lo = std::min(v1, v2);
        const double hi = std::max(v1, v2);
        return std::clamp(v0, lo, hi);
    }
    return 0.0;
}

} // namespace

void JoSIM::create_parameter(const tokens_t& s, param_map& parameters, string_o subc) {
    if (s.size() < 2) { Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s)); }
    if (Misc::vector_to_string(s).find('=') == std::string::npos) {
        Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s));
    }

    // Step 1: balance multi-token { ... } or ( ... ) fragments
    tokens_t balanced;
    for (size_t i = 1; i < s.size(); ++i) {
        if (s[i] == "PARAMS:" || s[i] == "PARAM:") { continue; }
        std::string cur   = s[i];
        int         depth = 0;
        for (char c : cur) {
            if (c == '(' || c == '{' || c == '[') {
                ++depth;
            } else if (c == ')' || c == '}' || c == ']') {
                --depth;
            }
        }
        while (depth > 0 && i + 1 < s.size()) {
            ++i;
            cur += s[i];
            for (char c : s[i]) {
                if (c == '(' || c == '{' || c == '[') {
                    ++depth;
                } else if (c == ')' || c == '}' || c == ']') {
                    --depth;
                }
            }
        }
        balanced.push_back(std::move(cur));
    }

    // Step 2: attach standalone, trailing, or leading '=' to adjacent tokens
    tokens_t merged;
    for (size_t i = 0; i < balanced.size(); ++i) {
        if (balanced[i] == "=" && !merged.empty() && i + 1 < balanced.size()) {
            merged.back() += "=" + balanced[i + 1];
            ++i;
        } else if (!merged.empty() && merged.back().back() == '=') {
            merged.back() += balanced[i];
        } else if (!balanced[i].empty() && balanced[i].front() == '=' && !merged.empty()) {
            merged.back() += balanced[i];
        } else {
            merged.push_back(balanced[i]);
        }
    }

    // Step 3: group into one or more NAME=EXPR assignments
    std::vector<std::pair<std::string, std::string>> assignments;
    for (const auto& tok : merged) {
        if (has_top_level_equals(tok)) {
            const size_t eqPos = tok.find('=');
            std::string  pName = tok.substr(0, eqPos);
            std::string  pExpr = tok.substr(eqPos + 1);
            while (!pName.empty() && pName.front() == ',') { pName.erase(pName.begin()); }
            if (!pName.empty()) {
                assignments.emplace_back(std::move(pName), std::move(pExpr));
            }
        } else if (!assignments.empty()) {
            assignments.back().second += tok;
        }
    }

    if (assignments.empty()) {
        Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s));
        return;
    }

    for (auto& kv : assignments) {
        std::string& pName = kv.first;
        std::string& pExpr = kv.second;
        while (!pExpr.empty() && pExpr.back() == ',') { pExpr.pop_back(); }
        pExpr = strip_outer_braces(pExpr);
        if (pName.empty() || pExpr.empty()) {
            Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s));
            continue;
        }
        Parameter temp;
        temp.set_expression(pExpr);
        const ParameterName key(pName, subc);
        if (parameters.count(key) != 0) {
            Errors::parsing_errors(ParsingErrors::EXPRESSION_ARLEADY_DEFINED, Misc::vector_to_string(s));
        }
        parameters[key] = temp;
    }
}

int64_t JoSIM::precedence_lvl(const std::string& op) {
    if (op.empty()) { return 4; }
    if (is_known_func(op)) { return 4; }
    switch (op.at(0)) {
        case '+':
        case '-': return 1;
        case '*':
        case '/': return 2;
        case '^': return 3;
        default: return 4;
    }
}

double JoSIM::parse_param(const std::string& expr, const param_map& params, string_o subc, bool single) {
    std::string expToEval = strip_outer_braces(expr);
    expToEval.erase(std::remove_if(expToEval.begin(), expToEval.end(), ::isspace), expToEval.end());
    if (expToEval.empty()) { return 0.0; }
    if (expToEval.front() == '.') { expToEval = "0" + expToEval; }

    std::vector<std::string> rpnQueue, rpnQueueCopy, opStack;
    std::vector<char>        qType, qTypeCopy;
    std::vector<int>         parenCommaCount;
    std::string              partToEval;
    int64_t                  popCount          = 0;
    double                   result            = 0.0;
    bool                     prevTokenWasValue = false;

    auto is_sci_exponent_sign = [&](int64_t pos) -> bool {
        if (pos <= 1 || pos + 1 >= static_cast<int64_t>(expToEval.size())) { return false; }
        if (expToEval[pos] != '-' && expToEval[pos] != '+') { return false; }
        if (expToEval[pos - 1] != 'E') { return false; }
        const char beforeE = expToEval[pos - 2];
        const char afterOp = expToEval[pos + 1];
        return (std::isdigit(static_cast<unsigned char>(beforeE)) || beforeE == '.')
               && std::isdigit(static_cast<unsigned char>(afterOp));
    };

    while (!expToEval.empty()) {
        int64_t opLoc = static_cast<int64_t>(expToEval.find_first_of("/*-+(){}[]^,"));

        if (opLoc == 0 && (expToEval[0] == '-' || expToEval[0] == '+') && !prevTokenWasValue) {
            if (expToEval.size() > 1
                && (std::isdigit(static_cast<unsigned char>(expToEval[1])) || expToEval[1] == '.')) {
                opLoc = static_cast<int64_t>(expToEval.find_first_of("/*-+(){}[]^,", 1));
                while (opLoc != -1 && is_sci_exponent_sign(opLoc)) {
                    opLoc = static_cast<int64_t>(expToEval.find_first_of("/*-+(){}[]^,", static_cast<size_t>(opLoc + 1)));
                }
            } else {
                if (expToEval[0] == '-') { opStack.push_back("NEG"); }
                expToEval = expToEval.substr(1);
                continue;
            }
        } else {
            while (opLoc != -1 && is_sci_exponent_sign(opLoc)) {
                opLoc = static_cast<int64_t>(expToEval.find_first_of("/*-+(){}[]^,", static_cast<size_t>(opLoc + 1)));
            }
        }

        if (opLoc == -1) {
            partToEval = expToEval;
        } else if (opLoc == 0) {
            partToEval = expToEval.substr(0, 1);
        } else {
            partToEval = expToEval.substr(0, static_cast<size_t>(opLoc));
        }

        if (partToEval.front() == '.' && partToEval.size() > 1) {
            partToEval = "0" + partToEval;
        } else if ((partToEval.front() == '-' || partToEval.front() == '+') && partToEval.size() > 2
                   && partToEval[1] == '.') {
            partToEval.insert(1, "0");
        }

        if (std::isdigit(static_cast<unsigned char>(partToEval.at(0)))
            || ((partToEval.at(0) == '-' || partToEval.at(0) == '+') && partToEval.size() > 1)) {
            rpnQueue.push_back(Misc::precise_to_string(Misc::modifier(partToEval)));
            qType.push_back('V');
            prevTokenWasValue = true;
        } else if (const Parameter* p = lookup_scoped_param(partToEval, params, subc)) {
            if (!p->get_value().has_value()) {
                if (!single) { return std::numeric_limits<double>::quiet_NaN(); }
                Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, expr);
                return 0.0;
            }
            rpnQueue.push_back(Misc::precise_to_string(p->get_value().value()));
            qType.push_back('V');
            prevTokenWasValue = true;
        } else if (is_known_func(partToEval)) {
            opStack.push_back(partToEval);
            prevTokenWasValue = false;
        } else if (Misc::string_constant(partToEval) != 0.0) {
            rpnQueue.push_back(Misc::precise_to_string(Misc::string_constant(partToEval)));
            qType.push_back('V');
            prevTokenWasValue = true;
        } else if (partToEval == ",") {
            while (!opStack.empty() && opStack.back().find_first_of("([{") == std::string::npos) {
                rpnQueue.push_back(opStack.back());
                qType.push_back('O');
                opStack.pop_back();
            }
            if (opStack.empty()) { Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr); }
            if (!parenCommaCount.empty()) {
                parenCommaCount.back()++;
                if (parenCommaCount.back() >= 2 && opStack.size() >= 2) {
                    const std::string& fn = opStack[opStack.size() - 2];
                    if (fn == "MIN" || fn == "MAX") {
                        rpnQueue.push_back(fn);
                        qType.push_back('O');
                    }
                }
            }
            prevTokenWasValue = false;
        } else if (partToEval.find_first_of("/*-+^") != std::string::npos) {
            while ((!opStack.empty())
                   && (((precedence_lvl(opStack.back()) == 4)
                        || (precedence_lvl(opStack.back()) > precedence_lvl(partToEval))
                        || (precedence_lvl(opStack.back()) == precedence_lvl(partToEval) && partToEval != "^"))
                       && (opStack.back().find_first_of("([{") == std::string::npos))) {
                rpnQueue.push_back(opStack.back());
                qType.push_back('O');
                opStack.pop_back();
            }
            opStack.push_back(partToEval);
            prevTokenWasValue = false;
        } else if (partToEval.find_first_of("([{") != std::string::npos) {
            opStack.push_back(partToEval);
            parenCommaCount.push_back(0);
            prevTokenWasValue = false;
        } else if (partToEval.find_first_of(")]}") != std::string::npos) {
            while ((!opStack.empty()) && (opStack.back().find_first_of("([{") == std::string::npos)) {
                rpnQueue.push_back(opStack.back());
                qType.push_back('O');
                opStack.pop_back();
            }
            if ((!opStack.empty()) && (opStack.back().find_first_of("([{") != std::string::npos)) {
                opStack.pop_back();
                if (!parenCommaCount.empty()) { parenCommaCount.pop_back(); }
                if (!opStack.empty() && is_known_func(opStack.back())) {
                    rpnQueue.push_back(opStack.back());
                    qType.push_back('O');
                    opStack.pop_back();
                }
            } else {
                Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr);
            }
            prevTokenWasValue = true;
        } else {
            if (!single) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, expr);
            }
        }

        if (opLoc == 0) {
            expToEval = expToEval.substr(1);
        } else if (opLoc == -1) {
            expToEval = "";
        } else {
            expToEval = expToEval.substr(static_cast<size_t>(opLoc));
        }
    }

    while (!opStack.empty()) {
        if (opStack.back().find_first_of("([{") != std::string::npos) {
            Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr);
            opStack.pop_back();
        } else {
            rpnQueue.push_back(opStack.back());
            qType.push_back('O');
            opStack.pop_back();
        }
    }

    if (rpnQueue.empty()) { return 0.0; }

    while (rpnQueue.size() > 1) {
        rpnQueueCopy.clear();
        qTypeCopy.clear();
        bool progressed = false;
        for (size_t i = 0; i < qType.size(); ++i) {
            if (qType[i] == 'V') {
                rpnQueueCopy.push_back(rpnQueue[i]);
                qTypeCopy.push_back('V');
            } else if (qType[i] == 'O') {
                const std::string& op = rpnQueue[i];
                if (i == 0) {
                    Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
                    return 0.0;
                } else if (is_func3(op)) {
                    if (i < 3) {
                        Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
                        return 0.0;
                    }
                    result = eval_func3(
                            op, Misc::modifier(rpnQueue[i - 3]), Misc::modifier(rpnQueue[i - 2]), Misc::modifier(rpnQueue[i - 1]));
                    for (int k = 0; k < 3; ++k) { rpnQueueCopy.pop_back(); }
                    qTypeCopy.pop_back();
                    qTypeCopy.pop_back();
                    rpnQueueCopy.push_back(Misc::precise_to_string(result));
                } else if (i < 2) {
                    rpnQueueCopy.pop_back();
                    rpnQueueCopy.push_back(
                            Misc::precise_to_string(parse_operator(op, 0.0, Misc::modifier(rpnQueue[i - 1]), popCount)));
                } else {
                    result = parse_operator(op, Misc::modifier(rpnQueue[i - 2]), Misc::modifier(rpnQueue[i - 1]), popCount);
                    for (int64_t k = 0; k < popCount; ++k) { rpnQueueCopy.pop_back(); }
                    if (popCount == 2) { qTypeCopy.pop_back(); }
                    rpnQueueCopy.push_back(Misc::precise_to_string(result));
                }
                if (rpnQueue.size() > i + 1) {
                    rpnQueueCopy.insert(rpnQueueCopy.end(), rpnQueue.begin() + static_cast<ptrdiff_t>(i + 1), rpnQueue.end());
                    qTypeCopy.insert(qTypeCopy.end(), qType.begin() + static_cast<ptrdiff_t>(i + 1), qType.end());
                }
                progressed = true;
                break;
            }
        }
        if (!progressed) { break; }
        rpnQueue = rpnQueueCopy;
        qType    = qTypeCopy;
    }
    return Misc::modifier(rpnQueue.back());
}

double JoSIM::parse_operator(const std::string& op, double val1, double val2, int64_t& popCount) {
    if (is_func1(op)) {
        popCount = 1;
        if (op == "SIN") {
            return std::sin(val2);
        } else if (op == "COS") {
            return std::cos(val2);
        } else if (op == "TAN") {
            return std::tan(val2);
        } else if (op == "ASIN") {
            return std::asin(val2);
        } else if (op == "ACOS") {
            return std::acos(val2);
        } else if (op == "ATAN") {
            return std::atan(val2);
        } else if (op == "SINH") {
            return std::sinh(val2);
        } else if (op == "COSH") {
            return std::cosh(val2);
        } else if (op == "TANH") {
            return std::tanh(val2);
        } else if (op == "ASINH") {
            return std::asinh(val2);
        } else if (op == "ACOSH") {
            return std::acosh(val2);
        } else if (op == "ATANH") {
            return std::atanh(val2);
        } else if (op == "EXP") {
            return std::exp(val2);
        } else if (op == "EXP2") {
            return std::exp2(val2);
        } else if (op == "LOG" || op == "LN") {
            return std::log(val2);
        } else if (op == "LOG10") {
            return std::log10(val2);
        } else if (op == "LOG2") {
            return std::log2(val2);
        } else if (op == "SQRT") {
            return std::sqrt(val2);
        } else if (op == "CBRT") {
            return std::cbrt(val2);
        } else if (op == "ABS" || op == "FABS") {
            return std::abs(val2);
        } else if (op == "SGN" || op == "SIGN") {
            return (val2 > 0.0) ? 1.0 : ((val2 < 0.0) ? -1.0 : 0.0);
        } else if (op == "FLOOR") {
            return std::floor(val2);
        } else if (op == "CEIL") {
            return std::ceil(val2);
        } else if (op == "ROUND" || op == "NINT") {
            return std::round(val2);
        } else if (op == "TRUNC" || op == "INT") {
            return std::trunc(val2);
        } else if (op == "FRAC") {
            return val2 - std::trunc(val2);
        } else if (op == "STEP" || op == "U") {
            return (val2 >= 0.0) ? 1.0 : 0.0;
        } else if (op == "URAMP") {
            return (val2 > 0.0) ? val2 : 0.0;
        } else if (op == "ERF") {
            return std::erf(val2);
        } else if (op == "ERFC") {
            return std::erfc(val2);
        } else if (op == "DB") {
            return 20.0 * std::log10(std::max(1e-300, std::abs(val2)));
        } else if (op == "NEG") {
            return -val2;
        }
    } else if (is_func2(op)) {
        popCount = 2;
        if (op == "MIN") {
            return std::min(val1, val2);
        } else if (op == "MAX") {
            return std::max(val1, val2);
        } else if (op == "ATAN2") {
            return std::atan2(val1, val2);
        } else if (op == "POW") {
            return std::pow(val1, val2);
        } else if (op == "PWR") {
            return std::pow(std::abs(val1), val2);
        } else if (op == "PWRS") {
            const double s = (val1 < 0.0) ? -1.0 : 1.0;
            return s * std::pow(std::abs(val1), val2);
        } else if (op == "HYPOT") {
            return std::hypot(val1, val2);
        } else if (op == "MOD" || op == "FMOD") {
            return (val2 != 0.0) ? std::fmod(val1, val2) : 0.0;
        }
    } else {
        popCount = 2;
        switch (op.at(0)) {
            case '+': return val1 + val2;
            case '-': return val1 - val2;
            case '*': return val1 * val2;
            case '/': return val1 / val2;
            case '^': return std::pow(val1, val2);
        }
    }
    return 0.0;
}

void JoSIM::parse_parameters(param_map& parameters) {
    double value;
    size_t parsedCounter = 0;
    for (const auto& i : parameters) {
        if (i.second.get_value()) { parsedCounter++; }
    }
    while (parsedCounter < parameters.size()) {
        int64_t previous_counter = static_cast<int64_t>(parsedCounter);
        for (auto& i : parameters) {
            if (!i.second.get_value()) {
                value = parse_param(i.second.get_expression(), parameters, i.first.subcircuit(), false);
                if (!std::isnan(value)) {
                    i.second.set_value(value);
                    parsedCounter++;
                }
            }
        }
        if (previous_counter == static_cast<int64_t>(parsedCounter)) {
            std::string unknownParams;
            for (auto& i : parameters) {
                if (!i.second.get_value()) {
                    unknownParams += i.first.name() + " " + i.first.subcircuit().value_or("") + "\n";
                }
            }
            Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, unknownParams);
        }
    }
}

void JoSIM::update_parameters(param_map& parameters) {
    for (auto& i : parameters) { i.second.reset_value(); }
    parse_parameters(parameters);
}

void JoSIM::expand_inline_parameters(std::pair<tokens_t, string_o>& s, param_map& parameters) {
    while (true) {
        int_o oPos, cPos;
        for (size_t i = 0; i < s.first.size(); ++i) {
            if (!oPos && s.first.at(i).find('{') != std::string::npos) { oPos = static_cast<int64_t>(i); }
            if (s.first.at(i).find('}') != std::string::npos) {
                cPos = static_cast<int64_t>(i);
                if (oPos) { break; }
            }
        }
        if (!oPos && !cPos) { break; }
        if (!oPos || !cPos || cPos.value() < oPos.value()) {
            Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s.first));
            break;
        }
        if (cPos.value() != oPos.value()) {
            s.first.at(oPos.value()) = Misc::vector_to_string(
                    tokens_t{s.first.begin() + oPos.value(), s.first.begin() + cPos.value() + 1}, "");
            s.first.erase(s.first.begin() + oPos.value() + 1, s.first.begin() + cPos.value() + 1);
        }
        const std::string& tok    = s.first.at(oPos.value());
        const size_t       lBrace = tok.find('{');
        const size_t       rBrace = tok.find('}', lBrace);
        if (lBrace == std::string::npos || rBrace == std::string::npos || rBrace <= lBrace + 1) {
            Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s.first));
            break;
        }
        const std::string prefix = tok.substr(0, lBrace);
        const std::string expr   = tok.substr(lBrace + 1, rBrace - lBrace - 1);
        const std::string suffix = tok.substr(rBrace + 1);
        const double      value  = parse_param(expr, parameters, s.second, false);
        if (std::isnan(value)) {
            Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, Misc::vector_to_string(s.first));
        }
        s.first.at(oPos.value()) = prefix + Misc::precise_to_string(value) + suffix;
    }
}
