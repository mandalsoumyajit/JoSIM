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

std::string strip_outer_braces(std::string s) {
    while (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
        int  depth       = 0;
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

// One-argument functions (upstream's set plus the fork's additions).
const std::vector<std::string> funcs1 = {"COS",   "SIN",   "TAN",   "ACOS",  "ASIN",  "ATAN",  "COSH",  "SINH",
                                         "TANH",  "ACOSH", "ASINH", "ATANH", "EXP",   "EXP2",  "LOG",   "LN",
                                         "LOG10", "LOG2",  "SQRT",  "CBRT",  "ABS",   "FABS",  "SGN",   "SIGN",
                                         "FLOOR", "CEIL",  "ROUND", "NINT",  "TRUNC", "INT",   "FRAC",  "STEP",
                                         "U",     "URAMP", "ERF",   "ERFC",  "DB",    "NEG"};

// Two-argument functions; MIN and MAX also take more than two.
const std::vector<std::string> funcs2 = {"MIN", "MAX", "ATAN2", "POW", "PWR", "PWRS", "HYPOT", "MOD", "FMOD"};

// Three-argument functions.
const std::vector<std::string> funcs3 = {"IF", "TERNARY", "LIMIT", "CLAMP"};

bool is_func1(const std::string& s) { return std::find(funcs1.begin(), funcs1.end(), s) != funcs1.end(); }
bool is_func2(const std::string& s) { return std::find(funcs2.begin(), funcs2.end(), s) != funcs2.end(); }
bool is_func3(const std::string& s) { return std::find(funcs3.begin(), funcs3.end(), s) != funcs3.end(); }
bool is_known_func(const std::string& s) { return is_func1(s) || is_func2(s) || is_func3(s); }
bool is_variadic(const std::string& s) { return s == "MIN" || s == "MAX"; }

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

bool is_params_keyword(const std::string& token) { return token == "PARAMS:" || token == "PARAM:"; }

} // namespace

size_t JoSIM::parameter_start(const tokens_t& tokens, size_t first) {
    for (size_t i = first; i < tokens.size(); ++i) {
        if (is_params_keyword(tokens[i]) || tokens[i].find('=') != std::string::npos
            || (i + 1 < tokens.size() && tokens[i + 1].rfind("=", 0) == 0)) { return i; }
    }
    return tokens.size();
}

std::vector<std::pair<std::string, std::string>> JoSIM::parameter_assignments(const tokens_t& tokens, size_t first) {
    std::vector<std::pair<std::string, std::string>> result;
    if (first >= tokens.size()) { return result; }
    const auto text = Misc::vector_to_string(tokens_t(tokens.begin() + first, tokens.end()));
    auto fail = [&] { Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, text); };
    auto space = [](unsigned char c) { return std::isspace(c) || c == ','; };
    auto name_char = [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '|' || c == '.'; };
    size_t pos = 0;
    if (text.rfind("PARAMS:", 0) == 0) { pos = 7; }
    else if (text.rfind("PARAM:", 0) == 0) { pos = 6; }
    while (pos < text.size()) {
        while (pos < text.size() && space(text[pos])) { ++pos; }
        if (pos == text.size()) { break; }
        const auto begin = pos;
        while (pos < text.size() && name_char(text[pos])) { ++pos; }
        const auto name = text.substr(begin, pos - begin);
        while (pos < text.size() && space(text[pos])) { ++pos; }
        if (name.empty() || pos == text.size() || text[pos++] != '=') { fail(); }
        const auto expression = pos;
        std::string brackets;
        while (pos < text.size()) {
            const char c = text[pos];
            if (std::string("({[").find(c) != std::string::npos) { brackets += c; }
            else if (std::string(")}]").find(c) != std::string::npos) {
                if (brackets.empty() || std::string("({[").find(brackets.back()) != std::string(")}]").find(c)) { fail(); }
                brackets.pop_back();
            } else if (brackets.empty() && space(c)) {
                size_t next = pos;
                while (next < text.size() && space(text[next])) { ++next; }
                const auto nextName = next;
                while (next < text.size() && name_char(text[next])) { ++next; }
                const auto nextEnd = next;
                while (next < text.size() && space(text[next])) { ++next; }
                if (nextEnd > nextName && next < text.size() && text[next] == '=') { break; }
            }
            ++pos;
        }
        auto value = text.substr(expression, pos - expression);
        while (!value.empty() && space(value.back())) { value.pop_back(); }
        if (!brackets.empty() || value.find_first_not_of(" \t,") == std::string::npos
            || std::any_of(result.begin(), result.end(), [&](const auto& kv) { return kv.first == name; })) { fail(); }
        result.emplace_back(name, value);
    }
    if (result.empty()) { fail(); }
    return result;
}

string_o JoSIM::parent_parameter_scope(const string_o& scope) {
    if (scope && !scope->empty() && scope->front() == '|') {
        const auto separator = scope->find('|', 1);
        if (separator != std::string::npos) { return scope->substr(separator); }
    }
    return std::nullopt;
}

void JoSIM::create_parameter(const tokens_t& s, param_map& parameters, string_o subc) {
    if (s.size() < 2) { Errors::parsing_errors(ParsingErrors::INVALID_DECLARATION, Misc::vector_to_string(s)); }
    for (const auto& assignment : parameter_assignments(s, 1)) {
        Parameter temp;
        // A wholly braced value ({expr}) is stored as the bare expression, as the fork always has.
        temp.set_expression(strip_outer_braces(assignment.second));
        const ParameterName name(assignment.first, subc);
        if (parameters.count(name)) {
            Errors::parsing_errors(ParsingErrors::EXPRESSION_ARLEADY_DEFINED, Misc::vector_to_string(s));
        }
        parameters[name] = temp;
    }
}

int64_t JoSIM::precedence_lvl(const std::string& op) {
    if (op.empty()) { return 4; }
    if (is_known_func(op)) { return 4; }
    switch (op.at(0)) {
            // + and - are lowest level
        case '+':
        case '-':
            return 1;
            // * and / are higher level
        case '*':
        case '/':
            return 2;
            // ^ (pow) is highest level
        case '^':
            return 3;
            // default (functions) are max level
        default: return 4;
    }
}

double JoSIM::parse_param(const std::string& expr, const param_map& params, string_o subc, bool single) {
    std::string text = expr;
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c); }), text.end());
    if (text.empty()) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
    auto uppercase = [](std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
        return name;
    };
    auto parameter = [&](const std::string& name) {
        auto scope = subc;
        const auto upper = uppercase(name);
        for (;;) {
            auto found = params.find(ParameterName(name, scope));
            if (found == params.end() && upper != name) { found = params.find(ParameterName(upper, scope)); }
            if (found != params.end() || !scope) { return found; }
            scope = parent_parameter_scope(scope);
        }
    };
    auto resolved = [&](const Parameter& value) {
        if (!value.get_value()) {
            if (!single) { return std::numeric_limits<double>::quiet_NaN(); }
            Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, expr);
        }
        if (!std::isfinite(value.get_value().value())) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
        return value.get_value().value();
    };
    // Preserve exact whole-parameter lookup, including names supplied directly
    // by embedding callers, before interpreting an expression's operators.
    auto whole = parameter(text);
    if (whole != params.end()) { return resolved(whole->second); }

    // An operator token carries its argument count (functions take 1-3 arguments, MIN/MAX two or more).
    struct Token {
        std::string op;
        double      value;
        int64_t     arity = 0;
    };
    std::vector<Token>       output;
    std::vector<std::string> operators;
    // Commas seen inside each open bracket, in the order the brackets were opened.
    std::vector<int64_t>     commas;
    bool operand = true;
    auto function = [](const std::string& name) { return is_known_func(name); };
    auto opening = [](const std::string& token) { return token.size() == 1 && std::string("([{").find(token[0]) != std::string::npos; };
    auto priority = [](const std::string& op) { return op == "U-" || op == "U+" ? int64_t(3) : precedence_lvl(op); };
    auto emit = [&] {
        const std::string op = operators.back();
        operators.pop_back();
        output.push_back({op, 0.0, (op == "U-" || op == "U+" || is_func1(op)) ? 1 : 2});
    };
    const std::string delimiters = "+-*/^()[]{},";
    size_t position = 0;
    while (position < text.size()) {
        const char c = text[position];
        if (std::isdigit(static_cast<unsigned char>(c))
            || (c == '.' && position + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[position + 1])))) {
            if (!operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
            size_t numeric = 0;
            try { std::stod(text.substr(position), &numeric); }
            catch (const std::exception&) { Errors::misc_errors(MiscErrors::STOD_ERROR, text.substr(position)); }
            // std::stod identifies exponent signs as part of the number. Only
            // subsequent operators delimit its optional engineering/unit suffix.
            size_t end = text.find_first_of(delimiters, position + numeric);
            if (end == std::string::npos) { end = text.size(); }
            const auto literal = text.substr(position, end - position);
            const auto suffix = literal.substr(numeric);
            if ((!suffix.empty() && (suffix.front() == 'E' || suffix.front() == 'e'))
                || std::any_of(suffix.begin(), suffix.end(), [](unsigned char ch) { return !std::isalpha(ch); })) {
                Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
            }
            const double value = Misc::modifier(literal);
            if (!std::isfinite(value)) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
            output.push_back({"", value});
            position = end;
            operand = false;
            continue;
        }
        if (c == '(' || c == '[' || c == '{') {
            if (!operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
            operators.emplace_back(1, c);
            commas.push_back(0);
            ++position;
            continue;
        }
        if (c == ',') {
            // Argument separator, only inside brackets (a function's, checked at the closing bracket).
            if (operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
            while (!operators.empty() && !opening(operators.back())) { emit(); }
            if (operators.empty() || commas.empty()) { Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr); }
            ++commas.back();
            ++position;
            operand = true;
            continue;
        }
        if (c == ')' || c == ']' || c == '}') {
            if (operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
            while (!operators.empty() && !opening(operators.back())) { emit(); }
            if (operators.empty()
                || std::string("([{").find(operators.back()[0]) != std::string(")]}").find(c)) {
                Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr);
            }
            operators.pop_back();
            const int64_t args = 1 + (commas.empty() ? 0 : commas.back());
            if (!commas.empty()) { commas.pop_back(); }
            if (!operators.empty() && function(operators.back())) {
                const std::string fn     = operators.back();
                const int64_t     wanted = is_func1(fn) ? 1 : is_func2(fn) ? 2 : 3;
                if (args != wanted && !(is_variadic(fn) && args >= 2)) {
                    Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
                }
                operators.pop_back();
                output.push_back({fn, 0.0, args});
            } else if (args != 1) {
                Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
            }
            ++position;
            operand = false;
            continue;
        }
        if (std::string("+-*/^").find(c) != std::string::npos) {
            if (operand) {
                if (c != '+' && c != '-') { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
                // Unary signs bind after exponentiation but before products;
                // pushing them directly also permits a signed exponent.
                operators.push_back(c == '-' ? "U-" : "U+");
            } else {
                const std::string op(1, c);
                while (!operators.empty() && !opening(operators.back())
                       && (priority(operators.back()) > priority(op)
                           || (priority(operators.back()) == priority(op) && c != '^'))) { emit(); }
                operators.push_back(op);
                operand = true;
            }
            ++position;
            continue;
        }
        if (!operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
        size_t end = text.find_first_of(delimiters, position);
        if (end == std::string::npos) { end = text.size(); }
        const auto name = text.substr(position, end - position);
        auto found = parameter(name);
        if (found != params.end()) {
            const double value = resolved(found->second);
            if (std::isnan(value)) { return value; }
            output.push_back({"", value});
            operand = false;
        } else {
            const auto canonical = uppercase(name);
            if (function(canonical)) {
                if (end == text.size() || std::string("([{").find(text[end]) == std::string::npos) {
                    Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr);
                }
                operators.push_back(canonical);
            } else if (Misc::string_constant(canonical) != 0.0) {
                output.push_back({"", Misc::string_constant(canonical)});
                operand = false;
            } else {
                if (!single) { return std::numeric_limits<double>::quiet_NaN(); }
                Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, expr);
            }
        }
        position = end;
    }
    if (operand) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
    while (!operators.empty()) {
        if (opening(operators.back())) { Errors::parsing_errors(ParsingErrors::MISMATCHED_PARENTHESIS, expr); }
        emit();
    }
    std::vector<double> values;
    for (const auto& token : output) {
        if (token.op.empty()) {
            values.push_back(token.value);
            continue;
        }
        const bool    sign  = token.op == "U-" || token.op == "U+";
        const int64_t arity = token.arity;
        if (arity < 1 || values.size() < static_cast<size_t>(arity)) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
        double result;
        if (sign) {
            const double right = values.back();
            values.pop_back();
            result = token.op == "U-" ? -right : right;
        } else if (is_variadic(token.op)) {
            // MIN/MAX fold over all their arguments, left to right.
            const auto first = values.end() - arity;
            result           = *first;
            for (auto it = first + 1; it != values.end(); ++it) {
                result = token.op == "MIN" ? std::min(result, *it) : std::max(result, *it);
            }
            values.erase(first, values.end());
        } else if (arity == 3) {
            const double v2 = values.back();
            values.pop_back();
            const double v1 = values.back();
            values.pop_back();
            const double v0 = values.back();
            values.pop_back();
            result = eval_func3(token.op, v0, v1, v2);
        } else {
            const double right = values.back();
            values.pop_back();
            double left = 0.0;
            if (arity == 2) {
                left = values.back();
                values.pop_back();
            }
            int64_t consumed = 0;
            result = parse_operator(token.op, left, right, consumed);
            if (consumed != arity) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
        }
        if (!std::isfinite(result)) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
        values.push_back(result);
    }
    if (values.size() != 1) { Errors::parsing_errors(ParsingErrors::INVALID_RPN, expr); }
    return values.back();
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
    std::vector<ParameterName> names;
    for (const auto& parameter : parameters) { names.push_back(parameter.first); }
    parse_parameters(parameters, names);
}

void JoSIM::parse_parameters(param_map& parameters, const std::vector<ParameterName>& names) {
    std::vector<ParameterName> pending;
    for (const auto& name : names) {
        if (!parameters.at(name).get_value()) { pending.push_back(name); }
    }
    while (!pending.empty()) {
        const auto previous = pending.size();
        pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const ParameterName& name) {
            auto& parameter = parameters.at(name);
            const double value = parse_param(parameter.get_expression(), parameters, name.subcircuit(), false);
            if (std::isnan(value)) { return false; }
            parameter.set_value(value);
            return true;
        }), pending.end());
        if (pending.size() == previous) {
            std::string unknown;
            for (const auto& name : pending) { unknown += name.name() + " " + name.subcircuit().value_or("") + "\n"; }
            Errors::parsing_errors(ParsingErrors::UNIDENTIFIED_PART, unknown);
        }
    }
}

void JoSIM::update_parameters(param_map& parameters) {
    // Re-evaluate global and per-instance ("|path") parameters. Subcircuit-definition defaults are templates
    // that may depend on an instance's bindings, so they are not evaluated on their own.
    std::vector<ParameterName> names;
    for (auto& parameter : parameters) {
        const auto& scope = parameter.first.subcircuit();
        if (!scope || (!scope->empty() && scope->front() == '|')) {
            parameter.second.reset_value();
            names.push_back(parameter.first);
        }
    }
    parse_parameters(parameters, names);
}

void JoSIM::expand_inline_parameters(std::pair<tokens_t, string_o>& s, param_map& parameters) {
    // Expand every {expression} in the line; a token may carry text before and after its braces.
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
