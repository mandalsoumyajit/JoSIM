// Copyright (c) 2026 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Simulation.hpp"
#include "JoSIM/Constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>

using namespace JoSIM;

void Matrix::validate_inductor_coupling() const {
    const auto& devices = components.devices;
    const auto& indices = components.inductorIndices;
    if (std::none_of(indices.begin(), indices.end(), [&](int64_t index) {
        return std::get<Inductor>(devices[index]).alpha_ > 0;
    })) { return; }

    std::unordered_set<int64_t> visited;
    for (auto start : indices) {
        if (visited.count(start)) { continue; }
        std::vector<int64_t> group, pending{start};
        bool nonlinear = false;
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (!visited.insert(index).second) { continue; }
            group.push_back(index);
            const auto& device = std::get<Inductor>(devices[index]);
            nonlinear = nonlinear || device.alpha_ > 0;
            for (const auto& mutual : device.mutualInductances_) { pending.push_back(mutual.first); }
        }
        if (!nonlinear || (group.size() == 1 && std::get<Inductor>(devices[start]).mutualInductances_.empty())) {
            continue;
        }
        const auto fail = [&] {
            Errors::invalid_component_errors(ComponentErrors::INVALID_COMPONENT_DECLARATION,
                std::get<Inductor>(devices[start]).netlistInfo.label_
                + ": nonlinear coupling requires a finite positive-definite base inductance matrix without self-coupling");
        };
        std::sort(group.begin(), group.end());
        std::unordered_map<int64_t, size_t> position;
        for (size_t i = 0; i < group.size(); ++i) { position[group[i]] = i; }
        // Sparse LDL elimination on D^-1/2 L D^-1/2. Positive base pivots and
        // ALPHA>=0 keep the differential inductance matrix positive definite.
        std::vector<std::map<size_t, double>> upper(group.size());
        for (size_t i = 0; i < group.size(); ++i) {
            const auto& device = std::get<Inductor>(devices[group[i]]);
            if (!std::isfinite(device.netlistInfo.value_) || device.netlistInfo.value_ <= 0) { fail(); }
            upper[i][i] = 1.0;
            for (const auto& mutual : device.mutualInductances_) {
                const auto j = position.at(mutual.first);
                const double other = std::get<Inductor>(devices[mutual.first]).netlistInfo.value_;
                if (i == j || !std::isfinite(other) || other <= 0 || !std::isfinite(mutual.second)) { fail(); }
                if (j > i) {
                    upper[i][j] += mutual.second / (std::sqrt(device.netlistInfo.value_) * std::sqrt(other));
                }
            }
        }
        for (size_t k = 0; k < upper.size(); ++k) {
            const double pivot = upper[k].at(k);
            if (!std::isfinite(pivot) || pivot <= 64 * std::numeric_limits<double>::epsilon() * group.size()) { fail(); }
            for (auto i = upper[k].upper_bound(k); i != upper[k].end(); ++i) {
                for (auto j = i; j != upper[k].end(); ++j) {
                    upper[i->first][j->first] -= (i->second / pivot) * j->second;
                }
            }
        }
    }
}
