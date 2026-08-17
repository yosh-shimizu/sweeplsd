// Named SweepLSD configurations for the bench harnesses (--config SPEC).
//
// SPEC = BASE (('+'|'-') TOGGLE)*
//   BASE   : "full" (the shipped defaults, Params{}) or "2014"
//            (Params::original2014()).
//   TOGGLE : tiebreak | subpixel | hysteresis | bboxend | curverej |
//            lattice | link | nfa
// '+' enables the toggle, '-' disables it. Examples: "full-subpixel" (the
// hardware configuration), "2014+lattice+tiebreak" (an ablation rung).
#pragma once

#include <string>

#include "sweeplsd/sweeplsd.hpp"

struct NamedConfig {
    std::string spec;
    sweeplsd::Params params;
};

inline bool parseConfigSpec(const std::string& spec, sweeplsd::Params& out) {
    std::size_t i = 0;
    auto word = [&]() {
        std::size_t j = i;
        while (j < spec.size() && spec[j] != '+' && spec[j] != '-') ++j;
        std::string w = spec.substr(i, j - i);
        i = j;
        return w;
    };
    std::string base = word();
    if (base == "full") out = sweeplsd::Params::improved();
    else if (base == "2014") out = sweeplsd::Params::original2014();
    else return false;
    while (i < spec.size()) {
        bool on = spec[i] == '+';
        ++i;
        std::string m = word();
        if (m == "tiebreak") out.nms_strict_tiebreak = on;
        else if (m == "subpixel") out.subpixel_nms = on;
        else if (m == "hysteresis") out.use_hysteresis = on;
        else if (m == "bboxend") out.endpoint_from_bbox = on;
        else if (m == "curverej") out.max_perp_spread = on ? 1.0 : 0.0;
        else if (m == "lattice") out.lattice_half_shift = on;
        else if (m == "link") out.link_collinear = on;
        else if (m == "nfa") out.use_nfa = on;
        else return false;
    }
    return true;
}
