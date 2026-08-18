// Quantitative evaluation of SweepLSD vs LSD vs EDLines on SYNTHETIC images with
// exact ground truth.
//
// The comparison tool (app/compare.cpp) only reports counts and timing. This
// tool fills the missing quality axis with a fair, controlled protocol:
//
//   * Synthetic ground truth — we draw known line segments (anti-aliased) onto
//     a canvas, so we know the exact GT. Orientations are sampled uniformly in
//     [0,180) to avoid biasing the axis-aligned methods (SweepLSD quantises the
//     gradient direction to H/V, so diagonal lines must be represented). The
//     geometry is fixed per image index and only Gaussian noise sigma varies,
//     isolating the effect of noise.
//
//   * Fair operating points — a single threshold comparison is unfair (one
//     method tuned aggressive, another conservative), so we sweep each method's
//     principal sensitivity knob and trace its precision/recall FRONTIER:
//         SweepLSD  : pixel_num_th  (min pixels per segment)
//         LSD    : eps           (the -log10(NFA) detection threshold)
//         EDLines: min_length    (shortest accepted chain)
//     and summarise with F-max and AP (area under the PR frontier).
//
//   * Matching — a detection matches a GT segment when their directions agree
//     (<= angle_th), the detection lies laterally within tau of the GT line,
//     and their projections overlap; matched greedily one-to-one by overlap.
//     TP = matched detections, FP = the rest, FN = unmatched GT.
//
//   * Geometric accuracy — for matched pairs we report the lateral
//     localization error and the angular error (robust to fragmentation).
//
// Speed/ISA fairness is handled by compare.cpp (all methods built at AVX2); the
// focus here is detection quality, so timing is reported only for context.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "config_spec.hpp"
#include "edlines.hpp"
#include "sweeplsd/io.hpp"
#include "sweeplsd/sweeplsd.hpp"
#include "edreal_io.hpp"
#include "mlsd_io.hpp"

#include "lsd.h"  // third_party/lsd (AGPL)

namespace {

constexpr double kPi = 3.14159265358979323846;
using sweeplsd::LineSegment;

// ---------------------------------------------------------------------------
// Segment geometry
// ---------------------------------------------------------------------------

double segLength(const LineSegment& s) {
    double dx = s.x1 - s.x0, dy = s.y1 - s.y0;
    return std::sqrt(dx * dx + dy * dy);
}

// Undirected orientation in [0, pi).
double segOrient(const LineSegment& s) {
    double a = std::atan2(s.y1 - s.y0, s.x1 - s.x0);
    if (a < 0) a += kPi;
    if (a >= kPi) a -= kPi;
    return a;
}

// Smallest angle between two undirected orientations, in [0, pi/2].
double orientDiff(double a, double b) {
    double d = std::fabs(a - b);
    if (d > kPi / 2) d = kPi - d;
    return d;
}

// Distance from point p to the infinite line through segment g, and the
// projection parameter t of p along g's unit direction (origin at g.p0).
void pointToLine(const LineSegment& g, double px, double py, double& perp, double& t) {
    double ux = g.x1 - g.x0, uy = g.y1 - g.y0;
    double len = std::sqrt(ux * ux + uy * uy);
    if (len < 1e-9) { perp = std::hypot(px - g.x0, py - g.y0); t = 0; return; }
    ux /= len; uy /= len;
    double vx = px - g.x0, vy = py - g.y0;
    t = vx * ux + vy * uy;
    perp = std::fabs(vx * uy - vy * ux);  // |cross|
}

// ---------------------------------------------------------------------------
// Matching and metrics
// ---------------------------------------------------------------------------

struct Stats {
    long tp = 0, fp = 0, fn = 0;
    double sum_lat = 0, sum_ang = 0;  // accumulated over matched pairs
    long n_matched = 0;
    void add(const Stats& o) {
        tp += o.tp; fp += o.fp; fn += o.fn;
        sum_lat += o.sum_lat; sum_ang += o.sum_ang; n_matched += o.n_matched;
    }
    double precision() const { return (tp + fp) ? double(tp) / (tp + fp) : 0.0; }
    double recall() const { return (tp + fn) ? double(tp) / (tp + fn) : 0.0; }
    double f1() const {
        double p = precision(), r = recall();
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
    }
    double latErr() const { return n_matched ? sum_lat / n_matched : 0.0; }
    double angErrDeg() const { return n_matched ? sum_ang / n_matched * 180.0 / kPi : 0.0; }
};

// Per-GT-segment match record (optional matchSegments output, for the
// intersection analysis of the randomized protocol).
struct PerGT { char matched = 0; float lat = 0, ang = 0; };

// Greedy one-to-one matching of detections to GT. If `per` is non-null it is
// resized to gt.size() and filled with the per-GT match outcome.
Stats matchSegments(const std::vector<LineSegment>& gt,
                    const std::vector<LineSegment>& det, double tau, double ang_th,
                    std::vector<PerGT>* per = nullptr) {
    if (per) per->assign(gt.size(), PerGT{});
    struct Cand { double overlap; int gi, di; double lat, ang; };
    std::vector<Cand> cands;
    for (int di = 0; di < (int)det.size(); ++di) {
        const LineSegment& d = det[di];
        double od = segOrient(d), ld = segLength(d);
        for (int gi = 0; gi < (int)gt.size(); ++gi) {
            const LineSegment& g = gt[gi];
            double ang = orientDiff(od, segOrient(g));
            if (ang > ang_th) continue;
            double perp0, t0, perp1, t1;
            pointToLine(g, d.x0, d.y0, perp0, t0);
            pointToLine(g, d.x1, d.y1, perp1, t1);
            if (perp0 > tau || perp1 > tau) continue;
            double lg = segLength(g);
            double lo = std::min(t0, t1), hi = std::max(t0, t1);
            double overlap = std::min(hi, lg) - std::max(lo, 0.0);
            if (overlap <= 0 || overlap < 0.5 * ld) continue;  // detection mostly on GT
            cands.push_back({overlap, gi, di, 0.5 * (perp0 + perp1), ang});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.overlap > b.overlap; });
    std::vector<char> g_used(gt.size(), 0), d_used(det.size(), 0);
    Stats s;
    for (const Cand& c : cands) {
        if (g_used[c.gi] || d_used[c.di]) continue;
        g_used[c.gi] = d_used[c.di] = 1;
        ++s.tp; ++s.n_matched;
        s.sum_lat += c.lat; s.sum_ang += c.ang;
        if (per) (*per)[c.gi] = {1, float(c.lat), float(c.ang)};
    }
    s.fp = (long)det.size() - s.tp;
    s.fn = (long)gt.size() - s.tp;
    return s;
}

// ---------------------------------------------------------------------------
// Synthetic image generation (exact ground truth)
// ---------------------------------------------------------------------------

constexpr double kBg = 210.0, kFg = 40.0;  // dark bars on a light background
// Bar level actually drawn; --contrast D sets it to kBg - D (default = kFg,
// i.e. the standard 170-level bars). Lets a low-contrast variant of the same
// protocol probe the faint-structure regime (where the hysteresis acts).
double gFg = kFg;
constexpr double kBarWidth = 3.0;          // bar width; each bar = TWO edges (its flanks)

double distToSegment(double px, double py, const LineSegment& s) {
    double vx = s.x1 - s.x0, vy = s.y1 - s.y0;
    double wx = px - s.x0, wy = py - s.y0;
    double c1 = vx * wx + vy * wy;
    double c2 = vx * vx + vy * vy;
    double t = c2 > 1e-9 ? c1 / c2 : 0.0;
    t = std::max(0.0, std::min(1.0, t));
    double dx = px - (s.x0 + t * vx), dy = py - (s.y0 + t * vy);
    return std::sqrt(dx * dx + dy * dy);
}

// A clean float canvas (light background, dark bars). Each bar is a filled
// rectangle of width kBarWidth, so it presents TWO parallel intensity edges
// (its flanks) — which is what an edge-based detector actually finds. The
// ground truth is therefore the two flank lines of every bar. Geometry depends
// only on `seed`. Returns the bar centerlines via `bars` and the GT flanks via
// `gt`.
std::vector<float> genClean(int w, int h, int n_seg, unsigned seed,
                            std::vector<LineSegment>& gt,
                            std::vector<LineSegment>& bars,
                            double width = kBarWidth, double fg = -1.0,
                            double par_prob = 0.0) {
    if (fg < 0) fg = gFg;  // default: the --contrast-controlled global level
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uAng(0.0, kPi);
    std::uniform_real_distribution<double> uLen(40.0, 200.0);
    std::uniform_real_distribution<double> uX(20.0, w - 20.0);
    std::uniform_real_distribution<double> uY(20.0, h - 20.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    bars.clear();
    gt.clear();
    const double off = 0.5 * width;
    auto addBar = [&](const LineSegment& s, double nx, double ny) {
        bars.push_back(s);
        // Two flank edges, offset along the bar normal.
        gt.push_back({float(s.x0 + off * nx), float(s.y0 + off * ny),
                      float(s.x1 + off * nx), float(s.y1 + off * ny)});
        gt.push_back({float(s.x0 - off * nx), float(s.y0 - off * ny),
                      float(s.x1 - off * nx), float(s.y1 - off * ny)});
    };
    for (int k = 0; k < n_seg; ++k) {
        for (int tries = 0; tries < 20; ++tries) {
            double a = uAng(rng), len = uLen(rng);
            double cx = uX(rng), cy = uY(rng);
            double hx = 0.5 * len * std::cos(a), hy = 0.5 * len * std::sin(a);
            LineSegment s{float(cx - hx), float(cy - hy), float(cx + hx), float(cy + hy)};
            if (s.x0 < 5 || s.y0 < 5 || s.x1 < 5 || s.y1 < 5 ||
                s.x0 > w - 5 || s.y0 > h - 5 || s.x1 > w - 5 || s.y1 > h - 5)
                continue;
            double nx = -std::sin(a), ny = std::cos(a);
            addBar(s, nx, ny);
            // Randomized protocol only: with probability par_prob, add a close
            // parallel neighbor (4-10 px lateral separation). The u01 draws
            // happen only when par_prob > 0, so the standard protocol's
            // geometry stream is untouched.
            if (par_prob > 0 && u01(rng) < par_prob) {
                double sep = 4.0 + 6.0 * u01(rng);
                LineSegment p{float(s.x0 + sep * nx), float(s.y0 + sep * ny),
                              float(s.x1 + sep * nx), float(s.y1 + sep * ny)};
                if (p.x0 >= 5 && p.y0 >= 5 && p.x1 >= 5 && p.y1 >= 5 &&
                    p.x0 <= w - 5 && p.y0 <= h - 5 && p.x1 <= w - 5 && p.y1 <= h - 5)
                    addBar(p, nx, ny);
            }
            break;
        }
    }

    const int pad = int(std::ceil(off)) + 2;
    std::vector<float> buf(std::size_t(w) * h, float(kBg));
    for (const LineSegment& s : bars) {
        int xmin = std::max(0, int(std::floor(std::min(s.x0, s.x1))) - pad);
        int xmax = std::min(w - 1, int(std::ceil(std::max(s.x0, s.x1))) + pad);
        int ymin = std::max(0, int(std::floor(std::min(s.y0, s.y1))) - pad);
        int ymax = std::min(h - 1, int(std::ceil(std::max(s.y0, s.y1))) + pad);
        for (int y = ymin; y <= ymax; ++y)
            for (int x = xmin; x <= xmax; ++x) {
                // Sample pixel (x,y) AT (x,y): the canvas, the GT lines and the
                // detectors all share the pixel-centre coordinate frame (canonical
                // LSD also reports in this frame after its own +0.5 correction).
                // Sampling at (x+0.5, y+0.5) would shift the rendered edges half a
                // pixel away from the GT lines and charge every detector a
                // spurious 0.5 px of lateral error.
                double d = distToSegment(double(x), double(y), s);
                double cov = std::max(0.0, std::min(1.0, off + 0.5 - d));  // filled bar, AA edges
                if (cov <= 0) continue;
                float& px = buf[std::size_t(y) * w + x];
                float v = float(kBg + (fg - kBg) * cov);
                if (v < px) px = v;  // darkest wins (dark bar on light bg)
            }
    }
    return buf;
}

// Separable Gaussian blur of the clean float canvas (pre-noise defocus for
// the randomized protocol). sigma_b < 0.05 is a no-op.
void blurClean(std::vector<float>& buf, int w, int h, double sigma_b) {
    if (sigma_b < 0.05) return;
    int r = int(std::ceil(3.0 * sigma_b));
    std::vector<double> k(2 * r + 1);
    double sum = 0;
    for (int i = -r; i <= r; ++i)
        sum += k[i + r] = std::exp(-0.5 * i * i / (sigma_b * sigma_b));
    for (double& v : k) v /= sum;
    std::vector<float> tmp(buf.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double a = 0;
            for (int i = -r; i <= r; ++i)
                a += k[i + r] * buf[std::size_t(y) * w + std::min(w - 1, std::max(0, x + i))];
            tmp[std::size_t(y) * w + x] = float(a);
        }
    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y) {
            double a = 0;
            for (int i = -r; i <= r; ++i)
                a += k[i + r] * tmp[std::size_t(std::min(h - 1, std::max(0, y + i))) * w + x];
            buf[std::size_t(y) * w + x] = float(a);
        }
}

// Add Gaussian noise and quantise to 8-bit.
sweeplsd::GrayImage addNoise(const std::vector<float>& clean, int w, int h,
                          double sigma, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    sweeplsd::GrayImage img(w, h);
    for (std::size_t i = 0; i < clean.size(); ++i) {
        double v = clean[i] + (sigma > 0 ? noise(rng) : 0.0);
        img.data[i] = std::uint8_t(std::max(0.0, std::min(255.0, v)) + 0.5);
    }
    return img;
}

// ---------------------------------------------------------------------------
// Method runners (each with its swept sensitivity knob)
// ---------------------------------------------------------------------------

std::vector<LineSegment> runSweeplsd(const sweeplsd::GrayImage& src, int pixel_num_th) {
    sweeplsd::Params p = sweeplsd::Params::original2014();
    p.pixel_num_th = pixel_num_th;
    return sweeplsd::detect(src, p);
}

std::vector<LineSegment> runSweeplsdImproved(const sweeplsd::GrayImage& src, int pixel_num_th) {
    sweeplsd::Params p = sweeplsd::Params::improved();
    p.pixel_num_th = pixel_num_th;
    return sweeplsd::detect(src, p);
}

std::vector<LineSegment> runSweeplsdImprovedLink(const sweeplsd::GrayImage& src, int pixel_num_th) {
    sweeplsd::Params p = sweeplsd::Params::improved();
    p.pixel_num_th = pixel_num_th;
    p.link_collinear = true;  // linker defaults: lateral tol, gap, two-stage admit
    return sweeplsd::detect(src, p);
}

std::vector<LineSegment> runLsdEps(const sweeplsd::GrayImage& src, double eps) {
    std::vector<double> buf(std::size_t(src.width) * src.height);
    for (int i = 0; i < src.width * src.height; ++i) buf[i] = double(src.data[i]);
    int n = 0;
    // Standard LSD defaults, varying only log_eps (the NFA detection threshold).
    double* out = LineSegmentDetection(&n, buf.data(), src.width, src.height,
                                       0.8, 0.6, 2.0, 22.5, eps, 0.7, 1024,
                                       NULL, NULL, NULL);
    std::vector<LineSegment> segs;
    segs.reserve(n);
    for (int j = 0; j < n; ++j)
        segs.push_back({(float)out[7 * j], (float)out[7 * j + 1],
                        (float)out[7 * j + 2], (float)out[7 * j + 3]});
    std::free(out);
    return segs;
}

std::vector<LineSegment> runEd(const sweeplsd::GrayImage& src, int min_length) {
    edlines::Params p;
    p.min_length = min_length;
    return edlines::detect(src, p);
}

// ---------------------------------------------------------------------------
// Line-free "clouds" texture for the negative (false-positive) probe: two
// octaves of value noise with smoothstep interpolation -- C1 across cell
// borders, so the lattice itself draws no straight creases -- normalized to
// nearly the full 8-bit range. Contains no straight structure by construction.
// ---------------------------------------------------------------------------

std::vector<float> genClouds(int w, int h, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<double> acc(std::size_t(w) * h, 0.0);
    auto octave = [&](int cell, double amp) {
        int gw = w / cell + 2, gh = h / cell + 2;
        std::vector<double> g(std::size_t(gw) * gh);
        for (double& v : g) v = uni(rng);
        for (int y = 0; y < h; ++y) {
            double fy = double(y) / cell;
            int cy = int(fy); double ty = fy - cy;
            double sy = ty * ty * (3 - 2 * ty);
            for (int x = 0; x < w; ++x) {
                double fx = double(x) / cell;
                int cx = int(fx); double tx = fx - cx;
                double sx = tx * tx * (3 - 2 * tx);
                double v00 = g[std::size_t(cy) * gw + cx], v10 = g[std::size_t(cy) * gw + cx + 1];
                double v01 = g[std::size_t(cy + 1) * gw + cx], v11 = g[std::size_t(cy + 1) * gw + cx + 1];
                double v0 = v00 + (v10 - v00) * sx;
                double v1 = v01 + (v11 - v01) * sx;
                acc[std::size_t(y) * w + x] += amp * (v0 + (v1 - v0) * sy);
            }
        }
    };
    octave(96, 1.0);
    octave(32, 0.35);
    double lo = 1e18, hi = -1e18;
    for (double v : acc) { lo = std::min(lo, v); hi = std::max(hi, v); }
    std::vector<float> img(std::size_t(w) * h);
    for (std::size_t i = 0; i < img.size(); ++i)
        img[i] = float(20.0 + 215.0 * (acc[i] - lo) / (hi - lo));
    return img;
}

// ---------------------------------------------------------------------------
// PR frontier
// ---------------------------------------------------------------------------

struct PrPoint { double knob, recall, precision, f1, lat, ang, dets; };
struct Curve {
    std::string method, color;
    std::vector<PrPoint> pts;  // one per knob value
    double fmax = 0, ap = 0;
    PrPoint best{};  // operating point with max F
};

double areaUnderPr(std::vector<PrPoint> pts) {
    std::sort(pts.begin(), pts.end(),
              [](const PrPoint& a, const PrPoint& b) { return a.recall < b.recall; });
    double ap = 0, prevR = 0;
    for (const PrPoint& p : pts) {
        ap += (p.recall - prevR) * p.precision;
        prevR = p.recall;
    }
    return ap;
}

void finalizeCurve(Curve& c) {
    c.fmax = 0;
    for (const PrPoint& p : c.pts)
        if (p.f1 > c.fmax) { c.fmax = p.f1; c.best = p; }
    c.ap = areaUnderPr(c.pts);
}

// ---------------------------------------------------------------------------
// HTML report
// ---------------------------------------------------------------------------

// Map (recall, precision) in [0,1]^2 to SVG plot coordinates.
struct Plot { double x0, y0, w, h; };
std::string svgXY(const Plot& pl, double r, double p) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.1f,%.1f", pl.x0 + r * pl.w, pl.y0 + (1 - p) * pl.h);
    return b;
}

std::string prCurveSvg(const std::vector<Curve>& curves, const std::string& title) {
    Plot pl{55, 20, 360, 300};
    std::string s;
    s += "<svg viewBox=\"0 0 460 380\" width=\"460\" height=\"380\">";
    s += "<rect x=\"55\" y=\"20\" width=\"360\" height=\"300\" fill=\"#fff\" stroke=\"#ccc\"/>";
    // gridlines + ticks
    for (int i = 0; i <= 5; ++i) {
        double f = i / 5.0;
        char g[512];
        std::snprintf(g, sizeof(g),
            "<line x1=\"%.0f\" y1=\"20\" x2=\"%.0f\" y2=\"320\" stroke=\"#eee\"/>"
            "<line x1=\"55\" y1=\"%.0f\" x2=\"415\" y2=\"%.0f\" stroke=\"#eee\"/>"
            "<text x=\"%.0f\" y=\"335\" font-size=\"11\" text-anchor=\"middle\" fill=\"#666\">%.1f</text>"
            "<text x=\"48\" y=\"%.0f\" font-size=\"11\" text-anchor=\"end\" fill=\"#666\">%.1f</text>",
            pl.x0 + f * pl.w, pl.x0 + f * pl.w, pl.y0 + f * pl.h, pl.y0 + f * pl.h,
            pl.x0 + f * pl.w, f, pl.y0 + (1 - f) * pl.h + 4, f);
        s += g;
    }
    s += "<text x=\"235\" y=\"358\" font-size=\"13\" text-anchor=\"middle\">Recall</text>";
    s += "<text x=\"16\" y=\"170\" font-size=\"13\" text-anchor=\"middle\" transform=\"rotate(-90 16 170)\">Precision</text>";
    s += "<text x=\"235\" y=\"14\" font-size=\"13\" text-anchor=\"middle\" font-weight=\"600\">" + title + "</text>";
    for (const Curve& c : curves) {
        std::vector<PrPoint> pts = c.pts;
        std::sort(pts.begin(), pts.end(),
                  [](const PrPoint& a, const PrPoint& b) { return a.recall < b.recall; });
        std::string poly;
        for (const PrPoint& p : pts) poly += svgXY(pl, p.recall, p.precision) + " ";
        s += "<polyline points=\"" + poly + "\" fill=\"none\" stroke=\"" + c.color + "\" stroke-width=\"2\"/>";
        for (const PrPoint& p : pts) {
            std::string xy = svgXY(pl, p.recall, p.precision);
            s += "<circle cx=\"" + xy.substr(0, xy.find(',')) + "\" cy=\"" +
                 xy.substr(xy.find(',') + 1) + "\" r=\"2.5\" fill=\"" + c.color + "\"/>";
        }
    }
    // legend
    double ly = 30;
    for (const Curve& c : curves) {
        char lg[256];
        std::snprintf(lg, sizeof(lg),
            "<line x1=\"320\" y1=\"%.0f\" x2=\"345\" y2=\"%.0f\" stroke=\"%s\" stroke-width=\"3\"/>"
            "<text x=\"350\" y=\"%.0f\" font-size=\"12\" fill=\"#333\">%s</text>",
            ly, ly, c.color.c_str(), ly + 4, c.method.c_str());
        s += lg;
        ly += 18;
    }
    s += "</svg>";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int w = 1280, h = 720, n_seg = 18, images = 4;
    std::string html_path, assets_dir, dump_dir, mlsd_dir, edreal_dir, elsed_dir;
    std::vector<double> sigmas = {0, 5, 10, 20};
    std::vector<NamedConfig> configs;
    bool negatives = false;
    bool junctions = false;
    int nrand = 0;
    std::string csv_path;
    std::string perseg_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--html" && i + 1 < argc) html_path = argv[++i];
        // --config SPEC : add an extra SweepLSD curve in the named
        //   configuration ("full"/"2014" plus +/- toggles over {tiebreak,
        //   subpixel, hysteresis, bboxend, curverej, lattice, link, nfa}),
        //   swept over the same pixel_num_th knob. Repeatable.
        else if (a == "--config" && i + 1 < argc) {
            NamedConfig nc;
            nc.spec = argv[++i];
            if (!parseConfigSpec(nc.spec, nc.params)) {
                std::fprintf(stderr, "bad --config spec: %s\n", nc.spec.c_str());
                return 1;
            }
            configs.push_back(nc);
        }
        // --negatives : run the line-free false-positive probe instead of the
        //   F-max protocol (see the negatives block below).
        else if (a == "--negatives") negatives = true;
        // --junctions : run the junction endpoint probe instead (see the
        //   junctions block below): projection-extreme vs first-contact
        //   endpoints on T/X-junction scenes, scored against the analytically
        //   visible sub-extents of the subject bar's flank edges.
        else if (a == "--junctions") junctions = true;
        // --contrast D : draw the bars at kBg - D instead of the standard
        //   170-level contrast (low-contrast probe for the hysteresis).
        else if (a == "--contrast" && i + 1 < argc) gFg = kBg - std::atof(argv[++i]);
        // --randomized N : run the randomized-scene protocol instead of the
        //   fixed-condition one: N scenes, each with drawn bar count, width,
        //   contrast, blur, noise, and parallel-neighbor pairs (see the
        //   randomized block below). --csv dumps per-scene counts for the
        //   bootstrap/slice analysis.
        else if (a == "--randomized" && i + 1 < argc) nrand = std::atoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        // --perseg CSV : with --randomized, also dump one row per matched GT
        //   segment per (method, knob) -- plus "gt" rows carrying each GT
        //   segment's length/orientation and a gi=-1 marker row per
        //   (method, knob, scene) that ran (distinguishes "no matches" from
        //   "no data" for the ingested detectors). For the matched-by-all
        //   intersection analysis.
        else if (a == "--perseg" && i + 1 < argc) perseg_path = argv[++i];
        else if (a == "--assets" && i + 1 < argc) assets_dir = argv[++i];
        else if (a == "--images" && i + 1 < argc) images = std::atoi(argv[++i]);
        else if (a == "--size" && i + 2 < argc) { w = std::atoi(argv[++i]); h = std::atoi(argv[++i]); }
        else if (a == "--nseg" && i + 1 < argc) n_seg = std::atoi(argv[++i]);
        // --dump-images DIR : write each (sigma,image) input PNG so M-LSD can be
        //   run on the exact same synthetic inputs offline (Python).
        else if (a == "--dump-images" && i + 1 < argc) dump_dir = argv[++i];
        // --mlsd-dir DIR : ingest precomputed M-LSD segments per (sigma,image,knob)
        //   as files "eval_s<sig>_im<im>_k<ki>.txt", scored by the SAME matcher.
        else if (a == "--mlsd-dir" && i + 1 < argc) mlsd_dir = argv[++i];
        // --edreal-dir DIR : ingest genuine ED_Lib EDLines segments per
        //   (sigma,image,knob) as "eval_s<sig>_im<im>_k<ki>.txt" (see
        //   tools/edlines_real.exe --minlen), scored by the SAME matcher.
        else if (a == "--edreal-dir" && i + 1 < argc) edreal_dir = argv[++i];
        // --elsed-dir DIR : ingest genuine ELSED segments per (sigma,image,knob)
        //   as "eval_s<sig>_im<im>_k<ki>.txt" (elsed_runner --minlen sweep over
        //   the same values as the EDLines minlen knob), same matcher.
        else if (a == "--elsed-dir" && i + 1 < argc) elsed_dir = argv[++i];
    }
    const bool have_mlsd = !mlsd_dir.empty();
    const bool have_edreal = !edreal_dir.empty();
    const bool have_elsed = !elsed_dir.empty();
    // M-LSD score-threshold sweep (its principal sensitivity knob; lower = more
    // detections). Must match the values mlsd_runner.py is invoked with.
    const std::vector<double> mlsd_knobs = {0.30, 0.20, 0.10, 0.05, 0.02};
    auto sigTag = [](double s) { char b[16]; std::snprintf(b, sizeof(b), "%d", (int)s); return std::string(b); };

    // ---- Negative (line-free) probe: false detections on images with no
    // straight structure -- uniform-background Gaussian noise at several
    // sigma, plus smooth "clouds" value-noise textures (line-free by
    // construction), with and without added noise. Every method runs at its
    // DEFAULT operating point (the question is the shipped behavior, not a
    // swept curve); genuine ED_Lib / ELSED results are ingested from
    // --edreal-dir / --elsed-dir files "neg_<kind>_im<i>.txt" produced on the
    // --dump-images PNGs. Cells are detections per image (mean/max).
    if (negatives) {
        struct Kind { std::string name; double sigma; bool clouds; };
        const std::vector<Kind> kinds = {
            {"noise5", 5, false},   {"noise10", 10, false},
            {"noise20", 20, false}, {"noise40", 40, false},
            {"clouds", 0, true},    {"clouds_n10", 10, true}};
        std::printf("Negative probe: %dx%d, %d images/kind; detections at default settings\n\n",
                    w, h, images);
        std::printf("  %-11s %13s %13s %13s %13s", "kind",
                    "sweeplsd", "sweeplsd+NFA", "LSD", "EDLines-style");
        if (have_edreal) std::printf(" %13s", "ED_Lib");
        if (have_elsed) std::printf(" %13s", "ELSED");
        std::printf("   (mean/max per image)\n");
        for (std::size_t ki = 0; ki < kinds.size(); ++ki) {
            const Kind& kind = kinds[ki];
            struct Acc {
                long tot = 0, mx = 0;
                void add(long n) { tot += n; mx = std::max(mx, n); }
            };
            Acc a_sw, a_nfa, a_lsd, a_ed, a_edr, a_el;
            for (int im = 0; im < images; ++im) {
                std::vector<float> canvas = kind.clouds
                    ? genClouds(w, h, 3000u + 17u * (unsigned)ki + im)
                    : std::vector<float>(std::size_t(w) * h, float(kBg));
                sweeplsd::GrayImage img =
                    addNoise(canvas, w, h, kind.sigma, 9000u + 100u * (unsigned)ki + im);
                if (!dump_dir.empty())
                    sweeplsd::saveGrayPng(dump_dir + "/neg_" + kind.name + "_im" +
                                          std::to_string(im) + ".png", img);
                sweeplsd::Params pdef;  // shipped defaults
                a_sw.add((long)sweeplsd::detect(img, pdef).size());
                sweeplsd::Params pnfa;  // shipped defaults + streaming NFA gate
                pnfa.use_nfa = true;
                a_nfa.add((long)sweeplsd::detect(img, pnfa).size());
                a_lsd.add((long)runLsdEps(img, 0.0).size());
                a_ed.add((long)runEd(img, 10).size());
                if (have_edreal) {
                    std::vector<LineSegment> v;
                    readEdRealFile(edreal_dir + "/neg_" + kind.name + "_im" +
                                   std::to_string(im) + ".txt", v);
                    a_edr.add((long)v.size());
                }
                if (have_elsed) {
                    std::vector<LineSegment> v;
                    readEdRealFile(elsed_dir + "/neg_" + kind.name + "_im" +
                                   std::to_string(im) + ".txt", v);
                    a_el.add((long)v.size());
                }
            }
            auto cell = [&](const Acc& a) {
                char b[32];
                std::snprintf(b, sizeof(b), "%.1f/%ld", double(a.tot) / images, a.mx);
                return std::string(b);
            };
            std::printf("  %-11s %13s %13s %13s %13s", kind.name.c_str(),
                        cell(a_sw).c_str(), cell(a_nfa).c_str(),
                        cell(a_lsd).c_str(), cell(a_ed).c_str());
            if (have_edreal) std::printf(" %13s", cell(a_edr).c_str());
            if (have_elsed) std::printf(" %13s", cell(a_el).c_str());
            std::printf("\n");
        }
        return 0;
    }

    const double tau = 2.0;                  // lateral matching tolerance (px); < bar width so
                                             // a flank detection matches its own edge, not the other
    const double ang_th = 10.0 * kPi / 180;  // angular matching tolerance
    const std::vector<int> sweeplsd_knobs = {64, 48, 32, 24, 16, 12, 8, 5};
    const std::vector<double> lsd_knobs = {4, 3, 2, 1, 0, -1, -2};
    const std::vector<int> ed_knobs = {40, 30, 20, 15, 10, 7, 5};

    // ---- Randomized-scene protocol: same generator, matcher, and knob
    // sweeps as the fixed protocol, but every scene draws its own bar count
    // (6-40), bar width (1.5-5 px), contrast (log-uniform 10-170 gray
    // levels), pre-noise Gaussian blur (sigma 0-1.5 px), noise (sigma 0-20),
    // and adds a close parallel neighbor to 30% of bars (4-10 px apart).
    // Per-scene match counts go to --csv for the bootstrap / slice analysis;
    // the table here reports the pooled F at each method's single pool-best
    // operating point (one knob for the whole pool, not per condition).
    if (nrand > 0) {
        struct RandScene { int nbars; double width, contrast, blur, noise; };
        std::vector<RandScene> sp(nrand);
        auto S3 = [&](std::size_t nk) {
            return std::vector<std::vector<Stats>>(nk, std::vector<Stats>(nrand));
        };
        auto st_sw = S3(sweeplsd_knobs.size());
        auto st_swl = S3(sweeplsd_knobs.size());
        auto st_lsd = S3(lsd_knobs.size());
        auto st_ed = S3(ed_knobs.size());
        auto st_edr = S3(ed_knobs.size());
        auto st_el = S3(ed_knobs.size());
        std::vector<std::vector<char>> got_el(ed_knobs.size(),
                                              std::vector<char>(nrand, 0));
        std::printf("Randomized synthetic evaluation: %dx%d, %d scenes\n\n", w, h, nrand);
        std::FILE* fper = nullptr;
        if (!perseg_path.empty()) {
            fper = std::fopen(perseg_path.c_str(), "w");
            std::fprintf(fper, "scene,method,knob,gi,lat,ang\n");
        }
        for (int i = 0; i < nrand; ++i) {
            std::mt19937 prng(5000u + i);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            RandScene& P = sp[i];
            P.nbars = 6 + int(U(prng) * 35.0);
            P.width = 1.5 + 3.5 * U(prng);
            P.contrast = std::exp(std::log(10.0) +
                                  (std::log(170.0) - std::log(10.0)) * U(prng));
            P.blur = 1.5 * U(prng);
            P.noise = 20.0 * U(prng);
            std::vector<LineSegment> gt, brs;
            std::vector<float> clean = genClean(w, h, P.nbars, 6000u + i, gt, brs,
                                                P.width, kBg - P.contrast, 0.3);
            blurClean(clean, w, h, P.blur);
            sweeplsd::GrayImage img = addNoise(clean, w, h, P.noise, 40000u + i);
            if (!dump_dir.empty())
                sweeplsd::saveGrayPng(dump_dir + "/rand_im" + std::to_string(i) + ".png", img);
            if (fper)
                for (int gi = 0; gi < (int)gt.size(); ++gi)
                    std::fprintf(fper, "%d,gt,0,%d,%.6f,%.6f\n", i, gi,
                                 segLength(gt[gi]), segOrient(gt[gi]));
            std::vector<PerGT> per;
            auto score = [&](const char* name, double knob,
                             const std::vector<LineSegment>& v) {
                Stats s = matchSegments(gt, v, tau, ang_th, fper ? &per : nullptr);
                if (fper) {
                    std::fprintf(fper, "%d,%s,%g,-1,0,0\n", i, name, knob);
                    for (int gi = 0; gi < (int)gt.size(); ++gi)
                        if (per[gi].matched)
                            std::fprintf(fper, "%d,%s,%g,%d,%.6f,%.6f\n", i, name,
                                         knob, gi, per[gi].lat, per[gi].ang);
                }
                return s;
            };
            for (std::size_t k = 0; k < sweeplsd_knobs.size(); ++k) {
                st_sw[k][i] = score("sweeplsd", sweeplsd_knobs[k], runSweeplsdImproved(img, sweeplsd_knobs[k]));
                st_swl[k][i] = score("sweeplsd_link", sweeplsd_knobs[k], runSweeplsdImprovedLink(img, sweeplsd_knobs[k]));
            }
            for (std::size_t k = 0; k < lsd_knobs.size(); ++k)
                st_lsd[k][i] = score("lsd", lsd_knobs[k], runLsdEps(img, lsd_knobs[k]));
            for (std::size_t k = 0; k < ed_knobs.size(); ++k)
                st_ed[k][i] = score("edstyle", ed_knobs[k], runEd(img, ed_knobs[k]));
            if (have_edreal)
                for (std::size_t k = 0; k < ed_knobs.size(); ++k) {
                    std::vector<LineSegment> v;
                    readEdRealFile(edreal_dir + "/rand_im" + std::to_string(i) +
                                   "_k" + std::to_string(k) + ".txt", v);
                    st_edr[k][i] = score("edreal", ed_knobs[k], v);
                }
            if (have_elsed)
                for (std::size_t k = 0; k < ed_knobs.size(); ++k) {
                    std::vector<LineSegment> v;
                    if (readEdRealFile(elsed_dir + "/rand_im" + std::to_string(i) +
                                       "_k" + std::to_string(k) + ".txt", v)) {
                        st_el[k][i] = score("elsed", ed_knobs[k], v);
                        got_el[k][i] = 1;
                    }
                }
            if ((i + 1) % 20 == 0) std::printf("  ...%d/%d\n", i + 1, nrand);
        }
        if (fper) {
            std::fclose(fper);
            std::printf("wrote %s\n", perseg_path.c_str());
        }

        // per-scene dump for the bootstrap/slice analysis
        if (!csv_path.empty()) {
            std::FILE* f = std::fopen(csv_path.c_str(), "w");
            std::fprintf(f, "scene,nbars,width,contrast,blur,noise,method,knob,"
                            "tp,fp,fn,sum_lat,sum_ang,nm\n");
            auto dump = [&](const char* name, auto& st, auto& knobs) {
                for (std::size_t k = 0; k < st.size(); ++k)
                    for (int i = 0; i < nrand; ++i) {
                        if (std::string(name) == "elsed" && !got_el[k][i]) continue;
                        const Stats& s = st[k][i];
                        const RandScene& P = sp[i];
                        std::fprintf(f, "%d,%d,%.3f,%.3f,%.3f,%.3f,%s,%g,%ld,%ld,%ld,%.6f,%.6f,%ld\n",
                                     i, P.nbars, P.width, P.contrast, P.blur, P.noise,
                                     name, double(knobs[k]), s.tp, s.fp, s.fn,
                                     s.sum_lat, s.sum_ang, s.n_matched);
                    }
            };
            dump("sweeplsd", st_sw, sweeplsd_knobs);
            dump("sweeplsd_link", st_swl, sweeplsd_knobs);
            dump("lsd", st_lsd, lsd_knobs);
            dump("edstyle", st_ed, ed_knobs);
            if (have_edreal) dump("edreal", st_edr, ed_knobs);
            if (have_elsed) dump("elsed", st_el, ed_knobs);
            std::fclose(f);
            std::printf("wrote %s\n", csv_path.c_str());
        }

        // pooled report at the pool-best knob
        std::printf("\n  %-14s %8s %7s %7s %7s %9s %8s\n",
                    "method", "bestKnob", "F", "P", "R", "latErr", "angErr");
        auto pooled = [&](const char* name, auto& st, auto& knobs) {
            double bestF = -1;
            std::size_t bk = 0;
            std::vector<Stats> agg(st.size());
            for (std::size_t k = 0; k < st.size(); ++k) {
                for (int i = 0; i < nrand; ++i) agg[k].add(st[k][i]);
                if (agg[k].f1() > bestF) { bestF = agg[k].f1(); bk = k; }
            }
            std::printf("  %-14s %8g %7.3f %7.3f %7.3f %7.2fpx %6.2fdeg\n",
                        name, double(knobs[bk]), agg[bk].f1(), agg[bk].precision(),
                        agg[bk].recall(), agg[bk].latErr(), agg[bk].angErrDeg());
        };
        pooled("SweepLSD", st_sw, sweeplsd_knobs);
        pooled("SweepLSD+link", st_swl, sweeplsd_knobs);
        pooled("LSD", st_lsd, lsd_knobs);
        pooled("EDLines-style", st_ed, ed_knobs);
        if (have_edreal) pooled("EDLines(ED_Lib)", st_edr, ed_knobs);
        if (have_elsed) pooled("ELSED", st_el, ed_knobs);
        return 0;
    }

    // ---- Junction endpoint probe (--junctions): the projection-extreme
    // endpoint rule (endpoint_from_bbox) vs the 2014 first-contact rule on
    // scenes where a component grazes more than two endpoint candidates.
    // One long "subject" bar is crossed (X) or abutted (T) by 1-4 short bars
    // at 60-90 deg; control scenes have no junction. The toggle changes only
    // the once-per-segment finalization, so BOTH rules run on identical
    // components -- a paired comparison. A junction locally occludes the
    // subject's flank edge (dark meets dark, no intensity step), so the
    // reference is the analytically visible sub-extents of each flank, and
    // the score per recovered interval [a,b] is the endpoint extent error
    // |t_lo - a| + |t_hi - b| of the best-overlap detection.
    if (junctions) {
        struct JKind { const char* name; int njmax; bool tee; bool graze; double sigma; };
        const std::vector<JKind> jkinds = {
            {"control", 0, false, false, 0}, {"X", 4, false, false, 0},
            {"T", 4, true, false, 0},        {"graze", 4, false, true, 0},
            {"X_n5", 4, false, false, 5},    {"T_n5", 4, true, false, 5},
            {"graze_n5", 4, false, true, 5}};
        const int nscene = 200;
        sweeplsd::Params p_ext = sweeplsd::Params::improved();
        sweeplsd::Params p_fc = p_ext;
        p_fc.endpoint_from_bbox = false;
        std::printf("Junction endpoint probe: %dx%d, %d scenes/kind, "
                    "projection-extreme (ext) vs first-contact (fc)\n\n",
                    w, h, nscene);
        auto med = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            std::size_t n = v.size();
            return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        };
        for (const JKind& jk : jkinds) {
            long total = 0;
            long rec[2] = {0, 0};
            std::vector<double> errs[2], diffs;
            long wins = 0, loss = 0;
            for (int i = 0; i < nscene; ++i) {
                std::mt19937 rng(unsigned(7000 + 1000 * (&jk - &jkinds[0]) + i));
                std::uniform_real_distribution<double> U(0.0, 1.0);
                // subject bar, fully inside a 40 px margin
                double sa = 0, slen = 0, sx0 = 0, sy0 = 0;
                for (int tries = 0; tries < 100; ++tries) {
                    sa = U(rng) * kPi;
                    slen = 300.0 + 200.0 * U(rng);
                    double cx = 40.0 + (w - 80.0) * U(rng);
                    double cy = 40.0 + (h - 80.0) * U(rng);
                    double hx = 0.5 * slen * std::cos(sa);
                    double hy = 0.5 * slen * std::sin(sa);
                    if (cx - std::abs(hx) < 40 || cx + std::abs(hx) > w - 40 ||
                        cy - std::abs(hy) < 40 || cy + std::abs(hy) > h - 40)
                        continue;
                    sx0 = cx - hx; sy0 = cy - hy;
                    break;
                }
                double ux = std::cos(sa), uy = std::sin(sa);
                LineSegment subject{float(sx0), float(sy0),
                                    float(sx0 + slen * ux), float(sy0 + slen * uy)};
                // per-bar half-widths: the subject keeps the protocol's 3 px
                // bar; crossers are WIDE (6-16 px) so the junction genuinely
                // interrupts the flank edge (a 3 px crosser's ~4 px occlusion
                // is washed out by the Gaussian and never cuts the run).
                struct JBar { LineSegment s; double halfw; };
                const double off = 0.5 * kBarWidth;
                std::vector<JBar> bars{{subject, off}};
                int nj = jk.njmax ? 1 + int(U(rng) * jk.njmax) : 0;
                if (nj > jk.njmax) nj = jk.njmax;
                double nx0 = -uy, ny0 = ux;
                for (int j = 0; j < nj; ++j) {
                    double t = (0.15 + 0.7 * U(rng)) * slen;
                    double jx = sx0 + t * ux, jy = sy0 + t * uy;
                    LineSegment c;
                    double chw;
                    if (jk.graze) {
                        // "graze" kind: a THIN bar whose tip stops 1-2.5 px
                        // short of the subject's flank surface -- no occlusion,
                        // but the tip's line-end candidates sit 8-adjacent to
                        // the flank's interior run (a mid-run contact without a
                        // cut: the >2-candidate graze the endpoint rule guards
                        // against).
                        chw = off;  // same 3 px bar as the subject
                        double sgn = U(rng) < 0.5 ? 1.0 : -1.0;
                        double gap = 1.0 + 1.5 * U(rng);
                        double tipx = jx + sgn * (off + chw + gap) * nx0;
                        double tipy = jy + sgn * (off + chw + gap) * ny0;
                        double base = std::atan2(sgn * ny0, sgn * nx0);
                        double ca = base + (U(rng) - 0.5) * (60.0 * kPi / 180.0);
                        double clen = 60.0 + 90.0 * U(rng);
                        c = LineSegment{float(tipx), float(tipy),
                                        float(tipx + clen * std::cos(ca)),
                                        float(tipy + clen * std::sin(ca))};
                    } else {
                        double d = (60.0 + 30.0 * U(rng)) * kPi / 180.0;
                        if (U(rng) < 0.5) d = -d;
                        double ca = sa + d, clen = 60.0 + 90.0 * U(rng);
                        chw = 3.0 + 5.0 * U(rng);
                        c = jk.tee
                            ? LineSegment{float(jx), float(jy),
                                          float(jx + clen * std::cos(ca)),
                                          float(jy + clen * std::sin(ca))}
                            : LineSegment{float(jx - 0.5 * clen * std::cos(ca)),
                                          float(jy - 0.5 * clen * std::sin(ca)),
                                          float(jx + 0.5 * clen * std::cos(ca)),
                                          float(jy + 0.5 * clen * std::sin(ca))};
                    }
                    bars.push_back({c, chw});
                }
                // render: same darkest-wins AA fill as genClean
                std::vector<float> buf(std::size_t(w) * h, float(kBg));
                for (const JBar& jb : bars) {
                    const LineSegment& s = jb.s;
                    const int pad = int(std::ceil(jb.halfw)) + 2;
                    int xmin = std::max(0, int(std::floor(std::min(s.x0, s.x1))) - pad);
                    int xmax = std::min(w - 1, int(std::ceil(std::max(s.x0, s.x1))) + pad);
                    int ymin = std::max(0, int(std::floor(std::min(s.y0, s.y1))) - pad);
                    int ymax = std::min(h - 1, int(std::ceil(std::max(s.y0, s.y1))) + pad);
                    for (int y = ymin; y <= ymax; ++y)
                        for (int x = xmin; x <= xmax; ++x) {
                            double d = distToSegment(double(x), double(y), s);
                            double cov = std::max(0.0, std::min(1.0, jb.halfw + 0.5 - d));
                            if (cov <= 0) continue;
                            float& px = buf[std::size_t(y) * w + x];
                            float v = float(kBg + (gFg - kBg) * cov);
                            if (v < px) px = v;
                        }
                }
                sweeplsd::GrayImage img =
                    addNoise(buf, w, h, jk.sigma, unsigned(45000 + i));
                if (!dump_dir.empty() && i < 4)
                    sweeplsd::saveGrayPng(dump_dir + "/junc_" + std::string(jk.name) +
                                          "_im" + std::to_string(i) + ".png", img);
                auto dets_ext = sweeplsd::detect(img, p_ext);
                auto dets_fc = sweeplsd::detect(img, p_fc);
                // visible sub-intervals of the two flank edges
                double nx = -uy, ny = ux;
                for (int f = 0; f < 2; ++f) {
                    double sgn = f ? -1.0 : 1.0;
                    LineSegment flank{
                        float(sx0 + sgn * off * nx), float(sy0 + sgn * off * ny),
                        float(sx0 + sgn * off * nx + slen * ux),
                        float(sy0 + sgn * off * ny + slen * uy)};
                    std::vector<std::pair<double, double>> iv;
                    double a0 = -1.0;
                    const double step = 0.25;
                    for (double t = 0.0; t <= slen + 1e-9; t += step) {
                        double px = flank.x0 + t * ux, py = flank.y0 + t * uy;
                        bool vis = true;
                        for (std::size_t b = 1; b < bars.size(); ++b)
                            if (distToSegment(px, py, bars[b].s) <=
                                bars[b].halfw + 0.5) {
                                vis = false;
                                break;
                            }
                        if (vis && a0 < 0) a0 = t;
                        if (!vis && a0 >= 0) { iv.push_back({a0, t - step}); a0 = -1; }
                    }
                    if (a0 >= 0) iv.push_back({a0, slen});
                    for (const auto& [ia, ib] : iv) {
                        if (ib - ia < 24.0) continue;
                        ++total;
                        double e[2];
                        bool ok[2];
                        const std::vector<LineSegment>* dd[2] = {&dets_ext, &dets_fc};
                        for (int cfg = 0; cfg < 2; ++cfg) {
                            double best_ov = 0, bt0 = 0, bt1 = 0;
                            double fo = segOrient(flank);
                            for (const LineSegment& ds : *dd[cfg]) {
                                if (orientDiff(segOrient(ds), fo) > ang_th) continue;
                                double q0, t0, q1, t1;
                                pointToLine(flank, ds.x0, ds.y0, q0, t0);
                                pointToLine(flank, ds.x1, ds.y1, q1, t1);
                                if (q0 > tau || q1 > tau) continue;
                                double lo = std::min(t0, t1), hi = std::max(t0, t1);
                                double ov = std::min(hi, ib) - std::max(lo, ia);
                                if (ov > best_ov) { best_ov = ov; bt0 = lo; bt1 = hi; }
                            }
                            ok[cfg] = best_ov >= 0.5 * (ib - ia);
                            e[cfg] = ok[cfg] ? std::abs(bt0 - ia) + std::abs(bt1 - ib) : 0.0;
                            if (ok[cfg]) { ++rec[cfg]; errs[cfg].push_back(e[cfg]); }
                        }
                        if (ok[0] && ok[1]) {
                            double dfc = e[1] - e[0];  // >0: extreme rule better
                            diffs.push_back(dfc);
                            if (dfc > 1e-9) ++wins;
                            else if (dfc < -1e-9) ++loss;
                        }
                    }
                }
            }
            double mean0 = 0, mean1 = 0, dmean = 0;
            for (double v : errs[0]) mean0 += v;
            for (double v : errs[1]) mean1 += v;
            for (double v : diffs) dmean += v;
            if (!errs[0].empty()) mean0 /= errs[0].size();
            if (!errs[1].empty()) mean1 /= errs[1].size();
            if (!diffs.empty()) dmean /= diffs.size();
            std::printf("  %-8s intervals %4ld | ext: rec %5.1f%% med %5.2f mean %5.2f px"
                        " | fc: rec %5.1f%% med %5.2f mean %5.2f px\n",
                        jk.name, total,
                        100.0 * rec[0] / total, med(errs[0]), mean0,
                        100.0 * rec[1] / total, med(errs[1]), mean1);
            std::printf("           paired dMean %+5.2f px, ext better/worse/tie"
                        " %ld/%ld/%ld (n=%ld)\n",
                        dmean, wins, loss, (long)diffs.size() - wins - loss,
                        (long)diffs.size());
        }
        return 0;
    }

    std::printf("Synthetic evaluation: %dx%d, %d segments/image, %d images/condition\n",
                w, h, n_seg, images);
    std::printf("Matching: lateral<=%.1fpx, angle<=%.1fdeg\n\n", tau, ang_th * 180 / kPi);

    // Pre-generate clean canvases + GT (geometry fixed across noise conditions).
    std::vector<std::vector<float>> cleans(images);
    std::vector<std::vector<LineSegment>> gts(images);
    std::vector<std::vector<LineSegment>> bars(images);
    for (int im = 0; im < images; ++im)
        cleans[im] = genClean(w, h, n_seg, 1000u + im, gts[im], bars[im]);

    struct Condition { double sigma; std::vector<Curve> curves; };
    std::vector<Condition> conditions;

    for (double sigma : sigmas) {
        // Accumulate stats per (method, knob) across all images (micro-average).
        std::vector<Stats> acc_sweeplsd(sweeplsd_knobs.size());
        std::vector<Stats> acc_sweeplsd_imp(sweeplsd_knobs.size());
        std::vector<Stats> acc_sweeplsd_implink(sweeplsd_knobs.size());
        std::vector<std::vector<Stats>> acc_cfg(
            configs.size(), std::vector<Stats>(sweeplsd_knobs.size()));
        std::vector<Stats> acc_lsd(lsd_knobs.size());
        std::vector<Stats> acc_ed(ed_knobs.size());
        std::vector<Stats> acc_mlsd(mlsd_knobs.size());
        std::vector<Stats> acc_edreal(ed_knobs.size());
        std::vector<Stats> acc_elsed(ed_knobs.size());
        // ELSED aborts (upstream assert) below minLineLen ~7-10, so some knob
        // files may be absent; a knob joins the curve only if every image of
        // the condition produced a result.
        std::vector<int> elsed_got(ed_knobs.size(), 0);

        for (int im = 0; im < images; ++im) {
            sweeplsd::GrayImage img = addNoise(cleans[im], w, h, sigma, 7000u + im);
            const std::vector<LineSegment>& gt = gts[im];
            if (!dump_dir.empty())
                sweeplsd::saveGrayPng(dump_dir + "/eval_s" + sigTag(sigma) + "_im" +
                                   std::to_string(im) + ".png", img);
            for (std::size_t k = 0; k < sweeplsd_knobs.size(); ++k)
                acc_sweeplsd[k].add(matchSegments(gt, runSweeplsd(img, sweeplsd_knobs[k]), tau, ang_th));
            for (std::size_t k = 0; k < sweeplsd_knobs.size(); ++k)
                acc_sweeplsd_imp[k].add(matchSegments(gt, runSweeplsdImproved(img, sweeplsd_knobs[k]), tau, ang_th));
            for (std::size_t k = 0; k < sweeplsd_knobs.size(); ++k)
                acc_sweeplsd_implink[k].add(matchSegments(gt, runSweeplsdImprovedLink(img, sweeplsd_knobs[k]), tau, ang_th));
            for (std::size_t ci = 0; ci < configs.size(); ++ci)
                for (std::size_t k = 0; k < sweeplsd_knobs.size(); ++k) {
                    sweeplsd::Params p = configs[ci].params;
                    p.pixel_num_th = sweeplsd_knobs[k];
                    acc_cfg[ci][k].add(matchSegments(gt, sweeplsd::detect(img, p), tau, ang_th));
                }
            for (std::size_t k = 0; k < lsd_knobs.size(); ++k)
                acc_lsd[k].add(matchSegments(gt, runLsdEps(img, lsd_knobs[k]), tau, ang_th));
            for (std::size_t k = 0; k < ed_knobs.size(); ++k)
                acc_ed[k].add(matchSegments(gt, runEd(img, ed_knobs[k]), tau, ang_th));
            if (have_mlsd)
                for (std::size_t k = 0; k < mlsd_knobs.size(); ++k) {
                    std::vector<LineSegment> ml;
                    readMlsdFile(mlsd_dir + "/eval_s" + sigTag(sigma) + "_im" +
                                 std::to_string(im) + "_k" + std::to_string(k) + ".txt", ml);
                    acc_mlsd[k].add(matchSegments(gt, ml, tau, ang_th));
                }
            if (have_edreal)
                for (std::size_t k = 0; k < ed_knobs.size(); ++k) {
                    std::vector<LineSegment> er;
                    readEdRealFile(edreal_dir + "/eval_s" + sigTag(sigma) + "_im" +
                                   std::to_string(im) + "_k" + std::to_string(k) + ".txt", er);
                    acc_edreal[k].add(matchSegments(gt, er, tau, ang_th));
                }
            if (have_elsed)
                for (std::size_t k = 0; k < ed_knobs.size(); ++k) {
                    std::vector<LineSegment> el;
                    if (readEdRealFile(elsed_dir + "/eval_s" + sigTag(sigma) + "_im" +
                                       std::to_string(im) + "_k" + std::to_string(k) + ".txt", el)) {
                        acc_elsed[k].add(matchSegments(gt, el, tau, ang_th));
                        ++elsed_got[k];
                    }
                }
        }

        auto buildCurve = [&](const std::string& name, const std::string& color,
                              const std::vector<Stats>& acc, auto& knobs) {
            Curve c; c.method = name; c.color = color;
            for (std::size_t k = 0; k < acc.size(); ++k)
                c.pts.push_back({double(knobs[k]), acc[k].recall(), acc[k].precision(),
                                 acc[k].f1(), acc[k].latErr(), acc[k].angErrDeg(),
                                 double(acc[k].tp + acc[k].fp) / images});
            finalizeCurve(c);
            return c;
        };
        Condition cond;
        cond.sigma = sigma;
        cond.curves.push_back(buildCurve("SweepLSD", "#1769aa", acc_sweeplsd, sweeplsd_knobs));
        cond.curves.push_back(buildCurve("SweepLSD-improved", "#7b2d8b", acc_sweeplsd_imp, sweeplsd_knobs));
        cond.curves.push_back(buildCurve("SweepLSD-imp+link", "#c0497a", acc_sweeplsd_implink, sweeplsd_knobs));
        static const char* kCfgColors[] = {"#555555", "#8a6d3b", "#3b8a6d",
                                           "#6d3b8a", "#a04040", "#4040a0",
                                           "#7a7a20", "#207a7a"};
        for (std::size_t ci = 0; ci < configs.size(); ++ci)
            cond.curves.push_back(buildCurve("cfg " + configs[ci].spec,
                                             kCfgColors[ci % 8],
                                             acc_cfg[ci], sweeplsd_knobs));
        cond.curves.push_back(buildCurve("LSD", "#e8820c", acc_lsd, lsd_knobs));
        if (have_edreal)
            cond.curves.push_back(buildCurve("EDLines (ED_Lib)", "#2e9e4f", acc_edreal, ed_knobs));
        if (have_elsed) {
            std::vector<Stats> acc_f;
            std::vector<int> knobs_f;
            for (std::size_t k = 0; k < ed_knobs.size(); ++k)
                if (elsed_got[k] == images) { acc_f.push_back(acc_elsed[k]); knobs_f.push_back(ed_knobs[k]); }
            if (!acc_f.empty())
                cond.curves.push_back(buildCurve("ELSED", "#b8860b", acc_f, knobs_f));
        }
        cond.curves.push_back(buildCurve("EDLines-style", "#6abf8a", acc_ed, ed_knobs));
        if (have_mlsd)
            cond.curves.push_back(buildCurve("M-LSD", "#d62878", acc_mlsd, mlsd_knobs));

        std::printf("noise sigma = %.0f\n", sigma);
        std::printf("  %-52s %6s %6s %7s %7s %9s %8s %7s\n",
                    "method", "F-max", "AP", "P@best", "R@best", "latErr", "angErr", "dets");
        for (const Curve& c : cond.curves)
            std::printf("  %-52s %6.3f %6.3f %7.3f %7.3f %7.2fpx %6.2fdeg %7.1f\n",
                        c.method.c_str(), c.fmax, c.ap, c.best.precision, c.best.recall,
                        c.best.lat, c.best.ang, c.best.dets);
        std::printf("\n");
        conditions.push_back(std::move(cond));
    }

    // --- HTML report (sample gallery at sigma=10, PR curves per condition) ---
    if (!html_path.empty() && !assets_dir.empty()) {
        // Representative sample image (image 0 at sigma=10) + overlays.
        double rep_sigma = 10;
        sweeplsd::GrayImage rep = addNoise(cleans[0], w, h, rep_sigma, 7000u);
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_src.png", rep, {});
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_gt.png", rep, gts[0]);
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_sweeplsd.png", rep, runSweeplsd(rep, 16));
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_sweeplsd_improved.png", rep, runSweeplsdImproved(rep, 16));
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_sweeplsd_implink.png", rep, runSweeplsdImprovedLink(rep, 16));
        sweeplsd::saveSegmentVisualization(assets_dir + "/eval_lsd.png", rep, runLsdEps(rep, 0));
        if (have_edreal) {
            std::vector<LineSegment> er;  // minlen=10 (knob index 4) on s=10, image 0
            readEdRealFile(edreal_dir + "/eval_s10_im0_k4.txt", er);
            sweeplsd::saveSegmentVisualization(assets_dir + "/eval_ed.png", rep, er);
        } else {
            sweeplsd::saveSegmentVisualization(assets_dir + "/eval_ed.png", rep, runEd(rep, 10));
        }
        if (have_elsed) {
            std::vector<LineSegment> el;  // minlen=10 (knob index 4) on s=10, image 0
            if (readEdRealFile(elsed_dir + "/eval_s10_im0_k4.txt", el))
                sweeplsd::saveSegmentVisualization(assets_dir + "/eval_elsed.png", rep, el);
        }
        if (have_mlsd) {
            std::vector<LineSegment> ml;  // score=0.10 (knob index 2) on s=10, image 0
            readMlsdFile(mlsd_dir + "/eval_s10_im0_k2.txt", ml);
            sweeplsd::saveSegmentVisualization(assets_dir + "/eval_mlsd.png", rep, ml);
        }

        std::ofstream o(html_path);
        if (o) {
            // Asset path relative to the HTML file's directory (e.g. html in
            // docs/, assets in docs/assets/eval/  ->  "assets/eval").
            std::string html_dir;
            std::string::size_type hs = html_path.find_last_of("/\\");
            if (hs != std::string::npos) html_dir = html_path.substr(0, hs);
            std::string ar = assets_dir;
            if (!html_dir.empty() && ar.size() > html_dir.size() &&
                ar.compare(0, html_dir.size(), html_dir) == 0 &&
                (ar[html_dir.size()] == '/' || ar[html_dir.size()] == '\\'))
                ar = ar.substr(html_dir.size() + 1);
            for (char& ch : ar) if (ch == '\\') ch = '/';

            o << "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
                 "<title>SweepLSD 定量評価（合成GT）</title><style>"
                 "body{font-family:'Segoe UI',Meiryo,sans-serif;margin:0;background:#f5f6f8;color:#1d2027;line-height:1.7}"
                 ".wrap{max-width:1000px;margin:0 auto;padding:32px 24px 64px}"
                 "h1{font-size:25px;margin:0 0 4px}h2{margin-top:38px;border-bottom:2px solid #d8dbe0;padding-bottom:6px}"
                 "table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:8px;overflow:hidden;margin-top:10px}"
                 "th,td{padding:8px 11px;text-align:right;border-bottom:1px solid #eef0f3}th:first-child,td:first-child{text-align:left}"
                 "thead th{background:#2b3245;color:#fff}.sweeplsd{color:#1769aa;font-weight:600}"
                 ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:14px}"
                 ".card{background:#fff;border-radius:8px;padding:9px;box-shadow:0 1px 3px rgba(0,0,0,.08)}"
                 ".card img{width:100%;border-radius:4px;background:#222}.card h3{margin:6px 2px;font-size:14px}"
                 ".curves{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}"
                 ".note{background:#fff8e1;border-left:4px solid #f5c518;padding:12px 16px;border-radius:4px;font-size:14px}"
                 "code{background:#eceff4;padding:1px 5px;border-radius:4px}</style></head><body><div class=\"wrap\">";

            o << "<h1>SweepLSD の定量評価 — 合成グラウンドトゥルースによる公平な比較</h1>"
                 "<p>既存手法（LSD・EDLines-style）との公平な比較のため、<b>既知の線分を描画した合成画像</b>で "
                 "Precision/Recall・幾何精度を測る。線分の向きは [0,180&deg;) 一様（SweepLSD は勾配方向を H/V に量子化"
                 "するため、斜め線を含めないと不公平）。幾何は固定し、ガウシアンノイズ &sigma; のみ変えてノイズ感度を分離。</p>";

            o << "<h2>公平化のための条件設定</h2><ul>"
                 "<li><b>同一入力</b>：全手法に同一の8bitグレー画像を渡す。</li>"
                 "<li><b>操作点の掃引</b>：単一しきい値の比較は不公平なので、各手法の主感度ノブを振って "
                 "<b>PR フロンティア</b>を描く（SweepLSD=<code>pixel_num_th</code>, LSD=<code>eps</code>(NFA), "
                 "EDLines=<code>min_length</code>）。要約は <b>F-max</b> と <b>AP</b>（PR曲線下面積）。</li>"
                 "<li><b>マッチング</b>：角度差&le;10&deg; かつ横方向距離&le;2px かつ射影の重なりが検出長の半分以上の検出を、"
                 "重なり長で1対1貪欲対応。TP=対応した検出, FP=残り, FN=未対応のGT。</li>"
                 "<li><b>幾何精度</b>：対応した対について横方向誤差・角度誤差（分断に頑健）。</li>"
                 "<li><b>速度/ISA</b> は <code>sweeplsd_compare</code> 側で AVX2 整合済み。本ツールは品質に集中。</li>"
                 "</ul>";

            o << "<h2>サンプル（&sigma;=10, 画像0）</h2><div class=\"gallery\">";
            // The unsuffixed "SweepLSD" card must show the shipped configuration
            // (eval_sweeplsd_improved.png — the image the public docs and the paper
            // use); eval_sweeplsd.png is the original2014() baseline.
            const char* keys[][2] = {{"src", "入力（合成＋ノイズ）"}, {"gt", "Ground Truth"},
                                     {"sweeplsd_improved", "SweepLSD"}, {"lsd", "LSD"},
                                     {"ed", "EDLines-style"}};
            for (auto& kv : keys)
                o << "<div class=\"card\"><img src=\"" << ar << "/eval_" << kv[0]
                  << ".png\"><h3>" << kv[1] << "</h3></div>";
            if (have_elsed)
                o << "<div class=\"card\"><img src=\"" << ar
                  << "/eval_elsed.png\"><h3>ELSED</h3></div>";
            o << "</div>";

            o << "<h2>PR フロンティア</h2><div class=\"curves\">";
            for (const Condition& c : conditions) {
                char t[64];
                std::snprintf(t, sizeof(t), "sigma = %.0f", c.sigma);
                o << prCurveSvg(c.curves, t);
            }
            o << "</div>";

            o << "<h2>メトリクス（ノイズ条件別, 画像横断のマイクロ平均）</h2>"
                 "<table><thead><tr><th>&sigma;</th><th>手法</th><th>F-max</th><th>AP</th>"
                 "<th>P@best</th><th>R@best</th><th>横方向誤差</th><th>角度誤差</th></tr></thead><tbody>";
            for (const Condition& c : conditions) {
                for (std::size_t i = 0; i < c.curves.size(); ++i) {
                    const Curve& cu = c.curves[i];
                    char row[512];
                    std::snprintf(row, sizeof(row),
                        "<tr>%s<td%s>%s</td><td%s>%.3f</td><td%s>%.3f</td><td>%.3f</td><td>%.3f</td>"
                        "<td>%.2f px</td><td>%.2f&deg;</td></tr>",
                        i == 0 ? ("<td rowspan=\"" + std::to_string(c.curves.size()) + "\">" +
                                  std::to_string((int)c.sigma) + "</td>").c_str() : "",
                        cu.method == "SweepLSD" ? " class=\"sweeplsd\"" : "", cu.method.c_str(),
                        cu.method == "SweepLSD" ? " class=\"sweeplsd\"" : "", cu.fmax,
                        cu.method == "SweepLSD" ? " class=\"sweeplsd\"" : "", cu.ap,
                        cu.best.precision, cu.best.recall, cu.best.lat, cu.best.ang);
                    o << row;
                }
            }
            o << "</tbody></table>";

            o << "<h2>読み方</h2><div class=\"note\">"
                 "<b>F-max</b> は各手法が達成できる最良の精度・再現バランス、<b>AP</b> は操作点全域での総合力。"
                 "<b>横方向誤差・角度誤差</b>は当たった線分の幾何精度（小さいほど正確）。"
                 "LSD/EDLines は a-contrario 検証で高精度寄り、SweepLSD の既定判定はより単純。"
                 "SweepLSD でも <code>--nfa</code>（improvement 4）を有効化すると同様の高精度側に寄せられる。"
                 "合成GTは制御性が高い反面、実写の質感エッジは表現しないため、実画像評価（York Urban 等）は別途必要。"
                 "</div>";
            o << "</div></body></html>";
            std::printf("HTML report: %s\n", html_path.c_str());
        }
    }
    return 0;
}
