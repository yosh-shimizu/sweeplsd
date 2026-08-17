// Per-configuration timing over an image directory: the runtime column of the
// refinement ablation. Each named configuration (--config, see
// config_spec.hpp) runs over every PNG in the directory with the ONE-PASS
// driver (the paper's timing driver), interleaved configuration-by-
// configuration within each image; per-image time is the median of --runs
// repetitions, and the table reports the per-configuration corpus median,
// p95, worst frame, and median segment count.
//
// Usage: config_timer <png dir> --config SPEC [--config SPEC ...] [--runs N]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#include "config_spec.hpp"
#include "sweeplsd/io.hpp"
#include "sweeplsd/sweeplsd.hpp"

namespace {

std::vector<std::string> listPngs(const std::string& dir) {
    std::vector<std::string> out;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*.png").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do out.push_back(dir + "/" + fd.cFileName);
        while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".png")
                out.push_back(dir + "/" + n);
        }
        closedir(d);
    }
#endif
    std::sort(out.begin(), out.end());
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double i = q * (v.size() - 1);
    std::size_t lo = (std::size_t)i;
    std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (i - lo);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir;
    int runs = 5;
    std::vector<NamedConfig> configs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            NamedConfig nc;
            nc.spec = argv[++i];
            if (!parseConfigSpec(nc.spec, nc.params)) {
                std::fprintf(stderr, "bad --config spec: %s\n", nc.spec.c_str());
                return 1;
            }
            configs.push_back(nc);
        } else if (a == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (!a.empty() && a[0] != '-') {
            dir = a;
        }
    }
    if (dir.empty() || configs.empty()) {
        std::printf("Usage: %s <png dir> --config SPEC [--config SPEC ...] [--runs N]\n", argv[0]);
        return 1;
    }
    std::vector<std::string> paths = listPngs(dir);
    if (paths.empty()) { std::fprintf(stderr, "no PNGs in %s\n", dir.c_str()); return 1; }
    std::printf("config timing: %zu images, %d runs/image, one-pass driver\n\n",
                paths.size(), runs);

    std::vector<std::vector<double>> times(configs.size());   // per-image medians
    std::vector<std::vector<double>> counts(configs.size());
    int done = 0;
    for (const std::string& p : paths) {
        sweeplsd::GrayImage img = sweeplsd::loadGray(p);
        if (img.width == 0) { std::fprintf(stderr, "skip %s\n", p.c_str()); continue; }
        for (std::size_t ci = 0; ci < configs.size(); ++ci) {
            std::vector<double> t(runs);
            std::size_t nseg = 0;
            for (int r = 0; r < runs; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                std::vector<sweeplsd::LineSegment> segs =
                    sweeplsd::detectOnePass(img, configs[ci].params);
                auto t1 = std::chrono::steady_clock::now();
                t[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                nseg = segs.size();
            }
            times[ci].push_back(median(t));
            counts[ci].push_back((double)nseg);
        }
        if (++done % 20 == 0) std::printf("  ...%d/%zu\n", done, paths.size());
    }

    std::printf("\n  %-52s %9s %9s %9s %9s\n", "config", "median", "p95", "worst", "medSegs");
    for (std::size_t ci = 0; ci < configs.size(); ++ci) {
        std::vector<double> t = times[ci];
        std::printf("  %-52s %7.2fms %7.2fms %7.2fms %9.0f\n", configs[ci].spec.c_str(),
                    median(t), quantile(t, 0.95),
                    *std::max_element(t.begin(), t.end()), median(counts[ci]));
    }
    return 0;
}
