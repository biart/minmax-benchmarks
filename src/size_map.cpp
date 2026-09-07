// Stage 1: the crossover map over the axes the heuristic can actually see.
//
// Sweeps sizeof(iterator) x sizeof(value) x track, holding everything else at a baseline:
// random data, a cheap reference-yielding dereference, ranges::min, ranges::less, identity
// projection. The question it answers is whether any single inequality in those two sizes fits.
//
// Two sizings are registered for every cell, because sweeping element size at fixed N confounds
// "big elements" with "more memory traffic":
//   /n8021  fixed element count (comparable with the STL's own minmax_element benchmark)
//   /kb256  fixed 256 KiB footprint, so N shrinks as elements grow
//
// The synthetic sweep is backed up by real anchors (vector, transform_view, deque, join, iota).
// If the synthetic curve is flat while the anchors are not, the optimizer dropped the dead padding
// in fat_iterator and the synthetic numbers are not measuring iterator size at all; rerun the
// `fat=live` row, where the extra bytes are consulted on every dereference.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <format>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "minmax_tracks.hpp"
#include "skewed_allocator.hpp"
#include "synthetic.hpp"

using namespace std;
using mmb::fat_iterator;
using mmb::fatness;
using mmb::sized_value;
using mmb::track;

namespace {

template <class T>
using buffer = vector<T, not_highly_aligned_allocator<T>>;

constexpr size_t fixed_n     = 8021;
constexpr size_t fixed_bytes = 256 * 1024;

template <size_t Bytes>
buffer<sized_value<Bytes>> random_values(const size_t n) {
    using V = sized_value<Bytes>;
    buffer<V> a(n);
    mt19937_64 gen(84710);
    uniform_int_distribution<uint64_t> dis(0, V::key_max);
    for (auto& e : a) {
        e.key = static_cast<typename V::key_type>(dis(gen));
    }
    return a;
}

buffer<double> random_doubles(const size_t n) {
    buffer<double> a(n);
    mt19937_64 gen(84710);
    normal_distribution<double> dis(0, 100000.0);
    for (auto& e : a) {
        e = dis(gen);
    }
    return a;
}

// ------------------------------------------------------------ synthetic grid

template <size_t ItBytes, size_t ValBytes, track Tr, fatness F>
void bm_grid(benchmark::State& state) {
    using V  = sized_value<ValBytes>;
    using It = fat_iterator<V, ItBytes, F>;

    const auto n = static_cast<size_t>(state.range(0));
    auto a       = random_values<ValBytes>(n);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto r = ranges::subrange(It::at(a.data(), 0), It::at(a.data(), static_cast<ptrdiff_t>(n)));
        auto found = mmb::min_of<Tr>(r);
        benchmark::DoNotOptimize(found);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

// ------------------------------------------------------------- real anchors

template <track Tr>
void bm_vector(benchmark::State& state) {
    auto a = random_doubles(static_cast<size_t>(state.range(0)));
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto found = mmb::min_of<Tr>(a);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <track Tr>
void bm_transform(benchmark::State& state) {
    auto a = random_doubles(static_cast<size_t>(state.range(0)));
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto v     = a | views::transform([](const double x) { return x * 1.0000001; });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <track Tr>
void bm_deque(benchmark::State& state) {
    const auto src = random_doubles(static_cast<size_t>(state.range(0)));
    deque<double> d(src.begin(), src.end());
    for (auto _ : state) {
        benchmark::DoNotOptimize(d);
        auto found = mmb::min_of<Tr>(d);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <track Tr>
void bm_join(benchmark::State& state) {
    constexpr size_t chunk = 64;
    const auto src         = random_doubles(static_cast<size_t>(state.range(0)));
    vector<vector<double>> chunks;
    for (size_t i = 0; i < src.size(); i += chunk) {
        chunks.emplace_back(src.begin() + static_cast<ptrdiff_t>(i),
            src.begin() + static_cast<ptrdiff_t>(min(i + chunk, src.size())));
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(chunks);
        auto v     = chunks | views::join;
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// NOTE: iota is ascending, so min never improves after the first element. It is here as the only
// anchor with an iterator *smaller* than a pointer and a dereference that touches no memory; its
// data shape is degenerate and belongs to the Stage 2 axis.
template <track Tr>
void bm_iota(benchmark::State& state) {
    const auto n = state.range(0);
    for (auto _ : state) {
        auto bound = n;
        benchmark::DoNotOptimize(bound);
        auto v     = views::iota(int64_t{0}, bound);
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ------------------------------------------------------------- registration

template <size_t ItBytes, size_t ValBytes, track Tr, fatness F>
void register_cell() {
    constexpr const char* fat = (F == fatness::dead) ? "dead" : "live";
    const auto stem           = format("min/{}/it{}/val{}/{}", fat, ItBytes, ValBytes, mmb::name_of(Tr));

    benchmark::RegisterBenchmark(stem + "/n8021", bm_grid<ItBytes, ValBytes, Tr, F>)
        ->Arg(static_cast<int64_t>(fixed_n));
    benchmark::RegisterBenchmark(stem + "/kb256", bm_grid<ItBytes, ValBytes, Tr, F>)
        ->Arg(static_cast<int64_t>(fixed_bytes / ValBytes));
}

template <size_t ItBytes, size_t ValBytes, fatness F = fatness::dead>
void register_tracks() {
    register_cell<ItBytes, ValBytes, track::iter, F>();
    register_cell<ItBytes, ValBytes, track::value, F>();
    register_cell<ItBytes, ValBytes, track::value_1x, F>();
    register_cell<ItBytes, ValBytes, track::iter_hoist, F>();
    register_cell<ItBytes, ValBytes, track::stl, F>();
}

template <size_t ItBytes, size_t... ValBytes>
void register_row() {
    (register_tracks<ItBytes, ValBytes>(), ...);
}

template <size_t... ItBytes>
void register_grid() {
    (register_row<ItBytes, 1, 2, 4, 8, 16, 32, 64>(), ...);
}

template <size_t... ItBytes>
void register_live_diagnostic() {
    (register_tracks<ItBytes, 8, fatness::live>(), ...);
}

constexpr int64_t anchor_n = static_cast<int64_t>(fixed_n);

// A template-template parameter cannot bind a function template, so this stays a macro.
#define MMB_REGISTER_ANCHOR(BM, NAME)                                                      \
    do {                                                                                   \
        benchmark::RegisterBenchmark(NAME "/iter", BM<track::iter>)->Arg(anchor_n);         \
        benchmark::RegisterBenchmark(NAME "/value", BM<track::value>)->Arg(anchor_n);       \
        benchmark::RegisterBenchmark(NAME "/value1x", BM<track::value_1x>)->Arg(anchor_n); \
        benchmark::RegisterBenchmark(NAME "/iterhoist", BM<track::iter_hoist>)->Arg(anchor_n); \
        benchmark::RegisterBenchmark(NAME "/ref1x", BM<track::ref_1x>)->Arg(anchor_n); \
        benchmark::RegisterBenchmark(NAME "/stl", BM<track::stl>)->Arg(anchor_n);           \
    } while (0)

// All four tracks must agree; a divergence means a transcription slip in minmax_tracks.hpp.
void verify() {
    auto a = random_values<16>(1001);
    using It = fat_iterator<sized_value<16>, 32>;
    auto r   = ranges::subrange(It::at(a.data(), 0), It::at(a.data(), 1001));

    const auto expected = mmb::min_of<track::stl>(r).key;
    const auto results  = {mmb::min_of<track::iter>(r).key, mmb::min_of<track::value>(r).key,
         mmb::min_of<track::value_1x>(r).key};
    for (const auto got : results) {
        if (got != expected) {
            fprintf(stderr, "verify: tracks disagree (%llu vs %llu)\n", static_cast<unsigned long long>(got),
                static_cast<unsigned long long>(expected));
            abort();
        }
    }
}

} // unnamed namespace

int main(int argc, char** argv) {
    verify();

    register_grid<8, 16, 32, 64>();
    register_live_diagnostic<16, 32, 64>();

    MMB_REGISTER_ANCHOR(bm_vector, "min/anchor/vector_double");
    MMB_REGISTER_ANCHOR(bm_transform, "min/anchor/transform_double");
    MMB_REGISTER_ANCHOR(bm_deque, "min/anchor/deque_double");
    MMB_REGISTER_ANCHOR(bm_join, "min/anchor/join_double");
    MMB_REGISTER_ANCHOR(bm_iota, "min/anchor/iota_int64");

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
