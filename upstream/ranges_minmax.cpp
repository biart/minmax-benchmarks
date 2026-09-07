// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// `ranges::min`, `ranges::max`, and `ranges::minmax` choose between scanning with an iterator and
// caching a `range_value_t` (see `_Prefer_iterator_copies` in <xutility>). `minmax_element.cpp`
// covers contiguous ranges of scalars, where the vectorized paths apply; this file covers the
// ranges where they do not, and where the choice is therefore visible.
//
// The number of times the running result is replaced determines what caching a value costs:
// zero for `Op::Min` over sorted input, `N - 1` for `Op::Max` over sorted input, and about
// `log N` for random input. `Kind` varies what a dereference and an assignment cost.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <format>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "skewed_allocator.hpp"

enum class Op { Min, Max, Both };

enum class Shape { Random, Sorted };

enum class Kind {
    Vec, // contiguous scalars: dereference is a load, assignment is a register move
    Transform, // dereference computes a prvalue; no vectorized path applies
    Deque, // segmented: dereference is an indirection through the block map
    Str, // dereference is cheap, assignment is not
    Join, // segmented, with a branchy ++ that dominates the loop
    Zip, // reference is a cheap proxy, but assigning it to range_value_t copies a string
};

using namespace std;

template <class T>
using not_highly_aligned_vector = vector<T, not_highly_aligned_allocator<T>>;

vector<size_t> ranks(const size_t n, const Shape shape) {
    vector<size_t> result(n);
    iota(result.begin(), result.end(), size_t{0});
    if (shape == Shape::Random) {
        ranges::shuffle(result, mt19937_64{84710});
    }

    return result;
}

not_highly_aligned_vector<double> make_doubles(const size_t n, const Shape shape) {
    const auto r = ranks(n, shape);
    not_highly_aligned_vector<double> result(n);
    ranges::transform(r, result.begin(), [](const size_t x) { return static_cast<double>(x); });
    return result;
}

// The seven-digit prefix is fixed width, so lexicographic order matches rank order. The suffix
// pushes every string onto the heap without making the comparison length-dependent.
vector<string> make_strings(const size_t n, const Shape shape) {
    const auto r = ranks(n, shape);
    vector<string> result;
    result.reserve(n);
    for (const auto rank : r) {
        result.push_back(format("{:07}", rank) + string(56, 'x'));
    }

    return result;
}

vector<vector<double>> make_chunks(const size_t n, const Shape shape) {
    const auto flat = make_doubles(n, shape);
    constexpr size_t chunk = 64;
    vector<vector<double>> result;
    for (size_t i = 0; i < n; i += chunk) {
        result.emplace_back(flat.begin() + static_cast<ptrdiff_t>(i),
            flat.begin() + static_cast<ptrdiff_t>(ranges::min(i + chunk, n)));
    }

    return result;
}

template <Op O, class Rng>
void invoke(Rng& rng) {
    if constexpr (O == Op::Min) {
        benchmark::DoNotOptimize(ranges::min(rng));
    } else if constexpr (O == Op::Max) {
        benchmark::DoNotOptimize(ranges::max(rng));
    } else {
        benchmark::DoNotOptimize(ranges::minmax(rng));
    }
}

template <Kind K, Op O, Shape S>
void bm(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range());

    if constexpr (K == Kind::Str) {
        auto data = make_strings(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            invoke<O>(data);
        }
    } else if constexpr (K == Kind::Deque) {
        const auto source = make_doubles(n, S);
        deque<double> data(source.begin(), source.end());
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            invoke<O>(data);
        }
    } else if constexpr (K == Kind::Transform) {
        auto data = make_doubles(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            auto transformed = data | views::transform([](const double x) { return x * 1.0000001; });
            invoke<O>(transformed);
        }
    } else if constexpr (K == Kind::Join) {
        auto chunks = make_chunks(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(chunks);
            auto joined = chunks | views::join;
            invoke<O>(joined);
        }
    } else if constexpr (K == Kind::Zip) {
        auto keys = make_strings(n, S);
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        for (auto _ : state) {
            benchmark::DoNotOptimize(keys);
            benchmark::DoNotOptimize(idx);
            auto zipped = views::zip(keys, idx);
            invoke<O>(zipped);
        }
    } else {
        auto data = make_doubles(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            invoke<O>(data);
        }
    }
}

void common_arg(benchmark::Benchmark* bm) {
    bm->Arg(20); // short ranges, where the running result is replaced comparatively often
    bm->Arg(8021);
}

#define BENCHMARK_KIND(Kind_)                                              \
    BENCHMARK(bm<Kind_, Op::Min, Shape::Random>)->Apply(common_arg);       \
    BENCHMARK(bm<Kind_, Op::Max, Shape::Random>)->Apply(common_arg);       \
    BENCHMARK(bm<Kind_, Op::Both, Shape::Random>)->Apply(common_arg);      \
    BENCHMARK(bm<Kind_, Op::Min, Shape::Sorted>)->Apply(common_arg);       \
    BENCHMARK(bm<Kind_, Op::Max, Shape::Sorted>)->Apply(common_arg);       \
    BENCHMARK(bm<Kind_, Op::Both, Shape::Sorted>)->Apply(common_arg)

BENCHMARK_KIND(Kind::Vec);
BENCHMARK_KIND(Kind::Transform);
BENCHMARK_KIND(Kind::Deque);
BENCHMARK_KIND(Kind::Str);
BENCHMARK_KIND(Kind::Join);
BENCHMARK_KIND(Kind::Zip);

BENCHMARK_MAIN();
