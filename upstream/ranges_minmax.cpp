// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// `ranges::min`, `ranges::max`, and `ranges::minmax` choose between scanning with an iterator and
// caching a `range_value_t` (`_Prefer_iterator_copies` in <xutility>). The iterator branch performs
// 2N - 1 dereferences; the value branch performs N + I, where I is the number of times the running
// result is replaced. Which is faster therefore depends on what a dereference costs, what an
// assignment costs, and how often the result is replaced -- none of which `minmax_element.cpp`
// varies, because it benchmarks contiguous ranges of scalars where the vectorized paths apply and
// the choice is invisible.
//
// This file covers the ranges where the choice is visible, and deliberately does not overlap:
// every `Kind` here is either non-contiguous, non-scalar, or a view.
//
// `Shape::Sorted` is ascending, which makes I depend on the operation: `Op::Min` never replaces its
// result (I = 0), `Op::Max` replaces it at every step (I = N - 1), and `Op::Both` does both at once.
// `Shape::Random` gives I ~ log N. Crossing that with `Op` covers the whole range of I without a
// separate axis for it.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <format>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "skewed_allocator.hpp"

using namespace std;

// An element of exactly `Bytes` bytes whose comparison touches only the leading key, so that
// comparison cost is constant across sizes and only the cost of copying scales.
//
// The relational operators are spelled out rather than defaulted from `operator<=>`. With only a
// three-way comparison, `ranges::less` rewrites `a < b` as `(a <=> b) < 0`, and MSVC does not
// always fold the materialized ordering back into flags; that adds several instructions to the
// loop-carried dependency chain and measures the element type rather than the algorithm.
template <size_t Bytes>
struct payload {
    static_assert(Bytes > sizeof(uint64_t) && Bytes % sizeof(uint64_t) == 0);

    uint64_t key;
    uint64_t rest[Bytes / sizeof(uint64_t) - 1];

    [[nodiscard]] friend bool operator<(const payload& a, const payload& b) {
        return a.key < b.key;
    }
    [[nodiscard]] friend bool operator>(const payload& a, const payload& b) {
        return a.key > b.key;
    }
    [[nodiscard]] friend bool operator<=(const payload& a, const payload& b) {
        return a.key <= b.key;
    }
    [[nodiscard]] friend bool operator>=(const payload& a, const payload& b) {
        return a.key >= b.key;
    }
    [[nodiscard]] friend bool operator==(const payload& a, const payload& b) {
        return a.key == b.key;
    }
    [[nodiscard]] friend bool operator!=(const payload& a, const payload& b) {
        return a.key != b.key;
    }
};

enum class Op { Min, Max, Both };

enum class Shape { Random, Sorted };

enum class Kind {
    Str, // reference; assignment allocates
    Deque, // reference; dereference indirects through the block map
    Join, // reference; 24-byte iterator with a branchy operator++
    Zip, // proxy reference; the value type is not what operator* yields
    Transform, // prvalue of the value type, cheaply produced
    TransformMid, // prvalue of a 16-byte value type
    TransformBig, // prvalue of a 64-byte value type
    TransformRef, // reference, but operator* does real work to produce it
    Iota, // prvalue of a scalar, from a view with no underlying storage
};

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
    const auto flat        = make_doubles(n, shape);
    constexpr size_t chunk = 64;
    vector<vector<double>> result;
    for (size_t i = 0; i < n; i += chunk) {
        result.emplace_back(flat.begin() + static_cast<ptrdiff_t>(i),
            flat.begin() + static_cast<ptrdiff_t>(ranges::min(i + chunk, n)));
    }

    return result;
}

template <size_t Bytes>
not_highly_aligned_vector<payload<Bytes>> make_payloads(const size_t n, const Shape shape) {
    const auto r = ranks(n, shape);
    not_highly_aligned_vector<payload<Bytes>> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i].key = r[i];
        ranges::fill(result[i].rest, i);
    }

    return result;
}

template <Op O, class Rng>
void invoke(Rng&& rng) {
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
    } else if constexpr (K == Kind::Transform) {
        auto data = make_doubles(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            auto transformed = data | views::transform([](const double x) { return x * 1.0000001; });
            invoke<O>(transformed);
        }
    } else if constexpr (K == Kind::TransformMid || K == Kind::TransformBig) {
        // The transform copies a live source rather than manufacturing a value. Only the leading
        // key is compared, so a value built from nothing lets the optimizer drop the payload write
        // and makes every size the same benchmark.
        constexpr size_t bytes = (K == Kind::TransformMid) ? 16 : 64;
        using P                = payload<bytes>;
        auto data              = make_payloads<bytes>(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            auto transformed = data | views::transform([](const P& p) {
                P copy = p;
                copy.key = p.key;
                return copy;
            });
            invoke<O>(transformed);
        }
    } else if constexpr (K == Kind::TransformRef) {
        auto chunks = make_chunks(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(chunks);
            // Yields a reference, but each dereference scans a whole chunk to produce it:
            // `is_reference_v<iter_reference_t<It>>` does not imply that `*it` is cheap.
            auto reduced = chunks | views::transform([](const vector<double>& row) -> const double& {
                return *ranges::min_element(row);
            });
            invoke<O>(reduced);
        }
    } else {
        static_assert(K == Kind::Iota);
        for (auto _ : state) {
            auto counted = views::iota(size_t{0}, n);
            invoke<O>(counted);
        }
    }
}

void common_arg(benchmark::Benchmark* bm) {
    bm->Arg(20); // short ranges, where the running result is replaced comparatively often
    bm->Arg(8021);
}

BENCHMARK(bm<Kind::Str, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::Deque, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::Join, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::Zip, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::Transform, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::TransformMid, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformMid, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformMid, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformMid, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformMid, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformMid, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::TransformBig, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformBig, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformBig, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformBig, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformBig, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformBig, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::TransformRef, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformRef, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformRef, Op::Both, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformRef, Op::Min, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformRef, Op::Max, Shape::Sorted>)->Apply(common_arg);
BENCHMARK(bm<Kind::TransformRef, Op::Both, Shape::Sorted>)->Apply(common_arg);

BENCHMARK(bm<Kind::Iota, Op::Min, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Iota, Op::Max, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Iota, Op::Both, Shape::Random>)->Apply(common_arg);

BENCHMARK_MAIN();
