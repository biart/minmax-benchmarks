// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The short, representative set behind the `_Prefer_iterator_copies` retune.
//
// This file calls the public API only, so the same source measures whichever STL it is built
// against. Point CMake at a built STL with -DSTL_BINARY_DIR and compare tips with the
// compare.py in google-benchmark/tools; nothing here knows which branch it is measuring.
//
// Eighteen rows: eight range kinds over two orderings, plus two singles. Each kind is present
// because it exercises a distinct reason to prefer one caching strategy over the other, and
// several are here specifically as guardrails -- they must NOT change.
//
//   Vec          reference, and the only kind reaching _Min_element_vectorized (the value path
//                excludes floating point unless _M_FP_FAST, so the branch choice gates SIMD)
//   Str          reference whose assignment allocates: the worst case for caching values
//   Deque        reference, segmented dereference, no vectorized path
//   Join         reference behind a 24-byte iterator with a branchy ++
//   Zip          proxy reference: remove_cvref_t<Ref> is not the value type
//   Transform    prvalue of the value type, cheap: the case the retune targets
//   Prvalue32    prvalue exactly at the guard: the only prvalue row that changes branch
//   Prvalue64    prvalue just over it (guardrail, Random only)
//   Prvalue256   prvalue far over it, where copies spill to the stack (guardrail, Random only)
//   Iota         prvalue of a tiny value type (ascending by construction)
//   ExpensiveRef reference, but operator* does O(chunk) work per element
//
// `Shape::Adverse` is descending, so `ranges::min` replaces its running result at every step
// (I = N - 1) -- the maximum-copy stress case. `Shape::Random` gives I ~ log N.

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
#include "synthetic.hpp"

enum class Shape { Random, Adverse };

enum class Kind { Vec, Str, Deque, Join, Zip, Transform, Prvalue32, Prvalue64, Prvalue256, Iota, ExpensiveRef };

using namespace std;

template <class T>
using not_highly_aligned_vector = vector<T, not_highly_aligned_allocator<T>>;

vector<size_t> ranks(const size_t n, const Shape shape) {
    vector<size_t> result(n);
    iota(result.begin(), result.end(), size_t{0});
    if (shape == Shape::Random) {
        ranges::shuffle(result, mt19937_64{84710});
    } else {
        ranges::reverse(result); // descending: the running minimum improves at every step
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

// The transform below copies its source rather than manufacturing a fresh value. Only the
// leading key is ever compared, so a value built from nothing lets the optimizer drop the
// payload write entirely -- which silently makes Prvalue64 and Prvalue256 the same benchmark.
// Copying a real source keeps the payload live and the sizes distinguishable.
template <size_t Bytes>
vector<mmb::sized_value<Bytes>, not_highly_aligned_allocator<mmb::sized_value<Bytes>>> make_values(
    const size_t n, const Shape shape) {
    using Big = mmb::sized_value<Bytes>;
    const auto r = ranks(n, shape);
    vector<Big, not_highly_aligned_allocator<Big>> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i].key = static_cast<typename Big::key_type>(r[i]);
        ranges::fill(result[i].tail.b, static_cast<unsigned char>(i));
    }

    return result;
}

template <Kind K, Shape S>
void bm(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range());

    if constexpr (K == Kind::Vec) {
        auto data = make_doubles(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            benchmark::DoNotOptimize(ranges::min(data));
        }
    } else if constexpr (K == Kind::Str) {
        auto data = make_strings(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            benchmark::DoNotOptimize(ranges::min(data));
        }
    } else if constexpr (K == Kind::Deque) {
        const auto source = make_doubles(n, S);
        deque<double> data(source.begin(), source.end());
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            benchmark::DoNotOptimize(ranges::min(data));
        }
    } else if constexpr (K == Kind::Join) {
        auto chunks = make_chunks(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(chunks);
            auto joined = chunks | views::join;
            benchmark::DoNotOptimize(ranges::min(joined));
        }
    } else if constexpr (K == Kind::Zip) {
        auto keys = make_strings(n, S);
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        for (auto _ : state) {
            benchmark::DoNotOptimize(keys);
            benchmark::DoNotOptimize(idx);
            auto zipped = views::zip(keys, idx);
            benchmark::DoNotOptimize(ranges::min(zipped));
        }
    } else if constexpr (K == Kind::Transform) {
        auto data = make_doubles(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            auto transformed = data | views::transform([](const double x) { return x * 1.0000001; });
            benchmark::DoNotOptimize(ranges::min(transformed));
        }
    } else if constexpr (K == Kind::Prvalue32 || K == Kind::Prvalue64 || K == Kind::Prvalue256) {
        constexpr size_t bytes = (K == Kind::Prvalue32) ? 32 : ((K == Kind::Prvalue64) ? 64 : 256);
        using Big              = mmb::sized_value<bytes>;
        auto data              = make_values<bytes>(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(data);
            auto widened = data | views::transform([](const Big& b) {
                Big c = b;
                c.key = b.key;
                return c;
            });
            benchmark::DoNotOptimize(ranges::min(widened));
        }
    } else if constexpr (K == Kind::Iota) {
        for (auto _ : state) {
            auto counted = views::iota(size_t{0}, n);
            benchmark::DoNotOptimize(ranges::min(counted));
        }
    } else {
        static_assert(K == Kind::ExpensiveRef);
        auto chunks = make_chunks(n, S);
        for (auto _ : state) {
            benchmark::DoNotOptimize(chunks);
            // Yields a reference, but each dereference scans a whole chunk to produce it.
            auto reduced = chunks | views::transform([](const vector<double>& row) -> const double& {
                return *ranges::min_element(row);
            });
            benchmark::DoNotOptimize(ranges::min(reduced));
        }
    }
}

void common_arg(benchmark::Benchmark* bm) {
    bm->Arg(8021);
}

BENCHMARK(bm<Kind::Vec, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Vec, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Str, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Deque, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Join, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Zip, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Transform, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Prvalue32, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Prvalue32, Shape::Adverse>)->Apply(common_arg);
BENCHMARK(bm<Kind::Prvalue64, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Prvalue256, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::Iota, Shape::Random>)->Apply(common_arg);
BENCHMARK(bm<Kind::ExpensiveRef, Shape::Random>)->Apply(common_arg);

BENCHMARK_MAIN();
