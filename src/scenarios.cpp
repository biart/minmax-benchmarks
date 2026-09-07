// Stage 2: does caching values ever actually lose?
//
// The cost model the heuristic lacks is
//     iterator branch: 2N x deref + I x (register move, free)
//     value branch:     N x deref + I x assign
// where I is the number of improvements: ~ln N for random data, N-1 for adversarially ordered
// data. So value caching can only lose when the assignment is expensive AND I is large. This
// benchmark adds the missing axis (data shape, i.e. I) and the element type that makes an
// assignment expensive (std::string, whose copy-assign may allocate).
//
// All benchmarks compute min, so:
//   ascending  -> the minimum is the first element     -> I = 0
//   descending -> every element is a new minimum       -> I = N-1
//   uniform    -> all elements equal                   -> I = 0
// (max over ascending is the same situation as min over descending.)

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <format>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#include "minmax_tracks.hpp"
#include "skewed_allocator.hpp"
#include "synthetic.hpp"

using namespace std;
using mmb::sized_value;
using mmb::track;

namespace {

enum class shape {
    random, // I ~ ln N
    ascending, // I = 0
    descending, // I = N-1, the adversarial case
    uniform, // I = 0, all elements equal
};

[[nodiscard]] constexpr const char* name_of(const shape s) noexcept {
    switch (s) {
    case shape::random:
        return "random";
    case shape::ascending:
        return "ascending";
    case shape::descending:
        return "descending";
    case shape::uniform:
        return "uniform";
    }
    return "?";
}

// String flavors, chosen for what they do to operator=:
enum class strkind {
    sso, // 7 chars: fits MSVC's small-string buffer, assignment never allocates
    heap, // 63 chars, all the same length: allocates once, then capacity is reused
    heap_grow, // lengths grow as values shrink, so every new minimum needs more capacity
};

[[nodiscard]] constexpr const char* name_of(const strkind k) noexcept {
    switch (k) {
    case strkind::sso:
        return "sso";
    case strkind::heap:
        return "heap";
    case strkind::heap_grow:
        return "heap_grow";
    }
    return "?";
}

// Ranks in the order the shape calls for. Rank 0 is the smallest element.
vector<size_t> ranks_for(const size_t n, const shape s) {
    vector<size_t> r(n);
    switch (s) {
    case shape::random:
        iota(r.begin(), r.end(), size_t{0});
        ranges::shuffle(r, mt19937_64(84710));
        break;
    case shape::ascending:
        iota(r.begin(), r.end(), size_t{0});
        break;
    case shape::descending:
        iota(r.rbegin(), r.rend(), size_t{0});
        break;
    case shape::uniform:
        ranges::fill(r, size_t{0});
        break;
    }
    return r;
}

// The 7-digit prefix is fixed width, so lexicographic order matches rank order regardless of
// what follows it.
vector<string> make_strings(const size_t n, const strkind k, const shape s) {
    const auto r = ranks_for(n, s);
    vector<string> a;
    a.reserve(n);
    for (const auto rank : r) {
        string str = format("{:07}", rank);
        switch (k) {
        case strkind::sso:
            break;
        case strkind::heap:
            str.append(56, 'x');
            break;
        case strkind::heap_grow:
            str.append(32 + (n - 1 - rank), 'x');
            break;
        }
        a.push_back(std::move(str));
    }
    return a;
}

template <size_t Bytes>
vector<sized_value<Bytes>, not_highly_aligned_allocator<sized_value<Bytes>>> make_values(
    const size_t n, const shape s) {
    using V = sized_value<Bytes>;
    const auto r = ranks_for(n, s);
    vector<V, not_highly_aligned_allocator<V>> a(n);
    for (size_t i = 0; i < n; ++i) {
        // Monotone rank -> key mapping, so the shape survives even when the key type is too
        // narrow to hold every rank distinctly.
        const uint64_t rank = r[i];
        const uint64_t key  = (V::key_max >= n) ? rank : (rank * V::key_max) / (n - 1);
        a[i].key            = static_cast<typename V::key_type>(key);
    }
    return a;
}

template <strkind K, shape S, track Tr>
void bm_string(benchmark::State& state) {
    auto a = make_strings(static_cast<size_t>(state.range(0)), K, S);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto found = mmb::min_of<Tr>(a);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// Step 1's case: iter_reference_t is a prvalue. Each dereference constructs a whole string, and
// the value branch's assignment is a MOVE from that temporary, not a copy.
template <strkind K, shape S, track Tr>
void bm_string_prvalue(benchmark::State& state) {
    auto a = make_strings(static_cast<size_t>(state.range(0)), K, S);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto v     = a | views::transform([](const string& s) { return s + "!"; });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// The structural counterexample: a range whose reference is a CHEAP proxy but whose value type
// is EXPENSIVE to assign. zip yields tuple<string&, int&> -- two pointers, trivial to materialize
// -- while range_value_t is tuple<string, int>, and assigning reference -> value COPIES the string
// (the source holds lvalue references, so there is nothing to move from). That decouples the cost
// of a dereference from the cost of an assignment, which transform_view cannot do.
template <shape S, track Tr>
void bm_zip(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    auto keys    = make_strings(n, strkind::heap, S);
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(keys);
        benchmark::DoNotOptimize(idx);
        auto v     = views::zip(keys, idx);
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// join: the one range where the current heuristic already prefers VALUES (24 > 2*8) and where
// iterators nevertheless looked competitive. Its dereference is a single load through the cached
// inner iterator, but its ++ is branchy, so the increment -- not the comparison chain -- limits
// the loop and the two branches nearly tie.
template <shape S, track Tr>
void bm_join(benchmark::State& state) {
    const auto n           = static_cast<size_t>(state.range(0));
    const auto r           = ranks_for(n, S);
    constexpr size_t chunk = 64;

    vector<vector<double>> chunks;
    for (size_t i = 0; i < n; i += chunk) {
        vector<double> c;
        for (size_t j = i; j < std::min(i + chunk, n); ++j) {
            c.push_back(static_cast<double>(r[j]));
        }
        chunks.push_back(std::move(c));
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(chunks);
        auto v     = chunks | views::join;
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// join | transform: the same branchy ++ that let iterators win on plain join, but now the
// dereference yields a prvalue OF THE VALUE TYPE, so it is inside Step 1's scope. The iterator
// branch must run the transform 2N-1 times where the value branch runs it N times.
template <shape S, track Tr>
void bm_join_transform(benchmark::State& state) {
    const auto n           = static_cast<size_t>(state.range(0));
    const auto r           = ranks_for(n, S);
    constexpr size_t chunk = 64;

    vector<vector<double>> chunks;
    for (size_t i = 0; i < n; i += chunk) {
        vector<double> c;
        for (size_t j = i; j < std::min(i + chunk, n); ++j) {
            c.push_back(static_cast<double>(r[j]));
        }
        chunks.push_back(std::move(c));
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(chunks);
        auto v     = chunks | views::join | views::transform([](const double x) { return x * 1.0000001; });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// The last candidate that could break Step 1: a prvalue that is LARGE. The iterator branch
// materializes it 2N-1 times, the value branch N times plus I stores -- but only the leading key
// is ever compared, so the optimizer may drop the payload on the comparison-only path (making the
// iterator branch's extra dereference nearly free) while the value branch's store stays live.
//
// build: the transform manufactures a fresh 256-byte value; the payload write may be elided.
// copy:  the transform copies a 256-byte source and edits the key; the payload write cannot be.
template <size_t Bytes, shape S, track Tr>
void bm_bigprv_build(benchmark::State& state) {
    using Big    = sized_value<Bytes>;
    const auto n = static_cast<size_t>(state.range(0));
    const auto r = ranks_for(n, S);
    vector<uint64_t> src(n);
    for (size_t i = 0; i < n; ++i) {
        src[i] = r[i];
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        auto v = src | views::transform([](const uint64_t k) {
            Big b{};
            b.key = k;
            return b;
        });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <shape S, track Tr>
void bm_bigprv_copy(benchmark::State& state) {
    using Big    = sized_value<256>;
    const auto n = static_cast<size_t>(state.range(0));
    auto src     = make_values<256>(n, S);

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        auto v = src | views::transform([](const Big& b) {
            Big c = b;
            c.key = b.key;
            return c;
        });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// is_reference_v<Ref> does NOT mean the dereference is cheap. A transform may return a real
// reference and still do arbitrary work to produce it -- here, a nested min_element per row.
// Ref is `const double&` and V is `double`, so this sits in Step 2's scope, not Step 1's.
template <shape S, track Tr>
void bm_refexpensive(benchmark::State& state) {
    const auto n           = static_cast<size_t>(state.range(0));
    const auto r           = ranks_for(n, S);
    constexpr size_t width = 8;

    vector<vector<double>> rows(n);
    for (size_t i = 0; i < n; ++i) {
        rows[i].resize(width);
        for (size_t j = 0; j < width; ++j) {
            rows[i][j] = static_cast<double>(r[i]) + static_cast<double>(j) + 1.0;
        }
        rows[i][width / 2] = static_cast<double>(r[i]); // row minimum, deliberately not first
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(rows);
        auto v = rows | views::transform(
            [](const vector<double>& row) -> const double& { return *ranges::min_element(row); });
        auto found = mmb::min_of<Tr>(v);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// The quadrant ref1x exists for: an expensive dereference (deque's block-map walk) AND an
// expensive-to-copy value (string) at the same time. iter pays 2N walks, value1x pays I string
// copies, ref1x pays neither.
template <shape S, track Tr>
void bm_deque_string(benchmark::State& state) {
    const auto src = make_strings(static_cast<size_t>(state.range(0)), strkind::heap, S);
    deque<string> d(src.begin(), src.end());

    for (auto _ : state) {
        benchmark::DoNotOptimize(d);
        auto found = mmb::min_of<Tr>(d);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <size_t Bytes, shape S, track Tr>
void bm_value(benchmark::State& state) {
    auto a = make_values<Bytes>(static_cast<size_t>(state.range(0)), S);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        auto found = mmb::min_of<Tr>(a);
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

// heap_grow is quadratic in total bytes, so it gets a smaller N.
constexpr int64_t n_default   = 8021;
constexpr int64_t n_heap_grow = 2048;

template <strkind K, shape S>
void register_string_row() {
    constexpr int64_t n = (K == strkind::heap_grow) ? n_heap_grow : n_default;
    const auto stem     = format("min/str/{}/{}", name_of(K), name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_string<K, S, track::iter>)->Arg(n);
    benchmark::RegisterBenchmark(stem + "/value", bm_string<K, S, track::value>)->Arg(n);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_string<K, S, track::value_1x>)->Arg(n);
    benchmark::RegisterBenchmark(stem + "/iterhoist", bm_string<K, S, track::iter_hoist>)->Arg(n);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_string<K, S, track::ref_1x>)->Arg(n);
    benchmark::RegisterBenchmark(stem + "/stl", bm_string<K, S, track::stl>)->Arg(n);
}

template <size_t Bytes, shape S>
void register_value_row() {
    const auto stem = format("min/val{}/{}", Bytes, name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_value<Bytes, S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_value<Bytes, S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_value<Bytes, S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/iterhoist", bm_value<Bytes, S, track::iter_hoist>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_value<Bytes, S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_value<Bytes, S, track::stl>)->Arg(n_default);
}

template <shape S>
void register_refexp_row() {
    const auto stem = format("min/refexp/{}", name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_refexpensive<S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_refexpensive<S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_refexpensive<S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/iterhoist", bm_refexpensive<S, track::iter_hoist>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_refexpensive<S, track::stl>)->Arg(n_default);
}

template <size_t Bytes, shape S>
void register_bigprv_size() {
    const auto stem = format("min/bigprv{}/{}", Bytes, name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_bigprv_build<Bytes, S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_bigprv_build<Bytes, S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_bigprv_build<Bytes, S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_bigprv_build<Bytes, S, track::stl>)->Arg(n_default);
}

template <shape S>
void register_bigprv_rows() {
    for (int variant = 0; variant < 2; ++variant) {
        const auto stem = format("min/bigprv_{}/{}", variant == 0 ? "build" : "copy", name_of(S));
        if (variant == 0) {
            register_bigprv_size<32, S>();
            register_bigprv_size<64, S>();
            register_bigprv_size<128, S>();
            register_bigprv_size<256, S>();
        } else {
            benchmark::RegisterBenchmark(stem + "/iter", bm_bigprv_copy<S, track::iter>)->Arg(n_default);
            benchmark::RegisterBenchmark(stem + "/value", bm_bigprv_copy<S, track::value>)->Arg(n_default);
            benchmark::RegisterBenchmark(stem + "/value1x", bm_bigprv_copy<S, track::value_1x>)->Arg(n_default);
            benchmark::RegisterBenchmark(stem + "/stl", bm_bigprv_copy<S, track::stl>)->Arg(n_default);
        }
    }
}

template <shape S>
void register_join_transform_row() {
    const auto stem = format("min/joinT/{}", name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_join_transform<S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_join_transform<S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_join_transform<S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_join_transform<S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_join_transform<S, track::stl>)->Arg(n_default);
}

template <shape S>
void register_deque_string_row() {
    const auto stem = format("min/dqstr/{}", name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_deque_string<S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_deque_string<S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_deque_string<S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_deque_string<S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_deque_string<S, track::stl>)->Arg(n_default);
}

template <shape S>
void register_join_row() {
    const auto stem = format("min/join/{}", name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_join<S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_join<S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_join<S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_join<S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_join<S, track::stl>)->Arg(n_default);
}

template <shape S>
void register_zip_row() {
    const auto stem = format("min/zip/{}", name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_zip<S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_zip<S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_zip<S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/iterhoist", bm_zip<S, track::iter_hoist>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_zip<S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_zip<S, track::stl>)->Arg(n_default);
}

template <strkind K, shape S>
void register_prvalue_row() {
    const auto stem = format("min/prv/{}/{}", name_of(K), name_of(S));
    benchmark::RegisterBenchmark(stem + "/iter", bm_string_prvalue<K, S, track::iter>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value", bm_string_prvalue<K, S, track::value>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/value1x", bm_string_prvalue<K, S, track::value_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/iterhoist", bm_string_prvalue<K, S, track::iter_hoist>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/ref1x", bm_string_prvalue<K, S, track::ref_1x>)->Arg(n_default);
    benchmark::RegisterBenchmark(stem + "/stl", bm_string_prvalue<K, S, track::stl>)->Arg(n_default);
}

template <strkind K>
void register_string_kind() {
    register_string_row<K, shape::random>();
    register_string_row<K, shape::ascending>();
    register_string_row<K, shape::descending>();
    register_string_row<K, shape::uniform>();
}

template <size_t Bytes>
void register_value_size() {
    register_value_row<Bytes, shape::random>();
    register_value_row<Bytes, shape::ascending>();
    register_value_row<Bytes, shape::descending>();
    register_value_row<Bytes, shape::uniform>();
}

void verify() {
    auto a = make_strings(1001, strkind::heap, shape::random);
    const auto expected = mmb::min_of<track::stl>(a);
    if (mmb::min_of<track::iter>(a) != expected || mmb::min_of<track::value>(a) != expected
        || mmb::min_of<track::value_1x>(a) != expected) {
        fprintf(stderr, "verify: tracks disagree on vector<string>\n");
        abort();
    }
}

} // unnamed namespace

int main(int argc, char** argv) {
    verify();

    register_string_kind<strkind::sso>();
    register_string_kind<strkind::heap>();
    register_string_kind<strkind::heap_grow>();

    register_prvalue_row<strkind::heap, shape::random>();
    register_prvalue_row<strkind::heap, shape::ascending>();
    register_prvalue_row<strkind::heap, shape::descending>();

    register_refexp_row<shape::random>();
    register_refexp_row<shape::ascending>();
    register_refexp_row<shape::descending>();

    register_bigprv_rows<shape::random>();
    register_bigprv_rows<shape::ascending>();
    register_bigprv_rows<shape::descending>();

    register_join_transform_row<shape::random>();
    register_join_transform_row<shape::ascending>();
    register_join_transform_row<shape::descending>();

    register_deque_string_row<shape::random>();
    register_deque_string_row<shape::descending>();

    register_join_row<shape::random>();
    register_join_row<shape::ascending>();
    register_join_row<shape::descending>();

    register_zip_row<shape::random>();
    register_zip_row<shape::ascending>();
    register_zip_row<shape::descending>();

    register_value_size<8>();
    register_value_size<128>();
    register_value_size<256>();
    register_value_size<64>();

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
