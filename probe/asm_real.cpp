// The synthetic fat_iterator has padding nothing reads, so the optimizer deletes it. Real fat
// iterators (deque, join) consult every field on each ++ and *, so they cannot be scalarized away.
// This probe disassembles and times those, to see what iterator size costs when it is real.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

#include "minmax_tracks.hpp"

using namespace mmb;

using DVec    = std::vector<double>;
using DDeque  = std::deque<double>;
using DChunks = std::vector<std::vector<double>>;

extern "C" __declspec(noinline) double min_iter_vector(DVec& a) {
    return mmb::min_of<track::iter>(a);
}
extern "C" __declspec(noinline) double min_value_vector(DVec& a) {
    return mmb::min_of<track::value>(a);
}

extern "C" __declspec(noinline) double min_iter_deque(DDeque& d) {
    return mmb::min_of<track::iter>(d);
}
extern "C" __declspec(noinline) double min_value_deque(DDeque& d) {
    return mmb::min_of<track::value>(d);
}

extern "C" __declspec(noinline) double min_iter_join(DChunks& c) {
    auto v = c | std::views::join;
    return mmb::min_of<track::iter>(v);
}
extern "C" __declspec(noinline) double min_value_join(DChunks& c) {
    auto v = c | std::views::join;
    return mmb::min_of<track::value>(v);
}

extern "C" __declspec(noinline) long long min_iter_iota(const long long n) {
    auto v = std::views::iota(0LL, n);
    return mmb::min_of<track::iter>(v);
}
extern "C" __declspec(noinline) long long min_value_iota(const long long n) {
    auto v = std::views::iota(0LL, n);
    return mmb::min_of<track::value>(v);
}

inline constexpr auto scale = [](const double x) { return x * 1.0000001; };

extern "C" __declspec(noinline) double min_iter_transform(DVec& a) {
    auto v = a | std::views::transform(scale);
    return mmb::min_of<track::iter>(v);
}
extern "C" __declspec(noinline) double min_value_transform(DVec& a) {
    auto v = a | std::views::transform(scale);
    return mmb::min_of<track::value>(v);
}
extern "C" __declspec(noinline) double min_value1x_transform(DVec& a) {
    auto v = a | std::views::transform(scale);
    return mmb::min_of<track::value_1x>(v);
}

// What MSVC would produce from the ITERATOR source if it hoisted `*found` into a register:
// the iterator is still maintained, but the compared value lives in a local. Operation counts
// become N + I dereferences -- identical to the MSVC value branch.
template <class Rng>
double min_iter_hoisted(Rng& r) {
    auto first = std::ranges::_Ubegin(r);
    auto last  = std::ranges::_Uend(r);
    auto found = first;
    double best = *first;
    while (++first != last) {
        if (*first < best) {
            found = first;
            best  = *first;
        }
    }
    return *found;
}

extern "C" __declspec(noinline) double min_hoist_vector(DVec& a) {
    return min_iter_hoisted(a);
}
extern "C" __declspec(noinline) double min_hoist_deque(DDeque& d) {
    return min_iter_hoisted(d);
}
extern "C" __declspec(noinline) double min_hoist_transform(DVec& a) {
    auto v = a | std::views::transform(scale);
    return min_iter_hoisted(v);
}

namespace {

constexpr std::size_t n = 8021;

template <class Fn, class Arg>
void time_it(const char* const name, Fn fn, Arg& arg, const std::size_t count) {
    double best   = 1e300;
    double sink   = 0;
    for (int rep = 0; rep < 200; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        sink += static_cast<double>(fn(arg));
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = ns < best ? ns : best;
    }
    std::printf("%-24s %8.3f ns/elem  (sink %g)\n", name, best / static_cast<double>(count), sink);
}

template <class Rng>
void show_size(const char* const name, Rng&& r) {
    using It = decltype(std::ranges::_Ubegin(r));
    std::printf("%-24s unwrapped iterator = %zu bytes, reference is %s\n", name, sizeof(It),
        std::is_reference_v<std::ranges::range_reference_t<std::remove_reference_t<Rng>>> ? "a reference" : "a prvalue");
}

} // unnamed namespace

int main() {
    DVec a(n);
    std::mt19937_64 gen(84710);
    std::normal_distribution<double> dis(0, 100000.0);
    for (auto& e : a) {
        e = dis(gen);
    }

    DDeque d(a.begin(), a.end());

    DChunks chunks;
    for (std::size_t i = 0; i < n; i += 64) {
        chunks.emplace_back(a.begin() + static_cast<std::ptrdiff_t>(i),
            a.begin() + static_cast<std::ptrdiff_t>(std::min(i + 64, n)));
    }

    long long iota_n = static_cast<long long>(n);

    auto joined    = chunks | std::views::join;
    auto iota_view = std::views::iota(0LL, iota_n);
    auto tview     = a | std::views::transform(scale);
    show_size("vector<double>", a);
    show_size("deque<double>", d);
    show_size("join", joined);
    show_size("iota", iota_view);
    show_size("transform", tview);
    std::puts("");

    time_it("vector/iter", min_iter_vector, a, n);
    time_it("vector/iter+hoist", min_hoist_vector, a, n);
    time_it("vector/value", min_value_vector, a, n);
    time_it("deque/iter", min_iter_deque, d, n);
    time_it("deque/iter+hoist", min_hoist_deque, d, n);
    time_it("deque/value", min_value_deque, d, n);
    time_it("join/iter", min_iter_join, chunks, n);
    time_it("join/value", min_value_join, chunks, n);
    time_it("iota/iter", min_iter_iota, iota_n, n);
    time_it("iota/value", min_value_iota, iota_n, n);
    time_it("transform/iter", min_iter_transform, a, n);
    time_it("transform/iter+hoist", min_hoist_transform, a, n);
    time_it("transform/value", min_value_transform, a, n);
    time_it("transform/value1x", min_value1x_transform, a, n);
}
