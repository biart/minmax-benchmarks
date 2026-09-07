// One noinline function per (track, iterator shape, element size) so that /FAcs produces a
// readable listing. extern "C" keeps the names unmangled.

#include <cstddef>
#include <cstdint>
#include <ranges>

#include "minmax_tracks.hpp"
#include "synthetic.hpp"

using namespace mmb;
using std::ptrdiff_t;
using std::uint64_t;

using V1  = sized_value<1>;
using V8  = sized_value<8>;
using V64 = sized_value<64>;

#define MMB_CASE(NAME, VAL, TRACK, ...)                                              \
    extern "C" __declspec(noinline) uint64_t NAME(VAL* const p, const ptrdiff_t n) {  \
        using It = __VA_ARGS__;                                                       \
        auto r   = std::ranges::subrange(It::at(p, 0), It::at(p, n));                 \
        return static_cast<uint64_t>(mmb::min_of<TRACK>(r).key);                      \
    }

// Raw pointer: what an unwrapped vector iterator actually is.
extern "C" __declspec(noinline) uint64_t min_iter_ptr(V8* const p, const ptrdiff_t n) {
    auto r = std::ranges::subrange(p, p + n);
    return static_cast<uint64_t>(mmb::min_of<track::iter>(r).key);
}

extern "C" __declspec(noinline) uint64_t min_value_ptr(V8* const p, const ptrdiff_t n) {
    auto r = std::ranges::subrange(p, p + n);
    return static_cast<uint64_t>(mmb::min_of<track::value>(r).key);
}

// iterator-size sweep, dead padding, 8-byte element
MMB_CASE(min_iter_dead8, V8, track::iter, fat_iterator<V8, 8, fatness::dead>)
MMB_CASE(min_iter_dead16, V8, track::iter, fat_iterator<V8, 16, fatness::dead>)
MMB_CASE(min_iter_dead32, V8, track::iter, fat_iterator<V8, 32, fatness::dead>)
MMB_CASE(min_iter_dead64, V8, track::iter, fat_iterator<V8, 64, fatness::dead>)

// iterator-size sweep, live base+offset, 8-byte element
MMB_CASE(min_iter_live16, V8, track::iter, fat_iterator<V8, 16, fatness::live>)
MMB_CASE(min_iter_live32, V8, track::iter, fat_iterator<V8, 32, fatness::live>)
MMB_CASE(min_iter_live64, V8, track::iter, fat_iterator<V8, 64, fatness::live>)

// value tracks over the same 8-byte element
MMB_CASE(min_value_dead8, V8, track::value, fat_iterator<V8, 8, fatness::dead>)
MMB_CASE(min_value1x_dead8, V8, track::value_1x, fat_iterator<V8, 8, fatness::dead>)
MMB_CASE(min_iterhoist_dead8, V8, track::iter_hoist, fat_iterator<V8, 8, fatness::dead>)
MMB_CASE(min_value_live16, V8, track::value, fat_iterator<V8, 16, fatness::live>)

// element-size sweep: why does the value track get faster as elements grow?
MMB_CASE(min_value_val1, V1, track::value, fat_iterator<V1, 8, fatness::dead>)
MMB_CASE(min_value_val64, V64, track::value, fat_iterator<V64, 8, fatness::dead>)
MMB_CASE(min_iter_val1, V1, track::iter, fat_iterator<V1, 8, fatness::dead>)
MMB_CASE(min_iter_val64, V64, track::iter, fat_iterator<V64, 8, fatness::dead>)

// Return the WHOLE value, not just the key, so the payload copy cannot be dead-code eliminated
// the way it can when only `.key` is returned.
#define MMB_FULL(NAME, TRACK)                                                                  extern "C" __declspec(noinline) void NAME(V64* const p, const ptrdiff_t n, V64* const out) {         using It = fat_iterator<V64, 8, fatness::dead>;                                            auto r   = std::ranges::subrange(It::at(p, 0), It::at(p, n));                              *out     = mmb::min_of<TRACK>(r);                                                      }

MMB_FULL(min_full_iter, track::iter)
MMB_FULL(min_full_value, track::value)
MMB_FULL(min_full_value1x, track::value_1x)
MMB_FULL(min_full_iterhoist, track::iter_hoist)

// --------------------------------------------------------------------------
// Timing harness, so the numbers can be tied to the exact loops listed above.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

template <class V>
std::vector<V> make(const std::size_t n) {
    std::vector<V> a(n);
    std::mt19937_64 gen(84710);
    std::uniform_int_distribution<std::uint64_t> dis(0, V::key_max);
    for (auto& e : a) {
        e.key = static_cast<typename V::key_type>(dis(gen));
    }
    return a;
}

template <class V, class Fn>
void time_it(const char* const name, Fn fn, std::vector<V>& a) {
    const auto n    = static_cast<ptrdiff_t>(a.size());
    double best_ns  = 1e300;
    uint64_t sink   = 0;
    for (int rep = 0; rep < 200; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        sink += fn(a.data(), n);
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best_ns = ns < best_ns ? ns : best_ns;
    }
    std::printf("%-20s %7.3f ns/elem   (sink %llu)\n", name, best_ns / static_cast<double>(n),
        static_cast<unsigned long long>(sink));
}

} // unnamed namespace

int main() {
    auto a1  = make<V1>(8021);
    auto a8  = make<V8>(8021);
    auto a64 = make<V64>(8021);

    time_it("iter_ptr", min_iter_ptr, a8);
    time_it("iter_dead8", min_iter_dead8, a8);
    time_it("iter_dead16", min_iter_dead16, a8);
    time_it("iter_dead32", min_iter_dead32, a8);
    time_it("iter_dead64", min_iter_dead64, a8);
    time_it("iter_live16", min_iter_live16, a8);
    time_it("iter_live32", min_iter_live32, a8);
    time_it("iter_live64", min_iter_live64, a8);
    std::puts("");
    time_it("value_ptr", min_value_ptr, a8);
    time_it("value_dead8", min_value_dead8, a8);
    time_it("value1x_dead8", min_value1x_dead8, a8);
    time_it("iterhoist_dead8", min_iterhoist_dead8, a8);
    time_it("value_live16", min_value_live16, a8);
    std::puts("");
    time_it("iter_val1", min_iter_val1, a1);
    time_it("value_val1", min_value_val1, a1);
    time_it("iter_val64", min_iter_val64, a64);
    time_it("value_val64", min_value_val64, a64);
}
