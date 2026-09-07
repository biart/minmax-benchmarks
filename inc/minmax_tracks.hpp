// Copyright (c) 2026. Benchmark support for microsoft/STL#6404.
//
// Faithful copies of the ranges::min / max / minmax loops, with the vectorized fast paths and the
// _Prefer_iterator_copies fork removed and replaced by an explicit compile-time `track` control.
// Bodies are transcribed as literally as possible so that codegen matches the shipping algorithms;
// only names and the internal _STD/_RANGES/_STL_ASSERT spellings differ.
//
// Sources (microsoft/STL @ eae5df4, gcc master):
//   track::iter     ranges::_Min_element_unchecked            stl/inc/xutility:7673
//                   ranges::_Max_element_unchecked            stl/inc/xutility:7454
//                   ranges::_Minmax_fn::_Minmax_fwd_unchecked stl/inc/algorithm:11507
//   track::value    the `else` branch of ranges::_Min_fn      stl/inc/xutility:7810
//                                        ranges::_Max_fn      stl/inc/xutility:7592
//                                        ranges::_Minmax_fn   stl/inc/algorithm:11462
//   track::value_1x libstdc++ __min_fn / __max_fn / __minmax_fn
//                   libstdc++-v3/include/bits/ranges_algo.h:4148, 4191, 4295
//   track::iter_hoist  NOT a candidate implementation: the iterator branch as it would compile
//                   if MSVC hoisted the loop-invariant `*found` into a register. Isolates how
//                   much of the iterator branch's cost is the missed optimization.
//   track::stl      whatever the installed std::ranges::* does today (control)

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>

namespace mmb {

enum class track {
    iter, // cache iterators: two dereferences per element, one iterator copy per improvement
    value, // cache values, MSVC: one dereference per element plus one more per improvement
    value_1x, // cache values, libstdc++: exactly one dereference per element
    iter_hoist, // the iterator branch as a compiler WOULD compile it if it hoisted `*found`
    ref_1x, // cache the cheapest handle to the element AND dereference exactly once per element
    stl, // control: the installed std::ranges::min/max/minmax
};

[[nodiscard]] constexpr const char* name_of(const track tr) noexcept {
    switch (tr) {
    case track::iter:
        return "iter";
    case track::value:
        return "value";
    case track::value_1x:
        return "value1x";
    case track::iter_hoist:
        return "iterhoist";
    case track::ref_1x:
        return "ref1x";
    case track::stl:
        return "stl";
    }
    return "?";
}

// The STL passes predicates through _Pass_fn; for the stateless ranges::less / identity used here
// that is a no-op, so the copies below take them by value.

// ---------------------------------------------------------------- track::iter

template <std::forward_iterator It, std::sentinel_for<It> Se, class Pr, class Pj>
[[nodiscard]] constexpr It min_element_unchecked(It first, const Se last, Pr pred, Pj proj) {
    auto found = first;
    if (first == last) {
        return found;
    }

    while (++first != last) {
        if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, *found))) {
            found = first;
        }
    }

    return found;
}

template <std::forward_iterator It, std::sentinel_for<It> Se, class Pr, class Pj>
[[nodiscard]] constexpr It max_element_unchecked(It first, const Se last, Pr pred, Pj proj) {
    auto found = first;
    if (first == last) {
        return found;
    }

    while (++first != last) {
        if (std::invoke(pred, std::invoke(proj, *found), std::invoke(proj, *first))) {
            found = first;
        }
    }

    return found;
}

template <std::forward_iterator It, std::sentinel_for<It> Se, class Pr, class Pj>
[[nodiscard]] constexpr std::ranges::minmax_result<std::iter_value_t<It>> minmax_fwd_unchecked(
    It first, const Se last, Pr pred, Pj proj) {
    using V = std::iter_value_t<It>;

    auto found_min = first;
    if (++first == last) {
        // This initialization is correct, similar to the N4950 [dcl.init.aggr]/6 example
        std::ranges::minmax_result<V> result = {static_cast<V>(*found_min), result.min};
        return result;
    }

    auto found_max = first;
    if (std::invoke(pred, std::invoke(proj, *found_max), std::invoke(proj, *found_min))) {
        std::ranges::swap(found_min, found_max);
    }

    while (++first != last) { // process one or two elements
        It prev = first;
        if (++first == last) { // process last element
            if (std::invoke(pred, std::invoke(proj, *prev), std::invoke(proj, *found_min))) {
                found_min = prev;
            } else if (!std::invoke(pred, std::invoke(proj, *prev), std::invoke(proj, *found_max))) {
                found_max = prev;
            }

            break;
        }

        // process two elements
        if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, *prev))) {
            // test first for new smallest
            if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, *found_min))) {
                found_min = first;
            }

            if (!std::invoke(pred, std::invoke(proj, *prev), std::invoke(proj, *found_max))) {
                found_max = prev;
            }
        } else { // test prev for new smallest
            if (std::invoke(pred, std::invoke(proj, *prev), std::invoke(proj, *found_min))) {
                found_min = prev;
            }

            if (!std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, *found_max))) {
                found_max = first;
            }
        }
    }

    return {static_cast<V>(*found_min), static_cast<V>(*found_max)};
}

// ------------------------------------------------------------------ dispatch

template <track Tr, std::ranges::forward_range Rng, class Pr = std::ranges::less, class Pj = std::identity>
[[nodiscard]] std::ranges::range_value_t<Rng> min_of(Rng& r, Pr pred = {}, Pj proj = {}) {
    using V = std::ranges::range_value_t<Rng>;

    if constexpr (Tr == track::stl) {
        return std::ranges::min(r, pred, proj);
    } else {
        // _Ubegin/_Uend, not begin/end: the heuristic inspects the *unwrapped* iterator, and for
        // e.g. deque that is a different (smaller) type than iterator_t.
        auto first = std::ranges::_Ubegin(r);
        auto last  = std::ranges::_Uend(r);

        if constexpr (Tr == track::iter) {
            return static_cast<V>(*mmb::min_element_unchecked(std::move(first), std::move(last), pred, proj));
        } else if constexpr (Tr == track::iter_hoist) {
            // Not a candidate implementation: a model of what the *iterator* branch would compile
            // to if MSVC hoisted the loop-invariant `*found` into a register instead of reloading
            // it through a cmov'd pointer every iteration (DevCom-11015032, DevCom-11139307).
            // The iterator is still maintained and still dereferenced once at the end, so this
            // performs N + I + 1 dereferences -- the same count as track::value.
            auto found = first;
            V best(*first);
            while (++first != last) {
                if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, best))) {
                    found = first;
                    best  = *first;
                }
            }

            return static_cast<V>(*found);
        } else if constexpr (Tr == track::ref_1x) {
            using Ref = std::iter_reference_t<decltype(first)>;
            if constexpr (std::is_reference_v<Ref>) {
                // A reference -- lvalue OR rvalue -- designates a persistent object we may point
                // at. Gating on is_lvalue_reference_v instead would send views::as_rvalue to the
                // value branch, where the forwarded store would MOVE OUT of the source range.
                // Bind once, then keep a pointer: exactly N dereferences of the range, and the
                // cached handle is a pointer, so a large V is never copied. Re-reading through
                // the pointer is a plain load, not the range's dereference machinery (deque's
                // block-map walk, join's two levels).
                auto&& head = *first;
                auto found  = std::addressof(head);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, tmp), std::invoke(proj, *found))) {
                        found = std::addressof(tmp);
                    }
                }

                // Cast back to Ref so the value category survives: as_rvalue_view yields V&&
                // and should move here, an ordinary container yields V& and should copy.
                return static_cast<V>(static_cast<Ref>(*found));
            } else if constexpr (std::same_as<std::remove_cvref_t<Ref>, V>) {
                // The reference IS the value, so caching it is exactly the libstdc++ loop.
                V result(*first);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, tmp), std::invoke(proj, result))) {
                        result = std::forward<decltype(tmp)>(tmp);
                    }
                }

                return result;
            } else {
                // A proxy (zip, vector<bool>): cache the proxy itself -- cheap by construction,
                // it is a handle -- but RE-CONSTRUCT rather than assign, since assigning a tuple
                // of lvalue references writes through into the range, which a non-modifying
                // algorithm must not do. Binding tmp first keeps this at N dereferences too.
                std::optional<Ref> found(*first);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, tmp), std::invoke(proj, *found))) {
                        found.emplace(std::forward<decltype(tmp)>(tmp));
                    }
                }

                return static_cast<V>(*found);
            }
        } else if constexpr (Tr == track::value) {
            V found(*first);
            while (++first != last) {
                if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, found))) {
                    found = *first;
                }
            }

            return found;
        } else {
            V result(*first);
            while (++first != last) {
                auto&& tmp = *first;
                if (std::invoke(pred, std::invoke(proj, tmp), std::invoke(proj, result))) {
                    result = std::forward<decltype(tmp)>(tmp);
                }
            }

            return result;
        }
    }
}

template <track Tr, std::ranges::forward_range Rng, class Pr = std::ranges::less, class Pj = std::identity>
[[nodiscard]] std::ranges::range_value_t<Rng> max_of(Rng& r, Pr pred = {}, Pj proj = {}) {
    using V = std::ranges::range_value_t<Rng>;

    if constexpr (Tr == track::stl) {
        return std::ranges::max(r, pred, proj);
    } else {
        auto first = std::ranges::_Ubegin(r);
        auto last  = std::ranges::_Uend(r);

        if constexpr (Tr == track::iter) {
            return static_cast<V>(*mmb::max_element_unchecked(std::move(first), std::move(last), pred, proj));
        } else if constexpr (Tr == track::iter_hoist) {
            auto found = first;
            V best(*first);
            while (++first != last) {
                if (std::invoke(pred, std::invoke(proj, best), std::invoke(proj, *first))) {
                    found = first;
                    best  = *first;
                }
            }

            return static_cast<V>(*found);
        } else if constexpr (Tr == track::ref_1x) {
            using Ref = std::iter_reference_t<decltype(first)>;
            if constexpr (std::is_reference_v<Ref>) {
                auto&& head = *first;
                auto found  = std::addressof(head);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, *found), std::invoke(proj, tmp))) {
                        found = std::addressof(tmp);
                    }
                }

                // Cast back to Ref so the value category survives: as_rvalue_view yields V&&
                // and should move here, an ordinary container yields V& and should copy.
                return static_cast<V>(static_cast<Ref>(*found));
            } else if constexpr (std::same_as<std::remove_cvref_t<Ref>, V>) {
                V result(*first);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, result), std::invoke(proj, tmp))) {
                        result = std::forward<decltype(tmp)>(tmp);
                    }
                }

                return result;
            } else {
                std::optional<Ref> found(*first);
                while (++first != last) {
                    auto&& tmp = *first;
                    if (std::invoke(pred, std::invoke(proj, *found), std::invoke(proj, tmp))) {
                        found.emplace(std::forward<decltype(tmp)>(tmp));
                    }
                }

                return static_cast<V>(*found);
            }
        } else if constexpr (Tr == track::value) {
            V found(*first);
            while (++first != last) {
                if (std::invoke(pred, std::invoke(proj, found), std::invoke(proj, *first))) {
                    found = *first;
                }
            }

            return found;
        } else {
            V result(*first);
            while (++first != last) {
                auto&& tmp = *first;
                if (std::invoke(pred, std::invoke(proj, result), std::invoke(proj, tmp))) {
                    result = std::forward<decltype(tmp)>(tmp);
                }
            }

            return result;
        }
    }
}

template <track Tr, std::ranges::forward_range Rng, class Pr = std::ranges::less, class Pj = std::identity>
[[nodiscard]] std::ranges::minmax_result<std::ranges::range_value_t<Rng>> minmax_of(
    Rng& r, Pr pred = {}, Pj proj = {}) {
    using V = std::ranges::range_value_t<Rng>;

    if constexpr (Tr == track::stl) {
        return std::ranges::minmax(r, pred, proj);
    } else {
        auto first = std::ranges::_Ubegin(r);
        auto last  = std::ranges::_Uend(r);

        if constexpr (Tr == track::iter) {
            return mmb::minmax_fwd_unchecked(std::move(first), std::move(last), pred, proj);
        } else if constexpr (Tr == track::iter_hoist) {
            // minmax_fwd_unchecked with *found_min, *found_max and the two elements of each pair
            // hoisted into locals, which is what CSE plus the missing hoist would produce.
            auto found_min = first;
            V best_min(*first);
            if (++first == last) {
                std::ranges::minmax_result<V> only = {best_min, only.min};
                return only;
            }

            auto found_max = first;
            V best_max(*first);
            if (std::invoke(pred, std::invoke(proj, best_max), std::invoke(proj, best_min))) {
                std::ranges::swap(found_min, found_max);
                std::ranges::swap(best_min, best_max);
            }

            while (++first != last) {
                auto prev = first;
                V a(*prev);
                if (++first == last) {
                    if (std::invoke(pred, std::invoke(proj, a), std::invoke(proj, best_min))) {
                        found_min = prev;
                        best_min  = std::move(a);
                    } else if (!std::invoke(pred, std::invoke(proj, a), std::invoke(proj, best_max))) {
                        found_max = prev;
                        best_max  = std::move(a);
                    }

                    break;
                }

                V b(*first);
                if (std::invoke(pred, std::invoke(proj, b), std::invoke(proj, a))) {
                    if (std::invoke(pred, std::invoke(proj, b), std::invoke(proj, best_min))) {
                        found_min = first;
                        best_min  = b;
                    }

                    if (!std::invoke(pred, std::invoke(proj, a), std::invoke(proj, best_max))) {
                        found_max = prev;
                        best_max  = std::move(a);
                    }
                } else {
                    if (std::invoke(pred, std::invoke(proj, a), std::invoke(proj, best_min))) {
                        found_min = prev;
                        best_min  = std::move(a);
                    }

                    if (!std::invoke(pred, std::invoke(proj, b), std::invoke(proj, best_max))) {
                        found_max = first;
                        best_max  = std::move(b);
                    }
                }
            }

            return {static_cast<V>(*found_min), static_cast<V>(*found_max)};
        } else if constexpr (Tr == track::value) {
            // This initialization is correct, similar to the N4950 [dcl.init.aggr]/6 example
            std::ranges::minmax_result<V> found = {static_cast<V>(*first), found.min};
            if (first == last) {
                return found;
            }

            while (++first != last) { // process one or two elements
                V prev(*first);
                if (++first == last) { // process last element
                    if (std::invoke(pred, std::invoke(proj, prev), std::invoke(proj, found.min))) {
                        found.min = std::move(prev);
                    } else if (!std::invoke(pred, std::invoke(proj, prev), std::invoke(proj, found.max))) {
                        found.max = std::move(prev);
                    }

                    break;
                }

                // process next two elements
                if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, prev))) {
                    // test first for new smallest
                    if (std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, found.min))) {
                        found.min = *first;
                    }

                    if (!std::invoke(pred, std::invoke(proj, prev), std::invoke(proj, found.max))) {
                        found.max = std::move(prev);
                    }
                } else { // test prev for new smallest
                    if (std::invoke(pred, std::invoke(proj, prev), std::invoke(proj, found.min))) {
                        found.min = std::move(prev);
                    }

                    if (!std::invoke(pred, std::invoke(proj, *first), std::invoke(proj, found.max))) {
                        found.max = *first;
                    }
                }
            }

            return found;
        } else {
            const auto comp_proj = [&](auto&& x, auto&& y) -> bool {
                return std::invoke(pred, std::invoke(proj, std::forward<decltype(x)>(x)),
                    std::invoke(proj, std::forward<decltype(y)>(y)));
            };

            std::ranges::minmax_result<V> result = {*first, result.min};
            if (++first == last) {
                return result;
            } else {
                // At this point result.min == result.max, so a single comparison with the next
                // element suffices.
                auto&& val = *first;
                if (comp_proj(val, result.min)) {
                    result.min = std::forward<decltype(val)>(val);
                } else {
                    result.max = std::forward<decltype(val)>(val);
                }
            }

            while (++first != last) {
                // Now process two elements at a time so that we perform at most 1 + 3*(N-2)/2
                // comparisons in total.
                V val1 = *first;
                if (++first == last) {
                    if (comp_proj(val1, result.min)) {
                        result.min = std::move(val1);
                    } else if (!comp_proj(val1, result.max)) {
                        result.max = std::move(val1);
                    }

                    break;
                }

                auto&& val2 = *first;
                if (!comp_proj(val2, val1)) {
                    if (comp_proj(val1, result.min)) {
                        result.min = std::move(val1);
                    }

                    if (!comp_proj(val2, result.max)) {
                        result.max = std::forward<decltype(val2)>(val2);
                    }
                } else {
                    if (comp_proj(val2, result.min)) {
                        result.min = std::forward<decltype(val2)>(val2);
                    }

                    if (!comp_proj(val1, result.max)) {
                        result.max = std::move(val1);
                    }
                }
            }

            return result;
        }
    }
}

} // namespace mmb
