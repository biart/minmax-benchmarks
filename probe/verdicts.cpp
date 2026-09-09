// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Which rows of bench-summary actually change branch?
//
// bench-summary compares whole binaries, and changing `_Prefer_iterator_copies` changes which
// template bodies get instantiated across the translation unit, which moves the code layout of
// every other loop. Rows whose branch does not change still shift by tens of percent as a result
// (measured: `deque<double>` moves -20% and +45% in the same comparison, on identical source).
//
// So the benchmark is only evidence for the rows where the branch genuinely differs. This probe
// evaluates both concepts over the same range types and prints which those are, so the claim is
// checked by the compiler rather than by hand.

#include <cstddef>
#include <cstdio>
#include <deque>
#include <iterator>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

#include "synthetic.hpp"

using namespace std;

template <class It>
concept legacy_prefers_iterators = sizeof(It) <= 2 * sizeof(iter_value_t<It>)
                                && (is_trivially_copyable_v<It> || !is_trivially_copyable_v<iter_value_t<It>>);

template <class It>
concept step1_prefers_iterators =
    sizeof(It) <= 2 * sizeof(iter_value_t<It>)
    && (is_reference_v<iter_reference_t<It>> || !same_as<remove_cvref_t<iter_reference_t<It>>, iter_value_t<It>>
        || sizeof(iter_value_t<It>) > 16);

// `ranges::min` inspects the *unwrapped* iterator, so match that.
template <class Rng>
using unwrapped = decltype(ranges::_Ubegin(declval<Rng&>()));

template <class Rng>
void report(const char* const name) {
    using It  = unwrapped<Rng>;
    using Ref = iter_reference_t<It>;
    using V   = iter_value_t<It>;

    const bool legacy = legacy_prefers_iterators<It>;
    const bool step1  = step1_prefers_iterators<It>;

    printf("%-26s %4zu %4zu  %-8s  %-6s %-6s  %s\n", name, sizeof(It), sizeof(V),
        is_reference_v<Ref> ? "ref" : (same_as<remove_cvref_t<Ref>, V> ? "prvalue" : "proxy"),
        legacy ? "iter" : "value", step1 ? "iter" : "value", legacy == step1 ? "" : "<-- CHANGES");
}

template <size_t Bytes>
using big = mmb::sized_value<Bytes>;

int main() {
    printf("%-26s %4s %4s  %-8s  %-6s %-6s\n", "range", "|It|", "|V|", "Ref", "legacy", "step1");
    printf("%.79s\n", "-------------------------------------------------------------------------------");

    using Chunks = vector<vector<double>>;

    report<vector<double>&>("vector<double>");
    report<vector<string>&>("vector<string>");
    report<deque<double>&>("deque<double>");
    report<decltype(declval<Chunks&>() | views::join)>("join");
    report<decltype(views::zip(declval<vector<string>&>(), declval<vector<int>&>()))>("zip");
    report<decltype(declval<vector<double>&>() | views::transform([](double x) { return x * 2.0; }))>(
        "transform -> double");

    report<decltype(declval<vector<big<16>>&>() | views::transform([](const big<16>& b) { return b; }))>(
        "transform -> prvalue 16 B");
    report<decltype(declval<vector<big<24>>&>() | views::transform([](const big<24>& b) { return b; }))>(
        "transform -> prvalue 24 B");
    report<decltype(declval<vector<big<32>>&>() | views::transform([](const big<32>& b) { return b; }))>(
        "transform -> prvalue 32 B");
    report<decltype(declval<vector<big<64>>&>() | views::transform([](const big<64>& b) { return b; }))>(
        "transform -> prvalue 64 B");
    report<decltype(declval<vector<big<256>>&>() | views::transform([](const big<256>& b) { return b; }))>(
        "transform -> prvalue 256 B");

    report<ranges::subrange<mmb::fat_iterator<double, 16, mmb::fatness::live>>>("fat_iterator 16 B");
    report<ranges::subrange<mmb::fat_iterator<double, 32, mmb::fatness::live>>>("fat_iterator 32 B");
    report<ranges::subrange<mmb::fat_iterator<double, 64, mmb::fatness::live>>>("fat_iterator 64 B");

    report<decltype(views::iota(size_t{0}, size_t{1}))>("iota");
    report<decltype(declval<Chunks&>() | views::transform([](const vector<double>& r) -> const double& { return r[0]; }))>(
        "transform -> const double&");
    return 0;
}
