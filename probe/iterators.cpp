// What do real ranges look like to _Prefer_iterator_copies?
// Prints, for each candidate range: unwrapped iterator size, value size, whether the
// reference is a real reference, and how the current heuristic votes.

#include <cstdio>
#include <deque>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <forward_list>
#include <list>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace std;

template <class Rng>
void report(const char* name, Rng&& r) {
    using It  = decltype(ranges::_Ubegin(r)); // what the STL actually inspects
    using Raw = ranges::iterator_t<remove_reference_t<Rng>>;
    using Val = ranges::range_value_t<remove_reference_t<Rng>>;
    using Ref = ranges::range_reference_t<remove_reference_t<Rng>>;

    // Current heuristic vs the predicate the measurements suggest.
    constexpr bool current  = ranges::_Prefer_iterator_copies<It>;
    constexpr bool proposed = !is_trivially_copyable_v<Val> && is_reference_v<Ref>;
    printf("%-36s It=%-2zu Val=%-3zu ref?=%d triv_val=%d | current=%-3s proposed=%-3s%s\n", name, sizeof(It),
        sizeof(Val), int(is_reference_v<Ref>), int(is_trivially_copyable_v<Val>), current ? "IT" : "val",
        proposed ? "IT" : "val", current == proposed ? "" : "   <-- differs");
    (void) sizeof(Raw);
}

int main() {
    vector<double> vd(4);
    vector<char> vc(4);
    vector<string> vs(4);
    deque<double> dd(4);
    list<double> ld(4);
    forward_list<double> fd(4);
    vector<vector<double>> vvd(2, vector<double>(2));

    auto by_val  = [](double x) { return x * 2.0; };   // prvalue
    auto by_ref  = [](double& x) -> double& { return x; }; // real reference
    auto str_len = [](const string& s) { return s.size(); };

    puts("== containers ==");
    report("vector<char>", vc);
    report("vector<double>", vd);
    report("vector<string>", vs);
    report("deque<double>", dd);
    report("list<double>", ld);
    report("forward_list<double>", fd);

    puts("\n== views ==");
    report("vector<double> | transform(prvalue)", vd | views::transform(by_val));
    report("vector<double> | transform(ref)", vd | views::transform(by_ref));
    report("vector<string> | transform(size)", vs | views::transform(str_len));
    report("vector<double> | filter", vd | views::filter([](double x) { return x > 0.0; }));
    report("vector<vector<double>> | join", vvd | views::join);
    report("zip(vector<double>, vector<char>)", views::zip(vd, vc));
    report("iota(0, 100)", views::iota(0, 100));
    report("vector<double> | reverse", vd | views::reverse);

    puts("");
    puts("== proxies and node-based ==");
    vector<bool> vb(4);
    set<int> si{1, 2, 3};
    map<int, double> mi{{1, 1.0}, {2, 2.0}};
    unordered_map<int, double> ui{{1, 1.0}, {2, 2.0}};
    vector<tuple<int, int>> vt(4);
    vector<pair<string, int>> vp(4);
    vector<string_view> vsv(4);
    report("vector<bool>", vb);
    report("set<int>", si);
    report("map<int,double> | values", mi | views::values);
    report("unordered_map<int,double> | values", ui | views::values);
    report("vector<tuple<int,int>>", vt);
    report("vector<pair<string,int>>", vp);
    report("vector<string_view>", vsv);
}
