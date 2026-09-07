// Why is zip's iter_value_t tuple<T, U> and not tuple<T&, U&>?
#include <cstdio>
#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

using namespace std;

int main() {
    vector<string> a{"aaa", "bbb"};
    vector<int> b{1, 2};
    auto z   = views::zip(a, b);
    using It = ranges::iterator_t<decltype(z)>;

    printf("iter_reference_t        : %s\n", typeid(iter_reference_t<It>).name());
    printf("iter_value_t            : %s\n", typeid(iter_value_t<It>).name());
    printf("iter_rvalue_reference_t : %s\n", typeid(iter_rvalue_reference_t<It>).name());
    printf("reference is an lvalue ref? %d\n", int(is_reference_v<iter_reference_t<It>>));

    // The decisive point: a tuple of lvalue references assigns THROUGH the references.
    tuple<string&, int&> proxy{a[0], b[0]};
    tuple<string, int> owned{"zzz", 99};
    proxy = owned; // looks like "copy a value"...
    printf("after `proxy = owned`, the CONTAINER holds: a[0]=%s b[0]=%d\n", a[0].c_str(), b[0]);

    // And iter_move yields rvalue refs INTO the container, so moving from them would steal.
    auto rv = ranges::iter_move(z.begin() + 1);
    printf("iter_move gives rvalue refs into the source; a[1] before = '%s'\n", a[1].c_str());
    string stolen = std::move(get<0>(rv));
    printf("after moving from it,  a[1] = '%s' (emptied)\n", a[1].c_str());
}
