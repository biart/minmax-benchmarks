// Which branch does each benchmarks/src/minmax_element.cpp case actually take?
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ranges>
#include <vector>

using namespace std;

template <class T>
void report(const char* name) {
    using It  = T*; // unwrapped vector<T>::iterator
    // Gate for ranges::min / ranges::max (value-returning)
    constexpr bool vec_val = _VECTORIZED_MINMAX && _Is_min_max_value_optimization_safe<const T*, ranges::less>;
    // Gate for ranges::min_element / max_element (position-returning)
    constexpr bool vec_pos = _VECTORIZED_MINMAX_ELEMENT && _Is_min_max_optimization_safe<It, ranges::less>;
    constexpr bool prefer_it = ranges::_Prefer_iterator_copies<It>;
    printf("%-9s sizeof=%zu  sizeof(It)=%zu | min/max_element vectorized=%d | ranges::min/max vectorized=%d"
           " | _Prefer_iterator_copies=%d -> non-vector branch would be: %s\n",
        name, sizeof(T), sizeof(It), int(vec_pos), int(vec_val), int(prefer_it),
        prefer_it ? "COPY ITERATORS" : "copy elements");
}

int main() {
    printf("_M_FP_FAST defined: %d\n",
#ifdef _M_FP_FAST
        1
#else
        0
#endif
    );
    printf("_VECTORIZED_MINMAX=%d _VECTORIZED_MINMAX_ELEMENT=%d _VECTORIZED_MINMAX_ELEMENT_64BIT_INT=%d\n\n",
        _VECTORIZED_MINMAX, _VECTORIZED_MINMAX_ELEMENT, _VECTORIZED_MINMAX_ELEMENT_64BIT_INT);
    report<uint8_t>("uint8_t");
    report<uint16_t>("uint16_t");
    report<uint32_t>("uint32_t");
    report<uint64_t>("uint64_t");
    report<int8_t>("int8_t");
    report<int16_t>("int16_t");
    report<int32_t>("int32_t");
    report<int64_t>("int64_t");
    report<float>("float");
    report<double>("double");
}
