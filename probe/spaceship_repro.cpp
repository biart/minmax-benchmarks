// Does `a < b` synthesized from operator<=> cost more than a direct operator< ?
#include <compare>
#include <cstddef>

struct Spaceship {
    long long k;
    friend auto operator<=>(const Spaceship&, const Spaceship&) = default;
};
struct SpaceshipMember {
    long long k;
    friend std::strong_ordering operator<=>(const SpaceshipMember& a, const SpaceshipMember& b) {
        return a.k <=> b.k;
    }
    friend bool operator==(const SpaceshipMember& a, const SpaceshipMember& b) { return a.k == b.k; }
};
struct Direct {
    long long k;
    friend bool operator<(const Direct& a, const Direct& b) { return a.k < b.k; }
};

#define LOOP(NAME, T)                                                    \
    extern "C" long long NAME(const T* const p, const std::size_t n) {    \
        T best = p[0];                                                    \
        for (std::size_t i = 1; i < n; ++i) {                             \
            if (p[i] < best) {                                            \
                best = p[i];                                              \
            }                                                             \
        }                                                                 \
        return best.k;                                                    \
    }

LOOP(min_spaceship_default, Spaceship)
LOOP(min_spaceship_member, SpaceshipMember)
LOOP(min_direct, Direct)
