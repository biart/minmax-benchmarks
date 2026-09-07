// Synthetic element and iterator types that sweep sizeof() independently of everything else.

#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>

#ifdef _MSC_VER
#define MMB_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define MMB_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace mmb {

template <std::size_t N>
struct tail_bytes {
    unsigned char b[N];
};

template <>
struct tail_bytes<0> {};

// ------------------------------------------------------------------ elements

// A trivially copyable element of exactly `Bytes` bytes whose comparison touches only the leading
// key. Comparison cost is therefore constant across sizes; only the cost of *copying* an element
// scales with Bytes, which is the axis the heuristic is supposed to be about.
template <std::size_t Bytes>
struct sized_value {
    static_assert(Bytes == 1 || Bytes == 2 || Bytes == 4 || (Bytes >= 8 && Bytes % 8 == 0),
        "sized_value supports 1, 2, 4, and multiples of 8");

    using key_type = std::conditional_t<Bytes == 1, std::uint8_t,
        std::conditional_t<Bytes == 2, std::uint16_t, std::conditional_t<Bytes == 4, std::uint32_t, std::uint64_t>>>;

    static constexpr std::uint64_t key_max = std::numeric_limits<key_type>::max();

    key_type key;
    MMB_NO_UNIQUE_ADDRESS tail_bytes<Bytes - sizeof(key_type)> tail{};

    // Spelled out rather than defaulted from operator<=>: with only a three-way comparison,
    // ranges::less rewrites `a < b` as `(a <=> b) < 0`, and MSVC does not always fold the
    // materialized ordering back into flags. That adds five instructions to the loop-carried
    // dependency chain and is an artifact of the benchmark type, not of the algorithms --
    // real elements (int, double, string) offer a direct operator<.
    [[nodiscard]] friend constexpr bool operator<(const sized_value& a, const sized_value& b) noexcept {
        return a.key < b.key;
    }
    [[nodiscard]] friend constexpr bool operator>(const sized_value& a, const sized_value& b) noexcept {
        return a.key > b.key;
    }
    [[nodiscard]] friend constexpr bool operator<=(const sized_value& a, const sized_value& b) noexcept {
        return a.key <= b.key;
    }
    [[nodiscard]] friend constexpr bool operator>=(const sized_value& a, const sized_value& b) noexcept {
        return a.key >= b.key;
    }
    [[nodiscard]] friend constexpr bool operator==(const sized_value& a, const sized_value& b) noexcept {
        return a.key == b.key;
    }
    [[nodiscard]] friend constexpr bool operator!=(const sized_value& a, const sized_value& b) noexcept {
        return a.key != b.key;
    }
};

static_assert(sizeof(sized_value<1>) == 1);
static_assert(sizeof(sized_value<2>) == 2);
static_assert(sizeof(sized_value<4>) == 4);
static_assert(sizeof(sized_value<8>) == 8);
static_assert(sizeof(sized_value<16>) == 16);
static_assert(sizeof(sized_value<64>) == 64);
static_assert(std::is_trivially_copyable_v<sized_value<64>>);

// ----------------------------------------------------------------- iterators

// How the bytes past the pointer behave.
enum class fatness {
    dead, // inert padding the optimizer is free to drop -> lower bound on the cost of a fat iterator
    live, // base + offset addressing, as in deque/join -> upper bound; every deref consults them
};

// A random access iterator over T of exactly `Bytes` bytes. fat_iterator<T, sizeof(T*)> is a plain
// pointer wrapper, i.e. the unwrapped vector iterator.
template <class T, std::size_t Bytes, fatness F = fatness::dead>
class fat_iterator;

template <class T, std::size_t Bytes>
class fat_iterator<T, Bytes, fatness::dead> {
public:
    using iterator_concept  = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;

    fat_iterator() = default;

    [[nodiscard]] static fat_iterator at(T* const base, const difference_type off) noexcept {
        fat_iterator it;
        it.ptr_ = base + off;
        return it;
    }

    [[nodiscard]] T& operator*() const noexcept {
        return *ptr_;
    }
    [[nodiscard]] T& operator[](const difference_type n) const noexcept {
        return ptr_[n];
    }

    fat_iterator& operator++() noexcept {
        ++ptr_;
        return *this;
    }
    fat_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++ptr_;
        return tmp;
    }
    fat_iterator& operator--() noexcept {
        --ptr_;
        return *this;
    }
    fat_iterator operator--(int) noexcept {
        auto tmp = *this;
        --ptr_;
        return tmp;
    }
    fat_iterator& operator+=(const difference_type n) noexcept {
        ptr_ += n;
        return *this;
    }
    fat_iterator& operator-=(const difference_type n) noexcept {
        ptr_ -= n;
        return *this;
    }

    [[nodiscard]] friend fat_iterator operator+(fat_iterator it, const difference_type n) noexcept {
        return it += n;
    }
    [[nodiscard]] friend fat_iterator operator+(const difference_type n, fat_iterator it) noexcept {
        return it += n;
    }
    [[nodiscard]] friend fat_iterator operator-(fat_iterator it, const difference_type n) noexcept {
        return it -= n;
    }
    [[nodiscard]] friend difference_type operator-(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.ptr_ - b.ptr_;
    }
    [[nodiscard]] friend bool operator==(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.ptr_ == b.ptr_;
    }
    [[nodiscard]] friend std::strong_ordering operator<=>(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.ptr_ <=> b.ptr_;
    }

private:
    static_assert(Bytes >= sizeof(T*), "fat_iterator cannot be smaller than a pointer");

    T* ptr_ = nullptr;
    MMB_NO_UNIQUE_ADDRESS tail_bytes<Bytes - sizeof(T*)> tail_{};
};

template <class T, std::size_t Bytes>
class fat_iterator<T, Bytes, fatness::live> {
public:
    using iterator_concept  = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;

    fat_iterator() = default;

    [[nodiscard]] static fat_iterator at(T* const base, const difference_type off) noexcept {
        fat_iterator it;
        it.base_ = base;
        it.off_  = off;
        return it;
    }

    [[nodiscard]] T& operator*() const noexcept {
        return base_[off_];
    }
    [[nodiscard]] T& operator[](const difference_type n) const noexcept {
        return base_[off_ + n];
    }

    fat_iterator& operator++() noexcept {
        ++off_;
        return *this;
    }
    fat_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++off_;
        return tmp;
    }
    fat_iterator& operator--() noexcept {
        --off_;
        return *this;
    }
    fat_iterator operator--(int) noexcept {
        auto tmp = *this;
        --off_;
        return tmp;
    }
    fat_iterator& operator+=(const difference_type n) noexcept {
        off_ += n;
        return *this;
    }
    fat_iterator& operator-=(const difference_type n) noexcept {
        off_ -= n;
        return *this;
    }

    [[nodiscard]] friend fat_iterator operator+(fat_iterator it, const difference_type n) noexcept {
        return it += n;
    }
    [[nodiscard]] friend fat_iterator operator+(const difference_type n, fat_iterator it) noexcept {
        return it += n;
    }
    [[nodiscard]] friend fat_iterator operator-(fat_iterator it, const difference_type n) noexcept {
        return it -= n;
    }
    [[nodiscard]] friend difference_type operator-(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.off_ - b.off_;
    }
    [[nodiscard]] friend bool operator==(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.off_ == b.off_;
    }
    [[nodiscard]] friend std::strong_ordering operator<=>(const fat_iterator& a, const fat_iterator& b) noexcept {
        return a.off_ <=> b.off_;
    }

private:
    static_assert(Bytes >= sizeof(T*) + sizeof(std::ptrdiff_t), "a live fat_iterator needs base + offset");

    T* base_             = nullptr;
    std::ptrdiff_t off_  = 0;
    MMB_NO_UNIQUE_ADDRESS tail_bytes<Bytes - sizeof(T*) - sizeof(std::ptrdiff_t)> tail_{};
};

static_assert(sizeof(fat_iterator<double, 8>) == 8);
static_assert(sizeof(fat_iterator<double, 16>) == 16);
static_assert(sizeof(fat_iterator<double, 64>) == 64);
static_assert(sizeof(fat_iterator<double, 16, fatness::live>) == 16);
static_assert(sizeof(fat_iterator<double, 64, fatness::live>) == 64);
static_assert(std::is_trivially_copyable_v<fat_iterator<double, 64>>);
static_assert(std::random_access_iterator<fat_iterator<double, 32>>);
static_assert(std::random_access_iterator<fat_iterator<double, 32, fatness::live>>);

} // namespace mmb
