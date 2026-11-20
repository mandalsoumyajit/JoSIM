// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_COPYBOX_HPP
#define JOSIM_COPYBOX_HPP

#include <memory>
#include <stdexcept>
#include <utility>

namespace JoSIM {

// An optional value held out of line with value semantics (copies are deep). Used for rarely set members of the
// device objects the fork keeps in contiguous hot-loop arrays (colored-noise state, Ic noise), so they cost one
// pointer instead of their full size. The interface follows the parts of std::optional these members use.
template <typename T> class CopyBox {
    std::unique_ptr<T> p_;

  public:
    CopyBox() = default;
    CopyBox(const CopyBox& o) : p_(o.p_ ? std::make_unique<T>(*o.p_) : nullptr) {}
    CopyBox(CopyBox&&) noexcept = default;
    CopyBox& operator=(const CopyBox& o) {
        if (this != &o) { p_ = o.p_ ? std::make_unique<T>(*o.p_) : nullptr; }
        return *this;
    }
    CopyBox& operator=(CopyBox&&) noexcept = default;
    CopyBox& operator=(T value) {
        p_ = std::make_unique<T>(std::move(value));
        return *this;
    }

    explicit operator bool() const noexcept { return static_cast<bool>(p_); }
    bool     has_value() const noexcept { return static_cast<bool>(p_); }
    void     reset() noexcept { p_.reset(); }
    T&       value() {
        if (!p_) { throw std::logic_error("CopyBox: no value"); }
        return *p_;
    }
    const T& value() const {
        if (!p_) { throw std::logic_error("CopyBox: no value"); }
        return *p_;
    }
    T&       operator*() { return *p_; }
    const T& operator*() const { return *p_; }
    T*       operator->() { return p_.get(); }
    const T* operator->() const { return p_.get(); }
};

} // namespace JoSIM

#endif // JOSIM_COPYBOX_HPP
