#pragma once
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA) || defined(__CUDACC__)
#include <cuda_runtime_api.h>
#endif

namespace ddc_transport {
inline bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string(value) == "1";
}

template<class Value> class HostBuffer {
    struct Allocation {
        Value* pointer = nullptr;
        size_t capacity = 0;
        explicit Allocation(size_t count) : capacity(count) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(__CUDACC__)
            if (count > std::numeric_limits<size_t>::max() / sizeof(Value))
                throw std::length_error("Pinned buffer size overflow");
            if (count && cudaHostAlloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value), cudaHostAllocPortable) != cudaSuccess)
                throw std::runtime_error("DDC pinned allocation failed");
#else
            throw std::runtime_error("DDC pinned reception requires CUDA");
#endif
        }
        ~Allocation() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(__CUDACC__)
            if (pointer) cudaFreeHost(pointer);
#endif
        }
    };
    struct Pool {
        std::mutex mutex;
        std::vector<Allocation*> free;
        size_t bytes = 0;
        ~Pool() { for (auto* allocation : free) delete allocation; }
        static Pool& instance() { static Pool pool; return pool; }
        static std::shared_ptr<Allocation> acquire(size_t count) {
            auto& pool = instance();
            Allocation* allocation = nullptr;
            {
                std::lock_guard<std::mutex> lock(pool.mutex);
                for (auto iterator = pool.free.begin(); iterator != pool.free.end(); ++iterator) {
                    if ((*iterator)->capacity >= count) {
                        allocation = *iterator; pool.bytes -= allocation->capacity*sizeof(Value);
                        pool.free.erase(iterator); break;
                    }
                }
            }
            if (!allocation) allocation = new Allocation(count);
            return std::shared_ptr<Allocation>(allocation, [](Allocation* value) {
                auto& destination = instance();
                const size_t bytes = value->capacity*sizeof(Value);
                {
                    std::lock_guard<std::mutex> lock(destination.mutex);
                    if (destination.bytes + bytes <= size_t(1024)*1024*1024) {
                        destination.free.push_back(value); destination.bytes += bytes; return;
                    }
                }
                delete value;
            });
        }
    };
    std::vector<Value> ordinary_;
    std::shared_ptr<Allocation> pinned_;
    size_t count_ = 0;
    size_t offset_ = 0;
public:
    using value_type = Value;
    HostBuffer() = default;
    HostBuffer(std::vector<Value> values) : ordinary_(std::move(values)) {}
    HostBuffer& operator=(std::vector<Value> values) {
        pinned_.reset(); count_ = offset_ = 0; ordinary_ = std::move(values); return *this;
    }
    void resize(size_t count, bool pinned = enabled("KPOLARIS_DDC_PINNED") || enabled("KPOLARIS_DDC_COMPACT")) {
        if (!pinned) { pinned_.reset(); ordinary_.resize(count); return; }
        ordinary_.clear();
        if (!pinned_ || pinned_.use_count() != 1 || pinned_->capacity < count || offset_)
            pinned_ = Pool::acquire(count);
        offset_ = 0; count_ = count;
    }
    HostBuffer slice(size_t offset, size_t count) const {
        if (!pinned_ || offset > count_ || count > count_ - offset)
            throw std::out_of_range("DDC pinned buffer slice");
        HostBuffer result; result.pinned_ = pinned_; result.offset_ = offset_ + offset;
        result.count_ = count; return result;
    }
    bool pinned() const { return bool(pinned_); }
    size_t size() const { return pinned_ ? count_ : ordinary_.size(); }
    Value* data() { return pinned_ ? pinned_->pointer + offset_ : ordinary_.data(); }
    const Value* data() const { return pinned_ ? pinned_->pointer + offset_ : ordinary_.data(); }
    Value* begin() { return data(); }
    Value* end() { return data() + size(); }
    const Value* begin() const { return data(); }
    const Value* end() const { return data() + size(); }
    Value& operator[](size_t index) { return data()[index]; }
    const Value& operator[](size_t index) const { return data()[index]; }
};
}
