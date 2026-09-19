#pragma once
#include "ddc_compact_types.hpp"
#include "grmhd/ddc_host_buffer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>

namespace ddc_transport {
struct CompactFrame {
    CompactMetadata metadata{};
    HostBuffer<unsigned char> packet;
    HostBuffer<float> first, last;
};
struct AnchorCache {
    std::string socket;
    std::map<uint64_t, HostBuffer<float>> values;
};
inline AnchorCache& anchor_cache(const std::string& socket) {
    thread_local AnchorCache cache;
    if (cache.socket != socket) { cache.values.clear(); cache.socket = socket; }
    return cache;
}
inline std::string compact_request(const std::string& socket, int sequence) {
    std::string request = "STAGEQ " + std::to_string(sequence);
    for (const auto& item : anchor_cache(socket).values) request += " " + std::to_string(item.first);
    return request + "\n";
}
template<class Receive>
std::shared_ptr<CompactFrame> receive_compact(Receive receive, const std::string& socket,
        size_t cells, int phi, int theta, int radius, size_t remaining_bytes, uint64_t sequence) {
    auto frame = std::make_shared<CompactFrame>();
    auto& meta = frame->metadata;
    if (remaining_bytes < sizeof(meta)) throw std::runtime_error("Truncated DDC compact metadata");
    receive(&meta, sizeof(meta)); remaining_bytes -= sizeof(meta);
    const size_t anchor_bytes = 10*cells*sizeof(float);
    if (meta.version != 1 || meta.reserved || meta.exact > 1 || meta.missing > 2 ||
        !std::isfinite(meta.tau) || meta.tau < 0 || meta.tau > 1 || meta.start > meta.end ||
        sequence < meta.start || sequence > meta.end ||
        (meta.exact && (meta.start != sequence || meta.end != sequence)) ||
        meta.packet_bytes > 160*cells || remaining_bytes != meta.missing*(8+anchor_bytes)+meta.packet_bytes)
        throw std::runtime_error("Invalid DDC compact metadata or payload length");
    auto& cache = anchor_cache(socket).values;
    for (size_t index = 0; index < meta.missing; ++index) {
        uint64_t key = 0; receive(&key, sizeof(key));
        if (key != meta.start && key != meta.end) throw std::runtime_error("Unrequested DDC compact anchor");
        HostBuffer<float> values; values.resize(10*cells, true);
        receive(values.data(), anchor_bytes);
        for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite DDC compact anchor");
        cache[key] = std::move(values);
    }
    if (!cache.count(meta.start) || !cache.count(meta.end)) throw std::runtime_error("Missing DDC compact anchor");
    frame->first = cache.at(meta.start); frame->last = cache.at(meta.end);
    if (frame->first.size() != 10*cells || frame->last.size() != 10*cells)
        throw std::runtime_error("Changed DDC compact anchor layout");
    for (auto iterator = cache.begin(); iterator != cache.end();) {
        if (iterator->first != meta.start && iterator->first != meta.end) iterator = cache.erase(iterator);
        else ++iterator;
    }
    frame->packet.resize(meta.packet_bytes, true);
    receive(frame->packet.data(), meta.packet_bytes);
    if (meta.exact) {
        if (meta.packet_bytes) throw std::runtime_error("Exact anchor has residual bytes");
        return frame;
    }
    size_t channel_index = 0;
    for (const auto& channel : meta.channels) {
        if (channel.count != (channel_index < 2 ? cells : 3*cells) ||
            (channel.code_bytes != 1 && channel.code_bytes != 2) ||
            (channel.scale_bytes != 4 && channel.scale_bytes != 8) ||
            !channel.tile_phi || !channel.tile_theta || !channel.tile_radius ||
            size_t(phi)%channel.tile_phi || size_t(theta)%channel.tile_theta || size_t(radius)%channel.tile_radius ||
            channel.scale_count != channel.count/(channel.tile_phi*channel.tile_theta*channel.tile_radius) ||
            channel.exceptions > channel.count)
            throw std::runtime_error("Invalid compact channel shape");
        auto check = [&](uint64_t offset, uint64_t count, uint64_t bytes) {
            if (offset%8 || offset > meta.packet_bytes || count > (meta.packet_bytes-offset)/bytes)
                throw std::runtime_error("Invalid compact channel offset");
        };
        check(channel.codes, channel.count, channel.code_bytes);
        check(channel.scales, channel.scale_count, channel.scale_bytes);
        check(channel.indices, channel.exceptions, 8); check(channel.values, channel.exceptions, 4);
        const auto* indices = reinterpret_cast<const int64_t*>(frame->packet.data()+channel.indices);
        const auto* values = reinterpret_cast<const float*>(frame->packet.data()+channel.values);
        int64_t previous = -1;
        for (size_t index = 0; index < channel.exceptions; ++index) {
            if (indices[index] <= previous || uint64_t(indices[index]) >= channel.count || !std::isfinite(values[index]))
                throw std::runtime_error("Invalid compact exception index or value");
            previous = indices[index];
        }
        for (size_t index = 0; index < channel.scale_count; ++index) {
            const double value = channel.scale_bytes == 4
                ? reinterpret_cast<const float*>(frame->packet.data()+channel.scales)[index]
                : reinterpret_cast<const double*>(frame->packet.data()+channel.scales)[index];
            if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid compact scale");
        }
        ++channel_index;
    }
    return frame;
}
}
