#pragma once
#include <cstdint>
namespace ddc_transport {
struct Channel {
    uint64_t codes, scales, indices, values, count, scale_count, exceptions;
    uint64_t code_bytes, scale_bytes, tile_phi, tile_theta, tile_radius;
};
struct CompactMetadata {
    uint64_t version, exact, start, end, missing, packet_bytes, reserved;
    double tau;
    Channel channels[4];
};
static_assert(sizeof(Channel) == 96 && sizeof(CompactMetadata) == 448);
}
