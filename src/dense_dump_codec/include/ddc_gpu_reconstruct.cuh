#pragma once
#include "ddc_compact_types.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace ddc_transport {
__device__ inline float prediction(const float* first, const float* last, size_t position, float left, float right) {
    return __fadd_rn(__fmul_rn(first[position], left), __fmul_rn(last[position], right));
}
__global__ void reconstruct_codes(float* output, const unsigned char* packet, const float* first,
        const float* last, CompactMetadata meta, size_t cells, int phi, int theta, int radius) {
    const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= 8*cells) return;
    const int channel_index = flat < cells ? 0 : flat < 2*cells ? 1 : flat < 5*cells ? 2 : 3;
    const size_t begin = (channel_index == 0 ? 0 : channel_index == 1 ? 1 : channel_index == 2 ? 2 : 5)*cells;
    const size_t index = flat - begin;
    const Channel channel = meta.channels[channel_index];
    const size_t radius_index = index % radius;
    const size_t theta_index = (index / radius) % theta;
    const size_t phi_index = (index / (size_t(radius)*theta)) % phi;
    const size_t prefix = index / (size_t(radius)*theta*phi);
    const size_t scale_index = (((prefix*(phi/channel.tile_phi) + phi_index/channel.tile_phi)
        * (theta/channel.tile_theta) + theta_index/channel.tile_theta)
        * (radius/channel.tile_radius) + radius_index/channel.tile_radius);
    const int code = channel.code_bytes == 1 ? reinterpret_cast<const int8_t*>(packet+channel.codes)[index]
                                             : reinterpret_cast<const int16_t*>(packet+channel.codes)[index];
    const float residual = channel.scale_bytes == 4
        ? __fmul_rn(float(code), reinterpret_cast<const float*>(packet+channel.scales)[scale_index])
        : float(double(code) * reinterpret_cast<const double*>(packet+channel.scales)[scale_index]);
    const size_t anchor_index = channel_index < 2 ? (8+channel_index)*cells+index : flat;
    output[flat] = __fadd_rn(prediction(first, last, anchor_index, float(1.0-meta.tau), float(meta.tau)), residual);
}
__global__ void reconstruct_exceptions(float* output, const unsigned char* packet, const float* first,
        const float* last, CompactMetadata meta, size_t cells) {
    size_t offset = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    int channel_index = 0;
    while (channel_index < 4 && offset >= meta.channels[channel_index].exceptions) {
        offset -= meta.channels[channel_index].exceptions; ++channel_index;
    }
    if (channel_index == 4) return;
    const Channel channel = meta.channels[channel_index];
    const size_t index = reinterpret_cast<const int64_t*>(packet+channel.indices)[offset];
    const size_t begin = (channel_index == 0 ? 0 : channel_index == 1 ? 1 : channel_index == 2 ? 2 : 5)*cells;
    const size_t anchor_index = channel_index < 2 ? (8+channel_index)*cells+index : begin+index;
    output[begin+index] = __fadd_rn(prediction(first, last, anchor_index, float(1.0-meta.tau), float(meta.tau)),
        reinterpret_cast<const float*>(packet+channel.values)[offset]);
}
__global__ void finish_and_validate(float* output, size_t cells, int* invalid) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= 8*cells) return;
    if (index < 2*cells) output[index] = expf(output[index]);
    if (invalid && !isfinite(output[index])) atomicExch(invalid,1);
}
inline void reconstruct(float* output, const unsigned char* packet, const float* first, const float* last,
        const CompactMetadata& meta, size_t cells, int phi, int theta, int radius, cudaStream_t stream, int* invalid = nullptr) {
    if (meta.exact) {
        if (cudaMemcpyAsync(output, first, 8*cells*sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            throw std::runtime_error("DDC exact GPU anchor copy failed");
        return;
    }
    reconstruct_codes<<<(8*cells+255)/256,256,0,stream>>>(output,packet,first,last,meta,cells,phi,theta,radius);
    size_t exceptions = 0;
    for (const auto& channel : meta.channels) exceptions += channel.exceptions;
    if (exceptions) reconstruct_exceptions<<<(exceptions+255)/256,256,0,stream>>>(output,packet,first,last,meta,cells);
    finish_and_validate<<<(8*cells+255)/256,256,0,stream>>>(output,cells,invalid);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DDC GPU reconstruction launch failed");
}
}
