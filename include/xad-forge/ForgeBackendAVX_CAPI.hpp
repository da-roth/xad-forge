#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendAVX_CAPI - AVX2 backend using Forge C API
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This backend processes 4 Monte Carlo paths per kernel execution using
//  AVX2 SIMD instructions (256-bit YMM registers = 4 doubles).
//
//  Uses the stable C API for binary compatibility across compilers.
//
//  https://github.com/da-roth/xad-forge
//
//////////////////////////////////////////////////////////////////////////////

#include <XAD/JITGraph.hpp>

// Forge C API - stable ABI
#include <forge_c_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>

namespace xad
{
namespace forge
{

/**
 * AVX2 Backend using Forge C API - standalone backend for 4-path SIMD execution.
 *
 * Uses the stable C API for binary compatibility with precompiled Forge packages.
 *
 * Usage pattern:
 *   ForgeBackendAVX_CAPI avxBackend;
 *   avxBackend.compile(jitGraph);
 *
 *   for (pathBatch = 0; pathBatch < nPaths; pathBatch += 4) {
 *       for (size_t i = 0; i < numInputs; ++i)
 *           avxBackend.setInputLanes(i, &pathInputs[pathBatch][i]);
 *
 *       double outputs[4], outputAdjoints[4] = {1.0, 1.0, 1.0, 1.0};
 *       std::vector<std::array<double, 4>> inputGradients(numInputs);
 *       avxBackend.forwardAndBackward(outputAdjoints, outputs, inputGradients);
 *   }
 */
class ForgeBackendAVX_CAPI
{
  public:
    static constexpr int VECTOR_WIDTH = 4;  // AVX2 processes 4 doubles

    explicit ForgeBackendAVX_CAPI(bool useGraphOptimizations = false)
        : useOptimizations_(useGraphOptimizations)
        , graph_(nullptr)
        , config_(nullptr)
        , kernel_(nullptr)
        , buffer_(nullptr)
    {
    }

    ~ForgeBackendAVX_CAPI()
    {
        cleanup();
    }

    ForgeBackendAVX_CAPI(ForgeBackendAVX_CAPI&& other) noexcept
        : useOptimizations_(other.useOptimizations_)
        , graph_(other.graph_)
        , config_(other.config_)
        , kernel_(other.kernel_)
        , buffer_(other.buffer_)
        , inputIds_(std::move(other.inputIds_))
        , outputIds_(std::move(other.outputIds_))
    {
        other.graph_ = nullptr;
        other.config_ = nullptr;
        other.kernel_ = nullptr;
        other.buffer_ = nullptr;
    }

    ForgeBackendAVX_CAPI& operator=(ForgeBackendAVX_CAPI&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            useOptimizations_ = other.useOptimizations_;
            graph_ = other.graph_;
            config_ = other.config_;
            kernel_ = other.kernel_;
            buffer_ = other.buffer_;
            inputIds_ = std::move(other.inputIds_);
            outputIds_ = std::move(other.outputIds_);
            other.graph_ = nullptr;
            other.config_ = nullptr;
            other.kernel_ = nullptr;
            other.buffer_ = nullptr;
        }
        return *this;
    }

    // No copy
    ForgeBackendAVX_CAPI(const ForgeBackendAVX_CAPI&) = delete;
    ForgeBackendAVX_CAPI& operator=(const ForgeBackendAVX_CAPI&) = delete;

    /**
     * Compile an xad::JITGraph with AVX2 instruction set.
     */
    void compile(const xad::JITGraph& jitGraph)
    {
        cleanup();

        // Create graph
        graph_ = forge_graph_create();
        if (!graph_)
            throw std::runtime_error(std::string("Forge graph creation failed: ") + forge_get_last_error());

        // Build graph from JITGraph
        inputIds_.clear();
        for (std::size_t i = 0; i < jitGraph.nodeCount(); ++i)
        {
            ForgeOpCode op = static_cast<ForgeOpCode>(jitGraph.nodes[i].op);
            uint32_t a = jitGraph.nodes[i].a;
            uint32_t b = jitGraph.nodes[i].b;
            uint32_t c = jitGraph.nodes[i].c;
            double imm = jitGraph.nodes[i].imm;
            int isActive = (jitGraph.nodes[i].flags & xad::JITNodeFlags::IsActive) != 0 ? 1 : 0;
            int needsGrad = 0;

            uint32_t nodeId = forge_graph_add_node(graph_, op, a, b, c, imm, isActive, needsGrad);
            if (nodeId == UINT32_MAX)
                throw std::runtime_error(std::string("Forge add_node failed: ") + forge_get_last_error());

            if (op == FORGE_OP_INPUT)
                inputIds_.push_back(nodeId);
        }

        // Mark outputs
        outputIds_.assign(jitGraph.output_ids.begin(), jitGraph.output_ids.end());
        for (auto outputId : outputIds_)
        {
            ForgeError err = forge_graph_mark_output(graph_, outputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_output failed: ") + forge_get_last_error());
        }

        // Mark diff inputs
        for (auto inputId : jitGraph.input_ids)
        {
            ForgeError err = forge_graph_mark_diff_input(graph_, inputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_diff_input failed: ") + forge_get_last_error());
        }

        // Create config with AVX2
        config_ = useOptimizations_ ? forge_config_create_fast() : forge_config_create_default();
        if (!config_)
            throw std::runtime_error("Forge config creation failed");

        forge_config_set_instruction_set(config_, FORGE_INSTRUCTION_SET_AVX2_PACKED);

        // Compile
        kernel_ = forge_compile(graph_, config_);
        if (!kernel_)
            throw std::runtime_error(std::string("Forge AVX2 compilation failed: ") + forge_get_last_error());

        // Create buffer
        buffer_ = forge_buffer_create(graph_, kernel_);
        if (!buffer_)
            throw std::runtime_error(std::string("Forge AVX2 buffer creation failed: ") + forge_get_last_error());
    }

    // =========================================================================
    // Lane-based API for 4-path batching
    // =========================================================================

    /**
     * Set 4 values for an input (one per SIMD lane = one per path)
     */
    void setInputLanes(std::size_t inputIndex, const double* values)
    {
        if (inputIndex >= inputIds_.size())
            throw std::runtime_error("Input index out of range");
        forge_buffer_set_lanes(buffer_, inputIds_[inputIndex], values);
    }

    /**
     * Get 4 output values (one per SIMD lane = one per path)
     */
    void getOutputLanes(std::size_t outputIndex, double* output) const
    {
        if (outputIndex >= outputIds_.size())
            throw std::runtime_error("Output index out of range");
        forge_buffer_get_lanes(buffer_, outputIds_[outputIndex], output);
    }

    /**
     * Execute forward + backward in one call
     */
    void forwardAndBackward(const double* outputAdjoints, double* outputs,
                           std::vector<std::array<double, VECTOR_WIDTH>>& inputGradients)
    {
        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (inputGradients.size() != inputIds_.size())
            throw std::runtime_error("Input gradients array size mismatch");

        (void)outputAdjoints;  // Forge auto-seeds to 1.0

        // Clear gradients and execute
        forge_buffer_clear_gradients(buffer_);
        ForgeError err = forge_execute(kernel_, buffer_);
        if (err != FORGE_SUCCESS)
            throw std::runtime_error(std::string("Forge execution failed: ") + forge_get_last_error());

        // Get outputs (first output only for now)
        forge_buffer_get_lanes(buffer_, outputIds_[0], outputs);

        // Get input gradients
        for (std::size_t i = 0; i < inputIds_.size(); ++i)
        {
            forge_buffer_get_gradient_lanes(buffer_, &inputIds_[i], 1, inputGradients[i].data());
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    std::size_t numInputs() const { return inputIds_.size(); }
    std::size_t numOutputs() const { return outputIds_.size(); }

    const std::vector<uint32_t>& inputIds() const { return inputIds_; }
    const std::vector<uint32_t>& outputIds() const { return outputIds_; }

    int getVectorWidth() const
    {
        return buffer_ ? forge_buffer_get_vector_width(buffer_) : 0;
    }

    /**
     * Get buffer index for a node ID (for compatibility with C++ API)
     */
    std::size_t getBufferIndex(uint32_t nodeId) const
    {
        return buffer_ ? forge_buffer_get_index(buffer_, nodeId) : SIZE_MAX;
    }

    /**
     * Returns this for buffer() compatibility (C++ API returns buffer pointer)
     */
    ForgeBackendAVX_CAPI* buffer() { return this; }
    const ForgeBackendAVX_CAPI* buffer() const { return this; }

    void reset()
    {
        cleanup();
        inputIds_.clear();
        outputIds_.clear();
    }

  private:
    void cleanup()
    {
        if (buffer_) { forge_buffer_destroy(buffer_); buffer_ = nullptr; }
        if (kernel_) { forge_kernel_destroy(kernel_); kernel_ = nullptr; }
        if (config_) { forge_config_destroy(config_); config_ = nullptr; }
        if (graph_) { forge_graph_destroy(graph_); graph_ = nullptr; }
    }

    bool useOptimizations_;
    ForgeGraphHandle graph_;
    ForgeConfigHandle config_;
    ForgeKernelHandle kernel_;
    ForgeBufferHandle buffer_;
    std::vector<uint32_t> inputIds_;
    std::vector<uint32_t> outputIds_;
};

}  // namespace forge
}  // namespace xad
