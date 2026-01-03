#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendAVX - AVX2 backend using Forge C API
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This backend processes 4 Monte Carlo paths per kernel execution using
//  AVX2 SIMD instructions (256-bit YMM registers = 4 doubles).
//
//  Uses the stable C API for binary compatibility across compilers.
//
//  Copyright (c) 2025 The xad-forge Authors
//  https://github.com/da-roth/xad-forge
//
//  This software is provided 'as-is', without any express or implied
//  warranty. In no event will the authors be held liable for any damages
//  arising from the use of this software.
//
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//
//  1. The origin of this software must not be misrepresented; you must not
//     claim that you wrote the original software. If you use this software
//     in a product, an acknowledgment in the product documentation would be
//     appreciated but is not required.
//  2. Altered source versions must be plainly marked as such, and must not
//     be misrepresented as being the original software.
//  3. This notice may not be removed or altered from any source distribution.
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
 *   ForgeBackendAVX avxBackend;
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
class ForgeBackendAVX
{
  public:
    static constexpr int VECTOR_WIDTH = 4;  // AVX2 processes 4 doubles

    explicit ForgeBackendAVX(bool useGraphOptimizations = false)
        : useOptimizations_(useGraphOptimizations)
        , graph_(nullptr)
        , config_(nullptr)
        , kernel_(nullptr)
        , buffer_(nullptr)
    {
    }

    ~ForgeBackendAVX()
    {
        cleanup();
    }

    ForgeBackendAVX(ForgeBackendAVX&& other) noexcept
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

    ForgeBackendAVX& operator=(ForgeBackendAVX&& other) noexcept
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
    ForgeBackendAVX(const ForgeBackendAVX&) = delete;
    ForgeBackendAVX& operator=(const ForgeBackendAVX&) = delete;

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

        // Pre-populate forge's constPool to match XAD's const_pool indices.
        // This is critical because:
        // 1. XAD stores constPool indices in CONSTANT nodes' imm field
        // 2. Multiple CONSTANT nodes can reference the same constPool index
        // 3. forge_graph_add_constant() creates NEW constPool entries
        //
        // By first adding all constants, we ensure forge's constPool matches XAD's.
        // Then for CONSTANT nodes, we reference these pre-created nodes.
        std::vector<uint32_t> constNodeIds;
        constNodeIds.reserve(jitGraph.const_pool.size());
        for (std::size_t i = 0; i < jitGraph.const_pool.size(); ++i)
        {
            uint32_t nodeId = forge_graph_add_constant(graph_, jitGraph.const_pool[i]);
            if (nodeId == UINT32_MAX)
                throw std::runtime_error(std::string("Forge add_constant failed: ") + forge_get_last_error());
            constNodeIds.push_back(nodeId);
        }

        // Now add the actual graph nodes.
        // For CONSTANT nodes, we reference the pre-created constant nodes.
        // For other nodes, we add them normally.
        inputIds_.clear();

        // Map from XAD node index to Forge node ID
        std::vector<uint32_t> nodeIdMap(jitGraph.nodeCount());

        for (std::size_t i = 0; i < jitGraph.nodeCount(); ++i)
        {
            ForgeOpCode op = static_cast<ForgeOpCode>(jitGraph.nodes[i].op);
            uint32_t nodeId;

            if (op == FORGE_OP_INPUT)
            {
                nodeId = forge_graph_add_input(graph_);
                if (nodeId == UINT32_MAX)
                    throw std::runtime_error(std::string("Forge add_input failed: ") + forge_get_last_error());
                inputIds_.push_back(nodeId);
            }
            else if (op == FORGE_OP_CONSTANT)
            {
                // XAD stores the constPool index in node.imm
                // Reference the pre-created constant node
                uint32_t constIndex = static_cast<uint32_t>(jitGraph.nodes[i].imm);
                if (constIndex >= constNodeIds.size())
                    throw std::runtime_error("Invalid constant pool index in JITGraph");
                nodeId = constNodeIds[constIndex];
            }
            else
            {
                // Remap operand indices from XAD to Forge node IDs
                uint32_t a = jitGraph.nodes[i].a;
                uint32_t b = jitGraph.nodes[i].b;
                uint32_t c = jitGraph.nodes[i].c;

                if (a < i) a = nodeIdMap[a];
                if (b < i) b = nodeIdMap[b];
                if (c < i) c = nodeIdMap[c];

                double imm = jitGraph.nodes[i].imm;
                int isActive = (jitGraph.nodes[i].flags & xad::JITNodeFlags::IsActive) != 0 ? 1 : 0;

                nodeId = forge_graph_add_node(graph_, op, a, b, c, imm, isActive, 0);
                if (nodeId == UINT32_MAX)
                    throw std::runtime_error(std::string("Forge add_node failed: ") + forge_get_last_error());
            }

            nodeIdMap[i] = nodeId;
        }

        // Mark outputs (remap from XAD indices to Forge node IDs)
        outputIds_.clear();
        for (auto xadOutputId : jitGraph.output_ids)
        {
            uint32_t forgeOutputId = nodeIdMap[xadOutputId];
            outputIds_.push_back(forgeOutputId);
            ForgeError err = forge_graph_mark_output(graph_, forgeOutputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_output failed: ") + forge_get_last_error());
        }

        // Mark diff inputs (remap from XAD indices to Forge node IDs)
        for (auto xadInputId : jitGraph.input_ids)
        {
            uint32_t forgeInputId = nodeIdMap[xadInputId];
            ForgeError err = forge_graph_mark_diff_input(graph_, forgeInputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_diff_input failed: ") + forge_get_last_error());
        }

        // Propagate needsGradient flags through the graph
        {
            ForgeError err = forge_graph_propagate_gradients(graph_);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge propagate_gradients failed: ") + forge_get_last_error());
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
    ForgeBackendAVX* buffer() { return this; }
    const ForgeBackendAVX* buffer() const { return this; }

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
