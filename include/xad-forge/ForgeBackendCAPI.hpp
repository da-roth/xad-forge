#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendCAPI - XAD JIT backend using Forge C API
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
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

#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>

// Forge C API - stable ABI
#include <forge_c_api.h>

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <string>

namespace xad
{
namespace forge
{

/**
 * JIT Backend using Forge C API for native code generation.
 *
 * This version uses the stable C API instead of the C++ API,
 * enabling binary compatibility with precompiled Forge packages
 * built with different compilers.
 */
class ForgeBackendCAPI : public xad::JITBackend
{
  public:
    // Constructor with optional graph optimizations (default: disabled)
    explicit ForgeBackendCAPI(bool useGraphOptimizations = false)
        : useOptimizations_(useGraphOptimizations)
        , graph_(nullptr)
        , config_(nullptr)
        , kernel_(nullptr)
        , buffer_(nullptr)
    {
    }

    ~ForgeBackendCAPI() override
    {
        cleanup();
    }

    ForgeBackendCAPI(ForgeBackendCAPI&& other) noexcept
        : useOptimizations_(other.useOptimizations_)
        , graph_(other.graph_)
        , config_(other.config_)
        , kernel_(other.kernel_)
        , buffer_(other.buffer_)
        , inputIds_(std::move(other.inputIds_))
        , outputIds_(std::move(other.outputIds_))
        , lastNodeCount_(other.lastNodeCount_)
    {
        other.graph_ = nullptr;
        other.config_ = nullptr;
        other.kernel_ = nullptr;
        other.buffer_ = nullptr;
    }

    ForgeBackendCAPI& operator=(ForgeBackendCAPI&& other) noexcept
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
            lastNodeCount_ = other.lastNodeCount_;
            other.graph_ = nullptr;
            other.config_ = nullptr;
            other.kernel_ = nullptr;
            other.buffer_ = nullptr;
        }
        return *this;
    }

    // No copy
    ForgeBackendCAPI(const ForgeBackendCAPI&) = delete;
    ForgeBackendCAPI& operator=(const ForgeBackendCAPI&) = delete;

    void compile(const xad::JITGraph& jitGraph) override
    {
        // Skip recompilation if already compiled with same graph
        if (kernel_ && lastNodeCount_ == jitGraph.nodeCount())
            return;

        // Clean up previous compilation
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
        // Then for CONSTANT nodes, we use forge_graph_add_node() with the correct
        // imm value (the constPool index) instead of forge_graph_add_constant().
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

        // Create config
        config_ = useOptimizations_ ? forge_config_create_fast() : forge_config_create_default();
        if (!config_)
            throw std::runtime_error("Forge config creation failed");

        // Set instruction set to SSE2 scalar
        forge_config_set_instruction_set(config_, FORGE_INSTRUCTION_SET_SSE2_SCALAR);

        // Compile
        kernel_ = forge_compile(graph_, config_);
        if (!kernel_)
            throw std::runtime_error(std::string("Forge compilation failed: ") + forge_get_last_error());

        // Create buffer
        buffer_ = forge_buffer_create(graph_, kernel_);
        if (!buffer_)
            throw std::runtime_error(std::string("Forge buffer creation failed: ") + forge_get_last_error());

        lastNodeCount_ = jitGraph.nodeCount();
    }

    void forward(const xad::JITGraph& graph,
                 const double* inputs, std::size_t numInputs,
                 double* outputs, std::size_t numOutputs) override
    {
        (void)graph;

        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (numInputs != inputIds_.size())
            throw std::runtime_error("Input count mismatch");
        if (numOutputs != outputIds_.size())
            throw std::runtime_error("Output count mismatch");

        // Set inputs
        for (std::size_t i = 0; i < numInputs; ++i)
        {
            forge_buffer_set_value(buffer_, inputIds_[i], inputs[i]);
        }

        // Clear gradients and execute
        forge_buffer_clear_gradients(buffer_);
        ForgeError err = forge_execute(kernel_, buffer_);
        if (err != FORGE_SUCCESS)
            throw std::runtime_error(std::string("Forge execution failed: ") + forge_get_last_error());

        // Get outputs
        for (std::size_t i = 0; i < numOutputs; ++i)
        {
            forge_buffer_get_value(buffer_, outputIds_[i], &outputs[i]);
        }
    }

    void forwardAndBackward(const xad::JITGraph& graph,
                            const double* inputs, std::size_t numInputs,
                            const double* outputAdjoints, std::size_t numOutputs,
                            double* outputs,
                            double* inputAdjoints) override
    {
        (void)graph;
        (void)outputAdjoints;  // Forge auto-seeds to 1.0

        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (numInputs != inputIds_.size())
            throw std::runtime_error("Input count mismatch");
        if (numOutputs != outputIds_.size())
            throw std::runtime_error("Output count mismatch");

        // Set inputs
        for (std::size_t i = 0; i < numInputs; ++i)
        {
            forge_buffer_set_value(buffer_, inputIds_[i], inputs[i]);
        }

        // Clear gradients and execute
        forge_buffer_clear_gradients(buffer_);
        ForgeError err = forge_execute(kernel_, buffer_);
        if (err != FORGE_SUCCESS)
            throw std::runtime_error(std::string("Forge execution failed: ") + forge_get_last_error());

        // Get outputs
        for (std::size_t i = 0; i < numOutputs; ++i)
        {
            forge_buffer_get_value(buffer_, outputIds_[i], &outputs[i]);
        }

        // Get input gradients
        for (std::size_t i = 0; i < numInputs; ++i)
        {
            forge_buffer_get_gradient(buffer_, inputIds_[i], &inputAdjoints[i]);
        }
    }

    void reset() override
    {
        cleanup();
        inputIds_.clear();
        outputIds_.clear();
        lastNodeCount_ = 0;
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
    std::size_t lastNodeCount_ = 0;
};

}  // namespace forge
}  // namespace xad
