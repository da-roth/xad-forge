#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackend - Scalar backend using Forge C API
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This backend processes one evaluation per kernel execution using SSE2
//  scalar instructions. For backends that support multiple parallel
//  evaluations per execution (e.g., ForgeBackendAVX), see ForgeBackendAVX.hpp.
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

// Shared utilities (custom backend loading)
#include <xad-forge/ForgeBackendCommon.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace xad
{
namespace forge
{

/**
 * Scalar Backend using Forge C API - implements xad::JITBackend interface.
 *
 * Uses the stable C API for binary compatibility with precompiled Forge packages.
 * Processes one evaluation per kernel execution using SSE2 scalar instructions.
 *
 * Note: Forge currently only supports double precision. This backend is templated
 * to match the JITBackend<Scalar> interface, but only Scalar=double is supported.
 * Using Scalar=float will result in a static_assert failure.
 *
 * Usage pattern (via JITCompiler):
 *   xad::JITCompiler<double> jit;
 *   // ... record graph ...
 *   jit.setBackend(std::make_unique<xad::forge::ForgeBackend<double>>());
 *   jit.compile();
 *   jit.setInput(0, &inputValue);
 *   double output, gradient;
 *   jit.forwardAndBackward(&output, &gradient);
 */
template <class Scalar>
class ForgeBackend : public xad::JITBackend<Scalar>
{
    static_assert(std::is_same<Scalar, double>::value,
                  "ForgeBackend only supports double precision. Forge does not currently support float.");

  public:
    explicit ForgeBackend(bool useGraphOptimizations = false)
        : useOptimizations_(useGraphOptimizations)
        , graph_(nullptr)
        , config_(nullptr)
        , kernel_(nullptr)
        , buffer_(nullptr)
    {
    }

    ~ForgeBackend() override
    {
        cleanup();
    }

    ForgeBackend(ForgeBackend&& other) noexcept
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

    ForgeBackend& operator=(ForgeBackend&& other) noexcept
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
    ForgeBackend(const ForgeBackend&) = delete;
    ForgeBackend& operator=(const ForgeBackend&) = delete;

    //=========================================================================
    // JITBackend interface implementation
    //=========================================================================

    /**
     * Compile an xad::JITGraph with SSE2 scalar instruction set.
     */
    void compile(const xad::JITGraph& jitGraph) override
    {
        cleanup();

        // Create graph
        graph_ = forge_graph_create();
        if (!graph_)
            throw std::runtime_error(std::string("Forge graph creation failed: ") + forge_get_last_error());

        // Pre-populate forge's constPool to match XAD's const_pool indices.
        std::vector<uint32_t> constNodeIds;
        constNodeIds.reserve(jitGraph.const_pool.size());
        for (std::size_t i = 0; i < jitGraph.const_pool.size(); ++i)
        {
            uint32_t nodeId = forge_graph_add_constant(graph_, jitGraph.const_pool[i]);
            if (nodeId == UINT32_MAX)
                throw std::runtime_error(std::string("Forge add_constant failed: ") + forge_get_last_error());
            constNodeIds.push_back(nodeId);
        }

        // Add graph nodes
        inputIds_.clear();
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
                uint32_t constIndex = static_cast<uint32_t>(jitGraph.nodes[i].imm);
                if (constIndex >= constNodeIds.size())
                    throw std::runtime_error("Invalid constant pool index in JITGraph");
                nodeId = constNodeIds[constIndex];
            }
            else
            {
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

        // Mark outputs
        outputIds_.clear();
        for (auto xadOutputId : jitGraph.output_ids)
        {
            uint32_t forgeOutputId = nodeIdMap[xadOutputId];
            outputIds_.push_back(forgeOutputId);
            ForgeError err = forge_graph_mark_output(graph_, forgeOutputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_output failed: ") + forge_get_last_error());
        }

        // Mark diff inputs
        for (auto xadInputId : jitGraph.input_ids)
        {
            uint32_t forgeInputId = nodeIdMap[xadInputId];
            ForgeError err = forge_graph_mark_diff_input(graph_, forgeInputId);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge mark_diff_input failed: ") + forge_get_last_error());
        }

        // Propagate needsGradient flags
        {
            ForgeError err = forge_graph_propagate_gradients(graph_);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Forge propagate_gradients failed: ") + forge_get_last_error());
        }

        // Load custom backend if specified via environment variable (thread-safe, once per process)
        detail::loadCustomBackendFromEnv();

        // Create config
        config_ = useOptimizations_ ? forge_config_create_fast() : forge_config_create_default();
        if (!config_)
            throw std::runtime_error("Forge config creation failed");

        // Select instruction set: use custom if specified, otherwise default SSE2 scalar
        const char* customInstSet = std::getenv("XAD_FORGE_INSTRUCTION_SET");
        if (customInstSet && customInstSet[0] != '\0')
        {
            ForgeError err = forge_config_set_instruction_set_by_name(config_, customInstSet);
            if (err != FORGE_SUCCESS)
                throw std::runtime_error(std::string("Instruction set '") + customInstSet +
                                         "' not found. Available sets can be listed with forge_get_instruction_set_name().");
        }
        else
        {
            forge_config_set_instruction_set(config_, FORGE_INSTRUCTION_SET_SSE2_SCALAR);
        }

        // Compile
        kernel_ = forge_compile(graph_, config_);
        if (!kernel_)
            throw std::runtime_error(std::string("Forge compilation failed: ") + forge_get_last_error());

        // Create buffer
        buffer_ = forge_buffer_create(graph_, kernel_);
        if (!buffer_)
            throw std::runtime_error(std::string("Forge buffer creation failed: ") + forge_get_last_error());
    }

    void reset() override
    {
        cleanup();
        inputIds_.clear();
        outputIds_.clear();
    }

    std::size_t vectorWidth() const override { return 1; }
    std::size_t numInputs() const override { return inputIds_.size(); }
    std::size_t numOutputs() const override { return outputIds_.size(); }

    /**
     * Set value for an input.
     */
    void setInput(std::size_t inputIndex, const Scalar* values) override
    {
        if (inputIndex >= inputIds_.size())
            throw std::runtime_error("Input index out of range");
        forge_buffer_set_lanes(buffer_, inputIds_[inputIndex], values);
    }

    /**
     * Execute forward pass only.
     */
    void forward(Scalar* outputs) override
    {
        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        // Clear gradients and execute (Forge always does forward+backward)
        forge_buffer_clear_gradients(buffer_);
        ForgeError err = forge_execute(kernel_, buffer_);
        if (err != FORGE_SUCCESS)
            throw std::runtime_error(std::string("Forge execution failed: ") + forge_get_last_error());

        // Get outputs
        for (std::size_t i = 0; i < outputIds_.size(); ++i)
        {
            forge_buffer_get_lanes(buffer_, outputIds_[i], outputs + i);
        }
    }

    /**
     * Execute forward + backward in one call.
     */
    void forwardAndBackward(Scalar* outputs, Scalar* inputGradients) override
    {
        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        // Clear gradients and execute
        forge_buffer_clear_gradients(buffer_);
        ForgeError err = forge_execute(kernel_, buffer_);
        if (err != FORGE_SUCCESS)
            throw std::runtime_error(std::string("Forge execution failed: ") + forge_get_last_error());

        // Get outputs
        for (std::size_t i = 0; i < outputIds_.size(); ++i)
        {
            forge_buffer_get_lanes(buffer_, outputIds_[i], outputs + i);
        }

        // Get input gradients
        for (std::size_t i = 0; i < inputIds_.size(); ++i)
        {
            forge_buffer_get_gradient_lanes(buffer_, &inputIds_[i], 1, inputGradients + i);
        }
    }

    // =========================================================================
    // Additional Accessors
    // =========================================================================

    const std::vector<uint32_t>& inputIds() const { return inputIds_; }
    const std::vector<uint32_t>& outputIds() const { return outputIds_; }

    int getVectorWidth() const
    {
        return buffer_ ? forge_buffer_get_vector_width(buffer_) : 0;
    }

    std::size_t getBufferIndex(uint32_t nodeId) const
    {
        return buffer_ ? forge_buffer_get_index(buffer_, nodeId) : SIZE_MAX;
    }

    ForgeBackend* buffer() { return this; }
    const ForgeBackend* buffer() const { return this; }

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
