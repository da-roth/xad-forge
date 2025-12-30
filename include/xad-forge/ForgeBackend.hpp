#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackend - XAD JIT backend using Forge for native code generation
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  When XAD_FORGE_USE_CAPI is defined, this header forwards to
//  ForgeBackendCAPI for binary compatibility across compilers.
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

#ifdef XAD_FORGE_USE_CAPI

#include <xad-forge/ForgeBackendCAPI.hpp>
namespace xad { namespace forge { using ForgeBackend = ForgeBackendCAPI; } }

#else

#include <XAD/JITBackendInterface.hpp>
#include <XAD/JITGraph.hpp>

// Forge library (https://github.com/da-roth/forge)
#include <graph/graph.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/compiler_config.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace xad
{
namespace forge
{

/**
 * JIT Backend using Forge for native code generation.
 * https://github.com/da-roth/forge
 *
 * Uses Forge's JIT compiler for fast forward pass execution.
 * Falls back to JITGraphInterpreter for adjoint computation.
 */
class ForgeBackend : public xad::JITBackend
{
  public:
    // Constructor with optional graph optimizations (default: disabled)
    explicit ForgeBackend(bool useGraphOptimizations = false)
        : config_(useGraphOptimizations ? optimizedConfig() : defaultConfig())
    {
    }

    ~ForgeBackend() override = default;

    ForgeBackend(ForgeBackend&&) noexcept = default;
    ForgeBackend& operator=(ForgeBackend&&) noexcept = default;

    // No copy
    ForgeBackend(const ForgeBackend&) = delete;
    ForgeBackend& operator=(const ForgeBackend&) = delete;

    void compile(const xad::JITGraph& graph) override
    {
        // Skip recompilation if already compiled with same graph
        if (kernel_ && lastNodeCount_ == graph.nodeCount())
            return;

        // Build ::forge::Graph from JITGraph
        forgeGraph_ = ::forge::Graph();
        forgeGraph_.nodes.reserve(graph.nodeCount());

        // First pass: create nodes without needsGradient
        for (std::size_t i = 0; i < graph.nodeCount(); ++i)
        {
            ::forge::Node n;
            n.op = static_cast<::forge::OpCode>(graph.nodes[i].op);
            n.dst = static_cast<uint32_t>(i);
            n.a = graph.nodes[i].a;
            n.b = graph.nodes[i].b;
            n.c = graph.nodes[i].c;
            n.imm = graph.nodes[i].imm;
            n.isActive = (graph.nodes[i].flags & xad::JITNodeFlags::IsActive) != 0;
            n.isDead = false;
            n.needsGradient = false;  // Will be set in propagation pass
            forgeGraph_.nodes.push_back(n);
        }

        // Copy constant pool and outputs
        forgeGraph_.constPool = graph.const_pool;
        forgeGraph_.outputs.assign(graph.output_ids.begin(), graph.output_ids.end());
        forgeGraph_.diff_inputs.assign(graph.input_ids.begin(), graph.input_ids.end());

        // Second pass: propagate needsGradient from diff_inputs through the graph
        // Mark all input nodes that are in diff_inputs as needing gradients
        for (auto inputId : graph.input_ids)
        {
            if (inputId < forgeGraph_.nodes.size())
                forgeGraph_.nodes[inputId].needsGradient = true;
        }

        // Forward propagation: if any operand needs gradient, result needs gradient
        for (std::size_t i = 0; i < forgeGraph_.nodes.size(); ++i)
        {
            auto& node = forgeGraph_.nodes[i];
            if (node.isDead) continue;

            bool operandNeedsGrad = false;
            if (node.a < forgeGraph_.nodes.size())
                operandNeedsGrad |= forgeGraph_.nodes[node.a].needsGradient;
            if (node.b < forgeGraph_.nodes.size())
                operandNeedsGrad |= forgeGraph_.nodes[node.b].needsGradient;
            if (node.c < forgeGraph_.nodes.size())
                operandNeedsGrad |= forgeGraph_.nodes[node.c].needsGradient;

            if (operandNeedsGrad)
                node.needsGradient = true;
        }

        // Extract input node IDs
        inputIds_.clear();
        for (std::size_t i = 0; i < forgeGraph_.nodes.size(); ++i)
        {
            if (forgeGraph_.nodes[i].op == ::forge::OpCode::Input)
                inputIds_.push_back(static_cast<uint32_t>(i));
        }
        outputIds_.assign(graph.output_ids.begin(), graph.output_ids.end());

        // Compile to native code using the stored config
        ::forge::ForgeEngine compiler(config_);
        kernel_ = compiler.compile(forgeGraph_);

        if (!kernel_)
            throw std::runtime_error("Forge kernel compilation failed");

        // Create node value buffer
        buffer_ = ::forge::NodeValueBufferFactory::create(forgeGraph_, *kernel_);

        if (!buffer_)
            throw std::runtime_error("Forge buffer creation failed");

        // Cache graph size to detect changes
        lastNodeCount_ = graph.nodeCount();
    }

    void forward(const xad::JITGraph& graph,
                 const double* inputs, std::size_t numInputs,
                 double* outputs, std::size_t numOutputs) override
    {
        (void)graph;  // unused, we use forgeGraph_

        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (numInputs != inputIds_.size())
            throw std::runtime_error("Input count mismatch");
        if (numOutputs != outputIds_.size())
            throw std::runtime_error("Output count mismatch");

        // Set inputs
        double inputLane[1];
        for (std::size_t i = 0; i < numInputs; ++i) {
            inputLane[0] = inputs[i];
            buffer_->setLanes(inputIds_[i], inputLane);
        }

        // Execute kernel (Forge always runs forward+backward, but we ignore gradients here)
        buffer_->clearGradients();
        kernel_->execute(*buffer_);

        // Get outputs
        double outputLane[1];
        for (std::size_t i = 0; i < numOutputs; ++i) {
            buffer_->getLanes(outputIds_[i], outputLane);
            outputs[i] = outputLane[0];
        }
    }

    void forwardAndBackward(const xad::JITGraph& graph,
                            const double* inputs, std::size_t numInputs,
                            const double* outputAdjoints, std::size_t numOutputs,
                            double* outputs,
                            double* inputAdjoints) override
    {
        (void)graph;  // unused, we use forgeGraph_
        (void)outputAdjoints;  // unused, Forge auto-seeds output gradients to 1.0

        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (numInputs != inputIds_.size())
            throw std::runtime_error("Input count mismatch");
        if (numOutputs != outputIds_.size())
            throw std::runtime_error("Output count mismatch");

        // Set inputs
        double inputLane[1];
        for (std::size_t i = 0; i < numInputs; ++i) {
            inputLane[0] = inputs[i];
            buffer_->setLanes(inputIds_[i], inputLane);
        }

        // Clear gradients - Forge will auto-seed output gradients to 1.0
        buffer_->clearGradients();

        // Execute kernel (forward + backward in one call)
        kernel_->execute(*buffer_);

        double* gradPtr = buffer_->getGradientsPtr();

        // Get outputs
        double outputLane[1];
        for (std::size_t i = 0; i < numOutputs; ++i) {
            buffer_->getLanes(outputIds_[i], outputLane);
            outputs[i] = outputLane[0];
        }

        // Get input gradients
        for (std::size_t i = 0; i < numInputs; ++i) {
            size_t bufferIdx = buffer_->getBufferIndex(inputIds_[i]);
            inputAdjoints[i] = gradPtr[bufferIdx];
        }
    }

    void reset() override
    {
        kernel_.reset();
        buffer_.reset();
        forgeGraph_ = ::forge::Graph();
        inputIds_.clear();
        outputIds_.clear();
        lastNodeCount_ = 0;
    }

    // =========================================================================
    // Accessors for graph reuse (e.g., by AVX backend)
    // =========================================================================
    const ::forge::Graph& forgeGraph() const { return forgeGraph_; }
    const std::vector<uint32_t>& inputIds() const { return inputIds_; }
    const std::vector<uint32_t>& outputIds() const { return outputIds_; }

  private:
    static ::forge::CompilerConfig defaultConfig()
    {
        // Default: only stability cleaning, no graph optimizations
        ::forge::CompilerConfig config;
        config.instructionSet = ::forge::CompilerConfig::InstructionSet::SSE2_SCALAR;
        config.enableOptimizations = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        config.enableStabilityCleaning = true;
        return config;
    }

    static ::forge::CompilerConfig optimizedConfig()
    {
        // Use Forge's Fast config with all graph optimizations enabled
        auto config = ::forge::CompilerConfig::Fast();
        config.instructionSet = ::forge::CompilerConfig::InstructionSet::SSE2_SCALAR;
        return config;
    }

    ::forge::CompilerConfig config_;
    ::forge::Graph forgeGraph_;
    std::unique_ptr<::forge::StitchedKernel> kernel_;
    std::unique_ptr<::forge::INodeValueBuffer> buffer_;
    std::vector<uint32_t> inputIds_;
    std::vector<uint32_t> outputIds_;
    std::size_t lastNodeCount_ = 0;
};

}  // namespace forge
}  // namespace xad

#endif  // XAD_FORGE_USE_CAPI
