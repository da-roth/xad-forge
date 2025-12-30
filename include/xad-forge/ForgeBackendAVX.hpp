#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendAVX - AVX2 backend for 4-path batching using Forge
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This backend processes 4 Monte Carlo paths per kernel execution using
//  AVX2 SIMD instructions (256-bit YMM registers = 4 doubles).
//
//  USAGE: This backend is standalone with lane-based API for manual batching
//
//  When XAD_FORGE_USE_CAPI is defined, this header forwards to
//  ForgeBackendAVX_CAPI for binary compatibility across compilers.
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

#include <xad-forge/ForgeBackendAVX_CAPI.hpp>
namespace xad { namespace forge { using ForgeBackendAVX = ForgeBackendAVX_CAPI; } }

#else

#include <XAD/JITGraph.hpp>

// Forge library (https://github.com/da-roth/forge)
#include <graph/graph.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/compiler_config.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace xad
{
namespace forge
{

/**
 * AVX2 Backend for Forge - standalone backend for 4-path SIMD execution.
 *
 * Takes an xad::JITGraph, converts it to forge::Graph, and compiles it
 * with AVX2_PACKED instruction set for 4-path batching.
 *
 * Usage pattern:
 *   ForgeBackendAVX avxBackend;
 *   avxBackend.compile(jitGraph);
 *
 *   for (pathBatch = 0; pathBatch < nPaths; pathBatch += 4) {
 *       // Set inputs for 4 paths
 *       for (size_t i = 0; i < numInputs; ++i)
 *           avxBackend.setInputLanes(i, &pathInputs[pathBatch][i]);
 *
 *       // Run forward + backward, get both outputs and gradients
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
        : config_(useGraphOptimizations ? optimizedConfig() : defaultConfig())
    {
    }

    ~ForgeBackendAVX() = default;

    ForgeBackendAVX(ForgeBackendAVX&&) noexcept = default;
    ForgeBackendAVX& operator=(ForgeBackendAVX&&) noexcept = default;

    // No copy
    ForgeBackendAVX(const ForgeBackendAVX&) = delete;
    ForgeBackendAVX& operator=(const ForgeBackendAVX&) = delete;

    /**
     * Compile an xad::JITGraph with AVX2 instruction set.
     *
     * @param jitGraph The XAD JIT graph to compile
     */
    void compile(const xad::JITGraph& jitGraph)
    {
        // Convert xad::JITGraph to forge::Graph
        forgeGraph_ = ::forge::Graph();
        forgeGraph_.nodes.reserve(jitGraph.nodeCount());

        // First pass: create nodes without needsGradient
        for (std::size_t i = 0; i < jitGraph.nodeCount(); ++i)
        {
            ::forge::Node n;
            n.op = static_cast<::forge::OpCode>(jitGraph.nodes[i].op);
            n.dst = static_cast<uint32_t>(i);
            n.a = jitGraph.nodes[i].a;
            n.b = jitGraph.nodes[i].b;
            n.c = jitGraph.nodes[i].c;
            n.imm = jitGraph.nodes[i].imm;
            n.isActive = (jitGraph.nodes[i].flags & xad::JITNodeFlags::IsActive) != 0;
            n.isDead = false;
            n.needsGradient = false;  // Will be set in propagation pass
            forgeGraph_.nodes.push_back(n);
        }

        // Copy constant pool and outputs
        forgeGraph_.constPool = jitGraph.const_pool;
        forgeGraph_.outputs.assign(jitGraph.output_ids.begin(), jitGraph.output_ids.end());
        forgeGraph_.diff_inputs.assign(jitGraph.input_ids.begin(), jitGraph.input_ids.end());

        // Second pass: propagate needsGradient from diff_inputs through the graph
        // Mark all input nodes that are in diff_inputs as needing gradients
        for (auto inputId : jitGraph.input_ids)
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
        outputIds_.assign(jitGraph.output_ids.begin(), jitGraph.output_ids.end());

        // Compile to native code using AVX2 config
        ::forge::ForgeEngine compiler(config_);
        kernel_ = compiler.compile(forgeGraph_);

        if (!kernel_)
            throw std::runtime_error("Forge AVX2 kernel compilation failed");

        // Create node value buffer (will be AVX2NodeValueBuffer due to AVX2_PACKED config)
        buffer_ = ::forge::NodeValueBufferFactory::create(forgeGraph_, *kernel_);

        if (!buffer_)
            throw std::runtime_error("Forge AVX2 buffer creation failed");

        // Pre-compute buffer indices for all inputs (for gradient retrieval)
        inputBufferIndices_.clear();
        inputBufferIndices_.reserve(inputIds_.size());
        for (auto id : inputIds_)
        {
            inputBufferIndices_.push_back(buffer_->getBufferIndex(id));
        }

        // Pre-compute buffer indices for outputs
        outputBufferIndices_.clear();
        outputBufferIndices_.reserve(outputIds_.size());
        for (auto id : outputIds_)
        {
            outputBufferIndices_.push_back(buffer_->getBufferIndex(id));
        }
    }

    // =========================================================================
    // Lane-based API for 4-path batching
    // =========================================================================

    /**
     * Set 4 values for an input (one per SIMD lane = one per path)
     * @param inputIndex Index into the input array (0 to numInputs-1)
     * @param values Pointer to 4 doubles [path0, path1, path2, path3]
     */
    void setInputLanes(std::size_t inputIndex, const double* values)
    {
        if (inputIndex >= inputIds_.size())
            throw std::runtime_error("Input index out of range");
        buffer_->setLanes(inputIds_[inputIndex], values);
    }

    /**
     * Get 4 output values (one per SIMD lane = one per path)
     * @param outputIndex Index into the output array (0 to numOutputs-1)
     * @param output Pointer to receive 4 doubles [path0, path1, path2, path3]
     */
    void getOutputLanes(std::size_t outputIndex, double* output) const
    {
        if (outputIndex >= outputIds_.size())
            throw std::runtime_error("Output index out of range");
        buffer_->getLanes(outputIds_[outputIndex], output);
    }

    /**
     * Execute forward + backward in one call (efficient path for Forge)
     * Forge always computes both forward and backward together.
     *
     * Usage:
     *   double inputs[4], outputs[4], outputAdjoints[4], inputGradients[4];
     *   for each input: setInputLanes(idx, inputs);
     *   forwardAndBackward(outputAdjoints, outputs, inputGradients);
     *
     * @param outputAdjoints Pointer to array of 4 output adjoint values (seeds for backward)
     * @param outputs Pointer to array to receive 4 output values
     * @param inputGradients Pointer to array arrays to receive input gradients (numInputs arrays of 4 values)
     */
    void forwardAndBackward(const double* outputAdjoints, double* outputs,
                           std::vector<std::array<double, VECTOR_WIDTH>>& inputGradients)
    {
        if (!kernel_ || !buffer_)
            throw std::runtime_error("Backend not compiled");

        if (inputGradients.size() != inputIds_.size())
            throw std::runtime_error("Input gradients array size mismatch");

        // Clear and seed output adjoints
        buffer_->clearGradients();
        double* gradPtr = buffer_->getGradientsPtr();
        size_t bufferIdx = buffer_->getBufferIndex(outputIds_[0]);
        std::memcpy(&gradPtr[bufferIdx], outputAdjoints, VECTOR_WIDTH * sizeof(double));

        // Execute kernel (forward + backward together)
        kernel_->execute(*buffer_);

        // Get outputs
        buffer_->getLanes(outputIds_[0], outputs);

        // Get all input gradients in a single batched call (much more efficient)
        // Reference: /docs/quantlib-forge benchmarks use single getGradientLanes call
        const std::size_t numInputs = inputIds_.size();
        std::vector<double> allGradients(numInputs * VECTOR_WIDTH);
        buffer_->getGradientLanes(inputBufferIndices_, allGradients.data());

        // Distribute interleaved gradients to per-input arrays
        // Layout: [input0_lane0..3, input1_lane0..3, ...]
        for (std::size_t i = 0; i < numInputs; ++i) {
            std::memcpy(inputGradients[i].data(), &allGradients[i * VECTOR_WIDTH],
                       VECTOR_WIDTH * sizeof(double));
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    std::size_t numInputs() const { return inputIds_.size(); }
    std::size_t numOutputs() const { return outputIds_.size(); }

    const std::vector<uint32_t>& inputIds() const { return inputIds_; }
    const std::vector<uint32_t>& outputIds() const { return outputIds_; }

    // Access to underlying forge graph (for debugging/inspection)
    const ::forge::Graph& forgeGraph() const { return forgeGraph_; }

    // Access to buffer for advanced usage
    ::forge::INodeValueBuffer* buffer() { return buffer_.get(); }
    const ::forge::INodeValueBuffer* buffer() const { return buffer_.get(); }

    void reset()
    {
        kernel_.reset();
        buffer_.reset();
        forgeGraph_ = ::forge::Graph();
        inputIds_.clear();
        outputIds_.clear();
        inputBufferIndices_.clear();
        outputBufferIndices_.clear();
    }

  private:
    static ::forge::CompilerConfig defaultConfig()
    {
        ::forge::CompilerConfig config;
        // Use AVX2 packed mode - 4 doubles per operation
        config.instructionSet = ::forge::CompilerConfig::InstructionSet::AVX2_PACKED;
        config.enableOptimizations = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        config.enableStabilityCleaning = true;
        return config;
    }

    static ::forge::CompilerConfig optimizedConfig()
    {
        auto config = ::forge::CompilerConfig::Fast();
        // Use AVX2 packed mode - 4 doubles per operation
        config.instructionSet = ::forge::CompilerConfig::InstructionSet::AVX2_PACKED;
        return config;
    }

    ::forge::CompilerConfig config_;
    ::forge::Graph forgeGraph_;
    std::unique_ptr<::forge::StitchedKernel> kernel_;
    std::unique_ptr<::forge::INodeValueBuffer> buffer_;
    std::vector<uint32_t> inputIds_;
    std::vector<uint32_t> outputIds_;
    std::vector<size_t> inputBufferIndices_;   // Pre-computed for gradient access
    std::vector<size_t> outputBufferIndices_;  // Pre-computed for output access
};

}  // namespace forge
}  // namespace xad

#endif  // XAD_FORGE_USE_CAPI
