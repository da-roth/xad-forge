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
//  https://github.com/da-roth/xad-forge
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
            int needsGrad = 0;  // Will be set via mark_diff_input

            uint32_t nodeId = forge_graph_add_node(graph_, op, a, b, c, imm, isActive, needsGrad);
            if (nodeId == UINT32_MAX)
                throw std::runtime_error(std::string("Forge add_node failed: ") + forge_get_last_error());

            // Track input nodes
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
