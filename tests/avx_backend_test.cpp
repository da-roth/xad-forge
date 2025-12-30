/*
 * xad-forge AVX Backend Test Suite
 *
 * Tests the ForgeBackendAVX (AVXBackend) with re-evaluation pattern:
 * - Compile once, evaluate multiple times with different inputs
 * - Tests 4-path SIMD batching with AVX2
 * - Tests forward pass and adjoint computation
 *
 * Copyright (c) 2025 The xad-forge Authors
 * https://github.com/da-roth/xad-forge
 * SPDX-License-Identifier: Zlib
 */

#include <xad-forge/ForgeBackends.hpp>
#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <memory>

namespace {

// f1: Simple linear function
// f(x) = x * 3 + 2, f'(x) = 3
template <class T>
T f1(const T& x)
{
    return x * 3.0 + 2.0;
}

// f2: Quadratic function
// f(x) = x^2 + 3x, f'(x) = 2x + 3
template <class T>
T f2(const T& x)
{
    return x * x + 3.0 * x;
}

// f3: Function with math operations
template <class T>
T f3(const T& x)
{
    using std::sin; using std::cos; using std::exp; using std::log;
    using std::sqrt;

    T result = sin(x) + cos(x) * 2.0;
    result = result + exp(x / 10.0) + log(x + 5.0);
    result = result + sqrt(x + 1.0);
    result = result + x * x;
    result = result + 1.0 / (x + 2.0);
    return result;
}

// f4: Branching with ABool::If
xad::AD f4ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

double f4ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

} // anonymous namespace

class AVXBackendTest : public ::testing::Test {
protected:
    static constexpr int LANES = xad::forge::AVXBackend::VECTOR_WIDTH;

    void SetUp() override {}
    void TearDown() override {}

    // Helper to get reference values using XAD Tape
    template<typename Func>
    void computeReference(Func func, const std::vector<double>& inputs,
                          std::vector<double>& outputs, std::vector<double>& derivatives)
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = func(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            outputs.push_back(value(y));
            derivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

};

// =============================================================================
// Basic AVX backend tests with 4-path batching
// =============================================================================

TEST_F(AVXBackendTest, LinearFunctionBatched)
{
    // 8 inputs = 2 batches of 4
    std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Reference values
    std::vector<double> refOutputs, refDerivatives;
    computeReference(f1<xad::AD>, inputs, refOutputs, refDerivatives);

    // Build JIT graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);

    // Compile AVX backend from JIT graph
    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    // Process in batches of 4
    for (std::size_t batch = 0; batch < inputs.size(); batch += LANES)
    {
        // Set 4 input lanes
        double inputLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
            inputLanes[lane] = inputs[batch + lane];
        avx.setInputLanes(0, inputLanes);

        // Execute forward + backward
        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Verify all 4 lanes
        for (int lane = 0; lane < LANES; ++lane)
        {
            std::size_t idx = batch + lane;
            EXPECT_NEAR(refOutputs[idx], outputs[lane], 1e-10)
                << "Output mismatch at index " << idx;
            EXPECT_NEAR(refDerivatives[idx], inputGradients[0][lane], 1e-10)
                << "Gradient mismatch at index " << idx;
        }
    }
}

TEST_F(AVXBackendTest, QuadraticFunctionBatched)
{
    std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5};

    std::vector<double> refOutputs, refDerivatives;
    computeReference(f2<xad::AD>, inputs, refOutputs, refDerivatives);

    // Build graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f2(x);
    jit.registerOutput(y);

    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    for (std::size_t batch = 0; batch < inputs.size(); batch += LANES)
    {
        double inputLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
            inputLanes[lane] = inputs[batch + lane];
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            std::size_t idx = batch + lane;
            EXPECT_NEAR(refOutputs[idx], outputs[lane], 1e-10)
                << "Output mismatch at index " << idx;
            EXPECT_NEAR(refDerivatives[idx], inputGradients[0][lane], 1e-10)
                << "Gradient mismatch at index " << idx;
        }
    }
}

TEST_F(AVXBackendTest, MathFunctionsBatched)
{
    // Positive inputs for log/sqrt
    std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5};

    std::vector<double> refOutputs, refDerivatives;
    computeReference(f3<xad::AD>, inputs, refOutputs, refDerivatives);

    // Build graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f3(x);
    jit.registerOutput(y);

    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    for (std::size_t batch = 0; batch < inputs.size(); batch += LANES)
    {
        double inputLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
            inputLanes[lane] = inputs[batch + lane];
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            std::size_t idx = batch + lane;
            EXPECT_NEAR(refOutputs[idx], outputs[lane], 1e-10)
                << "Output mismatch at index " << idx;
            EXPECT_NEAR(refDerivatives[idx], inputGradients[0][lane], 1e-10)
                << "Gradient mismatch at index " << idx;
        }
    }
}

TEST_F(AVXBackendTest, ABoolBranchingBatched)
{
    // Mix of values < 2 and >= 2 to test both branches
    std::vector<double> inputs = {1.0, 3.0, 0.5, 2.5, -1.0, 5.0, 1.5, 4.0};

    std::vector<double> refOutputs, refDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f4ABool(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(y));
            refDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Build graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f4ABool(x);
    jit.registerOutput(y);

    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    for (std::size_t batch = 0; batch < inputs.size(); batch += LANES)
    {
        double inputLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
            inputLanes[lane] = inputs[batch + lane];
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            std::size_t idx = batch + lane;
            double expected = f4ABool_double(inputs[idx]);
            EXPECT_NEAR(expected, outputs[lane], 1e-10)
                << "Output mismatch at index " << idx;
            EXPECT_NEAR(refOutputs[idx], outputs[lane], 1e-10)
                << "Output vs tape mismatch at index " << idx;
            EXPECT_NEAR(refDerivatives[idx], inputGradients[0][lane], 1e-10)
                << "Gradient mismatch at index " << idx;
        }
    }
}

// =============================================================================
// Re-evaluation tests (compile once, run many batches)
// =============================================================================

TEST_F(AVXBackendTest, ReEvaluateManyBatches)
{
    // Build graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(1.0);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = x * x + 3.0 * x + 2.0;  // f(x) = x^2 + 3x + 2
    jit.registerOutput(y);

    // Compile once
    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    // Run 100 batches (400 evaluations)
    const int NUM_BATCHES = 100;
    for (int batch = 0; batch < NUM_BATCHES; ++batch)
    {
        // Generate 4 different inputs per batch
        double inputLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
        {
            inputLanes[lane] = static_cast<double>(batch * LANES + lane) / 50.0 - 4.0;
        }
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Verify each lane
        for (int lane = 0; lane < LANES; ++lane)
        {
            double xval = inputLanes[lane];
            double expected = xval * xval + 3.0 * xval + 2.0;
            double expectedDeriv = 2.0 * xval + 3.0;

            EXPECT_NEAR(expected, outputs[lane], 1e-10);
            EXPECT_NEAR(expectedDeriv, inputGradients[0][lane], 1e-10);
        }
    }
}

// =============================================================================
// Two-input function with AVX
// =============================================================================

TEST_F(AVXBackendTest, TwoInputFunctionBatched)
{
    // f(x, y) = x*y + x^2
    // df/dx = y + 2x, df/dy = x

    // 4 pairs of (x, y) per batch
    std::vector<std::pair<double, double>> inputs = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 1.0}, {0.5, 4.0},
        {-1.0, 2.0}, {2.0, -1.0}, {1.5, 1.5}, {3.0, 3.0}
    };

    // Reference
    std::vector<double> refOutputs, refDx, refDy;
    {
        xad::Tape<double> tape;
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            double xval = inputs[i].first;
            double yval = inputs[i].second;
            xad::AD x(xval), y(yval);
            tape.registerInput(x);
            tape.registerInput(y);
            tape.newRecording();
            xad::AD z = x * y + x * x;
            tape.registerOutput(z);
            derivative(z) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(z));
            refDx.push_back(derivative(x));
            refDy.push_back(derivative(y));
            tape.clearAll();
        }
    }

    // Build graph using JITCompiler
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(1.0), y(2.0);
    jit.registerInput(x);
    jit.registerInput(y);
    jit.newRecording();
    xad::AD z = x * y + x * x;
    jit.registerOutput(z);

    xad::forge::AVXBackend avx;
    avx.compile(jit.getGraph());

    ASSERT_EQ(2u, avx.numInputs());

    for (std::size_t batch = 0; batch < inputs.size(); batch += LANES)
    {
        // Set x lanes
        double xLanes[LANES], yLanes[LANES];
        for (int lane = 0; lane < LANES; ++lane)
        {
            xLanes[lane] = inputs[batch + lane].first;
            yLanes[lane] = inputs[batch + lane].second;
        }
        avx.setInputLanes(0, xLanes);
        avx.setInputLanes(1, yLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(2);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            std::size_t idx = batch + lane;
            EXPECT_NEAR(refOutputs[idx], outputs[lane], 1e-10)
                << "Output mismatch at index " << idx;
            EXPECT_NEAR(refDx[idx], inputGradients[0][lane], 1e-10)
                << "dx mismatch at index " << idx;
            EXPECT_NEAR(refDy[idx], inputGradients[1][lane], 1e-10)
                << "dy mismatch at index " << idx;
        }
    }
}

// =============================================================================
// Reset and recompile test
// =============================================================================

TEST_F(AVXBackendTest, ResetAndRecompile)
{
    xad::forge::AVXBackend avx;

    // First function: f(x) = 2x
    {
        xad::JITCompiler<double, 1> jit(
            std::make_unique<xad::forge::ScalarBackend>());

        xad::AD x(1.0);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = 2.0 * x;
        jit.registerOutput(y);

        avx.compile(jit.getGraph());

        double inputLanes[LANES] = {1.0, 2.0, 3.0, 4.0};
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            EXPECT_NEAR(2.0 * inputLanes[lane], outputs[lane], 1e-10);
            EXPECT_NEAR(2.0, inputGradients[0][lane], 1e-10);
        }
    }

    // Reset
    avx.reset();

    // Second function: f(x) = x^2
    {
        xad::JITCompiler<double, 1> jit(
            std::make_unique<xad::forge::ScalarBackend>());

        xad::AD x(1.0);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = x * x;
        jit.registerOutput(y);

        avx.compile(jit.getGraph());

        double inputLanes[LANES] = {1.0, 2.0, 3.0, 4.0};
        avx.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0, 1.0, 1.0, 1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        for (int lane = 0; lane < LANES; ++lane)
        {
            EXPECT_NEAR(inputLanes[lane] * inputLanes[lane], outputs[lane], 1e-10);
            EXPECT_NEAR(2.0 * inputLanes[lane], inputGradients[0][lane], 1e-10);
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
