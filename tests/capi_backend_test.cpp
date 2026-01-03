/*
 * xad-forge C API Backend Test Suite
 *
 * Tests ForgeBackend (which uses the C API internally) with lane-based API:
 * - Forward pass values
 * - Backward pass derivatives (adjoint computation)
 * - Re-evaluation pattern (compile once, run many times)
 *
 * This test is critical for catching issues like missing needsGradient
 * propagation in the C API layer.
 *
 * Copyright (c) 2025 The xad-forge Authors
 * https://github.com/da-roth/xad-forge
 * SPDX-License-Identifier: Zlib
 */

#include <xad-forge/ForgeBackend.hpp>
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

// f3: Two-input function
// f(x, y) = x*y + x^2, df/dx = y + 2x, df/dy = x
template <class T>
T f3(const T& x, const T& y)
{
    return x * y + x * x;
}

} // anonymous namespace

class CAPIBackendTest : public ::testing::Test {
protected:
    static constexpr int LANES = xad::forge::ForgeBackend::VECTOR_WIDTH;

    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test that ForgeBackend computes correct forward values
// =============================================================================

TEST_F(CAPIBackendTest, ForwardLinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0};

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);

    // Compile backend directly
    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (double input : inputs)
    {
        double inputLanes[LANES] = {input};
        backend.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        backend.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Expected: f(x) = 3x + 2
        double expected = 3.0 * input + 2.0;
        EXPECT_NEAR(expected, outputs[0], 1e-10)
            << "Forward mismatch at input " << input;
    }
}

// =============================================================================
// Test that ForgeBackend computes correct derivatives (CRITICAL TEST)
// This catches the needsGradient propagation bug
// =============================================================================

TEST_F(CAPIBackendTest, DerivativeLinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0};

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (double input : inputs)
    {
        double inputLanes[LANES] = {input};
        backend.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        backend.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Expected derivative: f'(x) = 3 (constant for all inputs)
        double expectedDeriv = 3.0;
        EXPECT_NEAR(expectedDeriv, inputGradients[0][0], 1e-10)
            << "Derivative mismatch at input " << input
            << " - got " << inputGradients[0][0] << " expected " << expectedDeriv;
    }
}

TEST_F(CAPIBackendTest, DerivativeQuadraticFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0, 0.0, -3.0};

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f2(x);
    jit.registerOutput(y);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (double input : inputs)
    {
        double inputLanes[LANES] = {input};
        backend.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        backend.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Expected: f(x) = x^2 + 3x
        double expectedOutput = input * input + 3.0 * input;
        EXPECT_NEAR(expectedOutput, outputs[0], 1e-10)
            << "Forward mismatch at input " << input;

        // Expected derivative: f'(x) = 2x + 3
        double expectedDeriv = 2.0 * input + 3.0;
        EXPECT_NEAR(expectedDeriv, inputGradients[0][0], 1e-10)
            << "Derivative mismatch at input " << input
            << " - got " << inputGradients[0][0] << " expected " << expectedDeriv;
    }
}

TEST_F(CAPIBackendTest, DerivativeTwoInputFunction)
{
    std::vector<std::pair<double, double>> inputs = {
        {2.0, 3.0}, {1.0, 1.0}, {-1.0, 2.0}, {0.5, 0.5}, {3.0, -2.0}
    };

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0].first), y(inputs[0].second);
    jit.registerInput(x);
    jit.registerInput(y);
    jit.newRecording();
    xad::AD z = f3(x, y);
    jit.registerOutput(z);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double xval = inputs[i].first;
        double yval = inputs[i].second;

        double xLanes[LANES] = {xval};
        double yLanes[LANES] = {yval};
        backend.setInputLanes(0, xLanes);
        backend.setInputLanes(1, yLanes);

        double outputAdjoints[LANES] = {1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(2);
        backend.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        // Expected: f(x,y) = x*y + x^2
        double expectedOutput = xval * yval + xval * xval;
        EXPECT_NEAR(expectedOutput, outputs[0], 1e-10)
            << "Forward mismatch at (" << xval << ", " << yval << ")";

        // Expected: df/dx = y + 2x, df/dy = x
        double expectedDx = yval + 2.0 * xval;
        double expectedDy = xval;
        EXPECT_NEAR(expectedDx, inputGradients[0][0], 1e-10)
            << "dx mismatch at (" << xval << ", " << yval << ")"
            << " - got " << inputGradients[0][0] << " expected " << expectedDx;
        EXPECT_NEAR(expectedDy, inputGradients[1][0], 1e-10)
            << "dy mismatch at (" << xval << ", " << yval << ")"
            << " - got " << inputGradients[1][0] << " expected " << expectedDy;
    }
}

// =============================================================================
// Comparison test: verify C API matches XAD Tape reference
// =============================================================================

TEST_F(CAPIBackendTest, MatchesXADTapeReference)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0};

    // Get reference values from XAD Tape
    std::vector<double> refOutputs, refDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f2(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(y));
            refDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compare with ForgeBackend
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f2(x);
    jit.registerOutput(y);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double inputLanes[LANES] = {inputs[i]};
        backend.setInputLanes(0, inputLanes);

        double outputAdjoints[LANES] = {1.0};
        double outputs[LANES];
        std::vector<std::array<double, LANES>> inputGradients(1);
        backend.forwardAndBackward(outputAdjoints, outputs, inputGradients);

        EXPECT_NEAR(refOutputs[i], outputs[0], 1e-10)
            << "C API output doesn't match XAD Tape at input " << inputs[i];

        EXPECT_NEAR(refDerivatives[i], inputGradients[0][0], 1e-10)
            << "C API derivative doesn't match XAD Tape at input " << inputs[i]
            << " - C API: " << inputGradients[0][0] << ", Tape: " << refDerivatives[i];
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
