/*
 * xad-forge C API Backend Test Suite
 *
 * Tests ForgeBackendCAPI directly (not via the ScalarBackend alias)
 * to ensure the C API implementation works correctly for:
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

#include <xad-forge/ForgeBackendCAPI.hpp>
#include <XAD/XAD.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
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
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test that ForgeBackendCAPI computes correct forward values
// =============================================================================

TEST_F(CAPIBackendTest, ForwardLinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0};

    // Use JITCompiler with ForgeBackendCAPI explicitly
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackendCAPI>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);
    jit.compile();

    for (double input : inputs)
    {
        value(x) = input;
        double output;
        jit.forward(&output, 1);

        // Expected: f(x) = 3x + 2
        double expected = 3.0 * input + 2.0;
        EXPECT_NEAR(expected, output, 1e-10)
            << "Forward mismatch at input " << input;
    }
}

// =============================================================================
// Test that ForgeBackendCAPI computes correct derivatives (CRITICAL TEST)
// This catches the needsGradient propagation bug
// =============================================================================

TEST_F(CAPIBackendTest, DerivativeLinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0};

    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackendCAPI>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);
    jit.compile();

    for (double input : inputs)
    {
        value(x) = input;
        double output;
        jit.forward(&output, 1);

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        // Expected derivative: f'(x) = 3 (constant for all inputs)
        double expectedDeriv = 3.0;
        EXPECT_NEAR(expectedDeriv, derivative(x), 1e-10)
            << "Derivative mismatch at input " << input
            << " - got " << derivative(x) << " expected " << expectedDeriv;
    }
}

TEST_F(CAPIBackendTest, DerivativeQuadraticFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0, 0.0, -3.0};

    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackendCAPI>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f2(x);
    jit.registerOutput(y);
    jit.compile();

    for (double input : inputs)
    {
        value(x) = input;
        double output;
        jit.forward(&output, 1);

        // Expected: f(x) = x^2 + 3x
        double expectedOutput = input * input + 3.0 * input;
        EXPECT_NEAR(expectedOutput, output, 1e-10)
            << "Forward mismatch at input " << input;

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        // Expected derivative: f'(x) = 2x + 3
        double expectedDeriv = 2.0 * input + 3.0;
        EXPECT_NEAR(expectedDeriv, derivative(x), 1e-10)
            << "Derivative mismatch at input " << input
            << " - got " << derivative(x) << " expected " << expectedDeriv;
    }
}

TEST_F(CAPIBackendTest, DerivativeTwoInputFunction)
{
    std::vector<std::pair<double, double>> inputs = {
        {2.0, 3.0}, {1.0, 1.0}, {-1.0, 2.0}, {0.5, 0.5}, {3.0, -2.0}
    };

    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackendCAPI>());

    xad::AD x(inputs[0].first), y(inputs[0].second);
    jit.registerInput(x);
    jit.registerInput(y);
    jit.newRecording();
    xad::AD z = f3(x, y);
    jit.registerOutput(z);
    jit.compile();

    for (const auto& [xval, yval] : inputs)
    {
        value(x) = xval;
        value(y) = yval;
        double output;
        jit.forward(&output, 1);

        // Expected: f(x,y) = x*y + x^2
        double expectedOutput = xval * yval + xval * xval;
        EXPECT_NEAR(expectedOutput, output, 1e-10)
            << "Forward mismatch at (" << xval << ", " << yval << ")";

        jit.clearDerivatives();
        derivative(z) = 1.0;
        jit.computeAdjoints();

        // Expected: df/dx = y + 2x, df/dy = x
        double expectedDx = yval + 2.0 * xval;
        double expectedDy = xval;
        EXPECT_NEAR(expectedDx, derivative(x), 1e-10)
            << "dx mismatch at (" << xval << ", " << yval << ")"
            << " - got " << derivative(x) << " expected " << expectedDx;
        EXPECT_NEAR(expectedDy, derivative(y), 1e-10)
            << "dy mismatch at (" << xval << ", " << yval << ")"
            << " - got " << derivative(y) << " expected " << expectedDy;
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

    // Compare with ForgeBackendCAPI
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackendCAPI>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f2(x);
    jit.registerOutput(y);
    jit.compile();

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        value(x) = inputs[i];
        double output;
        jit.forward(&output, 1);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "C API output doesn't match XAD Tape at input " << inputs[i];

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDerivatives[i], derivative(x), 1e-10)
            << "C API derivative doesn't match XAD Tape at input " << inputs[i]
            << " - C API: " << derivative(x) << ", Tape: " << refDerivatives[i];
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
