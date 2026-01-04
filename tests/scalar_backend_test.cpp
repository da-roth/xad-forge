/*
 * xad-forge Scalar Backend Test Suite
 *
 * Tests the ForgeBackend with re-evaluation pattern:
 * - Compile once, evaluate multiple times with different inputs
 * - Tests forward pass and adjoint computation
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
// Uses: sin, cos, exp, log, sqrt
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

// f4: Branching with ABool::If for trackable branches
xad::AD f4ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

double f4ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

} // anonymous namespace

class ScalarBackendTest : public ::testing::Test {
protected:
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
// Re-evaluation Tests (compile once, run many times)
// =============================================================================

TEST_F(ScalarBackendTest, ReEvaluateLinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0, 10.0, -3.0};

    // Reference results using XAD Tape
    std::vector<double> refOutputs, refDerivatives;
    computeReference(f1<xad::AD>, inputs, refOutputs, refDerivatives);

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);

    // Compile with ForgeBackend
    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    // Re-evaluate for each input using lane-based API
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double inputVal = inputs[i];
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at input " << inputs[i];
        EXPECT_NEAR(refDerivatives[i], inputGradient, 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateQuadraticFunction)
{
    std::vector<double> inputs = {2.0, 5.0, -1.0, 0.0, 3.5, -2.5};

    std::vector<double> refOutputs, refDerivatives;
    computeReference(f2<xad::AD>, inputs, refOutputs, refDerivatives);

    // Record graph using JITCompiler
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
        double inputVal = inputs[i];
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at input " << inputs[i];
        EXPECT_NEAR(refDerivatives[i], inputGradient, 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateMathFunctions)
{
    // Use positive inputs to avoid domain issues with log/sqrt
    std::vector<double> inputs = {2.0, 0.5, 1.0, 3.0, 4.5};

    std::vector<double> refOutputs, refDerivatives;
    computeReference(f3<xad::AD>, inputs, refOutputs, refDerivatives);

    // Record graph using JITCompiler
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f3(x);
    jit.registerOutput(y);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double inputVal = inputs[i];
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at input " << inputs[i];
        EXPECT_NEAR(refDerivatives[i], inputGradient, 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateABoolBranching)
{
    // Test inputs that hit both branches (x < 2 and x >= 2)
    std::vector<double> inputs = {1.0, 3.0, 0.5, 2.5, -1.0, 5.0};

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

    // Record graph for ABool function
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f4ABool(x);
    jit.registerOutput(y);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double inputVal = inputs[i];
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        // Verify against plain double computation
        double expected = f4ABool_double(inputs[i]);
        EXPECT_NEAR(expected, output, 1e-10)
            << "Forward mismatch at input " << inputs[i];
        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward vs tape mismatch at input " << inputs[i];
        EXPECT_NEAR(refDerivatives[i], inputGradient, 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

// =============================================================================
// Multi-input re-evaluation tests
// =============================================================================

TEST_F(ScalarBackendTest, ReEvaluateTwoInputFunction)
{
    // f(x, y) = x*y + x^2 + y^2
    // df/dx = y + 2x, df/dy = x + 2y

    std::vector<std::pair<double, double>> inputs = {
        {2.0, 3.0}, {1.0, 1.0}, {-1.0, 2.0}, {0.5, 0.5}, {3.0, -2.0}
    };

    // Reference results
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
            xad::AD z = x * y + x * x + y * y;
            tape.registerOutput(z);
            derivative(z) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(z));
            refDx.push_back(derivative(x));
            refDy.push_back(derivative(y));
            tape.clearAll();
        }
    }

    // Record graph for two-input function
    xad::JITCompiler<double, 1> jit;
    xad::AD x(inputs[0].first), y(inputs[0].second);
    jit.registerInput(x);
    jit.registerInput(y);
    jit.newRecording();
    xad::AD z = x * y + x * x + y * y;
    jit.registerOutput(z);

    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    // Re-evaluate
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double xVal = inputs[i].first;
        double yVal = inputs[i].second;
        backend.setInput(0, &xVal);
        backend.setInput(1, &yVal);

        double output;
        double inputGradients[2];
        backend.forwardAndBackward(&output, inputGradients);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";
        EXPECT_NEAR(refDx[i], inputGradients[0], 1e-10)
            << "dx mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";
        EXPECT_NEAR(refDy[i], inputGradients[1], 1e-10)
            << "dy mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";
    }
}

// =============================================================================
// Stress test with many re-evaluations
// =============================================================================

TEST_F(ScalarBackendTest, ManyReEvaluations)
{
    // Record graph
    xad::JITCompiler<double, 1> jit;
    xad::AD x(1.0);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = x * x + 3.0 * x + 2.0;  // f(x) = x^2 + 3x + 2
    jit.registerOutput(y);

    // Compile once
    xad::forge::ForgeBackend backend;
    backend.compile(jit.getGraph());

    const int NUM_EVALUATIONS = 1000;
    for (int i = 0; i < NUM_EVALUATIONS; ++i)
    {
        double inputVal = static_cast<double>(i) / 100.0 - 5.0;  // Range: -5 to 5
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        // Expected: f(x) = x^2 + 3x + 2
        double expected = inputVal * inputVal + 3.0 * inputVal + 2.0;
        EXPECT_NEAR(expected, output, 1e-10);

        // Expected derivative: f'(x) = 2x + 3
        double expectedDeriv = 2.0 * inputVal + 3.0;
        EXPECT_NEAR(expectedDeriv, inputGradient, 1e-10);
    }
}

// =============================================================================
// Reset and recompile test
// =============================================================================

TEST_F(ScalarBackendTest, ResetAndRecompile)
{
    xad::forge::ForgeBackend backend;

    // First function: f(x) = 2x
    {
        xad::JITCompiler<double, 1> jit;
        xad::AD x(1.0);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = 2.0 * x;
        jit.registerOutput(y);

        backend.compile(jit.getGraph());

        double inputVal = 3.0;
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        EXPECT_NEAR(6.0, output, 1e-10);  // f(3) = 6
        EXPECT_NEAR(2.0, inputGradient, 1e-10);  // f'(x) = 2
    }

    // Reset
    backend.reset();

    // Second function: f(x) = x^2
    {
        xad::JITCompiler<double, 1> jit;
        xad::AD x(1.0);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = x * x;
        jit.registerOutput(y);

        backend.compile(jit.getGraph());

        double inputVal = 3.0;
        backend.setInput(0, &inputVal);

        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);

        EXPECT_NEAR(9.0, output, 1e-10);  // f(3) = 9
        EXPECT_NEAR(6.0, inputGradient, 1e-10);  // f'(3) = 6
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
