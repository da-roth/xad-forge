/*
 * xad-forge Scalar Backend Test Suite
 *
 * Tests the ForgeBackend (ScalarBackend) with re-evaluation pattern:
 * - Compile once, evaluate multiple times with different inputs
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
};

// =============================================================================
// Re-evaluation Tests (compile once, run many times)
// =============================================================================

TEST_F(ScalarBackendTest, ReEvaluateLinearFunction)
{
    // Test inputs for re-evaluation
    std::vector<double> inputs = {2.0, 0.5, -1.0, 5.0, 10.0, -3.0};

    // Reference results using XAD Tape
    std::vector<double> refOutputs, refDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f1(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(y));
            refDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compile ONCE with ScalarBackend, then re-evaluate many times
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f1(x);
    jit.registerOutput(y);
    jit.compile();  // Compile once!

    // Re-evaluate for each input
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        value(x) = inputs[i];
        double output;
        jit.forward(&output, 1);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at input " << inputs[i];

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDerivatives[i], derivative(x), 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateQuadraticFunction)
{
    std::vector<double> inputs = {2.0, 5.0, -1.0, 0.0, 3.5, -2.5};

    // Reference results
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

    // Compile once, re-evaluate many times
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

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
            << "Forward mismatch at input " << inputs[i];

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDerivatives[i], derivative(x), 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateMathFunctions)
{
    // Use positive inputs to avoid domain issues with log/sqrt
    std::vector<double> inputs = {2.0, 0.5, 1.0, 3.0, 4.5};

    // Reference results
    std::vector<double> refOutputs, refDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f3(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            refOutputs.push_back(value(y));
            refDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compile once, re-evaluate many times
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f3(x);
    jit.registerOutput(y);
    jit.compile();

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        value(x) = inputs[i];
        double output;
        jit.forward(&output, 1);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at input " << inputs[i];

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDerivatives[i], derivative(x), 1e-10)
            << "Adjoint mismatch at input " << inputs[i];
    }
}

TEST_F(ScalarBackendTest, ReEvaluateABoolBranching)
{
    // Test inputs that hit both branches (x < 2 and x >= 2)
    std::vector<double> inputs = {1.0, 3.0, 0.5, 2.5, -1.0, 5.0};

    // Reference results
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

    // Compile once, re-evaluate many times
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0]);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = f4ABool(x);
    jit.registerOutput(y);
    jit.compile();

    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        value(x) = inputs[i];
        double output;
        jit.forward(&output, 1);

        // Verify against plain double computation
        double expected = f4ABool_double(inputs[i]);
        EXPECT_NEAR(expected, output, 1e-10)
            << "Forward mismatch at input " << inputs[i];
        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward vs tape mismatch at input " << inputs[i];

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDerivatives[i], derivative(x), 1e-10)
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

    // Compile once
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(inputs[0].first), y(inputs[0].second);
    jit.registerInput(x);
    jit.registerInput(y);
    jit.newRecording();
    xad::AD z = x * y + x * x + y * y;
    jit.registerOutput(z);
    jit.compile();

    // Re-evaluate
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        value(x) = inputs[i].first;
        value(y) = inputs[i].second;

        double output;
        jit.forward(&output, 1);

        EXPECT_NEAR(refOutputs[i], output, 1e-10)
            << "Forward mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";

        jit.clearDerivatives();
        derivative(z) = 1.0;
        jit.computeAdjoints();

        EXPECT_NEAR(refDx[i], derivative(x), 1e-10)
            << "dx mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";
        EXPECT_NEAR(refDy[i], derivative(y), 1e-10)
            << "dy mismatch at inputs (" << inputs[i].first << ", " << inputs[i].second << ")";
    }
}

// =============================================================================
// Stress test with many re-evaluations
// =============================================================================

TEST_F(ScalarBackendTest, ManyReEvaluations)
{
    // Compile once, then run 1000 evaluations
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ScalarBackend>());

    xad::AD x(1.0);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = x * x + 3.0 * x + 2.0;  // f(x) = x^2 + 3x + 2
    jit.registerOutput(y);
    jit.compile();

    const int NUM_EVALUATIONS = 1000;
    for (int i = 0; i < NUM_EVALUATIONS; ++i)
    {
        double inputVal = static_cast<double>(i) / 100.0 - 5.0;  // Range: -5 to 5
        value(x) = inputVal;

        double output;
        jit.forward(&output, 1);

        // Expected: f(x) = x^2 + 3x + 2
        double expected = inputVal * inputVal + 3.0 * inputVal + 2.0;
        EXPECT_NEAR(expected, output, 1e-10);

        jit.clearDerivatives();
        derivative(y) = 1.0;
        jit.computeAdjoints();

        // Expected derivative: f'(x) = 2x + 3
        double expectedDeriv = 2.0 * inputVal + 3.0;
        EXPECT_NEAR(expectedDeriv, derivative(x), 1e-10);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
