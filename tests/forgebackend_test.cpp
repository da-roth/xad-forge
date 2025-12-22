/*
 * xad-forge test suite
 *
 * This file tests the Forge JIT backend integration with XAD.
 * Tests native code generation via Forge for automatic differentiation.
 *
 * https://github.com/da-roth/xad-forge
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

// f2: Function with supported math operations
// Uses: sin, cos, exp, log, sqrt, abs
template <class T>
T f2(const T& x)
{
    using std::sin; using std::cos; using std::exp; using std::log;
    using std::sqrt; using std::abs;

    T result = sin(x) + cos(x) * 2.0;
    result = result + exp(x / 10.0) + log(x + 5.0);
    result = result + sqrt(x + 1.0);
    result = result + abs(x - 1.0) + x * x;
    result = result + 1.0 / (x + 2.0);
    return result;
}

// f3ABool: Branching with ABool::If for trackable branches
xad::AD f3ABool(const xad::AD& x)
{
    return xad::less(x, 2.0).If(2.0 * x, 10.0 * x);
}

double f3ABool_double(double x)
{
    return (x < 2.0) ? 2.0 * x : 10.0 * x;
}

} // anonymous namespace

class ForgeBackendTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ForgeBackendTest, LinearFunction)
{
    std::vector<double> inputs = {2.0, 0.5, -1.0};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
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
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with ForgeBackend (native JIT)
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        xad::JITCompiler<double, 1> jit(
            std::make_unique<xad::forge::ForgeBackend>());

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
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f1(inputs[i]);
        EXPECT_NEAR(expected, tapeOutputs[i], 1e-10);
        EXPECT_NEAR(expected, forgeOutputs[i], 1e-10);
        EXPECT_NEAR(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

TEST_F(ForgeBackendTest, MathFunctions)
{
    std::vector<double> inputs = {2.0, 0.5};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
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
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with ForgeBackend
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        xad::JITCompiler<double, 1> jit(
            std::make_unique<xad::forge::ForgeBackend>());

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
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f2(inputs[i]);
        EXPECT_NEAR(expected, tapeOutputs[i], 1e-10);
        EXPECT_NEAR(expected, forgeOutputs[i], 1e-10);
        EXPECT_NEAR(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

TEST_F(ForgeBackendTest, ABoolBranching)
{
    std::vector<double> inputs = {1.0, 3.0};

    // Compute with Tape
    std::vector<double> tapeOutputs, tapeDerivatives;
    {
        xad::Tape<double> tape;
        for (double input : inputs)
        {
            xad::AD x(input);
            tape.registerInput(x);
            tape.newRecording();
            xad::AD y = f3ABool(x);
            tape.registerOutput(y);
            derivative(y) = 1.0;
            tape.computeAdjoints();
            tapeOutputs.push_back(value(y));
            tapeDerivatives.push_back(derivative(x));
            tape.clearAll();
        }
    }

    // Compute with ForgeBackend
    std::vector<double> forgeOutputs, forgeDerivatives;
    {
        xad::JITCompiler<double, 1> jit(
            std::make_unique<xad::forge::ForgeBackend>());

        xad::AD x(inputs[0]);
        jit.registerInput(x);
        jit.newRecording();
        xad::AD y = f3ABool(x);
        jit.registerOutput(y);
        jit.compile();

        for (double input : inputs)
        {
            value(x) = input;
            double output;
            jit.forward(&output, 1);
            forgeOutputs.push_back(output);

            jit.clearDerivatives();
            derivative(y) = 1.0;
            jit.computeAdjoints();
            forgeDerivatives.push_back(derivative(x));
        }
    }

    // Compare results
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        double expected = f3ABool_double(inputs[i]);
        EXPECT_NEAR(expected, tapeOutputs[i], 1e-10);
        EXPECT_NEAR(expected, forgeOutputs[i], 1e-10);
        EXPECT_NEAR(tapeDerivatives[i], forgeDerivatives[i], 1e-10);
    }
}

TEST_F(ForgeBackendTest, BasicInstantiation)
{
    xad::JITCompiler<double, 1> jit(
        std::make_unique<xad::forge::ForgeBackend>());

    xad::AD x(2.0);
    jit.registerInput(x);
    jit.newRecording();
    xad::AD y = x * x + 3.0 * x;  // f(x) = x^2 + 3x, f'(x) = 2x + 3
    jit.registerOutput(y);
    jit.compile();

    double output;
    jit.forward(&output, 1);
    EXPECT_NEAR(10.0, output, 1e-10);  // f(2) = 4 + 6 = 10

    value(x) = 5.0;
    jit.forward(&output, 1);
    EXPECT_NEAR(40.0, output, 1e-10);  // f(5) = 25 + 15 = 40

    jit.clearDerivatives();
    derivative(y) = 1.0;
    jit.computeAdjoints();
    EXPECT_NEAR(13.0, derivative(x), 1e-10);  // f'(5) = 10 + 3 = 13
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
