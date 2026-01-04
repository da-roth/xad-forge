/*******************************************************************************
 *
 *   xad-forge JIT Tutorial: Branching with Forge Backend
 *
 *   This extends the XAD jit_tutorial sample to demonstrate using the Forge
 *   JIT backend for native code compilation.
 *
 *   Demonstrates:
 *   - XAD's default interpreter backend vs Forge's native code backend
 *   - ABool::If for trackable branches that work with JIT
 *   - Compile-once, evaluate-many pattern with ForgeBackend
 *
 *   Copyright (c) 2025 The xad-forge Authors
 *   https://github.com/da-roth/xad-forge
 *   SPDX-License-Identifier: Zlib
 *
 ******************************************************************************/

#include <xad-forge/ForgeBackend.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>
#include <XAD/XAD.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

// f1: Plain C++ if - decision is made at record time based on current value.
// WARNING: This will NOT work correctly with JIT when input changes!
template <class AD>
AD piecewise_plain_if(const AD& x)
{
    if (xad::value(x) < 2.0)
        return 1.0 * x;
    return 7.0 * x;
}

// f2: ABool::If - records a conditional node so the branch can vary at runtime.
// This is the correct way to write branching code for JIT.
template <class AD>
AD piecewise_abool_if(const AD& x)
{
    auto cond = xad::less(x, 2.0);
    AD t = 1.0 * x;
    AD f = 7.0 * x;
    return cond.If(t, f);
}

}  // namespace

int main()
{
    std::cout << "=============================================================================\n";
    std::cout << "  xad-forge JIT Tutorial: Branching with Forge Backend\n";
    std::cout << "=============================================================================\n\n";

    std::cout << "Comparing JIT backends for the following two functions:\n";
    std::cout << "f1(x) = (x < 2) ? (1*x) : (7*x)          (plain C++ if)\n";
    std::cout << "f2(x) = less(x,2).If(1*x, 7*x)           (ABool::If)\n";
    std::cout << "(f2 is semantically the same as f1, but expressed in a way JIT can record)\n\n";

    std::cout << "Settings:\n";
    std::cout << "  Record with x=1, replay with x=3\n";
    std::cout << "  Expected: x=1 -> y=1, dy/dx=1 | x=3 -> y=21, dy/dx=7\n";

    struct Row
    {
        const char* scenario;
        double x;
        double y;
        double dydx;
        const char* note;
    };
    std::vector<Row> rows;

    // -------------------------------------------------------------------------
    // 1) JIT (default interpreter) with plain if - demonstrates the problem
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        std::cout << "\n1) JIT (default) with plain C++ if:\n";

        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;  // Record with x=1, so x<2 is true
        jit.registerInput(x);
        jit.newRecording();
        AD y = piecewise_plain_if(x);
        jit.registerOutput(y);
        jit.compile();

        // Evaluate at x=1 (same as recording)
        double out = 0.0;
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "   x=1: y=" << out << ", dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT default, plain if", 1.0, out, jit.getDerivative(x.getSlot()), ""});

        // Evaluate at x=3 (different branch should be taken)
        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "   x=3: y=" << out << ", dy/dx=" << jit.getDerivative(x.getSlot())
                  << "  (WRONG! expected y=21, dy/dx=7)\n";
        rows.push_back({"JIT default, plain if", 3.0, out, jit.getDerivative(x.getSlot()), "WRONG"});
    }

    // -------------------------------------------------------------------------
    // 2) JIT (default interpreter) with ABool::If - correct approach
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        std::cout << "\n2) JIT (default) with ABool::If:\n";

        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;
        jit.registerInput(x);
        jit.newRecording();
        AD y = piecewise_abool_if(x);
        jit.registerOutput(y);
        jit.compile();

        // Evaluate at x=1
        double out = 0.0;
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "   x=1: y=" << out << ", dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT default, ABool::If", 1.0, out, jit.getDerivative(x.getSlot()), ""});

        // Evaluate at x=3
        x = 3.0;
        jit.clearDerivatives();
        jit.forward(&out);
        jit.setDerivative(y.getSlot(), 1.0);
        jit.computeAdjoints();
        std::cout << "   x=3: y=" << out << ", dy/dx=" << jit.getDerivative(x.getSlot()) << "\n";
        rows.push_back({"JIT default, ABool::If", 3.0, out, jit.getDerivative(x.getSlot()), ""});
    }

    // -------------------------------------------------------------------------
    // 3) JIT with ForgeBackend (native code) and ABool::If
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        std::cout << "\n3) JIT with ForgeBackend (scalar) and ABool::If:\n";

        // Record graph using JITCompiler
        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;
        jit.registerInput(x);
        jit.newRecording();
        AD y = piecewise_abool_if(x);
        jit.registerOutput(y);

        // Compile to native x86 code via Forge
        xad::forge::ForgeBackend backend;
        backend.compile(jit.getGraph());

        // Evaluate at x=1
        double input = 1.0;
        backend.setInput(0, &input);
        double output;
        double inputGradient;
        backend.forwardAndBackward(&output, &inputGradient);
        std::cout << "   x=1: y=" << output << ", dy/dx=" << inputGradient << "\n";
        rows.push_back({"Forge ForgeBackend", 1.0, output, inputGradient, ""});

        // Evaluate at x=3
        input = 3.0;
        backend.setInput(0, &input);
        backend.forwardAndBackward(&output, &inputGradient);
        std::cout << "   x=3: y=" << output << ", dy/dx=" << inputGradient << "\n";
        rows.push_back({"Forge ForgeBackend", 3.0, output, inputGradient, ""});
    }

    // -------------------------------------------------------------------------
    // 4) AVX Backend - evaluate 4 inputs simultaneously
    // -------------------------------------------------------------------------
    {
        using AD = xad::AReal<double, 1>;

        std::cout << "\n4) ForgeBackendAVX - 4 inputs in parallel with ABool::If:\n";

        // First build the graph using JITCompiler
        xad::JITCompiler<double, 1> jit;
        AD x = 1.0;
        jit.registerInput(x);
        jit.newRecording();
        AD y = piecewise_abool_if(x);
        jit.registerOutput(y);

        // Compile AVX backend from the JIT graph
        xad::forge::ForgeBackendAVX avx;
        avx.compile(jit.getGraph());

        // Evaluate 4 different inputs simultaneously
        // x = {0.5, 1.5, 2.5, 3.5} - first two take true branch, last two take false branch
        constexpr int BATCH_SIZE = xad::forge::ForgeBackendAVX::VECTOR_WIDTH;
        double inputBatch[BATCH_SIZE] = {0.5, 1.5, 2.5, 3.5};
        avx.setInput(0, inputBatch);

        double outputs[BATCH_SIZE];
        double inputGradients[BATCH_SIZE];
        avx.forwardAndBackward(outputs, inputGradients);

        std::cout << "   Inputs:  x = {0.5, 1.5, 2.5, 3.5}\n";
        std::cout << "   Outputs: y = {" << outputs[0] << ", " << outputs[1]
                  << ", " << outputs[2] << ", " << outputs[3] << "}\n";
        std::cout << "   dy/dx:       {" << inputGradients[0] << ", " << inputGradients[1]
                  << ", " << inputGradients[2] << ", " << inputGradients[3] << "}\n";
        std::cout << "   Expected: y = {0.5, 1.5, 17.5, 24.5}, dy/dx = {1, 1, 7, 7}\n";

        for (int i = 0; i < BATCH_SIZE; ++i)
        {
            rows.push_back({"Forge ForgeBackendAVX", inputBatch[i], outputs[i], inputGradients[i], ""});
        }
    }

    // -------------------------------------------------------------------------
    // Summary table
    // -------------------------------------------------------------------------
    std::cout << "\nSummary:\n";
    std::cout << std::left << std::setw(26) << "Scenario"
              << std::right << std::setw(6) << "x"
              << std::setw(10) << "y"
              << std::setw(10) << "dy/dx"
              << "  " << "Note"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    for (const auto& r : rows)
    {
        std::cout << std::left << std::setw(26) << r.scenario
                  << std::right << std::setw(6) << r.x
                  << std::setw(10) << r.y
                  << std::setw(10) << r.dydx
                  << "  " << r.note
                  << "\n";
    }

    std::cout << "\nKey points:\n";
    std::cout << "  - Plain C++ if: branch is baked in at record time (incorrect for JIT)\n";
    std::cout << "  - ABool::If: records a conditional node (correct for JIT)\n";
    std::cout << "  - ForgeBackend: compiles to native x86 code, same correct behavior\n";
    std::cout << "  - ForgeBackendAVX: evaluates 4 inputs in parallel using SIMD\n";

    return 0;
}
