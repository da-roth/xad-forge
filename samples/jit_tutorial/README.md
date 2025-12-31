# JIT Tutorial: Branching with Forge Backend

This tutorial extends the [XAD JIT tutorial](https://github.com/auto-differentiation/xad/tree/main/samples/jit_tutorial) to demonstrate using the Forge JIT backend for native code compilation.

## Overview

When using XAD's JIT compiler, the computation graph is recorded once and can be re-evaluated many times with different inputs. This "compile once, evaluate many" pattern is efficient for scenarios like Monte Carlo simulations or sensitivity analysis.

However, there's an important caveat with **branching code**: plain C++ `if` statements are evaluated at record time, which means the branch decision gets "baked in" to the graph. If the input changes such that a different branch should be taken, the JIT-compiled code will still follow the original branch — producing incorrect results.

The solution is to use `ABool::If`, which records a conditional node into the graph, allowing the correct branch to be selected at runtime.

## What This Tutorial Demonstrates

1. **The branching problem** — Plain C++ `if` fails when inputs change
2. **The solution** — `ABool::If` records both branches with runtime selection
3. **ScalarBackend** — Forge compiles to native x86 code with correct behavior
4. **AVXBackend** — Evaluate 4 inputs in parallel using SIMD

## The Problem: Plain C++ `if` with JIT

Consider this piecewise function:

```cpp
template <class AD>
AD piecewise_plain_if(const AD& x)
{
    if (xad::value(x) < 2.0)
        return 1.0 * x;
    return 7.0 * x;
}
```

When recorded with `x=1`:
- The condition `x < 2` evaluates to `true`
- Only the `1.0 * x` branch is recorded
- The `7.0 * x` branch is never recorded

When replayed with `x=3`:
- The graph only contains `y = 1.0 * x`
- Returns `y = 3` instead of `y = 21` — **incorrect!**

## The Solution: `ABool::If`

```cpp
template <class AD>
AD piecewise_abool_if(const AD& x)
{
    auto cond = xad::less(x, 2.0);
    AD t = 1.0 * x;   // true branch
    AD f = 7.0 * x;   // false branch
    return cond.If(t, f);
}
```

This records:
- A comparison node for `x < 2`
- Both branch computations
- A conditional select node

At runtime, the correct branch is selected based on the actual input value.

## Using ForgeBackend

The default XAD JIT uses an interpreter to execute the recorded graph. xad-forge provides backends that compile the graph to native x86 machine code using the [Forge](https://github.com/da-roth/forge) JIT compiler.

### ScalarBackend

Processes one input at a time with native code:

```cpp
#include <xad-forge/ForgeBackends.hpp>

// Create JIT compiler with ForgeBackend
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());

xad::AReal<double, 1> x = 1.0;
jit.registerInput(x);
jit.newRecording();
auto y = piecewise_abool_if(x);
jit.registerOutput(y);
jit.compile();  // Compiles to native x86!

// Evaluate with different inputs
double out;
value(x) = 3.0;
jit.forward(&out, 1);  // out = 21
```

### AVXBackend

Processes 4 inputs simultaneously using AVX2 SIMD instructions:

```cpp
// Build graph using JITCompiler
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());
// ... record graph ...

// Compile for AVX
xad::forge::AVXBackend avx;
avx.compile(jit.getGraph());

// Evaluate 4 inputs at once
double inputs[4] = {0.5, 1.5, 2.5, 3.5};
avx.setInputLanes(0, inputs);

double outputs[4];
double outputAdjoints[4] = {1.0, 1.0, 1.0, 1.0};
std::vector<std::array<double, 4>> inputGradients(1);
avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);

// outputs = {0.5, 1.5, 17.5, 24.5}
// gradients = {1, 1, 7, 7}
```

Note how different lanes can take different branches: inputs 0.5 and 1.5 take the true branch (`1*x`), while 2.5 and 3.5 take the false branch (`7*x`).

## Expected Output

```
=============================================================================
  xad-forge JIT Tutorial: Branching with Forge Backend
=============================================================================

Comparing JIT backends for the following two functions:
f1(x) = (x < 2) ? (1*x) : (7*x)          (plain C++ if)
f2(x) = less(x,2).If(1*x, 7*x)           (ABool::If)
(f2 is semantically the same as f1, but expressed in a way JIT can record)

Settings:
  Record with x=1, replay with x=3
  Expected: x=1 -> y=1, dy/dx=1 | x=3 -> y=21, dy/dx=7

1) JIT (default) with plain C++ if:
   x=1: y=1, dy/dx=1
   x=3: y=3, dy/dx=1  (WRONG! expected y=21, dy/dx=7)

2) JIT (default) with ABool::If:
   x=1: y=1, dy/dx=1
   x=3: y=21, dy/dx=7

3) JIT with ForgeBackend (ScalarBackend) and ABool::If:
   x=1: y=1, dy/dx=1
   x=3: y=21, dy/dx=7

4) AVX Backend - 4 inputs in parallel with ABool::If:
   Inputs:  x = {0.5, 1.5, 2.5, 3.5}
   Outputs: y = {0.5, 1.5, 17.5, 24.5}
   dy/dx:       {1, 1, 7, 7}
   Expected: y = {0.5, 1.5, 17.5, 24.5}, dy/dx = {1, 1, 7, 7}

Summary:
Scenario                       x         y     dy/dx  Note
----------------------------------------------------------------------
JIT default, plain if          1         1         1
JIT default, plain if          3         3         1  WRONG
JIT default, ABool::If         1         1         1
JIT default, ABool::If         3        21         7
Forge ScalarBackend            1         1         1
Forge ScalarBackend            3        21         7
Forge AVXBackend             0.5       0.5         1
Forge AVXBackend             1.5       1.5         1
Forge AVXBackend             2.5      17.5         7
Forge AVXBackend             3.5      24.5         7

Key points:
  - Plain C++ if: branch is baked in at record time (incorrect for JIT)
  - ABool::If: records a conditional node (correct for JIT)
  - ScalarBackend: compiles to native x86 code, same correct behavior
  - AVXBackend: evaluates 4 inputs in parallel using SIMD
```

## Building

```bash
cmake -B build \
  -DXAD_FORGE_BUILD_SAMPLES=ON \
  -DCMAKE_PREFIX_PATH=/path/to/install

cmake --build build

./build/samples/jit_tutorial/xad-forge-jit-tutorial
```

## See Also

- [XAD JIT Tutorial](https://github.com/auto-differentiation/xad/tree/main/samples/jit_tutorial) — Original tutorial demonstrating Tape vs JIT
- [Forge](https://github.com/da-roth/forge) — The JIT compiler backend
- [xad-forge](https://github.com/da-roth/xad-forge) — Bridge between XAD and Forge
