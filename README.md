# xad-forge

Forge JIT backend for [XAD](https://github.com/da-roth/xad-jit) automatic differentiation.

xad-forge provides JIT compilation backends that compile XAD's computation graphs to native x86-64 machine code using the [Forge](https://github.com/da-roth/forge) JIT compiler. This enables fast re-evaluation of recorded computations — ideal for Monte Carlo simulations, sensitivity analysis, and model calibration.

## Features

- **ScalarBackend** — Compiles to native x86-64 code using SSE2 scalar instructions
- **AVXBackend** — Compiles to AVX2 SIMD code, evaluating 4 inputs in parallel using 256-bit YMM registers
- **C API mode** — Optional binary-compatible interface for cross-compiler usage
- **Header-only** — Simple integration, just include and link

## How It Works

xad-forge implements the `xad::JITBackend` interface, acting as a bridge between XAD and Forge. It translates XAD's `JITGraph` to Forge's `Graph` format:

```
xad::JITGraph  →  xad-forge  →  forge::Graph  →  Native x86-64 code
   (XAD)          (bridge)       (Forge)          (executable)
```

This keeps XAD and Forge independent — neither knows about the other. The translation is isolated in xad-forge, making it reusable across any project.

## Quick Start

```cpp
#include <xad-forge/ForgeBackends.hpp>
#include <XAD/XAD.hpp>

// Create JIT compiler with ForgeBackend instead of default interpreter
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());

xad::AReal<double, 1> x = 2.0;
jit.registerInput(x);
jit.newRecording();

// Record computation
auto y = x * x + 3.0 * x + 2.0;

jit.registerOutput(y);
jit.compile();  // Compiles to native x86-64!

// Re-evaluate with different inputs
double out;
for (double input : {1.0, 2.0, 3.0, 4.0, 5.0})
{
    value(x) = input;
    jit.forward(&out, 1);

    jit.clearDerivatives();
    derivative(y) = 1.0;
    jit.computeAdjoints();

    std::cout << "x=" << input << " y=" << out << " dy/dx=" << derivative(x) << "\n";
}
```

## AVXBackend — 4-Way SIMD

For batch evaluation, AVXBackend uses AVX2 instructions to process 4 inputs simultaneously using 256-bit YMM registers (4 packed doubles):

```cpp
#include <xad-forge/ForgeBackends.hpp>

// Record graph (using any backend)
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());
// ... register inputs, record computation, register outputs ...

// Compile for AVX2
xad::forge::AVXBackend avx;
avx.compile(jit.getGraph());

// Evaluate 4 inputs at once
double inputs[4] = {1.0, 2.0, 3.0, 4.0};
avx.setInputLanes(0, inputs);

double outputs[4];
double outputAdjoints[4] = {1.0, 1.0, 1.0, 1.0};
std::vector<std::array<double, 4>> inputGradients(1);
avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);
```

This is particularly useful for Monte Carlo simulations where many independent paths can be evaluated in parallel.

## Branching with JIT

When using JIT compilation, plain C++ `if` statements are evaluated at record time — the branch decision gets "baked in" to the graph. Use `ABool::If` instead to record a conditional node:

```cpp
// WRONG: Branch baked at record time
template <class AD>
AD piecewise_wrong(const AD& x)
{
    if (xad::value(x) < 2.0)
        return 1.0 * x;
    return 7.0 * x;
}

// CORRECT: Conditional node recorded
template <class AD>
AD piecewise_correct(const AD& x)
{
    auto cond = xad::less(x, 2.0);
    return cond.If(1.0 * x, 7.0 * x);
}
```

With AVXBackend, different SIMD lanes can take different branches — the conditional is evaluated per-lane at runtime.

See the [JIT Tutorial](samples/jit_tutorial/) for a complete example.

## Building

### As a Subdirectory

```cmake
add_subdirectory(xad-jit)
add_subdirectory(forge)
add_subdirectory(xad-forge)

target_link_libraries(your_target PRIVATE XADForge::xad-forge)
```

### With Pre-built Dependencies

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/path/to/xad;/path/to/forge" \
  -DXAD_FORGE_BUILD_TESTS=ON \
  -DXAD_FORGE_BUILD_SAMPLES=ON

cmake --build build
ctest --test-dir build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `XAD_FORGE_USE_CAPI` | OFF | Use Forge C API for binary compatibility |
| `XAD_FORGE_BUILD_TESTS` | OFF | Build test executables |
| `XAD_FORGE_BUILD_SAMPLES` | OFF | Build sample executables |

## Backends

### ScalarBackend

- Compiles to SSE2 scalar instructions
- Single input/output per evaluation
- Implements `xad::JITBackend` interface
- Works with `xad::JITCompiler`

### AVXBackend

- Compiles to AVX2 packed instructions (256-bit YMM registers)
- 4 inputs/outputs per evaluation (SIMD lanes)
- Standalone API with lane-based input/output
- Requires AVX2-capable CPU (Intel Haswell or later, AMD Excavator or later)

### C API Backends

When `XAD_FORGE_USE_CAPI=ON`, the backends use Forge's C API instead of the C++ API. This provides binary compatibility when xad-forge and Forge are compiled with different compilers or compiler versions.

## Samples

| Sample | Description |
|--------|-------------|
| [jit_tutorial](samples/jit_tutorial/) | Branching and graph reuse with ScalarBackend and AVXBackend |

## Dependencies

- [XAD](https://github.com/da-roth/xad-jit) with JIT enabled (`XAD_ENABLE_JIT=ON`)
- [Forge](https://github.com/da-roth/forge)
- CMake 3.20+

## License

AGPL-3.0-or-later

## See Also

- [XAD](https://github.com/da-roth/xad-jit) — Automatic differentiation library
- [Forge](https://github.com/da-roth/forge) — JIT compiler for computation graphs
- [QuantLib-Risks-Cpp-Forge](https://github.com/da-roth/QuantLib-Risks-Cpp-Forge) — XAD integration with QuantLib
