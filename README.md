# xad-forge

Forge JIT backends for [XAD](https://github.com/auto-differentiation/xad) automatic differentiation.

This library implements JIT backends for XAD using the [Forge](https://github.com/da-roth/forge) C API as the code generation engine. When XAD records a computation graph, xad-forge compiles it to native x86-64 machine code for fast re-evaluation.

All backends use the Forge C API for binary compatibility across compilers.

## When to Use JIT

XAD's default tape-based AAD is highly optimized for workflows where each computation is evaluated once or a few times. It uses expression templates and avoids recording overhead where possible.

However, for workflows requiring **repeated evaluation with different inputs**—such as Monte Carlo simulation, risk scenarios, or XVA calculations—a different approach is more efficient: record the computation once into a graph, compile it to native machine code, then re-evaluate as many times as needed.

This JIT approach has an upfront compilation cost, but each subsequent evaluation is significantly faster. The crossover point is typically around 5-20 evaluations depending on the workflow, after which the JIT approach outperforms tape replay.

**Use tape-based AAD when:**
- Each computation is evaluated once or a few times
- The computation structure changes between evaluations

**Use JIT when:**
- Evaluating the same computation many times with different inputs
- Running Monte Carlo simulations
- Computing risk sensitivities across many scenarios
- XVA and other batch pricing workloads

## Backends

xad-forge provides two backends:

| Backend | Description | Use case |
|---------|-------------|----------|
| `ScalarBackend` | Compiles to scalar x86-64 code | General purpose, replaces interpreter |
| `AVXBackend` | Compiles to AVX2 SIMD code | Batch evaluation, 4 inputs in parallel |

## Usage

XAD's JIT support allows recording a computation once and re-evaluating it with different inputs. By default, XAD uses an interpreter. xad-forge provides compiled backends instead.

### ScalarBackend

Replace XAD's default interpreter with Forge-compiled native code:

```cpp
#include <xad-forge/ForgeBackends.hpp>
#include <XAD/XAD.hpp>

// Use ForgeBackend instead of default interpreter
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());

xad::AReal<double, 1> x = 2.0;
jit.registerInput(x);
jit.newRecording();

auto y = x * x + 3.0 * x + 2.0;

jit.registerOutput(y);
jit.compile();  // Compiles to native x86-64

// Re-evaluate with different inputs
double out;
for (double input : {1.0, 2.0, 3.0, 4.0, 5.0})
{
    value(x) = input;
    jit.forward(&out);

    jit.clearDerivatives();
    derivative(y) = 1.0;
    jit.computeAdjoints();

    std::cout << "x=" << input << " y=" << out << " dy/dx=" << derivative(x) << "\n";
}
```

### AVXBackend

For batch evaluation, AVXBackend compiles to AVX2 SIMD instructions, processing 4 inputs in parallel:

```cpp
#include <xad-forge/ForgeBackends.hpp>

// Record graph using ScalarBackend
xad::JITCompiler<double, 1> jit(
    std::make_unique<xad::forge::ScalarBackend>());
// ... register inputs, record computation, register outputs ...

// Compile same graph for AVX2
xad::forge::AVXBackend avx;
avx.compile(jit.getGraph());

// Evaluate 4 inputs at once
double inputs[4] = {1.0, 2.0, 3.0, 4.0};
avx.setInput(0, inputs);

double outputs[4];
double inputGradients[4];
avx.forwardAndBackward(outputs, inputGradients);
```

## Building

xad-forge requires the Forge C API library (`forge_capi`).

```cmake
# Build Forge C API first
add_subdirectory(forge/api/c)

# Then XAD with JIT enabled
set(XAD_ENABLE_JIT ON)
add_subdirectory(xad)

# Finally xad-forge
add_subdirectory(xad-forge)

target_link_libraries(your_target PRIVATE XADForge::xad-forge)
```

## Dependencies

- [XAD](https://github.com/auto-differentiation/xad) with JIT enabled (`XAD_ENABLE_JIT=ON`)
- [Forge C API](https://github.com/da-roth/forge) (`forge_capi` target)
- CMake 3.20+

## License

xad-forge is licensed under the [zlib license](LICENSE.md).

[XAD](https://github.com/auto-differentiation/xad) is licensed under AGPL-3.0, which may affect your combined work.

## See Also

- [XAD](https://github.com/auto-differentiation/xad) — Automatic differentiation library
- [Forge](https://github.com/da-roth/forge) — JIT compiler for computation graphs
