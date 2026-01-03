# xad-forge

Forge JIT backends for [XAD](https://github.com/auto-differentiation/xad) automatic differentiation.

This library implements JIT backends for XAD using the [Forge](https://github.com/da-roth/forge) C API as the code generation engine. When XAD records a computation graph, xad-forge compiles it to native x86-64 machine code for fast re-evaluation.

All backends use the Forge C API for binary compatibility across compilers.

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
    jit.forward(&out, 1);

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
avx.setInputLanes(0, inputs);

double outputs[4];
double outputAdjoints[4] = {1.0, 1.0, 1.0, 1.0};
std::vector<std::array<double, 4>> inputGradients(1);
avx.forwardAndBackward(outputAdjoints, outputs, inputGradients);
```

## Building

```cmake
add_subdirectory(xad)
add_subdirectory(forge)
add_subdirectory(xad-forge)

target_link_libraries(your_target PRIVATE XADForge::xad-forge)
```

## Dependencies

- [XAD](https://github.com/auto-differentiation/xad) with JIT enabled (`XAD_ENABLE_JIT=ON`)
- [Forge](https://github.com/da-roth/forge)
- CMake 3.20+

## License

xad-forge is licensed under the [zlib license](LICENSE.md).

[XAD](https://github.com/auto-differentiation/xad) is licensed under AGPL-3.0, which may affect your combined work.

## See Also

- [XAD](https://github.com/auto-differentiation/xad) — Automatic differentiation library
- [Forge](https://github.com/da-roth/forge) — JIT compiler for computation graphs
