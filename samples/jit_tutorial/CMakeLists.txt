##############################################################################
#
#  xad-forge JIT Tutorial Sample
#
#  Demonstrates:
#    - ABool::If for trackable branches with JIT
#    - ForgeBackend for native code compilation
#    - AVXBackend for 4-way SIMD evaluation
#    - Compile-once, evaluate-many pattern
#
#  Copyright (c) 2025 The xad-forge Authors
#  SPDX-License-Identifier: Zlib
#
##############################################################################

add_executable(xad-forge-jit-tutorial main.cpp)

target_link_libraries(xad-forge-jit-tutorial PRIVATE
    xad-forge
)

# Ensure C++17 for structured bindings and other features
target_compile_features(xad-forge-jit-tutorial PRIVATE cxx_std_17)

# AVX2 flags for AVXBackend
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i686)")
    if(MSVC)
        target_compile_options(xad-forge-jit-tutorial PRIVATE /arch:AVX2)
    else()
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(xad-forge-jit-tutorial PRIVATE -mavx2)
        endif()
    endif()
endif()
