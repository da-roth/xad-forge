##############################################################################
#
#  xad-forge tests
#
#  Test executables:
#    - xad-forge-scalar-tests: Tests ForgeBackend (ScalarBackend)
#    - xad-forge-avx-tests: Tests ForgeBackendAVX (AVXBackend)
#
##############################################################################

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

##############################################################################
# Scalar Backend Tests (SSE2 scalar mode)
##############################################################################

add_executable(xad-forge-scalar-tests
    scalar_backend_test.cpp
)

target_link_libraries(xad-forge-scalar-tests PRIVATE
    xad-forge
    GTest::gtest
)

include(GoogleTest)
gtest_discover_tests(xad-forge-scalar-tests)

##############################################################################
# AVX Backend Tests (AVX2 packed mode - 4 doubles)
# Only built on x86/x64 with AVX2 support
##############################################################################

# Check for AVX2 support
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i686)")
    if(MSVC)
        set(AVX2_FLAG "/arch:AVX2")
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set(AVX2_FLAG "-mavx2")
        endif()
    endif()

    if(AVX2_FLAG)
        add_executable(xad-forge-avx-tests
            avx_backend_test.cpp
        )

        target_link_libraries(xad-forge-avx-tests PRIVATE
            xad-forge
            GTest::gtest
        )

        # AVX2 compile flags for the test file
        target_compile_options(xad-forge-avx-tests PRIVATE ${AVX2_FLAG})

        gtest_discover_tests(xad-forge-avx-tests)

        message(STATUS "xad-forge: AVX backend tests enabled with ${AVX2_FLAG}")
    else()
        message(STATUS "xad-forge: AVX backend tests disabled (no AVX2 support)")
    endif()
else()
    message(STATUS "xad-forge: AVX backend tests disabled (non-x86 platform)")
endif()
