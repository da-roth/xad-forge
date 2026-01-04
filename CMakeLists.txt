##############################################################################
#
#  xad-forge - Forge JIT backend for XAD automatic differentiation
#
#  This library provides JIT compilation backends using Forge for XAD.
#  It converts xad::JITGraph to forge::Graph and compiles to native code.
#
#  All backends use the Forge C API for binary compatibility across compilers.
#
#  Copyright (c) 2025 The xad-forge Authors
#  https://github.com/da-roth/xad-forge
#  SPDX-License-Identifier: Zlib
#
##############################################################################

cmake_minimum_required(VERSION 3.20)
project(xad-forge VERSION 0.1.0 LANGUAGES CXX)

option(XAD_FORGE_BUILD_TESTS "Build xad-forge tests" OFF)
option(XAD_FORGE_BUILD_SAMPLES "Build xad-forge samples" OFF)
option(XAD_FORGE_USE_STATIC_RUNTIME "Use static runtime library (/MT) instead of dynamic (/MD) on MSVC" OFF)

# Configure MSVC runtime for xad-forge targets
if(MSVC)
    if(XAD_FORGE_USE_STATIC_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        message(STATUS "xad-forge: Using static MSVC runtime (/MT, /MTd)")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        message(STATUS "xad-forge: Using dynamic MSVC runtime (/MD, /MDd)")
    endif()
endif()

##############################################################################
# Find dependencies
##############################################################################

# Find XAD (from xad-jit fork which includes JIT support)
# First check if xad target already exists (added as subdirectory)
if(NOT TARGET XAD::xad)
    # Try to find xad-jit as a sibling directory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../xad-jit")
        message(STATUS "xad-forge: Using xad-jit from sibling directory")
        set(XAD_ENABLE_JIT ON CACHE BOOL "Enable JIT support" FORCE)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../xad-jit" "${CMAKE_CURRENT_BINARY_DIR}/xad-jit")
    else()
        find_package(XAD CONFIG REQUIRED)
    endif()
endif()

# Find Forge C API
set(FORGE_FOUND FALSE)

# Option 1: Check if forge_capi was added as subdirectory
if(TARGET forge_capi)
    message(STATUS "xad-forge: Found forge_capi target (subdirectory mode)")
    set(FORGE_FOUND TRUE)
    set(FORGE_TARGET forge_capi)
    get_target_property(FORGE_CAPI_SOURCE_DIR forge_capi SOURCE_DIR)
endif()

# Option 2: Try find_package for pre-built ForgeCAPI
if(NOT FORGE_FOUND)
    find_package(ForgeCAPI CONFIG QUIET)
    if(ForgeCAPI_FOUND)
        message(STATUS "xad-forge: Found ForgeCAPI package (pre-built mode)")
        set(FORGE_FOUND TRUE)
        set(FORGE_TARGET Forge::forge_capi)
    endif()
endif()

if(NOT FORGE_FOUND)
    message(FATAL_ERROR "xad-forge: Forge C API not found. Please provide forge_capi via subdirectory or CMAKE_PREFIX_PATH.")
endif()

##############################################################################
# Create xad-forge interface library
##############################################################################

add_library(xad-forge INTERFACE)
add_library(XADForge::xad-forge ALIAS xad-forge)

target_include_directories(xad-forge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xad-forge INTERFACE
    XAD::xad
    ${FORGE_TARGET}
)

# Add C API header directory for subdirectory mode
if(FORGE_CAPI_SOURCE_DIR)
    target_include_directories(xad-forge INTERFACE
        $<BUILD_INTERFACE:${FORGE_CAPI_SOURCE_DIR}>
    )
endif()

message(STATUS "xad-forge: Configured with Forge C API, target ${FORGE_TARGET}")

##############################################################################
# Samples
##############################################################################

if(XAD_FORGE_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

##############################################################################
# Tests
##############################################################################

if(XAD_FORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

##############################################################################
# Installation
##############################################################################

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xad-forge
    EXPORT xad-forge-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/xad-forge
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT xad-forge-targets
    FILE xad-forge-targets.cmake
    NAMESPACE XADForge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xad-forge
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xad-forge-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/xad-forge-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xad-forge
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/xad-forge-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/xad-forge-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/xad-forge-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xad-forge
)
