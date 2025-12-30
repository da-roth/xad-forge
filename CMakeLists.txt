##############################################################################
#
#  xad-forge - Forge JIT backend for XAD automatic differentiation
#
#  This library provides JIT compilation backends using Forge for XAD.
#  It converts xad::JITGraph to forge::Graph and compiles to native code.
#
#  Copyright (c) 2025 The xad-forge Authors
#  https://github.com/da-roth/xad-forge
#  SPDX-License-Identifier: Zlib
#
##############################################################################

cmake_minimum_required(VERSION 3.20)
project(xad-forge VERSION 0.1.0 LANGUAGES CXX)

option(XAD_FORGE_USE_CAPI "Use Forge C API instead of C++ API for binary compatibility" OFF)
option(XAD_FORGE_BUILD_TESTS "Build xad-forge tests" OFF)
option(XAD_FORGE_BUILD_SAMPLES "Build xad-forge samples" OFF)
set(XAD_FORGE_FORGE_DIR "" CACHE PATH "Path to Forge source directory (for subdirectory build)")

##############################################################################
# Find dependencies
##############################################################################

# Find XAD
if(NOT TARGET XAD::xad)
    find_package(XAD CONFIG REQUIRED)
endif()

# Find Forge
set(FORGE_FOUND FALSE)

if(XAD_FORGE_USE_CAPI)
    ##########################################################################
    # C API Mode: Use forge_capi shared library for binary compatibility
    ##########################################################################
    message(STATUS "xad-forge: Using C API mode for binary compatibility")

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

else()
    ##########################################################################
    # C++ API Mode: Use original Forge library (requires matching compiler)
    ##########################################################################

    # Option 1: Check if forge was added as subdirectory
    if(TARGET forge)
        message(STATUS "xad-forge: Found forge target (subdirectory mode)")
        set(FORGE_FOUND TRUE)
        set(FORGE_TARGET forge)
    endif()

    # Option 2: If XAD_FORGE_FORGE_DIR is set, add it as a subdirectory
    if(NOT FORGE_FOUND AND XAD_FORGE_FORGE_DIR)
        if(EXISTS "${XAD_FORGE_FORGE_DIR}/CMakeLists.txt")
            message(STATUS "xad-forge: Adding Forge from ${XAD_FORGE_FORGE_DIR}")
            set(FORGE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            add_subdirectory("${XAD_FORGE_FORGE_DIR}" "${CMAKE_BINARY_DIR}/forge")
            if(TARGET forge)
                set(FORGE_FOUND TRUE)
                set(FORGE_TARGET forge)
            endif()
        else()
            message(WARNING "xad-forge: XAD_FORGE_FORGE_DIR set but no CMakeLists.txt found at ${XAD_FORGE_FORGE_DIR}")
        endif()
    endif()

    # Option 3: Try find_package for pre-built Forge
    if(NOT FORGE_FOUND)
        find_package(Forge CONFIG QUIET)
        if(Forge_FOUND)
            message(STATUS "xad-forge: Found Forge package (pre-built mode) version ${Forge_VERSION}")
            set(FORGE_FOUND TRUE)
            set(FORGE_TARGET Forge::forge)
        endif()
    endif()
endif()

if(NOT FORGE_FOUND)
    message(FATAL_ERROR "xad-forge: Forge not found. Please provide Forge via subdirectory or CMAKE_PREFIX_PATH.")
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

# Add C API compile definition if using C API mode
if(XAD_FORGE_USE_CAPI)
    target_compile_definitions(xad-forge INTERFACE XAD_FORGE_USE_CAPI=1)

    # Add C API header directory for subdirectory mode
    if(FORGE_CAPI_SOURCE_DIR)
        target_include_directories(xad-forge INTERFACE
            $<BUILD_INTERFACE:${FORGE_CAPI_SOURCE_DIR}>
        )
    endif()
else()
    # For C++ API subdirectory mode, add Forge source directories
    if(TARGET forge)
        get_target_property(FORGE_SOURCE_DIR forge SOURCE_DIR)
        if(FORGE_SOURCE_DIR)
            target_include_directories(xad-forge INTERFACE
                $<BUILD_INTERFACE:${FORGE_SOURCE_DIR}/src>
                $<BUILD_INTERFACE:${FORGE_SOURCE_DIR}/tools>
            )
        endif()
    endif()
endif()

message(STATUS "xad-forge: Configured with ${XAD_FORGE_USE_CAPI} C API mode, target ${FORGE_TARGET}")

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
