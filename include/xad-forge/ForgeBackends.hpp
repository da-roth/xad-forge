#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackends.hpp - Backend type selection based on API mode
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This header provides unified type aliases that automatically select
//  the appropriate backend implementation:
//
//    XAD_FORGE_USE_CAPI=1: Uses C API backends (binary compatible)
//    XAD_FORGE_USE_CAPI=0: Uses C++ API backends (requires matching compiler)
//
//  Usage:
//    #include <xad-forge/ForgeBackends.hpp>
//    auto backend = std::make_unique<xad::forge::ScalarBackend>();
//    xad::forge::AVXBackend avxBackend;
//
//  https://github.com/da-roth/xad-forge
//
//////////////////////////////////////////////////////////////////////////////

#ifdef XAD_FORGE_USE_CAPI

// C API mode - binary compatible across compilers
#include <xad-forge/ForgeBackendCAPI.hpp>
#include <xad-forge/ForgeBackendAVX_CAPI.hpp>

namespace xad
{
namespace forge
{

using ScalarBackend = ForgeBackendCAPI;
using AVXBackend = ForgeBackendAVX_CAPI;

}  // namespace forge
}  // namespace xad

#else

// C++ API mode - requires matching compiler/ABI
#include <xad-forge/ForgeBackend.hpp>
#include <xad-forge/ForgeBackendAVX.hpp>

namespace xad
{
namespace forge
{

using ScalarBackend = ForgeBackend;
using AVXBackend = ForgeBackendAVX;

}  // namespace forge
}  // namespace xad

#endif
