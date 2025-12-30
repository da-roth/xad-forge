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
//  Copyright (c) 2025 The xad-forge Authors
//  https://github.com/da-roth/xad-forge
//
//  This software is provided 'as-is', without any express or implied
//  warranty. In no event will the authors be held liable for any damages
//  arising from the use of this software.
//
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//
//  1. The origin of this software must not be misrepresented; you must not
//     claim that you wrote the original software. If you use this software
//     in a product, an acknowledgment in the product documentation would be
//     appreciated but is not required.
//  2. Altered source versions must be plainly marked as such, and must not
//     be misrepresented as being the original software.
//  3. This notice may not be removed or altered from any source distribution.
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
