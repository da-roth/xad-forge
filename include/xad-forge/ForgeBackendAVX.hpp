#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendAVX - AVX2 backend for 4-path batching using Forge
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
//
//  This backend processes 4 Monte Carlo paths per kernel execution using
//  AVX2 SIMD instructions (256-bit YMM registers = 4 doubles).
//
//  This header provides ForgeBackendAVX as an alias for ForgeBackendAVX_CAPI,
//  which uses the Forge C API for binary compatibility across compilers.
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

#include <xad-forge/ForgeBackendAVX_CAPI.hpp>

namespace xad
{
namespace forge
{

using ForgeBackendAVX = ForgeBackendAVX_CAPI;

}  // namespace forge
}  // namespace xad
