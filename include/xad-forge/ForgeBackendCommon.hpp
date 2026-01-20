#pragma once

//////////////////////////////////////////////////////////////////////////////
//
//  ForgeBackendCommon - Shared utilities for Forge backends
//
//  This file is part of xad-forge, providing Forge JIT compilation
//  as a backend for XAD automatic differentiation.
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

#include <forge_c_api.h>

#include <cstdlib>
#include <iostream>
#include <mutex>

namespace xad
{
namespace forge
{
namespace detail
{

/**
 * Thread-safe helper to load custom Forge backend from environment variable.
 *
 * Checks XAD_FORGE_BACKEND_PATH environment variable. If set, attempts to load
 * the specified shared library as a Forge backend. This is done only once per
 * process, regardless of how many backend instances are created.
 *
 * On failure, prints a warning to stderr but does not throw. The subsequent
 * instruction set selection will fail if the required backend isn't available.
 */
inline void loadCustomBackendFromEnv()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char* backendPath = std::getenv("XAD_FORGE_BACKEND_PATH");
        if (backendPath && backendPath[0] != '\0')
        {
            ForgeError err = forge_load_backend(backendPath);
            if (err != FORGE_SUCCESS)
            {
                std::cerr << "xad-forge: Warning: Failed to load custom backend from '"
                          << backendPath << "': " << forge_error_string(err) << std::endl;
            }
        }
    });
}

}  // namespace detail
}  // namespace forge
}  // namespace xad
