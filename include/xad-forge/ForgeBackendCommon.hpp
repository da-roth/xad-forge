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

//=============================================================================
// Debug logging - enabled by XAD_FORGE_DEBUG environment variable
//=============================================================================

/// Check if debug logging is enabled
inline bool isDebugEnabled()
{
    static bool checked = false;
    static bool enabled = false;
    if (!checked)
    {
        const char* env = std::getenv("XAD_FORGE_DEBUG");
        enabled = (env && env[0] != '\0' && env[0] != '0');
        checked = true;
    }
    return enabled;
}

/// Log a simple message
inline void debugLog(const char* msg)
{
    if (isDebugEnabled())
        std::cerr << "[xad-forge-debug] " << msg << std::endl;
}

/// Log a message with a string value
inline void debugLog(const char* msg, const char* detail)
{
    if (isDebugEnabled())
        std::cerr << "[xad-forge-debug] " << msg << ": " << detail << std::endl;
}

/// Log a message with a numeric value
inline void debugLog(const char* msg, size_t value)
{
    if (isDebugEnabled())
        std::cerr << "[xad-forge-debug] " << msg << ": " << value << std::endl;
}

/// Log a message with a pointer value
inline void debugLog(const char* msg, void* ptr)
{
    if (isDebugEnabled())
        std::cerr << "[xad-forge-debug] " << msg << ": " << ptr << std::endl;
}

//=============================================================================
// Backend loading
//=============================================================================

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
        debugLog("loadCustomBackendFromEnv() called");
        const char* backendPath = std::getenv("XAD_FORGE_BACKEND_PATH");
        if (backendPath && backendPath[0] != '\0')
        {
            debugLog("  Loading custom backend from", backendPath);
            ForgeError err = forge_load_backend(backendPath);
            if (err != FORGE_SUCCESS)
            {
                std::cerr << "xad-forge: Warning: Failed to load custom backend from '"
                          << backendPath << "': " << forge_error_string(err);
                const char* detailMsg = forge_get_last_error();
                if (detailMsg && detailMsg[0] != '\0')
                {
                    std::cerr << " (" << detailMsg << ")";
                }
                std::cerr << std::endl;
            }
            else
            {
                debugLog("  Custom backend loaded successfully");
            }
        }
        else
        {
            debugLog("  XAD_FORGE_BACKEND_PATH not set, using built-in backends");
        }

        // Log which instruction set will be used
        const char* instSet = std::getenv("XAD_FORGE_INSTRUCTION_SET");
        if (instSet && instSet[0] != '\0')
        {
            debugLog("  XAD_FORGE_INSTRUCTION_SET", instSet);
        }
    });
}

}  // namespace detail
}  // namespace forge
}  // namespace xad
