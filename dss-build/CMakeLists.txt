# CMakeLists.txt for building HEC-DSS as a shared library for Python
#
# This CMake file builds heclib as a shared library (DLL on Windows, .so on Linux)
# for use with pyiwfm via ctypes.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# The resulting library will be:
#   - Windows: hecdss.dll
#   - Linux/macOS: libhecdss.so / libhecdss.dylib

cmake_minimum_required(VERSION 3.16)
project(hecdss_python LANGUAGES C)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(HECDSS_INSTALL_PYTHON "Install library to Python package" ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find or fetch heclib source
# Option 1: Use existing heclib source from IWFM build
if(DEFINED HECLIB_SOURCE_DIR AND EXISTS "${HECLIB_SOURCE_DIR}")
    message(STATUS "Using heclib source from: ${HECLIB_SOURCE_DIR}")
    set(HECLIB_SRC_DIR "${HECLIB_SOURCE_DIR}")
else()
    # Option 2: Fetch from GitHub
    include(FetchContent)
    FetchContent_Declare(
        heclib
        GIT_REPOSITORY https://github.com/HydrologicEngineeringCenter/hec-dss.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(heclib)
    set(HECLIB_SRC_DIR "${heclib_SOURCE_DIR}/heclib/heclib_c")
endif()

# If FetchContent_MakeAvailable already created the hecdss target (the upstream
# hec-dss repo has its own CMakeLists.txt), skip building our own and just use it.
if(NOT TARGET hecdss)
    # Find zlib
    find_package(ZLIB QUIET)
    if(NOT ZLIB_FOUND)
        # Build zlib if not found
        include(FetchContent)
        FetchContent_Declare(
            zlib
            GIT_REPOSITORY https://github.com/madler/zlib.git
            GIT_TAG v1.3.1
        )
        FetchContent_MakeAvailable(zlib)
    endif()

    # Collect heclib source files
    file(GLOB_RECURSE HECLIB_SOURCES
        "${HECLIB_SRC_DIR}/src/*.c"
    )

    # Create shared library
    add_library(hecdss SHARED ${HECLIB_SOURCES})

    # Set library properties
    set_target_properties(hecdss PROPERTIES
        VERSION 7.0
        SOVERSION 7
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Platform-specific settings
    if(WIN32)
        set_target_properties(hecdss PROPERTIES
            OUTPUT_NAME "hecdss"
            SUFFIX ".dll"
            PREFIX ""
        )
        # Export all symbols on Windows for ctypes compatibility
        set_target_properties(hecdss PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        target_compile_definitions(hecdss PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_WARNINGS
        )
    elseif(APPLE)
        set_target_properties(hecdss PROPERTIES
            OUTPUT_NAME "hecdss"
            SUFFIX ".dylib"
        )
    else()
        set_target_properties(hecdss PROPERTIES
            OUTPUT_NAME "hecdss"
            SUFFIX ".so"
        )
    endif()

    # Compiler flags
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(hecdss PRIVATE
            -fPIC
            -std=gnu99
            -Wall
            -Wno-unused-variable
            -Wno-unused-function
        )
    elseif(MSVC)
        target_compile_options(hecdss PRIVATE
            /W3
            /MP
        )
    endif()

    # Include directories
    target_include_directories(hecdss PUBLIC
        $<BUILD_INTERFACE:${HECLIB_SRC_DIR}/src/headers>
        $<INSTALL_INTERFACE:include>
    )

    # Link zlib
    if(ZLIB_FOUND)
        target_link_libraries(hecdss PRIVATE ZLIB::ZLIB)
    else()
        target_link_libraries(hecdss PRIVATE zlibstatic)
    endif()
else()
    message(STATUS "Using upstream hecdss target (already defined by FetchContent)")
endif()

# Installation
install(TARGETS hecdss
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install to Python package if requested
if(HECDSS_INSTALL_PYTHON)
    # Determine Python package location
    set(PYIWFM_DSS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/pyiwfm/io/dss")
    if(EXISTS "${PYIWFM_DSS_DIR}")
        install(TARGETS hecdss
            RUNTIME DESTINATION "${PYIWFM_DSS_DIR}/lib"
            LIBRARY DESTINATION "${PYIWFM_DSS_DIR}/lib"
        )
        message(STATUS "Will install hecdss to: ${PYIWFM_DSS_DIR}/lib")
    endif()
endif()

# Post-build message (only for targets created in this directory)
get_property(_hecdss_dir TARGET hecdss PROPERTY SOURCE_DIR)
if(_hecdss_dir STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(WIN32)
        add_custom_command(TARGET hecdss POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Built hecdss.dll for Python ctypes"
            COMMENT "HEC-DSS shared library built successfully"
        )
    else()
        add_custom_command(TARGET hecdss POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Built libhecdss.so for Python ctypes"
            COMMENT "HEC-DSS shared library built successfully"
        )
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "HEC-DSS Python Library Configuration:")
message(STATUS "  Source directory: ${HECLIB_SRC_DIR}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "  Install to Python: ${HECDSS_INSTALL_PYTHON}")
message(STATUS "")
