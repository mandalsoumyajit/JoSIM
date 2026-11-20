cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
endif()

project(JoSIM VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SLU "Retired SuperLU selection (ON is unsupported)" OFF)
if(SLU)
  message(FATAL_ERROR "SuperLU backend has been retired. KLU is the supported solver; remove -DSLU=ON.")
endif()
option(BUILD_SHARED_LIBS "Build shared JoSIM (KLU dependencies remain static)" OFF)
option(MAKING_STATIC_BUILD "Build josim to be as static as possible" OFF)
if(MAKING_STATIC_BUILD)
  if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR "BUILD_SHARED_LIBS and MAKING_STATIC_BUILD are mutually exclusive.")
  endif()
  set(BUILD_SHARED_LIBS OFF)
  if(MSVC)
    if(CMAKE_VERSION VERSION_LESS 3.15)
      message(FATAL_ERROR "Static Windows runtimes require CMake 3.15 or newer")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/verification.cmake)
# KLU is linked statically into either form of JoSIM. The legacy wrappers do
# not implement portable shared solver libraries (duplicate DLONG symbols).
set(josim_build_shared "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)
include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)
set(BUILD_SHARED_LIBS "${josim_build_shared}")
if(BUILD_SHARED_LIBS)
  foreach(component config amd btf colamd klu)
    set_target_properties(${component} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
endif()
include(${PROJECT_SOURCE_DIR}/cmake/make_static_target.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/warnings.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/commit_hash.cmake)

add_definitions(-DVERSION="${CMAKE_PROJECT_VERSION}")
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
# Default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# See if OpenMP or CUDA was specified (the fork's parallel and GPU solvers)
option(USING_OPENMP "Allow parallel processing in JoSIM" OFF)
option(USING_CUDA "Enable CUDA GPU acceleration in JoSIM" OFF)

if(USING_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
      set(CMAKE_CUDA_ARCHITECTURES "native")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
    endif()
  endif()
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

# JoSIM library
# -------------

add_library(
  josim
  src/Capacitor.cpp
  src/CCCS.cpp
  src/CCVS.cpp
  src/CliOptions.cpp
  src/CurrentSource.cpp
  src/Errors.cpp
  src/Function.cpp
  src/Inductor.cpp
  src/InitialState.cpp
  src/Input.cpp
  src/JJ.cpp
  src/Matrix.cpp
  src/Misc.cpp
  src/Model.cpp
  src/Nanowire.cpp
  src/Netlist.cpp
  src/NonlinearInductor.cpp
  src/Output.cpp
  src/Parameters.cpp
  src/PhaseSource.cpp
  src/RelevantTrace.cpp
  src/Resistor.cpp
  src/Rng.cpp
  src/Simulation.cpp
  src/SimulationReference.cpp
  src/Transient.cpp
  src/TransmissionLine.cpp
  src/VCCS.cpp
  src/VCVS.cpp
  src/Verbose.cpp
  src/VoltageSource.cpp
  src/Noise.cpp
  src/Spread.cpp
  src/IV.cpp)

# Alias for projects including JoSIM
add_library(josim::josim ALIAS josim)

# Include directories
target_include_directories(
  josim PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include>)

# Dependencies
target_link_libraries(josim PUBLIC suitesparse::klu)
target_compile_features(josim PUBLIC cxx_std_17)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(josim PRIVATE Threads::Threads)
if(NOT MSVC AND NOT APPLE)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(filesystem_probe "#include <filesystem>\nint main() { return std::filesystem::exists(\".\") ? 0 : 1; }")
  check_cxx_source_compiles("${filesystem_probe}" JOSIM_FILESYSTEM_IN_STDLIB)
  if(NOT JOSIM_FILESYSTEM_IN_STDLIB)
    set(CMAKE_REQUIRED_LIBRARIES stdc++fs)
    check_cxx_source_compiles("${filesystem_probe}" JOSIM_FILESYSTEM_WITH_STDCXXFS)
    if(NOT JOSIM_FILESYSTEM_WITH_STDCXXFS)
      message(FATAL_ERROR "C++17 filesystem support could not be linked with this toolchain.")
    endif()
    target_link_libraries(josim PUBLIC stdc++fs)
  endif()
  cmake_pop_check_state()
endif()

# Properties
set_target_properties(josim PROPERTIES POSITION_INDEPENDENT_CODE ON WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(JOSIM_ABI_VERSION 4)
set_target_properties(josim PROPERTIES VERSION ${JOSIM_ABI_VERSION}.0.0 SOVERSION ${JOSIM_ABI_VERSION})
make_target_static(josim)
target_add_warnings(josim)
josim_instrument_target(josim)
# target_compile_features(josim PUBLIC cxx_std_20)

# OpenMP usage
if(USING_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(josim PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA usage
if(USING_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(josim PRIVATE src/CudaSolver.cu)
  target_compile_definitions(josim PUBLIC JOSIM_CUDA)
  target_link_libraries(josim PUBLIC CUDA::cudart CUDA::cusolver CUDA::cusparse)
endif()

# JoSIM command line interface
# ----------------------------

add_executable(josim-cli src/josim.cpp)

# Dependencies
target_link_libraries(josim-cli PRIVATE josim)
target_link_libraries(josim-cli PRIVATE suitesparse::klu)

# Properties
make_target_static(josim-cli)
target_add_warnings(josim-cli)
josim_instrument_target(josim-cli)

# JoSIM C / Python shared library (pyjosim)
# -----------------------------------------

add_library(josim_c SHARED src/josim_c_api.cpp)
target_link_libraries(josim_c PRIVATE josim suitesparse::klu Threads::Threads)
if(NOT MSVC AND NOT APPLE)
  target_link_libraries(josim_c PRIVATE stdc++fs)
endif()
if(USING_OPENMP)
  target_link_libraries(josim_c PRIVATE OpenMP::OpenMP_CXX)
endif()
target_add_warnings(josim_c)

# JoSIM Co-Simulation IPC Socket Server (Layer 2)
# -----------------------------------------------

add_executable(josim-cosim-server src/josim_cosim_server.cpp)
target_include_directories(josim-cosim-server PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(josim-cosim-server PRIVATE josim_c Threads::Threads)
if(WIN32)
  target_link_libraries(josim-cosim-server PRIVATE ws2_32)
endif()
target_add_warnings(josim-cosim-server)

# Cadence Spectre / SystemVerilog DPI-C Co-Simulation Library (Layer 3A)
# ----------------------------------------------------------------------

enable_language(C)
add_library(josim_spectre_dpi SHARED cosim/spectre/josim_cosim_dpi.c)
target_include_directories(josim_spectre_dpi PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(josim_spectre_dpi PRIVATE josim_c)
if(NOT MSVC)
  target_link_libraries(josim_spectre_dpi PRIVATE m)
endif()
if(WIN32)
  target_link_libraries(josim_spectre_dpi PRIVATE ws2_32)
endif()
set_target_properties(
  josim_spectre_dpi PROPERTIES
  BUILD_RPATH "$<TARGET_FILE_DIR:josim_c>")
target_add_warnings(josim_spectre_dpi)

# Optional ngspice XSPICE Co-Simulation Code Model (josim_cosim.cm)
# -----------------------------------------------------------------

set(_DEFAULT_NGSPICE_SRC_DIR "$ENV{NGSPICE_SRC_DIR}")
if(NOT _DEFAULT_NGSPICE_SRC_DIR)
  foreach(_CAND
    "$ENV{HOME}/local/src/ngspice-45.2"
    "$ENV{HOME}/ngspice-45.2")
    if(EXISTS "${_CAND}/src/include/ngspice/cm.h")
      set(_DEFAULT_NGSPICE_SRC_DIR "${_CAND}")
      break()
    endif()
  endforeach()
endif()
set(NGSPICE_SRC_DIR "${_DEFAULT_NGSPICE_SRC_DIR}" CACHE PATH "Path to ngspice source tree for XSPICE code model build")
if(EXISTS "${NGSPICE_SRC_DIR}/src/include/ngspice/cm.h")
  enable_language(C)
  add_library(
    josim_cosim_cm SHARED
    cosim/ngspice/josim_xspice.c
    ${NGSPICE_SRC_DIR}/src/xspice/icm/dlmain.c
    ${NGSPICE_SRC_DIR}/src/misc/dstring.c)
  target_include_directories(
    josim_cosim_cm PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/cosim/ngspice
    ${NGSPICE_SRC_DIR}/src/include)
  target_compile_definitions(josim_cosim_cm PRIVATE HAVE_CONFIG_H)
  target_compile_options(josim_cosim_cm PRIVATE -UVERSION -fvisibility=hidden)
  target_link_libraries(josim_cosim_cm PRIVATE josim_c m)
  set_target_properties(
    josim_cosim_cm PROPERTIES
    PREFIX ""
    OUTPUT_NAME "josim_cosim"
    SUFFIX ".cm"
    BUILD_RPATH "$<TARGET_FILE_DIR:josim_c>")
endif()

# Testing
# -------

# Neccesary includes
include(CTest)
include(${PROJECT_SOURCE_DIR}/cmake/integration_test.cmake)

# Enable testing
enable_testing()

# Add tests
add_subdirectory(test)

# Install
# -------

# Get install locations
include(GNUInstallDirs)
if(BUILD_SHARED_LIBS)
  if(APPLE)
    set_target_properties(josim-cli PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  elseif(UNIX)
    set_target_properties(josim-cli PROPERTIES INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
endif()

# Install headers
install(DIRECTORY include/JoSIM/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/JoSIM/")

# Legacy dependency wrappers add an extra include/ directory during install.
# Provide their headers at the advertised prefix/include as well.
foreach(component config amd btf colamd klu)
  install(DIRECTORY "${CMakeSuiteSparse_SOURCE_DIR}/${component}/include/"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endforeach()

install(FILES LICENSE README.md CHANGELOG.md DESTINATION .)
install(DIRECTORY docs/ DESTINATION docs PATTERN "context" EXCLUDE)

# Install targets
install(
  TARGETS josim josim_c josim-cli
  EXPORT josim-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT josim-targets
  FILE josim-targets.cmake
  NAMESPACE josim::
  DESTINATION share/josim/)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/josim-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/josim-config.cmake"
  INSTALL_DESTINATION share/josim)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/josim-config-version.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/josim-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/josim-config-version.cmake"
        DESTINATION share/josim/)
