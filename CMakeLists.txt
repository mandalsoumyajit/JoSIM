cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(JoSIM VERSION 2.7)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/make_static_target.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/warnings.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/commit_hash.cmake)

add_definitions(-DVERSION="${CMAKE_PROJECT_VERSION}")
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
option(SLU "Enable SuperLU instead of KLU")

# Default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# See if OpenMP or CUDA was specified
option(USING_OPENMP "Allow parallel processing in JoSIM" OFF)
option(USING_CUDA "Enable CUDA GPU acceleration in JoSIM" OFF)
option(MAKING_STATIC_BUILD "Build josim to be as static as possible" OFF)

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

if(MAKING_STATIC_BUILD)
  set(CMAKE_BUILD_SHARED OFF)
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
  src/Input.cpp
  src/JJ.cpp
  src/Matrix.cpp
  src/Misc.cpp
  src/Model.cpp
  src/Nanowire.cpp
  src/Netlist.cpp
  src/Output.cpp
  src/Parameters.cpp
  src/PhaseSource.cpp
  src/RelevantTrace.cpp
  src/Resistor.cpp
  src/Rng.cpp
  src/Simulation.cpp
  src/Transient.cpp
  src/TransmissionLine.cpp
  src/VCCS.cpp
  src/VCVS.cpp
  src/Verbose.cpp
  src/VoltageSource.cpp
  src/Noise.cpp
  src/Spread.cpp
  src/IV.cpp
  src/LUSolve.cpp)

# Alias for projects including JoSIM
add_library(josim::josim ALIAS josim)

# Include directories
target_include_directories(
  josim PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include>)

# Dependencies
target_link_libraries(josim PRIVATE suitesparse::klu)
target_link_libraries(josim PRIVATE superlu)
target_link_libraries(josim PRIVATE cblas)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(josim PRIVATE Threads::Threads)
if(NOT MSVC AND NOT APPLE)
  target_link_libraries(josim PRIVATE stdc++fs)
endif()

# Properties
set_target_properties(josim PROPERTIES POSITION_INDEPENDENT_CODE ON)
make_target_static(josim)
target_add_warnings(josim)
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
target_link_libraries(josim-cli PRIVATE superlu)
target_link_libraries(josim-cli PRIVATE cblas)

# Properties
make_target_static(josim-cli)
target_add_warnings(josim-cli)

# JoSIM C / Python shared library (pyjosim)
# -----------------------------------------

add_library(josim_c SHARED src/josim_c_api.cpp)
target_link_libraries(josim_c PRIVATE josim suitesparse::klu superlu cblas Threads::Threads)
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
target_link_libraries(josim_spectre_dpi PRIVATE josim_c m)
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

# Install headers
install(DIRECTORY include/JoSIM/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/JoSIM/")

install(FILES LICENSE README.md DESTINATION .)

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
  FILE josim-config.cmake
  NAMESPACE josim::
  DESTINATION share/josim/)
