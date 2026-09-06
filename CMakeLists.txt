cmake_minimum_required(VERSION 3.18)
project(analog_ecc_heights LANGUAGES CXX)

option(BUILD_TEST_MAIN "Build the standalone native test executable" OFF)
option(SANITIZE "Enable Address/UndefinedBehavior sanitizers" OFF)
option(USE_HIGHS "Require and enable the system HiGHS library" OFF)
option(USE_CUDA_FULL_SWEEP "Enable the experimental CUDA full sweep" OFF)

if (USE_CUDA_FULL_SWEEP)
  message(FATAL_ERROR
    "The CUDA full-sweep source is not part of this package. "
    "Build with USE_CUDA_FULL_SWEEP=OFF.")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Preserve the existing extension's cumulative GLPK dependency.
find_path(GLPK_INCLUDE_DIR glpk.h)
find_library(GLPK_LIBRARY NAMES glpk)
if (NOT GLPK_INCLUDE_DIR OR NOT GLPK_LIBRARY)
  message(FATAL_ERROR
    "System GLPK headers and library are required for every C++ build. "
    "Install GLPK development files, or set GLPK_INCLUDE_DIR and GLPK_LIBRARY.")
endif()

find_package(OpenMP COMPONENTS CXX QUIET)
set(HAS_OPENMP False)
if (OpenMP_CXX_FOUND)
  set(HAS_OPENMP True)
endif()

set(HAS_HIGHS False)
set(HIGHS_TARGET "")
if (USE_HIGHS)
  find_package(HiGHS CONFIG QUIET)
  find_package(highs CONFIG QUIET)
  foreach(candidate IN ITEMS HiGHS::highs highs::highs highs)
    if (TARGET ${candidate} AND NOT HIGHS_TARGET)
      set(HIGHS_TARGET ${candidate})
    endif()
  endforeach()

  # Existing sources include <highs/Highs.h>, so retain the include root
  # alongside any include/highs path supplied by a config target.
  find_path(HIGHS_INCLUDE_ROOT highs/Highs.h)
  if (NOT HIGHS_TARGET)
    find_library(HIGHS_LIBRARY NAMES highs)
    if (HIGHS_INCLUDE_ROOT AND HIGHS_LIBRARY)
      add_library(analog_ecc_system_highs UNKNOWN IMPORTED)
      set_target_properties(analog_ecc_system_highs PROPERTIES
        IMPORTED_LOCATION "${HIGHS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${HIGHS_INCLUDE_ROOT};${HIGHS_INCLUDE_ROOT}/highs")
      set(HIGHS_TARGET analog_ecc_system_highs)
    endif()
  endif()

  if (NOT HIGHS_TARGET)
    message(FATAL_ERROR
      "USE_HIGHS=ON requires a system HiGHS development installation. "
      "Set CMAKE_PREFIX_PATH to its installation prefix, or set "
      "HIGHS_INCLUDE_ROOT and HIGHS_LIBRARY explicitly. "
      "The Python highspy package alone is not sufficient.")
  endif()
  set(HAS_HIGHS True)
endif()

set(NATIVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/analog_ecc_heights/cpp_backend/src")
set(NATIVE_METHOD_SOURCES
  "${NATIVE_SOURCE_DIR}/methods_jiang_lp.cc"
  "${NATIVE_SOURCE_DIR}/methods_roth_lp.cc"
  "${NATIVE_SOURCE_DIR}/methods_roth_comb.cc"
  "${NATIVE_SOURCE_DIR}/methods_roth_mds_comb.cc")

# Keep the original module basename and PYBIND11_MODULE declaration unchanged.
pybind11_add_module(solve_m_height_cpp
  "${NATIVE_SOURCE_DIR}/py_binding.cc" ${NATIVE_METHOD_SOURCES})
set(NATIVE_TARGETS solve_m_height_cpp)
if (BUILD_TEST_MAIN)
  add_executable(test_main tests/native/test_main.cc ${NATIVE_METHOD_SOURCES})
  list(APPEND NATIVE_TARGETS test_main)
endif()

if (SANITIZE AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsanitize=address" ANALOG_ECC_HAS_ASAN)
  check_cxx_compiler_flag("-fsanitize=undefined" ANALOG_ECC_HAS_UBSAN)
endif()

get_filename_component(GLPK_LIBRARY_DIR "${GLPK_LIBRARY}" DIRECTORY)
foreach(native_target IN LISTS NATIVE_TARGETS)
  target_compile_features(${native_target} PRIVATE cxx_std_17)
  target_include_directories(${native_target} PRIVATE
    "${NATIVE_SOURCE_DIR}" "${GLPK_INCLUDE_DIR}")
  target_link_libraries(${native_target} PRIVATE Eigen3::Eigen "${GLPK_LIBRARY}")
  set_target_properties(${native_target} PROPERTIES
    BUILD_RPATH "${GLPK_LIBRARY_DIR}"
    INSTALL_RPATH "${GLPK_LIBRARY_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  if (OpenMP_CXX_FOUND)
    target_link_libraries(${native_target} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${native_target} PRIVATE USE_OPENMP=1)
  endif()
  if (USE_HIGHS)
    target_compile_definitions(${native_target} PRIVATE HAVE_HIGHS=1)
    target_link_libraries(${native_target} PRIVATE ${HIGHS_TARGET})
    if (HIGHS_INCLUDE_ROOT)
      target_include_directories(${native_target} PRIVATE "${HIGHS_INCLUDE_ROOT}")
    endif()
    set_property(TARGET ${native_target} APPEND PROPERTY
      BUILD_RPATH "$<TARGET_FILE_DIR:${HIGHS_TARGET}>")
    set_property(TARGET ${native_target} APPEND PROPERTY
      INSTALL_RPATH "$<TARGET_FILE_DIR:${HIGHS_TARGET}>")
  endif()
  if (SANITIZE AND NOT MSVC)
    if (ANALOG_ECC_HAS_ASAN)
      target_compile_options(${native_target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(${native_target} PRIVATE -fsanitize=address)
    endif()
    if (ANALOG_ECC_HAS_UBSAN)
      target_compile_options(${native_target} PRIVATE -fsanitize=undefined)
      target_link_options(${native_target} PRIVATE -fsanitize=undefined)
    endif()
  endif()
endforeach()

configure_file(cmake/build_info.py.in "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" @ONLY)
install(TARGETS solve_m_height_cpp
  LIBRARY DESTINATION analog_ecc_heights/cpp_backend
  RUNTIME DESTINATION analog_ecc_heights/cpp_backend)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
  DESTINATION analog_ecc_heights/cpp_backend)
