cmake_minimum_required(VERSION 3.18)
project(solve_m_height_cpp VERSION 0.0.1 LANGUAGES CXX)

option(BUILD_TEST_MAIN "Build a standalone test_main executable" ON)
option(SANITIZE "Enable Address/UB sanitizers (non-MSVC)" OFF)
option(USE_HIGHS "Build with HiGHS LP backend if available" ON)
option(USE_CUDA_FULL_SWEEP "Enable CUDA full-sweep implementation when CUDA is available" ON)

set(HAVE_CUDA_FULL_SWEEP OFF)
if (USE_CUDA_FULL_SWEEP)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA_FULL_SWEEP ON)
  else()
    message(STATUS "CUDA compiler not found; building test_main without CUDA full-sweep support")
  endif()
endif()

# -------------------- Dependencies --------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)

# GLPK (required)
find_path(GLPK_INCLUDE_DIR glpk.h)
find_library(GLPK_LIBRARY glpk)
if (NOT GLPK_INCLUDE_DIR OR NOT GLPK_LIBRARY)
  message(FATAL_ERROR "Could not find GLPK; set GLPK_INCLUDE_DIR/GLPK_LIBRARY or install libglpk-dev (or conda-forge: glpk).")
endif()

# OpenMP (optional)
find_package(OpenMP)

# -------------------- Sanitizers probe --------------------
include(CheckCXXCompilerFlag)
set(HAS_ASAN OFF)
set(HAS_UBSAN OFF)
if (SANITIZE AND NOT MSVC)
  check_cxx_compiler_flag("-fsanitize=address"   HAS_ASAN)
  check_cxx_compiler_flag("-fsanitize=undefined" HAS_UBSAN)
endif()

set(CXX_STD cxx_std_17)

# -------------------- pybind11 module --------------------
set(SOLVE_M_HEIGHT_CPP_SOURCES
  src/py_binding.cc
  src/methods_jiang_LP.cc
  src/methods_roth_LP.cc
  src/methods_roth_comb.cc
  src/methods_roth_mds_comb.cc
)
if (HAVE_CUDA_FULL_SWEEP)
  list(APPEND SOLVE_M_HEIGHT_CPP_SOURCES src/methods_roth_comb.cu)
endif()

if (HAVE_CUDA_FULL_SWEEP)
  pybind11_add_module(solve_m_height_cpp NO_EXTRAS ${SOLVE_M_HEIGHT_CPP_SOURCES})
else()
  pybind11_add_module(solve_m_height_cpp ${SOLVE_M_HEIGHT_CPP_SOURCES})
endif()
target_compile_features(solve_m_height_cpp PRIVATE ${CXX_STD})
target_include_directories(solve_m_height_cpp PRIVATE ${GLPK_INCLUDE_DIR})
target_link_libraries(solve_m_height_cpp PRIVATE ${GLPK_LIBRARY} Eigen3::Eigen)

if (HAVE_CUDA_FULL_SWEEP)
  set_target_properties(solve_m_height_cpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(solve_m_height_cpp PRIVATE HAVE_CUDA_FULL_SWEEP=1)
  target_link_libraries(solve_m_height_cpp PRIVATE CUDA::cublas CUDA::cusolver)
  if (NOT MSVC)
    # Avoid GCC LTO bytecode version mismatches between nvcc host compilation and
    # the final C++ linker toolchain (seen in conda envs).
    set_property(TARGET solve_m_height_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    target_compile_options(solve_m_height_cpp PRIVATE
      -fno-lto
      $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fno-lto>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_options(solve_m_height_cpp PRIVATE -fno-lto)
  endif()
endif()

if (OpenMP_CXX_FOUND)
  target_link_libraries(solve_m_height_cpp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(solve_m_height_cpp PRIVATE -DUSE_OPENMP=1)
endif()

# Compile flags / sanitizers
if (MSVC)
  target_compile_options(solve_m_height_cpp PRIVATE
    $<$<CONFIG:Debug>:/Zi /Od>
    $<$<NOT:$<CONFIG:Debug>>:/O2 /DNDEBUG>)
else()
  target_compile_options(solve_m_height_cpp PRIVATE
    $<$<CONFIG:Debug>:-O0 -g -fPIC -fno-omit-frame-pointer>
    $<$<NOT:$<CONFIG:Debug>>:-O3 -DNDEBUG -fPIC>)
  if (SANITIZE AND HAS_ASAN)
    target_compile_options(solve_m_height_cpp PRIVATE -fsanitize=address)
    target_link_options(solve_m_height_cpp PRIVATE -fsanitize=address)
  endif()
  if (SANITIZE AND HAS_UBSAN)
    target_compile_options(solve_m_height_cpp PRIVATE -fsanitize=undefined)
    target_link_options(solve_m_height_cpp PRIVATE -fsanitize=undefined)
  endif()
endif()

# -------------------- Standalone test executable --------------------
if (BUILD_TEST_MAIN)
  set(TEST_MAIN_SOURCES
    src/test_main.cc
    src/methods_jiang_LP.cc
    src/methods_roth_LP.cc
    src/methods_roth_comb.cc
    src/methods_roth_mds_comb.cc
  )
  if (HAVE_CUDA_FULL_SWEEP)
    list(APPEND TEST_MAIN_SOURCES src/methods_roth_comb.cu)
  endif()

  add_executable(test_main ${TEST_MAIN_SOURCES})
  target_compile_features(test_main PRIVATE ${CXX_STD})
  target_include_directories(test_main PRIVATE ${GLPK_INCLUDE_DIR})
  target_link_libraries(test_main PRIVATE ${GLPK_LIBRARY} Eigen3::Eigen)

  if (HAVE_CUDA_FULL_SWEEP)
    set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(test_main PRIVATE HAVE_CUDA_FULL_SWEEP=1)
    target_link_libraries(test_main PRIVATE CUDA::cublas CUDA::cusolver)
    if (NOT MSVC)
      set_property(TARGET test_main PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
      target_compile_options(test_main PRIVATE
        -fno-lto
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fno-lto>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      target_link_options(test_main PRIVATE -fno-lto)
    endif()
  endif()

  if (OpenMP_CXX_FOUND)
    target_link_libraries(test_main PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(test_main PRIVATE -DUSE_OPENMP=1)
  endif()

  if (MSVC)
    target_compile_options(test_main PRIVATE /Zi /Od)
  else()
    target_compile_options(test_main PRIVATE -O0 -g -fPIC -fno-omit-frame-pointer)
    if (HAS_ASAN)
      target_compile_options(test_main PRIVATE -fsanitize=address)
      target_link_options(test_main PRIVATE -fsanitize=address)
    endif()
    if (HAS_UBSAN)
      target_compile_options(test_main PRIVATE -fsanitize=undefined)
      target_link_options(test_main PRIVATE -fsanitize=undefined)
    endif()
  endif()
endif()

# -------------------- HiGHS (optional) --------------------
set(HAVE_HIGHS OFF)
set(HIGHS_TARGET "")

if (USE_HIGHS)
  # Prefer config package with imported target names
  find_package(HiGHS CONFIG QUIET)

  if (HiGHS_FOUND)
    if (TARGET highs)
      set(HIGHS_TARGET highs)
      set(HAVE_HIGHS ON)
    elseif (TARGET HiGHS::highs)
      set(HIGHS_TARGET HiGHS::highs)
      set(HAVE_HIGHS ON)
    endif()
  endif()

  # Fallback to raw include+lib if no imported target
  if(NOT HAVE_HIGHS)
    # Returns PREFIX/include (because we search for highs/Highs.h)
    find_path(HIGHS_INCLUDE_ROOT highs/Highs.h)
    find_library(HIGHS_LIBRARY NAMES highs)

    if(HIGHS_INCLUDE_ROOT AND HIGHS_LIBRARY)
      add_library(highs_fallback UNKNOWN IMPORTED)
      set_target_properties(highs_fallback PROPERTIES
        IMPORTED_LOCATION "${HIGHS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${HIGHS_INCLUDE_ROOT};${HIGHS_INCLUDE_ROOT}/highs"  # <-- add both
      )
      set(HIGHS_TARGET highs_fallback)
      set(HAVE_HIGHS ON)
    endif()
  endif()

  if (HAVE_HIGHS)
    message(STATUS "HiGHS usable target found: ${HIGHS_TARGET}")
    foreach(tgt IN ITEMS solve_m_height_cpp test_main)
      if (TARGET ${tgt})
        target_compile_definitions(${tgt} PRIVATE HAVE_HIGHS=1)
        target_link_libraries(${tgt} PRIVATE ${HIGHS_TARGET})
      endif()
    endforeach()
  else()
    message(STATUS "HiGHS not found; building without HiGHS backend")
  endif()
endif()

# -------------------- RPATH so the binaries find GLPK/HiGHS at runtime ------
get_filename_component(GLPK_DIR "${GLPK_LIBRARY}" DIRECTORY)
set(_extra_rpaths "")
if (DEFINED HIGHS_LIBRARY)
  get_filename_component(HIGHS_DIR "${HIGHS_LIBRARY}" DIRECTORY)
  list(APPEND _extra_rpaths "${HIGHS_DIR}")
endif()

if (TARGET solve_m_height_cpp)
  set_target_properties(solve_m_height_cpp PROPERTIES
    BUILD_RPATH "${GLPK_DIR};${_extra_rpaths}"
    INSTALL_RPATH "${GLPK_DIR};${_extra_rpaths}"
  )
endif()

if (TARGET test_main)
  set_target_properties(test_main PROPERTIES
    BUILD_RPATH "${GLPK_DIR};${_extra_rpaths}"
    INSTALL_RPATH "${GLPK_DIR};${_extra_rpaths}"
  )
endif()

# -------------------- Install --------------------
install(TARGETS solve_m_height_cpp DESTINATION .)
