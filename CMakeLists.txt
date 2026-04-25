cmake_minimum_required(VERSION 3.21)

project(fastawc LANGUAGES CXX)

include(CheckIPOSupported)
include(CTest)

option(FASTAWC_ENABLE_LTO "Enable interprocedural optimization in release builds" ON)
option(FASTAWC_ENABLE_AVX2_BACKEND "Build the AVX2 runtime backend when supported by the target architecture" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fastawc_options INTERFACE)
target_include_directories(fastawc_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/headers)

if(MSVC)
	target_compile_options(fastawc_options INTERFACE /W4 /O2 /Ob2 /Oi /Ot /GF /Gy /GS- /fp:fast /permissive-)
	target_compile_definitions(fastawc_options INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
	target_compile_options(fastawc_options INTERFACE -O3 -fomit-frame-pointer -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
		target_compile_options(fastawc_options INTERFACE -fno-semantic-interposition)
	endif()
endif()

set(FASTAWC_IPO_SUPPORTED FALSE)
if(FASTAWC_ENABLE_LTO)
	check_ipo_supported(RESULT FASTAWC_IPO_SUPPORTED)
endif()

function(fastawc_enable_ipo target_name)
	if(FASTAWC_ENABLE_LTO AND FASTAWC_IPO_SUPPORTED)
		set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
	endif()
endfunction()

add_library(fastawc_scalar OBJECT sources/engine_scalar.cpp)
target_link_libraries(fastawc_scalar PRIVATE fastawc_options)
fastawc_enable_ipo(fastawc_scalar)

set(FASTAWC_AVX2_AVAILABLE FALSE)
if(FASTAWC_ENABLE_AVX2_BACKEND AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$" OR CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "x64|X86"))
	add_library(fastawc_avx2 OBJECT sources/engine_avx2.cpp)
	target_link_libraries(fastawc_avx2 PRIVATE fastawc_options)
	if(MSVC)
		target_compile_options(fastawc_avx2 PRIVATE /arch:AVX2)
	else()
		target_compile_options(fastawc_avx2 PRIVATE -mavx2)
	endif()
	fastawc_enable_ipo(fastawc_avx2)
	set(FASTAWC_AVX2_AVAILABLE TRUE)
endif()

add_executable(fastawc
	sources/main.cpp
	sources/platform.cpp
	sources/runtime.cpp
	sources/thread_pool.cpp
	$<TARGET_OBJECTS:fastawc_scalar>
)
target_link_libraries(fastawc PRIVATE fastawc_options)
fastawc_enable_ipo(fastawc)

if(FASTAWC_AVX2_AVAILABLE)
	target_sources(fastawc PRIVATE $<TARGET_OBJECTS:fastawc_avx2>)
	target_compile_definitions(fastawc PRIVATE FASTAWC_HAS_AVX2_BACKEND=1)
else()
	target_compile_definitions(fastawc PRIVATE FASTAWC_HAS_AVX2_BACKEND=0)
endif()

if(UNIX AND NOT APPLE)
	target_compile_definitions(fastawc PRIVATE _FILE_OFFSET_BITS=64)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(BUILD_TESTING AND Python3_Interpreter_FOUND)
	add_test(
		NAME fastawc_regression
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py $<TARGET_FILE:fastawc>
	)
endif()
