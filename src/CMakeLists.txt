
# FNUFFT library
# Author: Falk Mayer
# TRON library target


cmake_minimum_required(VERSION 3.17)

# dependencies
find_package(CMakeshift 3.7 REQUIRED)
find_package(gsl-lite 0.35.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)


# TODO: add more package dependencies here

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/cmake/modules/")
find_package(FFTW REQUIRED)

# targets
	add_library(TRON tron.cu) # TODO: add more source files here
set_target_properties(TRON PROPERTIES CUDA_ARCHITECTURES 61)

# library properties
target_include_directories(TRON
	PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(TRON
	PUBLIC
        cxx_std_14)
	#set_target_properties(TRON
	#PROPERTIES
	#    CUDA_STANDARD 14)


# library dependencies
target_link_libraries(TRON
	PUBLIC
        gsl::gsl-lite-v1
		CUDA::cufft
		CUDA::cublas)
# installation: public header files, target
install(
    DIRECTORY
        "${PROJECT_SOURCE_DIR}/include/" # the trailing '/' is significant
    TYPE INCLUDE)
install(
    TARGETS TRON
	EXPORT TRON
    ARCHIVE INCLUDES)
