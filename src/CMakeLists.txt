
# FNUFFT library
# Author: Falk Mayer
# tron library target


cmake_minimum_required(VERSION 3.17)

# dependencies
find_package(CMakeshift 3.7 REQUIRED)
find_package(gsl-lite 0.35.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuda-api-wrappers REQUIRED)


# TODO: add more package dependencies here

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/cmake/modules/")
find_package(FFTW REQUIRED)

# targets
add_library(tron ra.cu float16.cu) # TODO: add more source files here
set_target_properties(tron PROPERTIES CUDA_ARCHITECTURES 61)

# library properties
target_include_directories(tron
	PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(tron
	PUBLIC
        cxx_std_14)
	#set_target_properties(tron
	#PROPERTIES
	#    CUDA_STANDARD 14)


# library dependencies
target_link_libraries(tron
	PUBLIC
        gsl::gsl-lite-v1
		CUDA::cufft
		CUDA::cublas
		cuda-api-wrappers::cuda-api-wrappers)
# installation: public header files, target
install(
    DIRECTORY
        "${PROJECT_SOURCE_DIR}/include/" # the trailing '/' is significant
    TYPE INCLUDE)
install(
    TARGETS tron
	EXPORT TRON
    ARCHIVE INCLUDES)
