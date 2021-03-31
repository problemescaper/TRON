
# TRON library
# Author: Falk Mayer


cmake_minimum_required(VERSION 3.17)

# Define the project.
project(TRON
    VERSION 1.0.0
	LANGUAGES CXX CUDA C)

# Define build options.
option(BUILD_APPS "Build applications" OFF)

# Include target definitions.
add_subdirectory(src)

if(BUILD_APPS)
	add_subdirectory(app)
endif()

# Install.
find_package(CMakeshift 3.7 REQUIRED)
include(CMakeshift/InstallBasicPackageFiles)
cmakeshift_install_basic_package_files(TRON
    COMPATIBILITY SameMajorVersion
    EXPORT TRON
    DEPENDENCIES
        "gsl-lite 0.35.3")
