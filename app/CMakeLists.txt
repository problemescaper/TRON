find_package(Boost REQUIRED COMPONENTS program_options filesystem)

find_package(gsl-lite REQUIRED)

if(BUILD_APPS)
	add_executable(tron-cli tron.cu)

target_compile_options(tron-cli PRIVATE --expt-relaxed-constexpr --expt-extended-lambda)
set_target_properties(tron-cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tron-cli PROPERTIES CUDA_ARCHITECTURES 61)

target_link_libraries(
	tron-cli PUBLIC TRON Boost::filesystem Boost::program_options
                    gsl::gsl-lite
)
endif()

