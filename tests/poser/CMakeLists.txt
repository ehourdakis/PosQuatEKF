enable_testing()

add_executable(example_poser example_poser.cpp)
target_include_directories(example_poser PUBLIC ${pqekf_INCLUDES})
target_link_libraries(example_poser pqekf ${EXTRAS})
if(USE_MATPLOTLIBCPP)
target_compile_definitions(example_poser PUBLIC USE_MATPLOTLIBCPP)
endif(USE_MATPLOTLIBCPP)

add_test(NAME poser_test COMMAND example_poser)