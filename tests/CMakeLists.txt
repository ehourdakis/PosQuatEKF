include(FetchContent)

# fetch gtest if needed
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
FetchContent_MakeAvailable(googletest)

option(USE_GNUPLOT "Use the gnuplot library" ON)
if(USE_GNUPLOT)
  set(pqekf_INCLUDES ${pqekf_INCLUDES} ${CMAKE_SOURCE_DIR}/3rdparty/matplotlibcpp/)
  find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)
  set(EXTRAS ${EXTRAS} ${Boost_LIBRARIES} -pthread)
endif(USE_GNUPLOT)

add_subdirectory(poser)