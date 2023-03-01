include(FetchContent)

# fetch gtest if needed
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
FetchContent_MakeAvailable(googletest)

# Enable matplotlibcpp for drawing
option(USE_MATPLOTLIBCPP "Use the matplotlibcpp library" ON)
if(USE_MATPLOTLIBCPP)
  find_package(Python 3.10.8 REQUIRED COMPONENTS Interpreter Development NumPy)
  include_directories(${Python_INCLUDE_DIRS})

  set(pqekf_INCLUDES ${pqekf_INCLUDES} ${CMAKE_SOURCE_DIR}/3rdparty/matplotlibcpp/)
  set(EXTRAS ${EXTRAS} python3.10)
endif(USE_MATPLOTLIBCPP)

add_subdirectory(poser)