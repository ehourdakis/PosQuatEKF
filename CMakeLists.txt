cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(PosQuatEKF
        VERSION 1.0
        DESCRIPTION "An EKF tracking the linearized position and orientation of a robot."
        LANGUAGES CXX C)

SET (CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build configuration" FORCE)

# Kalman filter CXX flags
set(CMAKE_CXX_FLAGS "-Wall -pedantic -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DEIGEN_NO_DEBUG")

# Inlude Eigen
find_package(Eigen3 REQUIRED)
set(EXTRAS ${EXTRAS} Eigen3::Eigen)

set(pqekf_INCLUDES 
  3rdparty/kalman
  include/posquatekf)

set(pqekf_HEADERS
  include/posquatekf/LocalizationMeasurementModel.hpp
  include/posquatekf/MotionModel.hpp
  include/posquatekf/PoseStruct.h
  include/posquatekf/pqekf.h
)

add_library(pqekf SHARED ${pqekf_HEADERS})
target_include_directories(pqekf PUBLIC ${pqekf_INCLUDES})
target_link_libraries(pqekf PUBLIC ${EXTRAS})
set_target_properties(pqekf PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pqekf PROPERTIES LINKER_LANGUAGE CXX)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif(BUILD_TESTS)