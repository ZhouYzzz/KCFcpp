cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src)

# FFTW3
include_directories("/usr/local/include")
link_directories("/usr/local/lib")

FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )
target_link_libraries( KCF ${OpenCV_LIBS} fftw3f)
