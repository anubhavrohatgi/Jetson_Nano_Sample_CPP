# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.0)

# declare my-recognition project
project(object_detection)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

#for OpenCV libraries version > 4
#Find and add opencv libraries
find_package(OpenCV REQUIRED core imgproc highgui imgcodecs video)

if(${OpenCV_VERSION} VERSION_LESS 4.0.0)
    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}")
else()
    MESSAGE("OpenCV version found : ${OpenCV_VERSION}")
    MESSAGE("OpenCV include path  : ${OpenCV_INCLUDE_DIRS}")
    MESSAGE("OpenCV libs          : ${OpenCV_LIBS}")
ENDIF()

include_directories( ${OpenCV_INCLUDE_DIRS} )

# CUDA 
find_package(CUDA)

# compile the my-recognition program
cuda_add_executable(object_detection object_detection.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(object_detection jetson-inference ${OpenCV_LIBS})
