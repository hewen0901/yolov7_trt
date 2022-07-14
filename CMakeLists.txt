cmake_minimum_required(VERSION 3.0)

project(yolov7_trt)

option(RTX3080TI " set ON COMPILE RTX3080TI SDK"  OFF)
IF(RTX3080TI)
  add_definitions(-DRTX3080TI)
  SET(CMAKE_BUILD_TYPE RELEASE)
  MESSAGE("EnPic Alg is running on RTX3080Ti!")
ELSE()
  SET(CMAKE_BUILD_TYPE RELEASE) #DEBUG
  MESSAGE("EnPic Alg is running on RTX2070Ti!")
ENDIF(RTX3080TI)


SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)
if(CMAKE_COMPILER_IS_GNUCXX)
   add_compile_options(-std=c++14)
   message(STATUS "optional:-std=c++14")
endif(CMAKE_COMPILER_IS_GNUCXX)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")


# TensorRT
set(TENSORRT_ROOT /media/hewen/TensorRT-8.4.1.5/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer 
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER} ) #nvinfer_plugin
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")


# set up Opencv CMake path
set(OpenCV_DIR /usr/local/opencv-3.4.14/build)
find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
if(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()


MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(OpenCV_DIR /usr/local/share/OpenCV)
    ADD_DEFINITIONS(-DLINUX_OS)
    MESSAGE(STATUS "current platform: Linux ")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(OpenCV_DIR F:/projects/camera_qt/opencv-3.4.14/build)
    MESSAGE(STATUS "current platform: Windows")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")


include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/inc
     ${CMAKE_CURRENT_SOURCE_DIR}/export
 )

#AUX_SOURCE_DIRECTORY(. SRCS)
AUX_SOURCE_DIRECTORY(utils SRCS)
AUX_SOURCE_DIRECTORY(src SRCS)
AUX_SOURCE_DIRECTORY(test TEST_SRCS)

SET(PROC_ALL_FILES ${SRCS})
SET(TEST_APP Test_app)


add_library(${PROJECT_NAME} SHARED ${PROC_ALL_FILES})
#add_library(${PROJECT_NAME} STATIC ${PROC_ALL_FILES})
add_executable(${TEST_APP} ${TEST_SRCS})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} pthread ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})
TARGET_LINK_LIBRARIES(${PROJECT_NAME})


TARGET_LINK_LIBRARIES(${TEST_APP} ${OpenCV_LIBS} pthread ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})
TARGET_LINK_LIBRARIES(${TEST_APP} ${PROJECT_NAME})


