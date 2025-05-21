cmake_minimum_required(VERSION 3.20)
project(onnx_inference)
set(CMAKE_CXX_STANDARD 20)

set(ONNX_RUNTIME_ROOT "D:/onnxruntime-win-x64-gpu-1.22.0")
set(ONNX_RUNTIME_INCLUDE_DIRS "${ONNX_RUNTIME_ROOT}/include")
set(ONNX_RUNTIME_LIBRARY_DIRS "${ONNX_RUNTIME_ROOT}/lib")

set(OPENCV_ROOT "D:/Program Files/opencv/build")
set(OPENCV_ROOT_INCLUDE_DIRS "${OPENCV_ROOT}/include")

include_directories(${ONNX_RUNTIME_INCLUDE_DIRS} ${OPENCV_ROOT_INCLUDE_DIRS})

# 查找库
find_library(ONNX_RUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNX_RUNTIME_LIBRARY_DIRS})
find_library(ONNX_RUNTIME_SHARED_LIBRARY NAMES onnxruntime_providers_shared HINTS ${ONNX_RUNTIME_LIBRARY_DIRS})
find_library(ONNX_RUNTIME_CUDA_LIBRARY NAMES onnxruntime_providers_cuda HINTS ${ONNX_RUNTIME_LIBRARY_DIRS})

# find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin_10 HINTS ${ONNX_RUNTIME_LIBRARY_DIRS})
# find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser_10 HINTS ${ONNX_RUNTIME_LIBRARY_DIRS})
# find_library(CUDA_LIBRARY NAMES cudart HINTS ${CUDA_LIBRARY_DIRS})
find_library(OPENCV_LIBRARY NAMES opencv_world470 HINTS "${OPENCV_ROOT}/x64/vc16/lib")

aux_source_directory(./includes DIR_INCLUDE)
aux_source_directory(./src DIR_SRCS)

add_executable(onnx_inference main.cpp ${DIR_SRCS} ${DIR_INCLUDE})

# 链接库
target_link_libraries(onnx_inference
    ${OPENCV_LIBRARY}
    ${ONNX_RUNTIME_LIBRARY}
    ${ONNX_RUNTIME_SHARED_LIBRARY}
    ${ONNX_RUNTIME_CUDA_LIBRARY}
)

file(COPY ${CMAKE_SOURCE_DIR}/models/ DESTINATION ${CMAKE_BINARY_DIR}/models/)

# 复制   DLL
set(All_DLLS
    "${ONNX_RUNTIME_ROOT}/lib/onnxruntime.dll"
    "${ONNX_RUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    "${ONNX_RUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
    "${OPENCV_ROOT}/x64/vc16/bin/opencv_world470.dll"
)

foreach(DLL ${All_DLLS})
    add_custom_command(TARGET onnx_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL}"
        $<TARGET_FILE_DIR:onnx_inference>)
endforeach()