set(EXAMPLE_FILES 
  3_HTTP_test2.cpp)

file(GLOB DARKLIST "/Net/darknet/obj/*.o")
file(GLOB DARKSRC "/Net/darknet/src/*.c")
file(GLOB DARKCUDA "/Net/darknet/src/*.cu")
file(GLOB proto_files *.proto)

find_package(CUDA)
find_package(Protobuf)
#find_package(CuDNN)
#find_package(OpenCV)

#enable_language(CUDA)
#include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")
#set(CMAKE_CUDA_DEVICE_LINK_LIBRARY 
#    "<CMAKE_CUDA_COMPILER> <CMAKE_CUDA_LINK_FLAGS> <LANGUAGE_COMPILE_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)
  
  if (UNIX AND NOT APPLE)
    set(EXE_NAME "${SOURCE_NAME}.bin")
  elseif (WIN32)
    set(EXE_NAME "${SOURCE_NAME}")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/Net/darknet/include/ -I/usr/local/cuda-8.0/include -fPIC -w -DOPENCV -DGPU -DCUDNN")
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/Net/darknet -L/usr/local/cuda-8.0/lib64")

  #add_library(darknet-cpp-shared SHARED IMPORTED)
  #set_property(TARGET darknet-cpp-shared PROPERTY IMPORTED_LOCATION "/Net/darknet/darknet-cpp-shared.so")
  set(DARKHEADER /Net/darknet/include/darknet.h)
  add_library(mylib ${DARKSRC} ${DARKHEADER})
  target_include_directories(mylib PRIVATE /Net/darknet/include)
  set_source_files_properties(${DARKSRC} PROPERTIES LANGUAGE CXX)

  protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_files})
  add_library(coordproto STATIC ${proto_hdrs} ${proto_srcs})
  target_link_libraries(coordproto PUBLIC ${PROTOBUF_LIBRARIES})
  target_include_directories(coordproto PUBLIC ${PROTOBUF_LIBRARIES})


  cuda_add_executable(${EXE_NAME} ${EXAMPLE_FILE} ${DARKCUDA} ${DARKSRC} ${proto_srcs})
  #target_include_directories(${EXE_NAME} PUBLIC "/Net/darknet/include" PUBLIC "/Net/darknet/src" PUBLIC ${CUDA_INCLUDE_DIRS} PUBLIC ${OpenCV_INCLUDE_DIRS})
  #cuda_add_library(dark_lib ${DARKCUDA})
  #cuda_link_library(${EXE_NAME} ${DARKLIST} ${DARKCUDA})
  #target_link_libraries(${EXE_NAME} ${DARKLIST})
  target_link_libraries(${EXE_NAME} openpose microhttpd ${coordproto} ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${Caffe_LIBS} ${MKL_LIBS} ${GLUT_LIBRARY} ${SPINNAKER_LIB} ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(${EXE_NAME} 
   /usr/local/cuda-8.0/nvvm/lib64/libnvvm.so
   /usr/local/cuda-8.0/lib64/libcudart.so
   /usr/local/cuda-8.0/lib64/libcublas.so
   /usr/local/cuda-8.0/lib64/libcurand.so
   /usr/lib/x86_64-linux-gnu/libcudnn.so.5.1.10
   /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
   /usr/lib/x86_64-linux-gnu/libprotobuf.so)

  if (WIN32)
    set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples/Tutorial/Pose")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${CL}.vcxproj.user
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
    # Properties->General->Output Directory
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
    set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  endif (WIN32)

endforeach()
