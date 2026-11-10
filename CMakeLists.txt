cmake_minimum_required(VERSION 3.20)
project(acvdlss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The NGX static library is built against the static runtime.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# fxc comes with the Windows SDK. Outside a developer prompt, take the newest
# SDK installed in the default place.
file(GLOB SDK_BIN LIST_DIRECTORIES true "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64")
list(SORT SDK_BIN)
list(REVERSE SDK_BIN)
find_program(FXC fxc REQUIRED HINTS "$ENV{WindowsSdkVerBinPath}x64" ${SDK_BIN})

set(SHADER_HEADERS)
foreach(shader prepare rcas downsample motionblur)
    set(header ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader}.h)
    add_custom_command(
        OUTPUT ${header}
        COMMAND ${FXC} /nologo /T cs_5_0 /E main /O3 /Vn ${shader}_cs /Fh ${header} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.hlsl
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.hlsl
        COMMENT "Compiling ${shader}.hlsl")
    list(APPEND SHADER_HEADERS ${header})
endforeach()

set(NGX_LIB ${CMAKE_CURRENT_SOURCE_DIR}/deps/dlss/lib/Windows_x86_64/x64)

add_library(acvdlss SHARED
    src/acvdlss.cpp
    src/taa.cpp
    src/dlss.cpp
    src/passes.cpp
    src/lod.cpp
    src/log.cpp
    src/acvdlss.rc
    deps/reshade/examples/utils/descriptor_tracking.cpp
    ${SHADER_HEADERS})
target_include_directories(acvdlss PRIVATE
    deps/reshade/include
    deps/reshade/examples/utils
    deps/imgui
    deps/dlss/include
    ${CMAKE_CURRENT_BINARY_DIR}/shaders)
target_link_libraries(acvdlss PRIVATE d3d12 dxgi shell32 ole32 version
    $<IF:$<CONFIG:Debug>,${NGX_LIB}/nvsdk_ngx_s_dbg.lib,${NGX_LIB}/nvsdk_ngx_s.lib>)
target_compile_definitions(acvdlss PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(acvdlss PRIVATE /W3 /utf-8)
set_target_properties(acvdlss PROPERTIES SUFFIX ".addon64")
