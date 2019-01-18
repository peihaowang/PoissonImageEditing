
cmake_minimum_required(VERSION 2.8)

project(PoissonImageEdit)

# Enable C++11 for new language features
set(CMAKE_CXX_STANDARD 11)
# Release/Debug are available. 'Release' by default
set(CMAKE_BUILD_TYPE Release)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Path to 3rdparty library
set(3RDPARTY_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib")

# Pre-define the extension name of library files on different platforms
if(WIN32)
	set(EXT_DYNAMIC_LIBRARY "dll")
elseif(APPLE)
	set(EXT_DYNAMIC_LIBRARY "dylib")
elseif(UNIX)
    set(EXT_DYNAMIC_LIBRARY "so")
endif()

if(WIN32)
	set(EXT_STATIC_LIBRARY "lib")
elseif(APPLE)
	set(EXT_STATIC_LIBRARY "a")
elseif(UNIX)
    set(EXT_STATIC_LIBRARY "a")
endif()

function(FILE_EXISTS FOUND FILES)
	set(${FOUND} TRUE PARENT_SCOPE)
	foreach(FILE ${FILES})
		if(NOT EXISTS ${FILE})
			set(${FOUND} FALSE PARENT_SCOPE)
			return()
		endif()
	endforeach()
endfunction()

# Eigen3
find_package(EIGEN3 QUIET)
if(NOT EIGEN3_FOUND)
	set(EIGEN3_INCLUDE_DIRS "${3RDPARTY_LIBRARY_DIR}/eigen3")
endif()
FILE_EXISTS(EIGEN3_FOUND ${EIGEN3_INCLUDE_DIRS})
if(NOT EIGEN3_FOUND)
	message(FATAL_ERROR "Failed to find Eigen3.")
endif()

# OpenCV
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "${3RDPARTY_LIBRARY_DIR}/opencv/include")
	set(OpenCV_LIBRARY_DIR "${3RDPARTY_LIBRARY_DIR}/opencv/lib")
	file(GLOB OpenCV_LIBRARIES LIST_DIRECTORIES FALSE "${OpenCV_LIBRARY_DIR}/*opencv*.${EXT_STATIC_LIBRARY}")
	if(NOT OpenCV_LIBRARIES)
		file(GLOB OpenCV_LIBRARIES LIST_DIRECTORIES FALSE "${OpenCV_LIBRARY_DIR}/*opencv*.${EXT_DYNAMIC_LIBRARY}")
	endif()
	FILE_EXISTS(OpenCV_FOUND ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBRARIES})
endif()
if(NOT OpenCV_FOUND)
	message(FATAL_ERROR "Failed to find OpenCV.")
endif()

set(3RDPARTY_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

set(3RDPARTY_LIBRARIES
	${OpenCV_LIBRARIES}
)

# Specify the directories to headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${3RDPARTY_INCLUDE_DIRS})

add_executable(POISSON_IMAGE_EDIT
    src/main.cpp
    src/PoissonImage.cpp
)

set_target_properties(POISSON_IMAGE_EDIT PROPERTIES OUTPUT_NAME PoissonImageEdit)
set_target_properties(POISSON_IMAGE_EDIT PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# The modules should placed in specified order because of the interdependencies
target_link_libraries(POISSON_IMAGE_EDIT
	${3RDPARTY_LIBRARIES}
)
