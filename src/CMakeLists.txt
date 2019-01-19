
add_library(POISSON_IMAGE_SHARED
	SHARED
    PoissonImage.cpp
)

set_target_properties(POISSON_IMAGE_SHARED PROPERTIES OUTPUT_NAME PoissonImage)
set_target_properties(POISSON_IMAGE_SHARED PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# The modules should placed in specified order because of the interdependencies
target_link_libraries(POISSON_IMAGE_SHARED
	${3RDPARTY_LIBRARIES}
)

###############################################################################

add_library(POISSON_IMAGE_STATIC
	STATIC
    PoissonImage.cpp
)

set_target_properties(POISSON_IMAGE_STATIC PROPERTIES OUTPUT_NAME PoissonImage)
set_target_properties(POISSON_IMAGE_STATIC PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# The modules should placed in specified order because of the interdependencies
target_link_libraries(POISSON_IMAGE_STATIC
	${3RDPARTY_LIBRARIES}
)
