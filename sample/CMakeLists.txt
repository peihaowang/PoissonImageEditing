
add_executable(POISSON_IMAGE_EXE
    main.cpp
)

set_target_properties(POISSON_IMAGE_EXE PROPERTIES OUTPUT_NAME PoissonImageEditor)
set_target_properties(POISSON_IMAGE_EXE PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# The modules should placed in specified order because of the interdependencies
target_link_libraries(POISSON_IMAGE_EXE
	POISSON_IMAGE_STATIC
	${3RDPARTY_LIBRARIES}
)

# Installation
install(TARGETS POISSON_IMAGE_EXE
    RUNTIME DESTINATION bin
)
