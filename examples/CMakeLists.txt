foreach(_example IN ITEMS fvm fem basix_orientation)
    add_executable(idmesh_${_example} ${_example}.cpp)
    target_link_libraries(idmesh_${_example} PRIVATE idmesh)
    idmesh_stage_runtime(idmesh_${_example})
endforeach()

if(IDMESH_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    add_test(NAME idmesh.example.fvm COMMAND idmesh_fvm)
    add_test(NAME idmesh.example.fem COMMAND idmesh_fem)
    set_tests_properties(idmesh.example.fvm idmesh.example.fem
        PROPERTIES LABELS "integration;example")
    if(IDMESH_TEST_BASIX)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME idmesh.example.basix_orientation
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/integration/test_basix.py
                    $<TARGET_FILE:idmesh_basix_orientation>)
        set_tests_properties(idmesh.example.basix_orientation
            PROPERTIES LABELS "integration;example;basix")
    endif()
endif()
