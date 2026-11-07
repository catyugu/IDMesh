function(idmesh_add_test target test_name labels)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE idmesh)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/support)
    idmesh_stage_runtime(${target})
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

add_subdirectory(unit)

if(IDMESH_BUILD_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()
