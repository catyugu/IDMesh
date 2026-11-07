idmesh_add_test(idmesh_integration_mesh idmesh.integration.mesh integration
    test_mesh.cpp)
idmesh_add_test(idmesh_integration_formats idmesh.integration.formats integration
    test_formats.cpp)
idmesh_add_test(idmesh_integration_vtk_data idmesh.integration.vtk_data integration
    test_vtk_data_io.cpp)
idmesh_add_test(idmesh_integration_vtk_manifest idmesh.integration.vtk_manifest integration
    test_vtk_manifest.cpp)
idmesh_add_test(idmesh_integration_quadratic_io idmesh.integration.quadratic_io integration
    test_quadratic_io.cpp)
idmesh_add_test(idmesh_integration_partition idmesh.integration.partition integration
    test_partition.cpp)
idmesh_add_test(idmesh_integration_mesh_pipeline idmesh.integration.mesh_pipeline integration
    test_mesh_pipeline.cpp)
target_compile_definitions(idmesh_integration_mesh_pipeline PRIVATE
    IDMESH_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# GPU-language source properties are directory-scoped.
if(IDMESH_DEVICE_LANGUAGE)
    set_source_files_properties(test_kokkos.cpp test_mpi_kokkos.cpp
        PROPERTIES LANGUAGE ${IDMESH_DEVICE_LANGUAGE})
endif()

if(IDMESH_ENABLE_KOKKOS)
    add_executable(idmesh_integration_kokkos test_kokkos.cpp)
    target_link_libraries(idmesh_integration_kokkos PRIVATE
        idmesh_kokkos Kokkos::kokkos)
    idmesh_stage_runtime(idmesh_integration_kokkos)
    add_test(NAME idmesh.integration.kokkos
        COMMAND idmesh_integration_kokkos)
    set_tests_properties(idmesh.integration.kokkos
        PROPERTIES LABELS "integration;kokkos")
endif()

if(IDMESH_ENABLE_MPI)
    add_executable(idmesh_integration_mpi_shards test_mpi_shards.cpp)
    target_link_libraries(idmesh_integration_mpi_shards PRIVATE
        idmesh_mpi MPI::MPI_CXX)
    idmesh_stage_runtime(idmesh_integration_mpi_shards)
    foreach(_ranks IN ITEMS 2 3)
        add_test(NAME idmesh.integration.mpi_shards.${_ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:idmesh_integration_mpi_shards>
                ${MPIEXEC_POSTFLAGS})
        set_tests_properties(idmesh.integration.mpi_shards.${_ranks}
            PROPERTIES LABELS "integration;mpi")
    endforeach()
endif()

if(IDMESH_ENABLE_MPI AND IDMESH_ENABLE_KOKKOS)
    add_executable(idmesh_integration_mpi_kokkos test_mpi_kokkos.cpp)
    target_link_libraries(idmesh_integration_mpi_kokkos PRIVATE
        idmesh_mpi_kokkos Kokkos::kokkos MPI::MPI_CXX)
    idmesh_stage_runtime(idmesh_integration_mpi_kokkos)
    add_test(NAME idmesh.integration.mpi_kokkos.2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:idmesh_integration_mpi_kokkos>
            ${MPIEXEC_POSTFLAGS})
    set_tests_properties(idmesh.integration.mpi_kokkos.2
        PROPERTIES LABELS "integration;mpi;kokkos")
endif()
