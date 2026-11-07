function(idmesh_add_core_test target source name)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE idmesh)
    idmesh_stage_runtime(${target})
    add_test(NAME ${name} COMMAND ${target})
endfunction()

idmesh_add_core_test(idmesh_test test_mesh.cpp idmesh.topology)
idmesh_add_core_test(idmesh_contracts_test test_contracts.cpp idmesh.contracts)
idmesh_add_core_test(idmesh_data_test test_data.cpp idmesh.data)
idmesh_add_core_test(idmesh_formats_test test_formats.cpp idmesh.formats)
idmesh_add_core_test(idmesh_vtk_common_test test_vtk_common.cpp idmesh.vtk_common)
target_include_directories(idmesh_vtk_common_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
idmesh_add_core_test(idmesh_geometry_test test_geometry.cpp idmesh.geometry)
idmesh_add_core_test(idmesh_quadratic_io_test test_quadratic_io.cpp idmesh.quadratic_io)
idmesh_add_core_test(idmesh_partition_test test_partition.cpp idmesh.partition)

# Source LANGUAGE properties are directory-scoped: set them where the
# CUDA/HIP test targets are declared, not only in the root CMakeLists.
if(IDMESH_DEVICE_LANGUAGE)
    set_source_files_properties(test_kokkos.cpp test_mpi_kokkos.cpp
        PROPERTIES LANGUAGE ${IDMESH_DEVICE_LANGUAGE})
endif()

if(IDMESH_ENABLE_KOKKOS)
    add_executable(idmesh_kokkos_test test_kokkos.cpp)
    target_link_libraries(idmesh_kokkos_test PRIVATE idmesh_kokkos Kokkos::kokkos)
    idmesh_stage_runtime(idmesh_kokkos_test)
    add_test(NAME idmesh.kokkos COMMAND idmesh_kokkos_test)
endif()

if(IDMESH_ENABLE_MPI)
    add_executable(idmesh_mpi_shards_test test_mpi_shards.cpp)
    target_link_libraries(idmesh_mpi_shards_test PRIVATE idmesh_mpi MPI::MPI_CXX)
    idmesh_stage_runtime(idmesh_mpi_shards_test)
    foreach(_ranks IN ITEMS 2 3)
        add_test(NAME idmesh.mpi.shards.${_ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:idmesh_mpi_shards_test> ${MPIEXEC_POSTFLAGS})
    endforeach()
endif()

if(IDMESH_ENABLE_MPI AND IDMESH_ENABLE_KOKKOS)
    add_executable(idmesh_mpi_kokkos_test test_mpi_kokkos.cpp)
    target_link_libraries(idmesh_mpi_kokkos_test PRIVATE
        idmesh_mpi_kokkos Kokkos::kokkos MPI::MPI_CXX)
    idmesh_stage_runtime(idmesh_mpi_kokkos_test)
    add_test(NAME idmesh.mpi.kokkos.2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:idmesh_mpi_kokkos_test> ${MPIEXEC_POSTFLAGS})
endif()
