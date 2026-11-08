cmake_minimum_required(VERSION 3.26)
project(IDMeshConsumer LANGUAGES CXX)
enable_testing()

# Intentionally lock the source-compatibility fixture to the current 0.3.x series.
find_package(IDMesh 0.3 CONFIG REQUIRED)

function(idmesh_add_installed_test target component)
    add_test(NAME idmesh.consumer.${component} COMMAND ${target})
    if(WIN32)
        # Installed Windows DLLs reside in bin/, outside the consumer build tree.
        # CTest prepends their directories without requiring a global PATH edit.
        set_tests_properties(idmesh.consumer.${component} PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:IDMesh::${component}>;PATH=path_list_prepend:$<TARGET_FILE_DIR:IDMesh::IDMesh>")
    endif()
endfunction()

add_executable(idmesh_consumer main.cpp io.cpp)
target_link_libraries(idmesh_consumer PRIVATE IDMesh::IDMesh)
idmesh_add_installed_test(idmesh_consumer IDMesh)

if(TARGET IDMesh::MPI)
    add_executable(idmesh_mpi_consumer mpi.cpp)
    target_link_libraries(idmesh_mpi_consumer PRIVATE IDMesh::MPI)
    idmesh_add_installed_test(idmesh_mpi_consumer MPI)
endif()
if(TARGET IDMesh::Kokkos)
    add_executable(idmesh_kokkos_consumer kokkos.cpp)
    target_link_libraries(idmesh_kokkos_consumer PRIVATE IDMesh::Kokkos)
    idmesh_add_installed_test(idmesh_kokkos_consumer Kokkos)
endif()
if(TARGET IDMesh::MPIKokkos)
    add_executable(idmesh_mpi_kokkos_consumer mpi_kokkos.cpp)
    target_link_libraries(idmesh_mpi_kokkos_consumer PRIVATE IDMesh::MPIKokkos)
    idmesh_add_installed_test(idmesh_mpi_kokkos_consumer MPIKokkos)
endif()
