# Public core and independently linkable optional backends.
function(idmesh_configure_target target export_name)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(${target} PROPERTIES
        EXPORT_NAME ${export_name}
        VERSION ${PROJECT_VERSION}
        SOVERSION ${IDMESH_ABI_VERSION}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# Tests and examples live in subdirectories, so stage transitively required
# DLLs beside each executable on Windows. On other platforms this is a no-op.
function(idmesh_stage_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${target}> $<TARGET_RUNTIME_DLLS:${target}>
        COMMAND_EXPAND_LISTS)
endfunction()

add_library(idmesh src/mesh.cpp src/geometry.cpp src/gmsh.cpp src/vtk.cpp src/medit.cpp)
add_library(IDMesh::IDMesh ALIAS idmesh)
idmesh_configure_target(idmesh IDMesh)
set_target_properties(idmesh PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(idmesh PRIVATE IDMESH_EXPORTS PUBLIC IDMESH_SHARED)
endif()

set(IDMESH_SDK_TARGETS idmesh)

if(IDMESH_ENABLE_KOKKOS)
    add_library(idmesh_kokkos SHARED src/kokkos.cpp)
    add_library(IDMesh::Kokkos ALIAS idmesh_kokkos)
    idmesh_configure_target(idmesh_kokkos Kokkos)
    target_compile_definitions(idmesh_kokkos PRIVATE IDMESH_KOKKOS_EXPORTS PUBLIC IDMESH_KOKKOS_SHARED)
    target_link_libraries(idmesh_kokkos PUBLIC idmesh PRIVATE Kokkos::kokkos)
    list(APPEND IDMESH_SDK_TARGETS idmesh_kokkos)
endif()

if(IDMESH_ENABLE_MPI)
    add_library(idmesh_mpi SHARED src/mpi.cpp src/mpi_shard.cpp src/mpi_partition.cpp)
    add_library(IDMesh::MPI ALIAS idmesh_mpi)
    idmesh_configure_target(idmesh_mpi MPI)
    target_compile_definitions(idmesh_mpi PRIVATE IDMESH_MPI_EXPORTS PUBLIC IDMESH_MPI_SHARED)
    target_link_libraries(idmesh_mpi PUBLIC idmesh PRIVATE MPI::MPI_CXX)
    list(APPEND IDMESH_SDK_TARGETS idmesh_mpi)
endif()

if(IDMESH_ENABLE_MPI AND IDMESH_ENABLE_KOKKOS)
    add_library(idmesh_mpi_kokkos SHARED src/mpi_kokkos.cpp)
    add_library(IDMesh::MPIKokkos ALIAS idmesh_mpi_kokkos)
    idmesh_configure_target(idmesh_mpi_kokkos MPIKokkos)
    target_compile_definitions(idmesh_mpi_kokkos PRIVATE IDMESH_MPI_KOKKOS_EXPORTS
        PUBLIC IDMESH_MPI_KOKKOS_SHARED)
    target_link_libraries(idmesh_mpi_kokkos PUBLIC idmesh_mpi idmesh_kokkos
        PRIVATE Kokkos::kokkos MPI::MPI_CXX)
    list(APPEND IDMESH_SDK_TARGETS idmesh_mpi_kokkos)
endif()
