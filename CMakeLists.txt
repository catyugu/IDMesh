cmake_minimum_required(VERSION 3.26)
if(TARGET IDMesh::IDMesh)
    return()
endif()
project(IDMesh VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(IDMESH_BUILD_TESTS "Build IDMesh tests" ${PROJECT_IS_TOP_LEVEL})
option(IDMESH_BUILD_EXAMPLES "Build IDMesh examples" ${PROJECT_IS_TOP_LEVEL})
option(IDMESH_BUILD_BENCHMARKS "Build development-only performance benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(IDMESH_ENABLE_KOKKOS "Build optional Kokkos backend" OFF)
option(IDMESH_ENABLE_MPI "Build optional MPI distribution backend" OFF)
option(IDMESH_INSTALL "Install IDMesh SDK" ${PROJECT_IS_TOP_LEVEL})

add_library(idmesh src/mesh.cpp src/geometry.cpp src/gmsh.cpp src/vtk.cpp src/medit.cpp)
add_library(IDMesh::IDMesh ALIAS idmesh)
target_compile_features(idmesh PUBLIC cxx_std_20)
target_include_directories(idmesh PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(idmesh PROPERTIES
    EXPORT_NAME IDMesh POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION} SOVERSION 0)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(idmesh PRIVATE IDMESH_EXPORTS PUBLIC IDMESH_SHARED)
endif()

if(IDMESH_ENABLE_KOKKOS)
    # Kokkos static archives are linked into the SDK's shared adapter.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    include(cmake/CPM.cmake)
    CPMFindPackage(NAME Kokkos VERSION 4.5.01
        GITHUB_REPOSITORY kokkos/kokkos GIT_TAG 4.5.01
        OPTIONS "Kokkos_ENABLE_SERIAL ON" "Kokkos_ENABLE_TESTS OFF"
                "Kokkos_ENABLE_EXAMPLES OFF")
    add_library(idmesh_kokkos SHARED src/kokkos.cpp)
    add_library(IDMesh::Kokkos ALIAS idmesh_kokkos)
    target_compile_features(idmesh_kokkos PUBLIC cxx_std_20)
    target_compile_definitions(idmesh_kokkos PRIVATE IDMESH_KOKKOS_EXPORTS PUBLIC IDMESH_KOKKOS_SHARED)
    target_link_libraries(idmesh_kokkos PUBLIC idmesh PRIVATE Kokkos::kokkos)
    target_include_directories(idmesh_kokkos PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(idmesh_kokkos PROPERTIES EXPORT_NAME Kokkos)
endif()

if(IDMESH_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(idmesh_mpi SHARED src/mpi.cpp src/mpi_shard.cpp src/mpi_partition.cpp)
    add_library(IDMesh::MPI ALIAS idmesh_mpi)
    target_compile_features(idmesh_mpi PUBLIC cxx_std_20)
    target_compile_definitions(idmesh_mpi PRIVATE IDMESH_MPI_EXPORTS PUBLIC IDMESH_MPI_SHARED)
    target_link_libraries(idmesh_mpi PUBLIC idmesh PRIVATE MPI::MPI_CXX)
    target_include_directories(idmesh_mpi PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(idmesh_mpi PROPERTIES EXPORT_NAME MPI)
endif()

if(IDMESH_BUILD_TESTS)
    if(PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_executable(idmesh_test tests/test_mesh.cpp)
    target_link_libraries(idmesh_test PRIVATE idmesh)
    add_test(NAME idmesh.topology COMMAND idmesh_test)
    add_executable(idmesh_contracts_test tests/test_contracts.cpp)
    target_link_libraries(idmesh_contracts_test PRIVATE idmesh)
    add_test(NAME idmesh.contracts COMMAND idmesh_contracts_test)
    add_executable(idmesh_formats_test tests/test_formats.cpp)
    target_link_libraries(idmesh_formats_test PRIVATE idmesh)
    add_test(NAME idmesh.formats COMMAND idmesh_formats_test)
    add_executable(idmesh_geometry_test tests/test_geometry.cpp)
    target_link_libraries(idmesh_geometry_test PRIVATE idmesh)
    add_test(NAME idmesh.geometry COMMAND idmesh_geometry_test)
    add_executable(idmesh_quadratic_io_test tests/test_quadratic_io.cpp)
    target_link_libraries(idmesh_quadratic_io_test PRIVATE idmesh)
    add_test(NAME idmesh.quadratic_io COMMAND idmesh_quadratic_io_test)
    if(IDMESH_ENABLE_KOKKOS)
        add_executable(idmesh_kokkos_test tests/test_kokkos.cpp)
        target_link_libraries(idmesh_kokkos_test PRIVATE idmesh_kokkos Kokkos::kokkos)
        add_test(NAME idmesh.kokkos COMMAND idmesh_kokkos_test)
    endif()
    if(IDMESH_ENABLE_MPI)
        add_executable(idmesh_mpi_test tests/test_mpi.cpp)
        target_link_libraries(idmesh_mpi_test PRIVATE idmesh_mpi MPI::MPI_CXX)
        add_test(NAME idmesh.mpi.2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:idmesh_mpi_test> ${MPIEXEC_POSTFLAGS})
        add_test(NAME idmesh.mpi.3
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:idmesh_mpi_test> ${MPIEXEC_POSTFLAGS})
        add_executable(idmesh_mpi_shards_test tests/test_mpi_shards.cpp)
        target_link_libraries(idmesh_mpi_shards_test PRIVATE idmesh_mpi MPI::MPI_CXX)
        foreach(_ranks IN ITEMS 2 3)
            add_test(NAME idmesh.mpi.shards.${_ranks}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:idmesh_mpi_shards_test> ${MPIEXEC_POSTFLAGS})
        endforeach()
    endif()
endif()

if(IDMESH_BUILD_EXAMPLES)
    add_executable(idmesh_example examples/main.cpp)
    target_link_libraries(idmesh_example PRIVATE idmesh)
    add_executable(idmesh_fvm examples/fvm.cpp)
    target_link_libraries(idmesh_fvm PRIVATE idmesh)
    add_executable(idmesh_feec examples/feec.cpp)
    target_link_libraries(idmesh_feec PRIVATE idmesh)
    add_executable(idmesh_orientation_export examples/basix_orientations.cpp)
    target_link_libraries(idmesh_orientation_export PRIVATE idmesh)
    if(IDMESH_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
        add_test(NAME idmesh.example.fvm COMMAND idmesh_fvm)
        add_test(NAME idmesh.example.feec COMMAND idmesh_feec)
    endif()
endif()

if(IDMESH_BUILD_BENCHMARKS)
    add_executable(idmesh_benchmark benchmarks/topology.cpp)
    target_link_libraries(idmesh_benchmark PRIVATE idmesh)
endif()

if(IDMESH_INSTALL)
    set(_targets idmesh)
    if(IDMESH_ENABLE_KOKKOS)
        list(APPEND _targets idmesh_kokkos)
    endif()
    if(IDMESH_ENABLE_MPI)
        list(APPEND _targets idmesh_mpi)
    endif()
    install(TARGETS ${_targets} EXPORT IDMeshTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/idmesh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/IDMesh)
    install(EXPORT IDMeshTargets NAMESPACE IDMesh:: FILE IDMeshTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)
    configure_package_config_file(cmake/IDMeshConfig.cmake.in
        ${PROJECT_BINARY_DIR}/IDMeshConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)
    write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/IDMeshConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
    install(FILES ${PROJECT_BINARY_DIR}/IDMeshConfig.cmake
        ${PROJECT_BINARY_DIR}/IDMeshConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)
endif()
