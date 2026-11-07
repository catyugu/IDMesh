# Optional backend dependencies and their build-language requirements.
if(IDMESH_ENABLE_KOKKOS)
    if(IDMESH_KOKKOS_DEVICE_BACKEND STREQUAL "CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
        set(IDMESH_DEVICE_LANGUAGE CUDA)
    elseif(IDMESH_KOKKOS_DEVICE_BACKEND STREQUAL "HIP")
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 20)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)
        set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
        set(IDMESH_DEVICE_LANGUAGE HIP)
    elseif(NOT IDMESH_KOKKOS_DEVICE_BACKEND STREQUAL "SERIAL")
        message(FATAL_ERROR "Unsupported IDMESH_KOKKOS_DEVICE_BACKEND")
    endif()

    if(IDMESH_DEVICE_LANGUAGE)
        set_source_files_properties(src/device/kokkos.cpp src/device/mpi_kokkos.cpp
            PROPERTIES LANGUAGE ${IDMESH_DEVICE_LANGUAGE})
    endif()

    # Kokkos static archives must be suitable for linking into shared adapters.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(NOT TARGET Kokkos::kokkos)
        include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")
        CPMFindPackage(NAME Kokkos VERSION 4.5.01
            GITHUB_REPOSITORY kokkos/kokkos GIT_TAG 4.5.01
            OPTIONS "Kokkos_ENABLE_SERIAL ON" "Kokkos_ENABLE_TESTS OFF"
                    "Kokkos_ENABLE_EXAMPLES OFF")
    endif()
endif()

if(IDMESH_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()
