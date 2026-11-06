# Install targets together so the exported package retains the same namespaces.
include(CMakePackageConfigHelpers)

install(TARGETS ${IDMESH_SDK_TARGETS} EXPORT IDMeshTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/idmesh"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/IDMesh)
install(EXPORT IDMeshTargets NAMESPACE IDMesh:: FILE IDMeshTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/IDMeshConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/IDMeshConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/IDMeshConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES
    "${PROJECT_BINARY_DIR}/IDMeshConfig.cmake"
    "${PROJECT_BINARY_DIR}/IDMeshConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/IDMesh)
