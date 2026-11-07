idmesh_add_test(idmesh_unit_contracts idmesh.unit.contracts unit
    test_contracts.cpp)
idmesh_add_test(idmesh_unit_data idmesh.unit.data unit
    test_data.cpp)
idmesh_add_test(idmesh_unit_geometry idmesh.unit.geometry unit
    test_geometry.cpp)

idmesh_add_test(idmesh_unit_vtk_common idmesh.unit.vtk_common unit
    test_vtk_common.cpp ${PROJECT_SOURCE_DIR}/src/vtk_common.cpp)
target_include_directories(idmesh_unit_vtk_common PRIVATE
    ${PROJECT_SOURCE_DIR}/src)
