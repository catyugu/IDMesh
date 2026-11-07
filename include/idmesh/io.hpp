#pragma once

#include "idmesh/mesh.hpp"

#include <string>

namespace idmesh {

    // Mesh-only format adapters. Field/state and VTU/PVD/PVTU APIs live
    // in idmesh/vtk.hpp.
    // Gmsh MSH 2.2 / 4.1 ASCII: physical groups and P2 geometry.
    IDMESH_API Mesh read_gmsh(const std::string& filename, BuildOptions options = {});

    // Legacy VTK UNSTRUCTURED_GRID: reads ASCII/BINARY and writes ASCII.
    IDMESH_API Mesh read_vtk_legacy(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_vtk_legacy(const Mesh& mesh, const std::string& filename);

    // Medit ASCII .mesh: linear cells and supported native P2 sections.
    IDMESH_API Mesh read_medit(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_medit(const Mesh& mesh, const std::string& filename);

} // namespace idmesh
