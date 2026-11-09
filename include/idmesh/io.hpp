#pragma once

#include "idmesh/mesh.hpp"

#include <string>

namespace idmesh {

    // Mesh-only format adapters. Field/state and VTU/PVD/PVTU APIs live
    // in idmesh/vtk.hpp.
    // Gmsh MSH 2.2 / 4.1, ASCII or binary. Physical labels and supported
    // first/second-order cell geometry are retained by the importer.
    IDMESH_API Mesh read_gmsh(const std::string& filename, BuildOptions options = {});

    enum class GmshVersion : std::uint8_t { v22, v41 };
    enum class GmshEncoding : std::uint8_t { ascii, binary };

    struct GmshWriteOptions {
        GmshVersion version = GmshVersion::v41;
        GmshEncoding encoding = GmshEncoding::binary;
    };

    // Writes all cells and labeled lower-dimensional entities. MSH 4.1
    // retains multiple physical groups per entity; MSH 2.2 cannot encode
    // multiple groups on a top-dimensional cell and rejects that case.
    IDMESH_API void write_gmsh(const Mesh& mesh, const std::string& filename,
        GmshWriteOptions options = {});

    // Legacy VTK UNSTRUCTURED_GRID: reads ASCII/BINARY and writes ASCII.
    IDMESH_API Mesh read_vtk_legacy(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_vtk_legacy(const Mesh& mesh, const std::string& filename);

    // Medit ASCII .mesh: linear cells and supported native P2 sections.
    IDMESH_API Mesh read_medit(const std::string& filename,
        BuildOptions options = {});
    IDMESH_API void write_medit(const Mesh& mesh, const std::string& filename);

} // namespace idmesh
