#pragma once

#include "idmesh/data.hpp"
#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"

#include <cstdint>
#include <string>

namespace idmesh::vtk {

    enum class LegacyEncoding : std::uint8_t {
        ascii,
        binary
    };

    enum class VtuEncoding : std::uint8_t {
        ascii,
        inline_binary,
        appended_raw
    };

    struct VtuWriteOptions {
        VtuEncoding encoding = VtuEncoding::appended_raw;
    };

    IDMESH_API data::Dataset read_legacy(
        const std::string& filename,
        BuildOptions options = {});

    IDMESH_API void write_legacy(
        const Mesh& mesh, const data::StateView& state,
        const std::string& filename,
        LegacyEncoding encoding = LegacyEncoding::ascii);

    IDMESH_API void write_legacy(
        const data::Dataset& dataset,
        const std::string& filename,
        LegacyEncoding encoding = LegacyEncoding::ascii);

    IDMESH_API data::Dataset read_vtu(
        const std::string& filename,
        BuildOptions options = {});

    IDMESH_API void write_vtu(
        const Mesh& mesh, const data::StateView& state,
        const std::string& filename,
        VtuWriteOptions options = {});

    IDMESH_API void write_vtu(
        const data::Dataset& dataset,
        const std::string& filename,
        VtuWriteOptions options = {});

} // namespace idmesh::vtk
