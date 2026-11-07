#pragma once

#include "idmesh/data.hpp"
#include "idmesh/export.hpp"
#include "idmesh/mesh.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

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

    struct PvdEntry {
        double time = 0.0;
        std::uint32_t part = 0;
        std::string group;
        std::string file;
    };

    struct PvdCollection {
        std::vector<PvdEntry> entries;
    };

    IDMESH_API PvdCollection read_pvd(
        const std::string& filename);
    IDMESH_API void write_pvd(
        const PvdCollection& collection,
        const std::string& filename);

    // Serial time-series convenience layer. Each StateView must carry a finite,
    // strictly increasing time. The writer creates sibling .vtu files and a
    // relative-path .pvd manifest; the reader accepts only one serial .vtu
    // dataset per timestep (group="", part=0).
    IDMESH_API void write_vtu_series(
        const Mesh& mesh,
        std::span<const data::StateView> states,
        const std::string& pvd_filename,
        VtuWriteOptions options = {});
    IDMESH_API std::vector<data::Dataset> read_vtu_series(
        const std::string& pvd_filename,
        BuildOptions options = {});

    struct ArraySchema {
        std::string name;
        data::ScalarType type = data::ScalarType::float64;
        std::uint32_t components = 1;
    };

    struct PvtuManifest {
        std::uint32_t ghost_level = 0;
        data::ScalarType points_type = data::ScalarType::float64;
        std::vector<ArraySchema> point_data;
        std::vector<ArraySchema> cell_data;
        std::vector<std::string> pieces;
    };

    // Build the PUnstructuredGrid schema for a set of .vtu pieces. The caller
    // is responsible for ensuring that every piece has the same array schema.
    // Whole-Mesh fields and state time/cycle have no PVTU schema entry and are
    // therefore rejected rather than discarded.
    IDMESH_API PvtuManifest make_pvtu_manifest(
        const Mesh& representative_mesh,
        const data::StateView& representative_state,
        std::span<const std::string> pieces,
        std::uint32_t ghost_level = 0);

    IDMESH_API PvtuManifest read_pvtu(
        const std::string& filename);
    IDMESH_API void write_pvtu(
        const PvtuManifest& manifest,
        const std::string& filename);

} // namespace idmesh::vtk
