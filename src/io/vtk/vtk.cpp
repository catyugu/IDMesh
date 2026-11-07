#include "contracts.hpp"
#include "idmesh/io.hpp"
#include "idmesh/vtk.hpp"
#include "vtk_common.hpp"
#include "vtk_data_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idmesh {
    namespace {

        constexpr std::string_view physical_tag_name = "physical_tag";
        constexpr std::string_view time_name = "__idmesh_time";
        constexpr std::string_view cycle_name = "__idmesh_cycle";

        enum class LegacyDomain {
            mesh,
            point,
            cell
        };

        struct RawField {
            LegacyDomain domain = LegacyDomain::mesh;
            std::string name;
            data::Array values;
        };

        std::string trim(std::string text)
        {
            while (!text.empty()
                && (text.back() == '\r'
                    || std::isspace(
                        static_cast<unsigned char>(text.back()))))
                text.pop_back();
            std::size_t first = 0;
            while (first < text.size()
                && std::isspace(
                    static_cast<unsigned char>(text[first])))
                ++first;
            return text.substr(first);
        }

        bool next_line(std::istream& in, std::string& line)
        {
            while (std::getline(in, line)) {
                line = trim(std::move(line));
                if (!line.empty())
                    return true;
            }
            return false;
        }

        template <class T>
        std::vector<T> read_binary_values(
            std::istream& in, std::size_t count)
        {
            detail::require_length(
                count <= std::numeric_limits<std::size_t>::max()
                        / sizeof(T),
                "VTK binary block size overflow");
            std::vector<std::byte> bytes(count * sizeof(T));
            in.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
            detail::require_runtime(
                static_cast<std::size_t>(in.gcount())
                    == bytes.size(),
                "truncated VTK binary block");
            if constexpr (sizeof(T) > 1) {
                if (std::endian::native == std::endian::little) {
                    for (std::size_t offset = 0;
                         offset < bytes.size();
                         offset += sizeof(T))
                        std::reverse(
                            bytes.begin() + offset,
                            bytes.begin() + offset
                                + sizeof(T));
                }
            }
            std::vector<T> result(count);
            std::memcpy(
                result.data(), bytes.data(), bytes.size());
            if (in.peek() == '\r')
                in.get();
            if (in.peek() == '\n')
                in.get();
            return result;
        }

        data::Array read_binary_array(
            std::istream& in, data::ScalarType type,
            std::size_t tuples, std::uint32_t components)
        {
            const auto bytes = data::scalar_size(type);
            detail::require_length(
                components > 0
                    && tuples
                        <= std::numeric_limits<std::size_t>::max()
                            / components
                    && tuples * components
                        <= std::numeric_limits<std::size_t>::max()
                            / bytes,
                "VTK binary array size overflow");
            std::vector<std::byte> raw(
                tuples * components * bytes);
            in.read(reinterpret_cast<char*>(raw.data()),
                static_cast<std::streamsize>(raw.size()));
            detail::require_runtime(
                static_cast<std::size_t>(in.gcount())
                    == raw.size(),
                "truncated VTK binary array");
            if (in.peek() == '\r')
                in.get();
            if (in.peek() == '\n')
                in.get();
            return detail::vtk::array_from_endian_bytes(
                type, tuples, components,
                std::move(raw), std::endian::big);
        }

        data::Array read_legacy_array(
            std::istream& in, bool binary,
            data::ScalarType type, std::size_t tuples,
            std::uint32_t components)
        {
            if (binary)
                return read_binary_array(
                    in, type, tuples, components);
            auto result = detail::vtk::read_ascii_values(
                in, type, tuples, components);
            std::string rest;
            std::getline(in, rest);
            return result;
        }

        void write_binary_array(
            std::ostream& out,
            const data::ArrayView& array)
        {
            const auto bytes = detail::vtk::endian_bytes(
                array, std::endian::big);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
            out.put('\n');
        }

        void write_legacy_array(
            std::ostream& out, bool binary,
            const data::ArrayView& array)
        {
            if (binary)
                write_binary_array(out, array);
            else {
                detail::vtk::write_ascii_values(out, array);
                out.put('\n');
            }
        }

        void validate_legacy_name(std::string_view name)
        {
            detail::require_argument(
                !name.empty(),
                "VTK field name cannot be empty");
            detail::require_argument(
                name != physical_tag_name
                    && name != time_name
                    && name != cycle_name,
                "VTK field name is reserved");
            detail::require_argument(
                std::none_of(name.begin(), name.end(),
                    [](char ch) {
                        return std::isspace(
                            static_cast<unsigned char>(ch));
                    }),
                "VTK legacy field names cannot contain whitespace");
        }

        struct OutputCell {
            int code = 0;
            std::vector<Index> points;
            int tag = 0;
            bool top = false;
        };

        std::vector<OutputCell> legacy_cells(
            const Mesh& mesh)
        {
            std::vector<OutputCell> cells;
            const Index d = mesh.dimension();

            for (const auto& label : mesh.labels()) {
                detail::require_runtime(
                    label.value != 0,
                    "VTK legacy uses zero as the unlabeled sentinel");
                detail::require_runtime(
                    mesh.label_name(
                        label.dimension, label.value).empty(),
                    "VTK legacy cannot preserve label names");
            }

            for (Index cell = 0;
                 cell < mesh.entity_count(d); ++cell) {
                const auto labels = mesh.labels(d, cell);
                detail::require_runtime(
                    labels.size() <= 1,
                    "VTK legacy cannot store multiple labels on one cell");
                auto encoded
                    = detail::vtk::encode_cell(mesh, cell);
                cells.push_back(
                    {encoded.code, std::move(encoded.points),
                        labels.empty() ? 0
                                       : labels.front().value,
                        true});
            }

            for (const auto& label : mesh.labels()) {
                if (label.dimension >= d)
                    continue;
                const int code = detail::vtk::entity_code(
                    label.dimension,
                    static_cast<Index>(label.vertices.size()));
                std::vector<Index> points(label.vertices.begin(), label.vertices.end());
                cells.push_back(
                    {code, std::move(points),
                        label.value, false});
            }
            return cells;
        }

        void write_field_block(
            std::ostream& out, bool binary,
            std::span<const data::MeshDataView> fields)
        {
            if (fields.empty())
                return;
            out << "FIELD FieldData "
                << fields.size() << '\n';
            for (const auto& field : fields) {
                const auto type
                    = detail::vtk::legacy_scalar_name(
                        field.values.type);
                out << field.name << ' '
                    << field.values.components.size() << ' '
                    << field.values.tuples << ' '
                    << type << '\n';
                write_legacy_array(
                    out, binary, field.values);
            }
        }

        void apply_physical_tags(
            std::vector<detail::vtk::ImportedCell>& cells,
            const data::Array& tags)
        {
            detail::require_runtime(
                tags.tuples == cells.size()
                    && tags.components == 1,
                "VTK physical_tag shape mismatch");
            for (std::size_t i = 0;
                 i < cells.size(); ++i) {
                const auto value
                    = detail::vtk::signed_integer(tags, i);
                detail::require_runtime(
                    value >= std::numeric_limits<int>::min()
                        && value
                            <= std::numeric_limits<int>::max(),
                    "VTK physical_tag out of int range");
                cells[i].tag
                    = static_cast<int>(value);
            }
        }

    } // namespace

    namespace vtk {

        data::Dataset read_legacy(
            const std::string& filename,
            BuildOptions options)
        {
            std::ifstream in(
                filename, std::ios::binary);
            detail::require_runtime(
                static_cast<bool>(in),
                "cannot open VTK legacy file");

            std::string line;
            detail::require_runtime(
                static_cast<bool>(std::getline(in, line))
                    && line.rfind(
                        "# vtk DataFile", 0) == 0,
                "invalid VTK legacy header");
            detail::require_runtime(
                static_cast<bool>(std::getline(in, line)),
                "missing VTK title");
            detail::require_runtime(
                static_cast<bool>(std::getline(in, line)),
                "missing VTK encoding");
            line = trim(std::move(line));
            const bool binary = line == "BINARY";
            detail::require_runtime(
                binary || line == "ASCII",
                "unsupported VTK legacy encoding");
            detail::require_runtime(
                static_cast<bool>(std::getline(in, line))
                    && trim(std::move(line))
                        == "DATASET UNSTRUCTURED_GRID",
                "expected VTK UNSTRUCTURED_GRID");

            detail::require_runtime(
                next_line(in, line),
                "missing VTK POINTS");
            std::istringstream point_header(line);
            std::string keyword, point_type;
            std::size_t point_count = 0;
            detail::require_runtime(
                static_cast<bool>(
                    point_header >> keyword
                    >> point_count >> point_type)
                    && keyword == "POINTS"
                    && point_count < invalid_index
                    && (point_type == "float"
                        || point_type == "double"),
                "invalid VTK POINTS");
            const auto coordinate_type
                = point_type == "float"
                ? data::ScalarType::float32
                : data::ScalarType::float64;
            data::Array coordinate_values
                = binary
                ? read_binary_array(
                      in, coordinate_type,
                      point_count, 3)
                : detail::vtk::read_ascii_values(
                      in, coordinate_type,
                      point_count, 3);
            if (!binary)
                std::getline(in, line);
            std::vector<std::array<double, 3>>
                points(point_count);
            for (std::size_t i = 0;
                 i < point_count; ++i) {
                for (std::size_t c = 0; c < 3; ++c) {
                    points[i][c] = detail::vtk::floating_value(
                        coordinate_values, i, c);
                    detail::require_runtime(
                        std::isfinite(points[i][c]),
                        "nonfinite VTK point");
                }
            }

            detail::require_runtime(
                next_line(in, line),
                "missing VTK CELLS");
            std::istringstream cell_header(line);
            std::size_t cell_count = 0, words = 0;
            detail::require_runtime(
                static_cast<bool>(
                    cell_header >> keyword
                    >> cell_count >> words)
                    && keyword == "CELLS"
                    && cell_count < invalid_index,
                "invalid VTK CELLS");
            std::vector<detail::vtk::ImportedCell> cells(cell_count);
            std::size_t actual = 0;
            if (binary) {
                const auto packed
                    = read_binary_values<std::int32_t>(
                        in, words);
                std::size_t cursor = 0;
                for (auto& cell : cells) {
                    detail::require_runtime(
                        cursor < packed.size()
                            && packed[cursor] >= 0
                            && packed[cursor] <= 27,
                        "invalid VTK cell arity");
                    const auto n
                        = static_cast<std::size_t>(
                            packed[cursor++]);
                    actual += n + 1;
                    detail::require_runtime(
                        cursor + n <= packed.size(),
                        "truncated VTK CELLS");
                    cell.points.resize(n);
                    for (auto& id : cell.points) {
                        const auto value
                            = packed[cursor++];
                        detail::require_runtime(
                            value >= 0
                                && static_cast<std::size_t>(
                                       value)
                                    < points.size(),
                            "invalid VTK point ID");
                        id = static_cast<Index>(value);
                    }
                }
                detail::require_runtime(
                    cursor == packed.size(),
                    "VTK CELLS word count mismatch");
            }
            else {
                for (auto& cell : cells) {
                    std::size_t n = 0;
                    detail::require_runtime(
                        static_cast<bool>(in >> n)
                            && n <= 27,
                        "invalid VTK cell arity");
                    actual += n + 1;
                    cell.points.resize(n);
                    for (auto& id : cell.points) {
                        std::uint64_t value = 0;
                        detail::require_runtime(
                            static_cast<bool>(in >> value)
                                && value < points.size(),
                            "invalid VTK point ID");
                        id = static_cast<Index>(value);
                    }
                }
                std::getline(in, line);
            }
            detail::require_runtime(
                words == actual,
                "VTK CELLS word count mismatch");

            detail::require_runtime(
                next_line(in, line),
                "missing VTK CELL_TYPES");
            std::istringstream type_header(line);
            std::size_t type_count = 0;
            detail::require_runtime(
                static_cast<bool>(
                    type_header >> keyword >> type_count)
                    && keyword == "CELL_TYPES"
                    && type_count == cells.size(),
                "invalid VTK CELL_TYPES");
            std::vector<std::int32_t> codes;
            if (binary)
                codes = read_binary_values<std::int32_t>(
                    in, cells.size());
            else {
                codes.resize(cells.size());
                for (auto& code : codes)
                    detail::require_runtime(
                        static_cast<bool>(in >> code),
                        "truncated VTK CELL_TYPES");
                std::getline(in, line);
            }
            for (std::size_t i = 0;
                 i < cells.size(); ++i) {
                const auto info
                    = detail::vtk::cell_info(codes[i]);
                detail::require_runtime(
                    info.nodes == cells[i].points.size(),
                    "invalid VTK cell node count");
                cells[i].type = info.type;
                cells[i].dimension = info.dimension;
            }

            LegacyDomain domain = LegacyDomain::mesh;
            std::size_t domain_tuples = 1;
            std::vector<RawField> raw_fields;
            std::optional<data::Array> physical_tags;

            while (next_line(in, line)) {
                std::istringstream header(line);
                header >> keyword;
                if (keyword == "POINT_DATA"
                    || keyword == "CELL_DATA") {
                    std::size_t count = 0;
                    detail::require_runtime(
                        static_cast<bool>(header >> count),
                        "invalid VTK data header");
                    domain = keyword == "POINT_DATA"
                        ? LegacyDomain::point
                        : LegacyDomain::cell;
                    domain_tuples = domain == LegacyDomain::point
                        ? points.size() : cells.size();
                    detail::require_runtime(
                        count == domain_tuples,
                        "VTK data size mismatch");
                    continue;
                }

                if (keyword == "FIELD") {
                    std::string field_name;
                    std::size_t arrays = 0;
                    detail::require_runtime(
                        static_cast<bool>(
                            header >> field_name >> arrays),
                        "invalid VTK FIELD header");
                    for (std::size_t array = 0;
                         array < arrays; ++array) {
                        detail::require_runtime(
                            next_line(in, line),
                            "truncated VTK FIELD");
                        std::istringstream array_header(line);
                        std::string name, type_name;
                        std::uint32_t components = 0;
                        std::size_t tuples = 0;
                        detail::require_runtime(
                            static_cast<bool>(
                                array_header >> name
                                >> components >> tuples
                                >> type_name),
                            "invalid VTK FIELD array header");
                        const auto type
                            = detail::vtk::legacy_scalar_type(
                                type_name);
                        auto values = read_legacy_array(
                            in, binary, type,
                            tuples, components);
                        if (domain != LegacyDomain::mesh)
                            detail::require_runtime(
                                tuples == domain_tuples,
                                "VTK FIELD tuple count mismatch");
                        else
                            detail::require_runtime(
                                tuples == 1,
                                "IDMesh whole-mesh VTK fields require one tuple");
                        raw_fields.push_back(
                            {domain, std::move(name),
                                std::move(values)});
                    }
                    continue;
                }

                if (keyword == "SCALARS") {
                    detail::require_runtime(
                        domain != LegacyDomain::mesh,
                        "SCALARS outside POINT_DATA/CELL_DATA");
                    std::string name, type_name;
                    std::uint32_t components = 1;
                    detail::require_runtime(
                        static_cast<bool>(
                            header >> name >> type_name),
                        "invalid VTK SCALARS");
                    if (!(header >> components))
                        components = 1;
                    detail::require_runtime(
                        components >= 1 && components <= 4,
                        "invalid VTK SCALARS component count");
                    detail::require_runtime(
                        next_line(in, line),
                        "missing VTK LOOKUP_TABLE");
                    std::istringstream lookup(line);
                    std::string lookup_keyword, table;
                    detail::require_runtime(
                        static_cast<bool>(
                            lookup >> lookup_keyword >> table)
                            && lookup_keyword == "LOOKUP_TABLE",
                        "missing VTK scalar lookup table");
                    const auto type
                        = detail::vtk::legacy_scalar_type(
                            type_name);
                    auto values = read_legacy_array(
                        in, binary, type,
                        domain_tuples, components);
                    if (domain == LegacyDomain::cell
                        && name == physical_tag_name
                        && components == 1) {
                        detail::require_runtime(
                            !physical_tags.has_value(),
                            "duplicate VTK physical_tag array");
                        physical_tags = std::move(values);
                    }
                    else
                        raw_fields.push_back(
                            {domain, std::move(name),
                                std::move(values)});
                    continue;
                }

                throw std::runtime_error(
                    "unsupported VTK attribute section");
            }

            if (physical_tags)
                apply_physical_tags(cells, *physical_tags);

            auto built = detail::vtk::build_imported_mesh(
                std::move(points), cells, options);

            data::State state;
            std::set<std::pair<LegacyDomain, std::string>>
                names;
            for (auto& raw : raw_fields) {
                detail::require_runtime(
                    names.emplace(
                        raw.domain, raw.name).second,
                    "duplicate VTK field name in one association");

                if (raw.domain == LegacyDomain::mesh
                    && detail::vtk::read_state_metadata(
                        state, raw.name, raw.values))
                    continue;
                detail::require_runtime(
                    raw.name != physical_tag_name
                        && raw.name != time_name
                        && raw.name != cycle_name,
                    "reserved VTK metadata name used as a field");

                data::MeshData field;
                field.name = std::move(raw.name);
                field.selection
                    = data::SelectionKind::all;
                if (raw.domain == LegacyDomain::point) {
                    field.location
                        = data::geometry_nodes();
                    field.values = detail::vtk::reorder_point_array(
                        std::move(raw.values),
                        built.point_to_geometry);
                }
                else if (raw.domain == LegacyDomain::cell) {
                    detail::require_runtime(
                        !built.has_lower_cells,
                        "VTK CellData cannot be mapped exactly when lower-dimensional cells are present");
                    field.location
                        = data::entities(
                            built.mesh.dimension());
                    field.values = std::move(raw.values);
                }
                else {
                    field.location = data::whole_mesh();
                    field.values = std::move(raw.values);
                }
                data::validate(built.mesh, field.view());
                state.fields.push_back(std::move(field));
            }

            return {std::move(built.mesh),
                std::move(state)};
        }

        void write_legacy(
            const Mesh& mesh, const data::StateView& state,
            const std::string& filename,
            LegacyEncoding encoding)
        {
            data::validate(mesh, state);
            const bool binary
                = encoding == LegacyEncoding::binary;
            auto cells = legacy_cells(mesh);

            std::vector<data::MeshDataView>
                point_fields, cell_fields, mesh_fields;
            for (const auto& field : state.fields) {
                validate_legacy_name(field.name);
                const auto association
                    = detail::vtk::attribute_association(
                        mesh, field);
                switch (association) {
                case detail::vtk::AttributeAssociation::point:
                    point_fields.push_back(field);
                    break;
                case detail::vtk::AttributeAssociation::cell:
                    cell_fields.push_back(field);
                    break;
                case detail::vtk::AttributeAssociation::field:
                    mesh_fields.push_back(field);
                    break;
                }
                (void)detail::vtk::legacy_scalar_name(
                    field.values.type);
            }

            const bool has_lower = std::any_of(
                cells.begin(), cells.end(),
                [](const OutputCell& cell) {
                    return !cell.top;
                });
            detail::require_argument(
                cell_fields.empty() || !has_lower,
                "VTK legacy CellData cannot coexist with exported lower-dimensional label cells");

            std::array<double, 1> time_storage {};
            std::array<std::int32_t, 1> cycle_storage {};
            std::vector<data::MeshDataView> top_fields
                = mesh_fields;
            data::ArrayView time_values, cycle_values;
            if (state.time) {
                detail::require_argument(
                    std::isfinite(*state.time),
                    "VTK state time must be finite");
                time_storage[0] = *state.time;
                time_values = data::make_scalar_array(
                    std::span(time_storage));
                top_fields.push_back(
                    {time_name, data::whole_mesh(),
                        data::all(), time_values});
            }
            if (state.cycle) {
                detail::require_argument(
                    *state.cycle
                            >= std::numeric_limits<std::int32_t>::min()
                        && *state.cycle
                            <= std::numeric_limits<std::int32_t>::max(),
                    "VTK legacy cycle exceeds portable int32 range");
                cycle_storage[0]
                    = static_cast<std::int32_t>(*state.cycle);
                cycle_values = data::make_scalar_array(
                    std::span(cycle_storage));
                top_fields.push_back(
                    {cycle_name, data::whole_mesh(),
                        data::all(), cycle_values});
            }

            std::ofstream out(
                filename, std::ios::binary);
            detail::require_runtime(
                static_cast<bool>(out),
                "cannot open VTK output");
            out << "# vtk DataFile Version 3.0\n"
                   "IDMesh\n"
                << (binary ? "BINARY\n" : "ASCII\n")
                << "DATASET UNSTRUCTURED_GRID\n";

            out << "POINTS " << mesh.geometry_node_count()
                << " double\n";
            const auto coordinates = detail::vtk::point_coordinates(mesh);
            const auto coordinate_array
                = data::make_interleaved_array(
                    std::span(coordinates), 3);
            write_legacy_array(
                out, binary, coordinate_array);

            std::size_t words = 0;
            for (const auto& cell : cells)
                words += cell.points.size() + 1;
            out << "CELLS " << cells.size()
                << ' ' << words << '\n';
            if (binary) {
                std::vector<std::int32_t> packed;
                packed.reserve(words);
                for (const auto& cell : cells) {
                    detail::require_length(
                        cell.points.size()
                            <= static_cast<std::size_t>(
                                std::numeric_limits<std::int32_t>::max()),
                        "VTK cell arity exceeds int32");
                    packed.push_back(
                        static_cast<std::int32_t>(
                            cell.points.size()));
                    for (Index point : cell.points) {
                        detail::require_length(
                            point <= static_cast<Index>(
                                std::numeric_limits<std::int32_t>::max()),
                            "VTK point ID exceeds int32");
                        packed.push_back(
                            static_cast<std::int32_t>(point));
                    }
                }
                auto array = data::make_scalar_array(
                    std::span(packed));
                write_binary_array(out, array);
            }
            else {
                for (const auto& cell : cells) {
                    out << cell.points.size();
                    for (Index point : cell.points)
                        out << ' ' << point;
                    out << '\n';
                }
            }

            out << "CELL_TYPES " << cells.size()
                << '\n';
            std::vector<std::int32_t> codes;
            codes.reserve(cells.size());
            for (const auto& cell : cells)
                codes.push_back(cell.code);
            if (binary) {
                auto array = data::make_scalar_array(
                    std::span(codes));
                write_binary_array(out, array);
            }
            else
                for (auto code : codes)
                    out << code << '\n';

            write_field_block(out, binary, top_fields);

            if (!point_fields.empty()) {
                out << "POINT_DATA "
                    << mesh.geometry_node_count() << '\n';
                write_field_block(
                    out, binary, point_fields);
            }

            const bool any_tag = std::any_of(
                cells.begin(), cells.end(),
                [](const OutputCell& cell) {
                    return cell.tag != 0;
                });
            if (any_tag || !cell_fields.empty()) {
                out << "CELL_DATA "
                    << cells.size() << '\n';
                if (any_tag) {
                    out << "SCALARS physical_tag int 1\n"
                           "LOOKUP_TABLE default\n";
                    std::vector<std::int32_t> tags;
                    tags.reserve(cells.size());
                    for (const auto& cell : cells)
                        tags.push_back(cell.tag);
                    auto array = data::make_scalar_array(
                        std::span(tags));
                    write_legacy_array(
                        out, binary, array);
                }
                write_field_block(
                    out, binary, cell_fields);
            }

            detail::require_runtime(
                static_cast<bool>(out),
                "VTK output failure");
        }

        void write_legacy(
            const data::Dataset& dataset,
            const std::string& filename,
            LegacyEncoding encoding)
        {
            std::vector<data::MeshDataView> views;
            views.reserve(dataset.state.fields.size());
            for (const auto& field : dataset.state.fields)
                views.push_back(field.view());
            const data::StateView state {
                dataset.state.time,
                dataset.state.cycle,
                views};
            write_legacy(
                dataset.mesh, state,
                filename, encoding);
        }

    } // namespace vtk

    Mesh read_vtk_legacy(
        const std::string& filename,
        BuildOptions options)
    {
        return vtk::read_legacy(
            filename, options).mesh;
    }

    void write_vtk_legacy(
        const Mesh& mesh, const std::string& filename)
    {
        const data::StateView state {
            std::nullopt, std::nullopt, {}};
        vtk::write_legacy(
            mesh, state, filename,
            vtk::LegacyEncoding::ascii);
    }

} // namespace idmesh
