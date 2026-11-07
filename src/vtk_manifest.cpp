#include "contracts.hpp"
#include "idmesh/vtk.hpp"
#include "vtk_common.hpp"
#include "vtk_data_io.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace idmesh::vtk {
    namespace {

        constexpr std::string_view physical_tag_name = "physical_tag";

        std::map<std::string, std::string> attributes(
            std::string_view tag)
        {
            std::map<std::string, std::string> result;
            const auto space = tag.find_first_of(" \t\r\n");
            if (space == std::string_view::npos)
                return result;
            std::size_t i = space;
            while (i < tag.size()) {
                while (i < tag.size()
                    && std::isspace(
                        static_cast<unsigned char>(tag[i])))
                    ++i;
                if (i >= tag.size()
                    || tag[i] == '>' || tag[i] == '/')
                    break;
                const auto key_begin = i;
                while (i < tag.size()
                    && !std::isspace(
                        static_cast<unsigned char>(tag[i]))
                    && tag[i] != '=')
                    ++i;
                const std::string key(
                    tag.substr(key_begin, i - key_begin));
                while (i < tag.size()
                    && std::isspace(
                        static_cast<unsigned char>(tag[i])))
                    ++i;
                ::idmesh::detail::require_runtime(
                    i < tag.size() && tag[i] == '=',
                    "invalid VTK XML attribute");
                ++i;
                while (i < tag.size()
                    && std::isspace(
                        static_cast<unsigned char>(tag[i])))
                    ++i;
                ::idmesh::detail::require_runtime(
                    i < tag.size()
                        && (tag[i] == '"' || tag[i] == '\''),
                    "invalid VTK XML attribute quoting");
                const char quote = tag[i++];
                const auto value_begin = i;
                const auto value_end = tag.find(quote, i);
                ::idmesh::detail::require_runtime(
                    value_end != std::string_view::npos,
                    "unterminated VTK XML attribute");
                result.emplace(
                    key,
                    detail::vtk::xml_unescape(
                        tag.substr(
                            value_begin,
                            value_end - value_begin)));
                i = value_end + 1;
            }
            return result;
        }

        std::string_view required(
            const std::map<std::string, std::string>& attrs,
            std::string_view name)
        {
            const auto it = attrs.find(std::string(name));
            ::idmesh::detail::require_runtime(
                it != attrs.end(),
                "missing required VTK XML attribute");
            return it->second;
        }

        std::string file_text(const std::string& filename)
        {
            std::ifstream in(filename, std::ios::binary);
            ::idmesh::detail::require_runtime(
                static_cast<bool>(in),
                "cannot open VTK manifest");
            return {std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>()};
        }

        std::string_view opening_tag(
            std::string_view xml, std::string_view name)
        {
            const std::string prefix = "<" + std::string(name);
            const auto begin = xml.find(prefix);
            ::idmesh::detail::require_runtime(
                begin != std::string_view::npos,
                "missing VTK XML element");
            const auto end = xml.find('>', begin);
            ::idmesh::detail::require_runtime(
                end != std::string_view::npos,
                "unterminated VTK XML element");
            return xml.substr(begin, end - begin + 1);
        }

        std::string_view element_body(
            std::string_view xml, std::string_view name,
            bool required_element = true)
        {
            const std::string open = "<" + std::string(name);
            const std::string close
                = "</" + std::string(name) + ">";
            const auto begin = xml.find(open);
            if (begin == std::string_view::npos) {
                if (required_element)
                    throw std::runtime_error(
                        "missing VTK XML element");
                return {};
            }
            const auto head_end = xml.find('>', begin);
            const auto end = xml.find(close, head_end);
            ::idmesh::detail::require_runtime(
                head_end != std::string_view::npos
                    && end != std::string_view::npos,
                "unterminated VTK XML element");
            return xml.substr(
                head_end + 1, end - head_end - 1);
        }

        std::vector<std::map<std::string, std::string>>
        self_closing_tags(
            std::string_view body, std::string_view name)
        {
            std::vector<std::map<std::string, std::string>> result;
            const std::string prefix = "<" + std::string(name);
            std::size_t cursor = 0;
            while (true) {
                const auto begin = body.find(prefix, cursor);
                if (begin == std::string_view::npos)
                    break;
                const auto end = body.find('>', begin);
                ::idmesh::detail::require_runtime(
                    end != std::string_view::npos,
                    "unterminated VTK XML tag");
                const auto tag = body.substr(
                    begin, end - begin + 1);
                ::idmesh::detail::require_runtime(
                    end > begin && body[end - 1] == '/',
                    "expected self-closing VTK XML tag");
                result.push_back(attributes(tag));
                cursor = end + 1;
            }
            return result;
        }

        double parse_double(std::string_view text)
        {
            std::istringstream in {std::string(text)};
            double value = 0;
            ::idmesh::detail::require_runtime(
                static_cast<bool>(in >> value)
                    && std::isfinite(value),
                "invalid finite VTK floating attribute");
            std::string trailing;
            ::idmesh::detail::require_runtime(
                !(in >> trailing),
                "invalid VTK floating attribute");
            return value;
        }

        std::uint32_t parse_u32(std::string_view text)
        {
            std::uint64_t value = 0;
            std::istringstream in {std::string(text)};
            ::idmesh::detail::require_runtime(
                static_cast<bool>(in >> value)
                    && value
                        <= std::numeric_limits<std::uint32_t>::max(),
                "invalid VTK UInt32 attribute");
            std::string trailing;
            ::idmesh::detail::require_runtime(
                !(in >> trailing),
                "invalid VTK UInt32 attribute");
            return static_cast<std::uint32_t>(value);
        }

        void validate_pvd(const PvdCollection& collection)
        {
            std::set<std::tuple<double, std::string, std::uint32_t>>
                keys;
            for (const auto& entry : collection.entries) {
                ::idmesh::detail::require_argument(
                    std::isfinite(entry.time),
                    "PVD timestep must be finite");
                ::idmesh::detail::require_argument(
                    !entry.file.empty(),
                    "PVD DataSet file cannot be empty");
                ::idmesh::detail::require_argument(
                    keys.emplace(
                        entry.time, entry.group,
                        entry.part).second,
                    "duplicate PVD timestep/group/part entry");
            }
        }

        void validate_schema(
            std::span<const ArraySchema> arrays,
            const char* message)
        {
            std::set<std::string> names;
            for (const auto& array : arrays) {
                ::idmesh::detail::require_argument(
                    !array.name.empty(), message);
                ::idmesh::detail::require_argument(
                    array.components > 0,
                    "PVTU array requires components");
                ::idmesh::detail::require_argument(
                    names.insert(array.name).second,
                    "duplicate PVTU array name");
            }
        }

        void validate_pvtu(const PvtuManifest& manifest)
        {
            ::idmesh::detail::require_argument(
                manifest.points_type == data::ScalarType::float32
                    || manifest.points_type
                        == data::ScalarType::float64,
                "PVTU points type must be Float32 or Float64");
            ::idmesh::detail::require_argument(
                !manifest.pieces.empty(),
                "PVTU requires at least one Piece");
            validate_schema(
                manifest.point_data,
                "PVTU PointData name cannot be empty");
            validate_schema(
                manifest.cell_data,
                "PVTU CellData name cannot be empty");
            for (const auto& piece : manifest.pieces)
                ::idmesh::detail::require_argument(
                    !piece.empty(),
                    "PVTU Piece source cannot be empty");
        }

        ArraySchema schema(const data::MeshDataView& field)
        {
            return {
                std::string(field.name),
                field.values.type,
                static_cast<std::uint32_t>(
                    field.values.components.size())};
        }

        std::string indexed_vtu_name(
            const std::filesystem::path& pvd,
            std::size_t index)
        {
            std::ostringstream name;
            name << pvd.stem().string()
                 << '-' << std::setw(6)
                 << std::setfill('0') << index
                 << ".vtu";
            return name.str();
        }

    } // namespace

    PvdCollection read_pvd(const std::string& filename)
    {
        const auto xml = file_text(filename);
        const auto root = attributes(
            opening_tag(xml, "VTKFile"));
        ::idmesh::detail::require_runtime(
            required(root, "type") == "Collection",
            "PVD VTKFile type must be Collection");

        const auto body = element_body(xml, "Collection");
        PvdCollection result;
        for (const auto& attrs
            : self_closing_tags(body, "DataSet")) {
            PvdEntry entry;
            entry.time = parse_double(
                required(attrs, "timestep"));
            if (const auto it = attrs.find("group");
                it != attrs.end())
                entry.group = it->second;
            if (const auto it = attrs.find("part");
                it != attrs.end())
                entry.part = parse_u32(it->second);
            entry.file = std::string(
                required(attrs, "file"));
            result.entries.push_back(std::move(entry));
        }
        validate_pvd(result);
        return result;
    }

    void write_pvd(
        const PvdCollection& collection,
        const std::string& filename)
    {
        validate_pvd(collection);
        std::ofstream out(filename, std::ios::binary);
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "cannot open PVD output");
        out << "<?xml version=\"1.0\"?>\n"
               "<VTKFile type=\"Collection\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n"
               "  <Collection>\n";
        out << std::setprecision(17);
        for (const auto& entry : collection.entries)
            out << "    <DataSet timestep=\""
                << entry.time
                << "\" group=\""
                << detail::vtk::xml_escape(entry.group)
                << "\" part=\"" << entry.part
                << "\" file=\""
                << detail::vtk::xml_escape(entry.file)
                << "\"/>\n";
        out << "  </Collection>\n"
               "</VTKFile>\n";
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "PVD output failure");
    }

    void write_vtu_series(
        const Mesh& mesh,
        std::span<const data::StateView> states,
        const std::string& pvd_filename,
        VtuWriteOptions options)
    {
        ::idmesh::detail::require_argument(
            !states.empty(),
            "VTU series requires at least one state");

        double previous = 0.0;
        bool first = true;
        for (const auto& state : states) {
            data::validate(mesh, state);
            ::idmesh::detail::require_argument(
                state.time.has_value()
                    && std::isfinite(*state.time),
                "VTU series states require finite time");
            if (!first)
                ::idmesh::detail::require_argument(
                    *state.time > previous,
                    "VTU series times must be strictly increasing");
            previous = *state.time;
            first = false;
        }

        const std::filesystem::path pvd(pvd_filename);
        const auto directory = pvd.parent_path();
        PvdCollection collection;
        collection.entries.reserve(states.size());

        for (std::size_t i = 0; i < states.size(); ++i) {
            const auto relative = indexed_vtu_name(pvd, i);
            const auto path = directory / relative;
            write_vtu(mesh, states[i], path.string(), options);
            collection.entries.push_back(
                {*states[i].time, 0, {}, relative});
        }
        write_pvd(collection, pvd_filename);
    }

    std::vector<data::Dataset> read_vtu_series(
        const std::string& pvd_filename,
        BuildOptions options)
    {
        const auto collection = read_pvd(pvd_filename);
        ::idmesh::detail::require_runtime(
            !collection.entries.empty(),
            "PVD series contains no DataSet entries");

        const std::filesystem::path pvd(pvd_filename);
        const auto directory = pvd.parent_path();
        std::vector<data::Dataset> result;
        result.reserve(collection.entries.size());

        double previous = 0.0;
        bool first = true;
        for (const auto& entry : collection.entries) {
            ::idmesh::detail::require_runtime(
                entry.part == 0 && entry.group.empty(),
                "serial VTU series requires group=\"\" and part=0");
            ::idmesh::detail::require_runtime(
                std::filesystem::path(entry.file).extension()
                    == ".vtu",
                "serial VTU series requires .vtu DataSet files");
            if (!first)
                ::idmesh::detail::require_runtime(
                    entry.time > previous,
                    "PVD series times must be strictly increasing");
            previous = entry.time;
            first = false;

            auto dataset = read_vtu(
                (directory / entry.file).string(), options);
            if (dataset.state.time)
                ::idmesh::detail::require_runtime(
                    *dataset.state.time == entry.time,
                    "PVD timestep conflicts with VTU state time");
            else
                dataset.state.time = entry.time;
            result.push_back(std::move(dataset));
        }
        return result;
    }

    PvtuManifest make_pvtu_manifest(
        const Mesh& representative_mesh,
        const data::StateView& representative_state,
        std::span<const std::string> pieces,
        std::uint32_t ghost_level)
    {
        data::validate(
            representative_mesh, representative_state);
        ::idmesh::detail::require_argument(
            !representative_state.time.has_value()
                && !representative_state.cycle.has_value(),
            "PVTU schema cannot encode state time or cycle");

        PvtuManifest manifest;
        manifest.ghost_level = ghost_level;
        manifest.points_type = data::ScalarType::float64;
        manifest.pieces.assign(
            pieces.begin(), pieces.end());

        std::set<std::string> point_names, cell_names;
        for (const auto& field : representative_state.fields) {
            const auto association
                = detail::vtk::attribute_association(
                    representative_mesh, field);
            if (association
                == detail::vtk::AttributeAssociation::field)
                throw std::invalid_argument(
                    "PVTU schema cannot encode whole-Mesh fields");

            auto item = schema(field);
            auto& names = association
                    == detail::vtk::AttributeAssociation::point
                ? point_names : cell_names;
            ::idmesh::detail::require_argument(
                names.insert(item.name).second,
                "duplicate PVTU field name");
            if (association
                == detail::vtk::AttributeAssociation::point)
                manifest.point_data.push_back(std::move(item));
            else
                manifest.cell_data.push_back(std::move(item));
        }

        bool any_tag = false;
        const Index d = representative_mesh.dimension();
        for (const auto& label : representative_mesh.labels()) {
            ::idmesh::detail::require_argument(
                label.dimension == d,
                "PVTU helper cannot represent lower-dimensional labels");
            ::idmesh::detail::require_argument(
                label.value != 0,
                "PVTU physical_tag uses zero as unlabeled sentinel");
            ::idmesh::detail::require_argument(
                representative_mesh.label_name(
                    label.dimension, label.value).empty(),
                "PVTU helper cannot preserve label names");
            any_tag = true;
        }
        for (Index cell = 0;
             cell < representative_mesh.entity_count(d); ++cell)
            ::idmesh::detail::require_argument(
                representative_mesh.labels(d, cell).size() <= 1,
                "PVTU helper cannot preserve multiple labels per cell");

        if (any_tag) {
            ::idmesh::detail::require_argument(
                cell_names.insert(
                    std::string(physical_tag_name)).second,
                "physical_tag conflicts with a CellData field");
            manifest.cell_data.push_back(
                {std::string(physical_tag_name),
                    data::ScalarType::int32, 1});
        }

        validate_pvtu(manifest);
        return manifest;
    }

    void write_pvtu(
        const PvtuManifest& manifest,
        const std::string& filename)
    {
        validate_pvtu(manifest);
        std::ofstream out(filename, std::ios::binary);
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "cannot open PVTU output");

        out << "<?xml version=\"1.0\"?>\n"
               "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
               "byte_order=\"LittleEndian\">\n"
               "  <PUnstructuredGrid GhostLevel=\""
            << manifest.ghost_level << "\">\n";

        if (!manifest.point_data.empty()) {
            out << "    <PPointData>\n";
            for (const auto& field : manifest.point_data)
                out << "      <PDataArray type=\""
                    << detail::vtk::xml_scalar_name(field.type)
                    << "\" Name=\""
                    << detail::vtk::xml_escape(field.name)
                    << "\" NumberOfComponents=\""
                    << field.components << "\"/>\n";
            out << "    </PPointData>\n";
        }

        if (!manifest.cell_data.empty()) {
            out << "    <PCellData>\n";
            for (const auto& field : manifest.cell_data)
                out << "      <PDataArray type=\""
                    << detail::vtk::xml_scalar_name(field.type)
                    << "\" Name=\""
                    << detail::vtk::xml_escape(field.name)
                    << "\" NumberOfComponents=\""
                    << field.components << "\"/>\n";
            out << "    </PCellData>\n";
        }

        out << "    <PPoints>\n"
               "      <PDataArray type=\""
            << detail::vtk::xml_scalar_name(
                   manifest.points_type)
            << "\" NumberOfComponents=\"3\"/>\n"
               "    </PPoints>\n";

        for (const auto& piece : manifest.pieces)
            out << "    <Piece Source=\""
                << detail::vtk::xml_escape(piece)
                << "\"/>\n";

        out << "  </PUnstructuredGrid>\n"
               "</VTKFile>\n";
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "PVTU output failure");
    }

    PvtuManifest read_pvtu(const std::string& filename)
    {
        const auto xml = file_text(filename);
        const auto root = attributes(
            opening_tag(xml, "VTKFile"));
        ::idmesh::detail::require_runtime(
            required(root, "type") == "PUnstructuredGrid",
            "PVTU VTKFile type must be PUnstructuredGrid");
        ::idmesh::detail::require_runtime(
            root.find("compressor") == root.end(),
            "compressed PVTU manifests are not supported");

        const auto grid_tag
            = opening_tag(xml, "PUnstructuredGrid");
        const auto grid_attrs = attributes(grid_tag);
        PvtuManifest result;
        if (const auto it = grid_attrs.find("GhostLevel");
            it != grid_attrs.end())
            result.ghost_level = parse_u32(it->second);

        const auto grid = element_body(
            xml, "PUnstructuredGrid");

        const auto point_body = element_body(
            grid, "PPointData", false);
        for (const auto& attrs
            : self_closing_tags(
                point_body, "PDataArray")) {
            ArraySchema field;
            field.name = std::string(
                required(attrs, "Name"));
            field.type = detail::vtk::xml_scalar_type(
                required(attrs, "type"));
            if (const auto it
                = attrs.find("NumberOfComponents");
                it != attrs.end())
                field.components = parse_u32(it->second);
            result.point_data.push_back(std::move(field));
        }

        const auto cell_body = element_body(
            grid, "PCellData", false);
        for (const auto& attrs
            : self_closing_tags(
                cell_body, "PDataArray")) {
            ArraySchema field;
            field.name = std::string(
                required(attrs, "Name"));
            field.type = detail::vtk::xml_scalar_type(
                required(attrs, "type"));
            if (const auto it
                = attrs.find("NumberOfComponents");
                it != attrs.end())
                field.components = parse_u32(it->second);
            result.cell_data.push_back(std::move(field));
        }

        const auto points = self_closing_tags(
            element_body(grid, "PPoints"),
            "PDataArray");
        ::idmesh::detail::require_runtime(
            points.size() == 1,
            "PVTU PPoints must contain one PDataArray");
        result.points_type = detail::vtk::xml_scalar_type(
            required(points.front(), "type"));
        const auto components = points.front().find(
            "NumberOfComponents");
        ::idmesh::detail::require_runtime(
            components != points.front().end()
                && parse_u32(components->second) == 3,
            "PVTU points require three components");

        for (const auto& attrs
            : self_closing_tags(grid, "Piece"))
            result.pieces.push_back(
                std::string(required(attrs, "Source")));

        validate_pvtu(result);
        return result;
    }

} // namespace idmesh::vtk
