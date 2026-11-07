#include "contracts.hpp"
#include "idmesh/vtk.hpp"
#include "vtk_common.hpp"
#include "vtk_data_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace idmesh::vtk {
    namespace {

        constexpr std::string_view physical_tag_name = "physical_tag";
        constexpr std::string_view time_name = "__idmesh_time";
        constexpr std::string_view cycle_name = "__idmesh_cycle";

        struct XmlArray {
            std::map<std::string, std::string> attributes;
            std::string_view content;
        };

        std::vector<XmlArray> data_arrays(
            std::string_view body)
        {
            std::vector<XmlArray> result;
            std::size_t cursor = 0;
            while (true) {
                const auto begin
                    = body.find("<DataArray", cursor);
                if (begin == std::string_view::npos)
                    break;
                const auto head_end = body.find('>', begin);
                ::idmesh::detail::require_runtime(
                    head_end != std::string_view::npos,
                    "unterminated VTU DataArray");
                const auto tag = body.substr(
                    begin, head_end - begin + 1);
                const bool self_closing
                    = head_end > begin
                    && body[head_end - 1] == '/';
                XmlArray array;
                array.attributes = detail::vtk::xml_attributes(tag);
                if (self_closing) {
                    array.content = {};
                    cursor = head_end + 1;
                }
                else {
                    const auto close
                        = body.find("</DataArray>", head_end);
                    ::idmesh::detail::require_runtime(
                        close != std::string_view::npos,
                        "unterminated VTU DataArray body");
                    array.content = body.substr(
                        head_end + 1,
                        close - head_end - 1);
                    cursor = close
                        + std::string_view("</DataArray>").size();
                }
                result.push_back(std::move(array));
            }
            return result;
        }

        std::uint64_t parse_u64(std::string_view text)
        {
            std::uint64_t value = 0;
            std::istringstream in {std::string(text)};
            ::idmesh::detail::require_runtime(
                static_cast<bool>(in >> value),
                "invalid VTU integer attribute");
            std::string trailing;
            ::idmesh::detail::require_runtime(
                !(in >> trailing),
                "invalid VTU integer attribute");
            return value;
        }

        std::uint64_t read_uint(
            std::span<const std::byte> bytes,
            std::size_t offset, std::size_t width,
            std::endian endian)
        {
            ::idmesh::detail::require_runtime(
                (width == 4 || width == 8)
                    && offset <= bytes.size()
                    && bytes.size() - offset >= width,
                "truncated VTU block header");
            std::array<std::byte, 8> local {};
            std::copy_n(
                bytes.data() + offset, width, local.data());
            if (endian != std::endian::native)
                std::reverse(
                    local.begin(), local.begin() + width);
            if (width == 4) {
                std::uint32_t value = 0;
                std::memcpy(&value, local.data(), 4);
                return value;
            }
            std::uint64_t value = 0;
            std::memcpy(&value, local.data(), 8);
            return value;
        }

        void append_u64_le(
            std::vector<std::byte>& bytes, std::uint64_t value)
        {
            std::array<std::byte, 8> raw {};
            std::memcpy(raw.data(), &value, 8);
            if (std::endian::native != std::endian::little)
                std::reverse(raw.begin(), raw.end());
            bytes.insert(bytes.end(), raw.begin(), raw.end());
        }

        std::size_t token_count(std::string_view text)
        {
            std::istringstream in {std::string(text)};
            std::size_t count = 0;
            std::string token;
            while (in >> token)
                ++count;
            return count;
        }

        data::Array decode_array(
            const XmlArray& source,
            std::optional<std::size_t> expected_tuples,
            std::span<const std::byte> appended,
            std::endian endian, std::size_t header_width)
        {
            const auto type = detail::vtk::xml_scalar_type(
                detail::vtk::xml_required_attribute(source.attributes, "type"));
            std::uint32_t components = 1;
            if (const auto it
                = source.attributes.find("NumberOfComponents");
                it != source.attributes.end()) {
                const auto parsed = parse_u64(it->second);
                ::idmesh::detail::require_runtime(
                    parsed > 0
                        && parsed
                            <= std::numeric_limits<std::uint32_t>::max(),
                    "invalid VTU NumberOfComponents");
                components = static_cast<std::uint32_t>(parsed);
            }

            std::string format = "ascii";
            if (const auto it = source.attributes.find("format");
                it != source.attributes.end())
                format = it->second;

            if (format == "ascii") {
                const auto values = token_count(source.content);
                ::idmesh::detail::require_runtime(
                    values % components == 0,
                    "VTU ASCII tuple shape mismatch");
                const auto tuples = values / components;
                if (expected_tuples)
                    ::idmesh::detail::require_runtime(
                        tuples == *expected_tuples,
                        "VTU ASCII tuple count mismatch");
                return detail::vtk::parse_ascii_values(
                    source.content, type, tuples, components);
            }

            std::vector<std::byte> block;
            if (format == "binary") {
                block = detail::vtk::base64_decode(
                    source.content);
            }
            else if (format == "appended") {
                const auto offset_it
                    = source.attributes.find("offset");
                ::idmesh::detail::require_runtime(
                    offset_it != source.attributes.end(),
                    "VTU appended DataArray lacks offset");
                const auto offset
                    = parse_u64(offset_it->second);
                ::idmesh::detail::require_runtime(
                    offset <= appended.size()
                        && appended.size() - offset >= header_width,
                    "VTU appended offset out of range");
                const auto payload
                    = read_uint(appended,
                        static_cast<std::size_t>(offset),
                        header_width, endian);
                ::idmesh::detail::require_runtime(
                    payload
                        <= std::numeric_limits<std::size_t>::max()
                    && static_cast<std::size_t>(offset)
                            + header_width
                        <= appended.size()
                    && appended.size()
                            - (static_cast<std::size_t>(offset)
                                + header_width)
                        >= payload,
                    "truncated VTU appended block");
                block.resize(header_width
                    + static_cast<std::size_t>(payload));
                std::copy_n(
                    appended.data()
                        + static_cast<std::size_t>(offset),
                    block.size(), block.data());
            }
            else
                throw std::runtime_error(
                    "unsupported VTU DataArray format");

            ::idmesh::detail::require_runtime(
                block.size() >= header_width,
                "truncated VTU binary block");
            const auto payload
                = read_uint(block, 0, header_width, endian);
            ::idmesh::detail::require_runtime(
                payload
                    == block.size() - header_width,
                "VTU binary block length mismatch");
            const auto scalar_bytes
                = data::scalar_size(type);
            ::idmesh::detail::require_runtime(
                payload % scalar_bytes == 0,
                "VTU binary scalar alignment mismatch");
            const auto scalars
                = static_cast<std::size_t>(payload)
                / scalar_bytes;
            ::idmesh::detail::require_runtime(
                scalars % components == 0,
                "VTU binary tuple shape mismatch");
            const auto tuples = scalars / components;
            if (expected_tuples)
                ::idmesh::detail::require_runtime(
                    tuples == *expected_tuples,
                    "VTU binary tuple count mismatch");

            std::vector<std::byte> raw(
                static_cast<std::size_t>(payload));
            std::copy_n(
                block.data() + header_width,
                raw.size(), raw.data());
            return detail::vtk::array_from_endian_bytes(
                type, tuples, components,
                std::move(raw), endian);
        }

        template <class T>
        T scalar_at(
            const data::Array& array, std::size_t tuple,
            std::size_t component = 0)
        {
            ::idmesh::detail::require_runtime(
                tuple < array.tuples
                    && component < array.components,
                "VTU scalar index out of range");
            T value {};
            const auto width = data::scalar_size(array.type);
            ::idmesh::detail::require_runtime(
                width == sizeof(T),
                "VTU scalar type mismatch");
            const auto offset
                = (tuple * array.components + component) * width;
            std::memcpy(
                &value, array.values.data() + offset, width);
            return value;
        }

        std::uint64_t nonnegative_integer(
            const data::Array& array, std::size_t tuple,
            std::size_t component = 0)
        {
            switch (array.type) {
            case data::ScalarType::int8: {
                const auto value = scalar_at<std::int8_t>(
                    array, tuple, component);
                ::idmesh::detail::require_runtime(
                    value >= 0, "negative VTU integer");
                return value;
            }
            case data::ScalarType::uint8:
                return scalar_at<std::uint8_t>(
                    array, tuple, component);
            case data::ScalarType::int16: {
                const auto value = scalar_at<std::int16_t>(
                    array, tuple, component);
                ::idmesh::detail::require_runtime(
                    value >= 0, "negative VTU integer");
                return value;
            }
            case data::ScalarType::uint16:
                return scalar_at<std::uint16_t>(
                    array, tuple, component);
            case data::ScalarType::int32: {
                const auto value = scalar_at<std::int32_t>(
                    array, tuple, component);
                ::idmesh::detail::require_runtime(
                    value >= 0, "negative VTU integer");
                return static_cast<std::uint64_t>(value);
            }
            case data::ScalarType::uint32:
                return scalar_at<std::uint32_t>(
                    array, tuple, component);
            case data::ScalarType::int64: {
                const auto value = scalar_at<std::int64_t>(
                    array, tuple, component);
                ::idmesh::detail::require_runtime(
                    value >= 0, "negative VTU integer");
                return static_cast<std::uint64_t>(value);
            }
            case data::ScalarType::uint64:
                return scalar_at<std::uint64_t>(
                    array, tuple, component);
            case data::ScalarType::float32:
            case data::ScalarType::float64:
                throw std::runtime_error(
                    "VTU connectivity uses floating type");
            }
            throw std::runtime_error("invalid VTU scalar type");
        }

        std::int64_t signed_integer(
            const data::Array& array, std::size_t tuple)
        {
            switch (array.type) {
            case data::ScalarType::int8:
                return scalar_at<std::int8_t>(array, tuple);
            case data::ScalarType::uint8:
                return scalar_at<std::uint8_t>(array, tuple);
            case data::ScalarType::int16:
                return scalar_at<std::int16_t>(array, tuple);
            case data::ScalarType::uint16:
                return scalar_at<std::uint16_t>(array, tuple);
            case data::ScalarType::int32:
                return scalar_at<std::int32_t>(array, tuple);
            case data::ScalarType::uint32:
                return scalar_at<std::uint32_t>(array, tuple);
            case data::ScalarType::int64:
                return scalar_at<std::int64_t>(array, tuple);
            case data::ScalarType::uint64: {
                const auto value
                    = scalar_at<std::uint64_t>(array, tuple);
                ::idmesh::detail::require_runtime(
                    value <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()),
                    "VTU integer out of int64 range");
                return static_cast<std::int64_t>(value);
            }
            case data::ScalarType::float32:
            case data::ScalarType::float64:
                throw std::runtime_error(
                    "VTU integer metadata has floating type");
            }
            throw std::runtime_error("invalid VTU scalar type");
        }

        double floating_value(
            const data::Array& array, std::size_t tuple,
            std::size_t component = 0)
        {
            if (array.type == data::ScalarType::float32)
                return scalar_at<float>(
                    array, tuple, component);
            if (array.type == data::ScalarType::float64)
                return scalar_at<double>(
                    array, tuple, component);
            throw std::runtime_error(
                "VTU coordinate/time data must be floating point");
        }

        void validate_vtu_name(std::string_view name)
        {
            ::idmesh::detail::require_argument(
                !name.empty(), "VTU field name cannot be empty");
            ::idmesh::detail::require_argument(
                name != physical_tag_name
                    && name != time_name
                    && name != cycle_name,
                "VTU field name is reserved");
        }

        struct PendingBlock {
            std::vector<std::byte> bytes;
        };

        std::string data_array_xml(
            std::string_view name,
            const data::ArrayView& array,
            VtuEncoding encoding,
            std::vector<PendingBlock>& appended,
            bool include_name = true)
        {
            data::validate_array(array);
            std::ostringstream out;
            out << "<DataArray type=\""
                << detail::vtk::xml_scalar_name(array.type)
                << "\"";
            if (include_name)
                out << " Name=\""
                    << detail::vtk::xml_escape(name) << "\"";
            out << " NumberOfComponents=\""
                << array.components.size() << "\"";

            if (encoding == VtuEncoding::ascii) {
                out << " format=\"ascii\">";
                detail::vtk::write_ascii_values(out, array);
                out << "</DataArray>";
                return out.str();
            }

            const auto payload = detail::vtk::endian_bytes(
                array, std::endian::little);
            std::vector<std::byte> block;
            block.reserve(8 + payload.size());
            append_u64_le(
                block,
                static_cast<std::uint64_t>(payload.size()));
            block.insert(
                block.end(), payload.begin(), payload.end());

            if (encoding == VtuEncoding::inline_binary) {
                out << " format=\"binary\">"
                    << detail::vtk::base64_encode(block)
                    << "</DataArray>";
                return out.str();
            }

            std::size_t offset = 0;
            for (const auto& existing : appended)
                offset += existing.bytes.size();
            out << " format=\"appended\" offset=\""
                << offset << "\"/>";
            appended.push_back({std::move(block)});
            return out.str();
        }

        void validate_vtu_labels(const Mesh& mesh)
        {
            const Index d = mesh.dimension();
            for (const auto& label : mesh.labels()) {
                ::idmesh::detail::require_argument(
                    label.dimension == d,
                    "VTU writer cannot preserve lower-dimensional labels in a single Piece");
                ::idmesh::detail::require_argument(
                    label.value != 0,
                    "VTU physical_tag uses zero as the unlabeled sentinel");
                ::idmesh::detail::require_argument(
                    mesh.label_name(
                        label.dimension, label.value).empty(),
                    "VTU writer cannot preserve label names");
            }
            for (Index cell = 0;
                 cell < mesh.entity_count(d); ++cell)
                ::idmesh::detail::require_argument(
                    mesh.labels(d, cell).size() <= 1,
                    "VTU writer cannot store multiple labels on one cell");
        }

        std::string read_file(const std::string& filename)
        {
            std::ifstream in(filename, std::ios::binary);
            ::idmesh::detail::require_runtime(
                static_cast<bool>(in),
                "cannot open VTU file");
            return {std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>()};
        }

        std::vector<std::byte> bytes_from_chars(
            const char* first, std::size_t size)
        {
            std::vector<std::byte> result(size);
            std::memcpy(result.data(), first, size);
            return result;
        }

        data::Array find_named_array(
            const std::vector<XmlArray>& arrays,
            std::string_view name,
            std::optional<std::size_t> tuples,
            std::span<const std::byte> appended,
            std::endian endian, std::size_t header_width)
        {
            const auto it = std::find_if(
                arrays.begin(), arrays.end(),
                [&](const XmlArray& array) {
                    const auto found
                        = array.attributes.find("Name");
                    return found != array.attributes.end()
                        && found->second == name;
                });
            ::idmesh::detail::require_runtime(
                it != arrays.end(),
                "missing required VTU DataArray");
            return decode_array(
                *it, tuples, appended, endian, header_width);
        }

        data::MeshData make_field(
            std::string name, data::DataLocation location,
            data::Array values)
        {
            data::MeshData field;
            field.name = std::move(name);
            field.location = location;
            field.selection = data::SelectionKind::all;
            field.values = std::move(values);
            return field;
        }

    } // namespace

    data::Dataset read_vtu(
        const std::string& filename, BuildOptions options)
    {
        std::string file = read_file(filename);
        std::vector<std::byte> appended_storage;
        std::string xml = file;

        const auto appended_open
            = file.find("<AppendedData");
        if (appended_open != std::string::npos) {
            const auto head_end = file.find('>', appended_open);
            ::idmesh::detail::require_runtime(
                head_end != std::string::npos,
                "unterminated VTU AppendedData");
            const auto attrs = detail::vtk::xml_attributes(
                std::string_view(file).substr(
                    appended_open, head_end - appended_open + 1));
            const auto encoding
                = detail::vtk::xml_required_attribute(attrs, "encoding");
            ::idmesh::detail::require_runtime(
                encoding == "raw",
                "only raw VTU AppendedData is supported");
            const auto underscore = file.find('_', head_end + 1);
            const auto close = file.rfind("</AppendedData>");
            ::idmesh::detail::require_runtime(
                underscore != std::string::npos
                    && close != std::string::npos
                    && close > underscore,
                "invalid VTU AppendedData body");
            appended_storage = bytes_from_chars(
                file.data() + underscore + 1,
                close - underscore - 1);
            xml = file.substr(0, underscore + 1)
                + file.substr(close);
        }
        const auto appended
            = std::span<const std::byte>(
                appended_storage.data(),
                appended_storage.size());

        const auto root_attrs = detail::vtk::xml_attributes(
            detail::vtk::xml_opening_tag(xml, "VTKFile"));
        ::idmesh::detail::require_runtime(
            detail::vtk::xml_required_attribute(root_attrs, "type")
                == "UnstructuredGrid",
            "VTU file is not an UnstructuredGrid");
        ::idmesh::detail::require_runtime(
            root_attrs.find("compressor")
                == root_attrs.end(),
            "compressed VTU is not supported");

        std::endian endian = std::endian::little;
        if (const auto it = root_attrs.find("byte_order");
            it != root_attrs.end()) {
            if (it->second == "LittleEndian")
                endian = std::endian::little;
            else if (it->second == "BigEndian")
                endian = std::endian::big;
            else
                throw std::runtime_error(
                    "unsupported VTU byte_order");
        }
        std::size_t header_width = 4;
        if (const auto it = root_attrs.find("header_type");
            it != root_attrs.end()) {
            if (it->second == "UInt32")
                header_width = 4;
            else if (it->second == "UInt64")
                header_width = 8;
            else
                throw std::runtime_error(
                    "unsupported VTU header_type");
        }

        const auto piece_tag = detail::vtk::xml_opening_tag(xml, "Piece");
        const auto next_piece = xml.find(
            "<Piece", xml.find("<Piece") + 1);
        ::idmesh::detail::require_runtime(
            next_piece == std::string::npos,
            "multi-Piece VTU is not supported");
        const auto piece_attrs = detail::vtk::xml_attributes(piece_tag);
        const auto point_count_u64 = parse_u64(
            detail::vtk::xml_required_attribute(piece_attrs, "NumberOfPoints"));
        const auto cell_count_u64 = parse_u64(
            detail::vtk::xml_required_attribute(piece_attrs, "NumberOfCells"));
        ::idmesh::detail::require_runtime(
            point_count_u64 < invalid_index
                && cell_count_u64 < invalid_index,
            "VTU mesh exceeds IDMesh index range");
        const auto point_count
            = static_cast<std::size_t>(point_count_u64);
        const auto cell_count
            = static_cast<std::size_t>(cell_count_u64);

        const auto piece = detail::vtk::xml_element_body(xml, "Piece");

        const auto point_arrays
            = data_arrays(detail::vtk::xml_element_body(piece, "Points"));
        ::idmesh::detail::require_runtime(
            point_arrays.size() == 1,
            "VTU Points must contain exactly one DataArray");
        auto coordinates = decode_array(
            point_arrays.front(), point_count,
            appended, endian, header_width);
        ::idmesh::detail::require_runtime(
            coordinates.components == 3
                && (coordinates.type == data::ScalarType::float32
                    || coordinates.type == data::ScalarType::float64),
            "VTU Points requires Float32/Float64 with three components");
        std::vector<std::array<double, 3>> points(point_count);
        for (std::size_t point = 0;
             point < point_count; ++point) {
            for (std::size_t component = 0;
                 component < 3; ++component) {
                points[point][component] = floating_value(
                    coordinates, point, component);
                ::idmesh::detail::require_runtime(
                    std::isfinite(points[point][component]),
                    "nonfinite VTU point");
            }
        }

        const auto cell_arrays
            = data_arrays(detail::vtk::xml_element_body(piece, "Cells"));
        const auto connectivity = find_named_array(
            cell_arrays, "connectivity", std::nullopt,
            appended, endian, header_width);
        const auto offsets = find_named_array(
            cell_arrays, "offsets", cell_count,
            appended, endian, header_width);
        const auto types = find_named_array(
            cell_arrays, "types", cell_count,
            appended, endian, header_width);
        ::idmesh::detail::require_runtime(
            connectivity.components == 1
                && offsets.components == 1
                && types.components == 1,
            "VTU cell arrays must be scalar");

        std::vector<detail::vtk::ImportedCell> cells(cell_count);
        std::size_t previous = 0;
        for (std::size_t cell = 0;
             cell < cell_count; ++cell) {
            const auto end_u64
                = nonnegative_integer(offsets, cell);
            ::idmesh::detail::require_runtime(
                end_u64 >= previous
                    && end_u64 <= connectivity.tuples,
                "invalid VTU cell offset");
            const auto end
                = static_cast<std::size_t>(end_u64);
            const auto code_u64
                = nonnegative_integer(types, cell);
            ::idmesh::detail::require_runtime(
                code_u64
                    <= static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()),
                "VTU cell type out of range");
            const auto info = detail::vtk::cell_info(
                static_cast<int>(code_u64));
            ::idmesh::detail::require_runtime(
                info.nodes == end - previous,
                "VTU cell node count does not match cell type");
            cells[cell].type = info.type;
            cells[cell].dimension = info.dimension;
            cells[cell].points.resize(end - previous);
            for (std::size_t i = previous; i < end; ++i) {
                const auto point
                    = nonnegative_integer(connectivity, i);
                ::idmesh::detail::require_runtime(
                    point < point_count,
                    "VTU connectivity point out of range");
                cells[cell].points[i - previous]
                    = static_cast<Index>(point);
            }
            previous = end;
        }
        ::idmesh::detail::require_runtime(
            previous == connectivity.tuples,
            "VTU connectivity has trailing entries");

        std::vector<std::pair<std::string, data::Array>>
            point_fields, cell_fields, mesh_fields;
        std::optional<data::Array> physical_tags;

        const auto read_field_section =
            [&](std::string_view name, std::size_t tuples,
                auto& destination, bool cell_domain) {
                const auto body = detail::vtk::xml_element_body(
                    piece, name, false);
                if (body.empty())
                    return;
                for (const auto& array : data_arrays(body)) {
                    const auto field_name = std::string(
                        detail::vtk::xml_required_attribute(array.attributes, "Name"));
                    auto values = decode_array(
                        array, tuples, appended,
                        endian, header_width);
                    if (cell_domain
                        && field_name == physical_tag_name) {
                        ::idmesh::detail::require_runtime(
                            !physical_tags.has_value()
                                && values.components == 1,
                            "duplicate or invalid VTU physical_tag");
                        physical_tags = std::move(values);
                    }
                    else
                        destination.emplace_back(
                            field_name, std::move(values));
                }
            };

        read_field_section(
            "PointData", point_count,
            point_fields, false);
        read_field_section(
            "CellData", cell_count,
            cell_fields, true);

        const auto field_body
            = detail::vtk::xml_element_body(piece, "FieldData", false);
        if (!field_body.empty()) {
            for (const auto& array : data_arrays(field_body)) {
                const auto field_name = std::string(
                    detail::vtk::xml_required_attribute(array.attributes, "Name"));
                auto values = decode_array(
                    array, std::size_t {1}, appended,
                    endian, header_width);
                mesh_fields.emplace_back(
                    field_name, std::move(values));
            }
        }

        if (physical_tags) {
            for (std::size_t cell = 0;
                 cell < cells.size(); ++cell) {
                const auto value
                    = signed_integer(*physical_tags, cell);
                ::idmesh::detail::require_runtime(
                    value >= std::numeric_limits<int>::min()
                        && value <= std::numeric_limits<int>::max(),
                    "VTU physical_tag out of int range");
                cells[cell].tag = static_cast<int>(value);
            }
        }

        auto built = detail::vtk::build_imported_mesh(
            std::move(points), cells, options);

        data::State state;
        std::set<std::pair<int, std::string>> seen;
        for (auto& [name, values] : point_fields) {
            ::idmesh::detail::require_runtime(
                name != physical_tag_name
                    && name != time_name
                    && name != cycle_name
                    && seen.emplace(0, name).second,
                "reserved or duplicate VTU PointData name");
            auto field = make_field(
                std::move(name), data::geometry_nodes(),
                detail::vtk::reorder_point_array(
                    std::move(values),
                    built.point_to_geometry));
            data::validate(built.mesh, field.view());
            state.fields.push_back(std::move(field));
        }

        for (auto& [name, values] : cell_fields) {
            ::idmesh::detail::require_runtime(
                !built.has_lower_cells,
                "VTU CellData cannot be mapped exactly when lower-dimensional cells are present");
            ::idmesh::detail::require_runtime(
                name != physical_tag_name
                    && name != time_name
                    && name != cycle_name
                    && seen.emplace(1, name).second,
                "reserved or duplicate VTU CellData name");
            auto field = make_field(
                std::move(name),
                data::entities(built.mesh.dimension()),
                std::move(values));
            data::validate(built.mesh, field.view());
            state.fields.push_back(std::move(field));
        }

        for (auto& [name, values] : mesh_fields) {
            if (name == time_name) {
                ::idmesh::detail::require_runtime(
                    !state.time.has_value()
                        && values.tuples == 1
                        && values.components == 1,
                    "invalid VTU time metadata");
                state.time = floating_value(values, 0);
                ::idmesh::detail::require_runtime(
                    std::isfinite(*state.time),
                    "nonfinite VTU time metadata");
                continue;
            }
            if (name == cycle_name) {
                ::idmesh::detail::require_runtime(
                    !state.cycle.has_value()
                        && values.tuples == 1
                        && values.components == 1,
                    "invalid VTU cycle metadata");
                state.cycle = signed_integer(values, 0);
                continue;
            }
            ::idmesh::detail::require_runtime(
                name != physical_tag_name
                    && seen.emplace(2, name).second,
                "reserved or duplicate VTU FieldData name");
            auto field = make_field(
                std::move(name), data::whole_mesh(),
                std::move(values));
            data::validate(built.mesh, field.view());
            state.fields.push_back(std::move(field));
        }

        return {
            std::move(built.mesh), std::move(state)};
    }

    void write_vtu(
        const Mesh& mesh, const data::StateView& state,
        const std::string& filename,
        VtuWriteOptions options)
    {
        data::validate(mesh, state);
        validate_vtu_labels(mesh);

        std::vector<data::MeshDataView>
            point_fields, cell_fields, mesh_fields;
        std::set<std::pair<int, std::string>> seen;
        for (const auto& field : state.fields) {
            validate_vtu_name(field.name);
            const auto association
                = detail::vtk::attribute_association(
                    mesh, field);
            int domain = 0;
            switch (association) {
            case detail::vtk::AttributeAssociation::point:
                point_fields.push_back(field);
                domain = 0;
                break;
            case detail::vtk::AttributeAssociation::cell:
                cell_fields.push_back(field);
                domain = 1;
                break;
            case detail::vtk::AttributeAssociation::field:
                mesh_fields.push_back(field);
                domain = 2;
                break;
            }
            ::idmesh::detail::require_argument(
                seen.emplace(
                    domain, std::string(field.name)).second,
                "duplicate VTU field name in one association");
        }

        std::vector<double> coordinates;
        coordinates.reserve(
            static_cast<std::size_t>(mesh.geometry_node_count()) * 3);
        for (Index point = 0; point < mesh.geometry_node_count(); ++point) {
            const auto p = mesh.geometry_node_position(point);
            coordinates.insert(
                coordinates.end(), p.begin(), p.end());
        }
        const auto points_array
            = data::make_interleaved_array(
                std::span(coordinates), 3);

        const Index d = mesh.dimension();
        const Index cell_count = mesh.entity_count(d);
        std::vector<std::int64_t>
            connectivity, offsets;
        std::vector<std::uint8_t> types;
        std::vector<std::int32_t> tags;
        offsets.reserve(cell_count);
        types.reserve(cell_count);
        tags.reserve(cell_count);

        bool any_tag = false;
        for (Index cell = 0; cell < cell_count; ++cell) {
            auto encoded = detail::vtk::encode_cell(
                mesh, cell);
            for (Index point : encoded.points)
                connectivity.push_back(
                    static_cast<std::int64_t>(point));
            offsets.push_back(
                static_cast<std::int64_t>(
                    connectivity.size()));
            ::idmesh::detail::require_runtime(
                encoded.code >= 0
                    && encoded.code
                        <= std::numeric_limits<std::uint8_t>::max(),
                "VTU cell type exceeds UInt8");
            types.push_back(
                static_cast<std::uint8_t>(encoded.code));
            const auto labels = mesh.labels(d, cell);
            const int tag = labels.empty()
                ? 0 : labels.front().value;
            any_tag = any_tag || tag != 0;
            tags.push_back(tag);
        }

        auto connectivity_array
            = data::make_scalar_array(
                std::span(connectivity));
        auto offsets_array
            = data::make_scalar_array(
                std::span(offsets));
        auto types_array
            = data::make_scalar_array(
                std::span(types));
        auto tags_array
            = data::make_scalar_array(
                std::span(tags));

        std::array<double, 1> time_storage {};
        std::array<std::int64_t, 1> cycle_storage {};
        std::vector<data::MeshDataView> all_mesh_fields
            = mesh_fields;
        if (state.time) {
            time_storage[0] = *state.time;
            all_mesh_fields.push_back(
                {time_name, data::whole_mesh(), data::all(),
                    data::make_scalar_array(
                        std::span(time_storage))});
        }
        if (state.cycle) {
            cycle_storage[0] = *state.cycle;
            all_mesh_fields.push_back(
                {cycle_name, data::whole_mesh(), data::all(),
                    data::make_scalar_array(
                        std::span(cycle_storage))});
        }

        std::vector<PendingBlock> appended;
        std::ostringstream xml;
        xml << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
               "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
            << "  <UnstructuredGrid>\n"
            << "    <Piece NumberOfPoints=\""
            << mesh.geometry_node_count()
            << "\" NumberOfCells=\""
            << cell_count << "\">\n";

        xml << "      <Points>\n        "
            << data_array_xml(
                "", points_array, options.encoding,
                appended, false)
            << "\n      </Points>\n";

        xml << "      <Cells>\n        "
            << data_array_xml(
                "connectivity", connectivity_array,
                options.encoding, appended)
            << "\n        "
            << data_array_xml(
                "offsets", offsets_array,
                options.encoding, appended)
            << "\n        "
            << data_array_xml(
                "types", types_array,
                options.encoding, appended)
            << "\n      </Cells>\n";

        if (!point_fields.empty()) {
            xml << "      <PointData>\n";
            for (const auto& field : point_fields)
                xml << "        "
                    << data_array_xml(
                        field.name, field.values,
                        options.encoding, appended)
                    << "\n";
            xml << "      </PointData>\n";
        }

        if (any_tag || !cell_fields.empty()) {
            xml << "      <CellData>\n";
            if (any_tag)
                xml << "        "
                    << data_array_xml(
                        physical_tag_name, tags_array,
                        options.encoding, appended)
                    << "\n";
            for (const auto& field : cell_fields)
                xml << "        "
                    << data_array_xml(
                        field.name, field.values,
                        options.encoding, appended)
                    << "\n";
            xml << "      </CellData>\n";
        }

        if (!all_mesh_fields.empty()) {
            xml << "      <FieldData>\n";
            for (const auto& field : all_mesh_fields)
                xml << "        "
                    << data_array_xml(
                        field.name, field.values,
                        options.encoding, appended)
                    << "\n";
            xml << "      </FieldData>\n";
        }

        xml << "    </Piece>\n"
            << "  </UnstructuredGrid>\n";

        std::ofstream out(filename, std::ios::binary);
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "cannot open VTU output");
        out << xml.str();
        if (options.encoding == VtuEncoding::appended_raw) {
            out << "  <AppendedData encoding=\"raw\">_";
            for (const auto& block : appended)
                out.write(
                    reinterpret_cast<const char*>(
                        block.bytes.data()),
                    static_cast<std::streamsize>(
                        block.bytes.size()));
            out << "</AppendedData>\n";
        }
        out << "</VTKFile>\n";
        ::idmesh::detail::require_runtime(
            static_cast<bool>(out),
            "VTU output failure");
    }

    void write_vtu(
        const data::Dataset& dataset,
        const std::string& filename,
        VtuWriteOptions options)
    {
        std::vector<data::MeshDataView> views;
        views.reserve(dataset.state.fields.size());
        for (const auto& field : dataset.state.fields)
            views.push_back(field.view());
        const data::StateView state {
            dataset.state.time,
            dataset.state.cycle, views};
        write_vtu(
            dataset.mesh, state, filename, options);
    }

} // namespace idmesh::vtk
