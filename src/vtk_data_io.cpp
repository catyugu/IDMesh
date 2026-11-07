#include "vtk_data_io.hpp"

#include "contracts.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace idmesh::detail::vtk {

    namespace {

        template <class T>
        void write_value(std::ostream& out, const std::byte* bytes)
        {
            T value {};
            std::memcpy(&value, bytes, sizeof(T));
            if constexpr (std::is_same_v<T, std::int8_t>
                || std::is_same_v<T, std::uint8_t>)
                out << static_cast<int>(value);
            else
                out << value;
        }

        template <class T>
        T read_value(std::istream& in)
        {
            if constexpr (std::is_same_v<T, std::int8_t>
                || std::is_same_v<T, std::uint8_t>) {
                int value = 0;
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(in >> value),
                    "truncated VTK numeric data");
                if constexpr (std::is_same_v<T, std::int8_t>)
                    ::idmesh::detail::require_runtime(
                        value >= std::numeric_limits<std::int8_t>::min()
                            && value <= std::numeric_limits<std::int8_t>::max(),
                        "VTK int8 value out of range");
                else
                    ::idmesh::detail::require_runtime(
                        value >= 0
                            && value <= std::numeric_limits<std::uint8_t>::max(),
                        "VTK uint8 value out of range");
                return static_cast<T>(value);
            }
            else {
                T value {};
                ::idmesh::detail::require_runtime(
                    static_cast<bool>(in >> value),
                    "truncated VTK numeric data");
                return value;
            }
        }

        template <class T>
        void append_value(
            std::vector<std::byte>& bytes, T value)
        {
            const auto* first
                = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), first, first + sizeof(T));
        }

        void swap_scalars(
            std::span<std::byte> bytes, std::size_t width)
        {
            if (width == 1)
                return;
            ::idmesh::detail::require_runtime(
                bytes.size() % width == 0,
                "misaligned VTK binary scalar block");
            for (std::size_t offset = 0;
                 offset < bytes.size(); offset += width)
                std::reverse(bytes.begin() + offset,
                    bytes.begin() + offset + width);
        }

        template <class T>
        data::Array read_values_t(
            std::istream& in, data::ScalarType type,
            std::size_t tuples, std::uint32_t components)
        {
            data::Array result;
            result.type = type;
            result.tuples = tuples;
            result.components = components;
            ::idmesh::detail::require_length(
                tuples <= std::numeric_limits<std::size_t>::max()
                        / components
                    && tuples * components
                        <= std::numeric_limits<std::size_t>::max()
                            / sizeof(T),
                "VTK array size overflow");
            result.values.reserve(
                tuples * components * sizeof(T));
            for (std::size_t i = 0;
                 i < tuples * components; ++i)
                append_value(result.values, read_value<T>(in));
            return result;
        }

    } // namespace

    std::string_view legacy_scalar_name(data::ScalarType type)
    {
        switch (type) {
        case data::ScalarType::int8:
            return "char";
        case data::ScalarType::uint8:
            return "unsigned_char";
        case data::ScalarType::int16:
            return "short";
        case data::ScalarType::uint16:
            return "unsigned_short";
        case data::ScalarType::int32:
            return "int";
        case data::ScalarType::uint32:
            return "unsigned_int";
        case data::ScalarType::float32:
            return "float";
        case data::ScalarType::float64:
            return "double";
        case data::ScalarType::int64:
        case data::ScalarType::uint64:
            throw std::invalid_argument(
                "VTK legacy has no portable fixed-width 64-bit integer type");
        }
        throw std::invalid_argument("invalid scalar type");
    }

    data::ScalarType legacy_scalar_type(std::string_view name)
    {
        if (name == "char" || name == "signed_char")
            return data::ScalarType::int8;
        if (name == "unsigned_char")
            return data::ScalarType::uint8;
        if (name == "short")
            return data::ScalarType::int16;
        if (name == "unsigned_short")
            return data::ScalarType::uint16;
        if (name == "int")
            return data::ScalarType::int32;
        if (name == "unsigned_int")
            return data::ScalarType::uint32;
        if (name == "float")
            return data::ScalarType::float32;
        if (name == "double")
            return data::ScalarType::float64;
        throw std::runtime_error(
            "unsupported VTK legacy scalar type");
    }

    std::string_view xml_scalar_name(data::ScalarType type)
    {
        switch (type) {
        case data::ScalarType::int8:
            return "Int8";
        case data::ScalarType::uint8:
            return "UInt8";
        case data::ScalarType::int16:
            return "Int16";
        case data::ScalarType::uint16:
            return "UInt16";
        case data::ScalarType::int32:
            return "Int32";
        case data::ScalarType::uint32:
            return "UInt32";
        case data::ScalarType::int64:
            return "Int64";
        case data::ScalarType::uint64:
            return "UInt64";
        case data::ScalarType::float32:
            return "Float32";
        case data::ScalarType::float64:
            return "Float64";
        }
        throw std::invalid_argument("invalid scalar type");
    }

    data::ScalarType xml_scalar_type(std::string_view name)
    {
        if (name == "Int8")
            return data::ScalarType::int8;
        if (name == "UInt8")
            return data::ScalarType::uint8;
        if (name == "Int16")
            return data::ScalarType::int16;
        if (name == "UInt16")
            return data::ScalarType::uint16;
        if (name == "Int32")
            return data::ScalarType::int32;
        if (name == "UInt32")
            return data::ScalarType::uint32;
        if (name == "Int64")
            return data::ScalarType::int64;
        if (name == "UInt64")
            return data::ScalarType::uint64;
        if (name == "Float32")
            return data::ScalarType::float32;
        if (name == "Float64")
            return data::ScalarType::float64;
        throw std::runtime_error(
            "unsupported VTU scalar type");
    }

    std::vector<std::byte> interleave(
        const data::ArrayView& array)
    {
        data::validate_array(array);
        const auto width = data::scalar_size(array.type);
        ::idmesh::detail::require_length(
            array.tuples <= std::numeric_limits<std::size_t>::max()
                    / array.components.size()
                && array.tuples * array.components.size()
                    <= std::numeric_limits<std::size_t>::max()
                        / width,
            "VTK array size overflow");
        std::vector<std::byte> result(
            array.tuples * array.components.size() * width);
        for (std::size_t tuple = 0;
             tuple < array.tuples; ++tuple) {
            for (std::size_t component = 0;
                 component < array.components.size(); ++component) {
                const auto& source = array.components[component];
                const auto input = source.offset_bytes
                    + tuple * source.stride_bytes;
                const auto output
                    = (tuple * array.components.size() + component)
                    * width;
                std::copy_n(
                    source.storage.data() + input, width,
                    result.data() + output);
            }
        }
        return result;
    }

    void write_ascii_values(
        std::ostream& out, const data::ArrayView& array)
    {
        const auto bytes = interleave(array);
        const auto width = data::scalar_size(array.type);
        out << std::setprecision(17);
        for (std::size_t i = 0;
             i < array.tuples * array.components.size(); ++i) {
            if (i != 0)
                out << ' ';
            const auto* value = bytes.data() + i * width;
            switch (array.type) {
            case data::ScalarType::int8:
                write_value<std::int8_t>(out, value);
                break;
            case data::ScalarType::uint8:
                write_value<std::uint8_t>(out, value);
                break;
            case data::ScalarType::int16:
                write_value<std::int16_t>(out, value);
                break;
            case data::ScalarType::uint16:
                write_value<std::uint16_t>(out, value);
                break;
            case data::ScalarType::int32:
                write_value<std::int32_t>(out, value);
                break;
            case data::ScalarType::uint32:
                write_value<std::uint32_t>(out, value);
                break;
            case data::ScalarType::int64:
                write_value<std::int64_t>(out, value);
                break;
            case data::ScalarType::uint64:
                write_value<std::uint64_t>(out, value);
                break;
            case data::ScalarType::float32:
                write_value<float>(out, value);
                break;
            case data::ScalarType::float64:
                write_value<double>(out, value);
                break;
            }
        }
    }

    data::Array read_ascii_values(
        std::istream& in, data::ScalarType type,
        std::size_t tuples, std::uint32_t components)
    {
        ::idmesh::detail::require_runtime(
            components > 0, "VTK array has zero components");
        switch (type) {
        case data::ScalarType::int8:
            return read_values_t<std::int8_t>(
                in, type, tuples, components);
        case data::ScalarType::uint8:
            return read_values_t<std::uint8_t>(
                in, type, tuples, components);
        case data::ScalarType::int16:
            return read_values_t<std::int16_t>(
                in, type, tuples, components);
        case data::ScalarType::uint16:
            return read_values_t<std::uint16_t>(
                in, type, tuples, components);
        case data::ScalarType::int32:
            return read_values_t<std::int32_t>(
                in, type, tuples, components);
        case data::ScalarType::uint32:
            return read_values_t<std::uint32_t>(
                in, type, tuples, components);
        case data::ScalarType::int64:
            return read_values_t<std::int64_t>(
                in, type, tuples, components);
        case data::ScalarType::uint64:
            return read_values_t<std::uint64_t>(
                in, type, tuples, components);
        case data::ScalarType::float32:
            return read_values_t<float>(
                in, type, tuples, components);
        case data::ScalarType::float64:
            return read_values_t<double>(
                in, type, tuples, components);
        }
        throw std::runtime_error("invalid scalar type");
    }

    data::Array parse_ascii_values(
        std::string_view text, data::ScalarType type,
        std::size_t tuples, std::uint32_t components)
    {
        std::istringstream in {std::string(text)};
        auto result
            = read_ascii_values(in, type, tuples, components);
        std::string trailing;
        ::idmesh::detail::require_runtime(
            !(in >> trailing),
            "extra values in VTK numeric array");
        return result;
    }

    std::vector<std::byte> endian_bytes(
        const data::ArrayView& array, std::endian target)
    {
        auto result = interleave(array);
        if (target != std::endian::native)
            swap_scalars(result, data::scalar_size(array.type));
        return result;
    }

    data::Array array_from_endian_bytes(
        data::ScalarType type, std::size_t tuples,
        std::uint32_t components, std::vector<std::byte> bytes,
        std::endian source)
    {
        const auto width = data::scalar_size(type);
        ::idmesh::detail::require_length(
            components > 0
                && tuples <= std::numeric_limits<std::size_t>::max()
                    / components
                && tuples * components
                    <= std::numeric_limits<std::size_t>::max()
                        / width,
            "VTK array size overflow");
        ::idmesh::detail::require_runtime(
            bytes.size() == tuples * components * width,
            "VTK binary array byte count mismatch");
        if (source != std::endian::native)
            swap_scalars(bytes, width);
        return {type, tuples, components, std::move(bytes)};
    }

    std::string base64_encode(
        std::span<const std::byte> bytes)
    {
        static constexpr char alphabet[]
            = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "abcdefghijklmnopqrstuvwxyz"
              "0123456789+/";
        std::string out;
        out.reserve((bytes.size() + 2) / 3 * 4);
        for (std::size_t i = 0; i < bytes.size(); i += 3) {
            const auto a = std::to_integer<unsigned>(bytes[i]);
            const auto b = i + 1 < bytes.size()
                ? std::to_integer<unsigned>(bytes[i + 1]) : 0;
            const auto c = i + 2 < bytes.size()
                ? std::to_integer<unsigned>(bytes[i + 2]) : 0;
            const unsigned value = (a << 16) | (b << 8) | c;
            out.push_back(alphabet[(value >> 18) & 63]);
            out.push_back(alphabet[(value >> 12) & 63]);
            out.push_back(i + 1 < bytes.size()
                    ? alphabet[(value >> 6) & 63] : '=');
            out.push_back(i + 2 < bytes.size()
                    ? alphabet[value & 63] : '=');
        }
        return out;
    }

    std::vector<std::byte> base64_decode(
        std::string_view text)
    {
        std::array<int, 256> decode;
        decode.fill(-1);
        const std::string alphabet
            = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "abcdefghijklmnopqrstuvwxyz"
              "0123456789+/";
        for (std::size_t i = 0; i < alphabet.size(); ++i)
            decode[static_cast<unsigned char>(alphabet[i])]
                = static_cast<int>(i);

        std::string compact;
        compact.reserve(text.size());
        for (char ch : text)
            if (ch != ' ' && ch != '\t'
                && ch != '\r' && ch != '\n')
                compact.push_back(ch);
        ::idmesh::detail::require_runtime(
            compact.size() % 4 == 0,
            "invalid VTK base64 length");

        std::vector<std::byte> out;
        out.reserve(compact.size() / 4 * 3);
        for (std::size_t i = 0; i < compact.size(); i += 4) {
            int values[4] {};
            int padding = 0;
            for (int j = 0; j < 4; ++j) {
                if (compact[i + j] == '=') {
                    values[j] = 0;
                    ++padding;
                }
                else {
                    values[j] = decode[
                        static_cast<unsigned char>(
                            compact[i + j])];
                    ::idmesh::detail::require_runtime(
                        values[j] >= 0,
                        "invalid VTK base64 character");
                }
            }
            ::idmesh::detail::require_runtime(
                padding == 0
                    || (i + 4 == compact.size()
                        && padding <= 2
                        && compact[i + 3] == '='),
                "invalid VTK base64 padding");
            const unsigned value
                = (static_cast<unsigned>(values[0]) << 18)
                | (static_cast<unsigned>(values[1]) << 12)
                | (static_cast<unsigned>(values[2]) << 6)
                | static_cast<unsigned>(values[3]);
            out.push_back(
                static_cast<std::byte>((value >> 16) & 255));
            if (padding < 2)
                out.push_back(
                    static_cast<std::byte>((value >> 8) & 255));
            if (padding < 1)
                out.push_back(
                    static_cast<std::byte>(value & 255));
        }
        return out;
    }

    std::string xml_escape(std::string_view text)
    {
        std::string out;
        for (char ch : text) {
            switch (ch) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&quot;";
                break;
            case '\'':
                out += "&apos;";
                break;
            default:
                out.push_back(ch);
                break;
            }
        }
        return out;
    }

    std::string xml_unescape(std::string_view text)
    {
        std::string out;
        for (std::size_t i = 0; i < text.size();) {
            if (text[i] != '&') {
                out.push_back(text[i++]);
                continue;
            }
            const auto semi = text.find(';', i);
            ::idmesh::detail::require_runtime(
                semi != std::string_view::npos,
                "unterminated XML entity");
            const auto entity = text.substr(i, semi - i + 1);
            if (entity == "&amp;")
                out.push_back('&');
            else if (entity == "&lt;")
                out.push_back('<');
            else if (entity == "&gt;")
                out.push_back('>');
            else if (entity == "&quot;")
                out.push_back('"');
            else if (entity == "&apos;")
                out.push_back('\'');
            else
                throw std::runtime_error(
                    "unsupported XML entity");
            i = semi + 1;
        }
        return out;
    }


    XmlAttributes xml_attributes(std::string_view tag)
    {
        XmlAttributes result;
        const auto space = tag.find_first_of(" \t\r\n");
        if (space == std::string_view::npos)
            return result;
        std::size_t i = space;
        while (i < tag.size()) {
            while (i < tag.size()
                && std::isspace(static_cast<unsigned char>(tag[i])))
                ++i;
            if (i >= tag.size() || tag[i] == '>' || tag[i] == '/')
                break;
            const auto key_begin = i;
            while (i < tag.size()
                && !std::isspace(static_cast<unsigned char>(tag[i]))
                && tag[i] != '=')
                ++i;
            const std::string key(tag.substr(key_begin, i - key_begin));
            while (i < tag.size()
                && std::isspace(static_cast<unsigned char>(tag[i])))
                ++i;
            ::idmesh::detail::require_runtime(
                i < tag.size() && tag[i] == '=',
                "invalid VTK XML attribute");
            ++i;
            while (i < tag.size()
                && std::isspace(static_cast<unsigned char>(tag[i])))
                ++i;
            ::idmesh::detail::require_runtime(
                i < tag.size() && (tag[i] == '"' || tag[i] == '\''),
                "invalid VTK XML attribute quoting");
            const char quote = tag[i++];
            const auto value_begin = i;
            const auto value_end = tag.find(quote, i);
            ::idmesh::detail::require_runtime(
                value_end != std::string_view::npos,
                "unterminated VTK XML attribute");
            result.emplace(
                key, xml_unescape(tag.substr(
                         value_begin, value_end - value_begin)));
            i = value_end + 1;
        }
        return result;
    }

    std::string_view xml_required_attribute(
        const XmlAttributes& attributes, std::string_view name)
    {
        const auto it = attributes.find(std::string(name));
        ::idmesh::detail::require_runtime(
            it != attributes.end(),
            "missing required VTK XML attribute");
        return it->second;
    }

    std::string_view xml_opening_tag(
        std::string_view xml, std::string_view name)
    {
        const std::string open = "<" + std::string(name);
        const auto begin = xml.find(open);
        ::idmesh::detail::require_runtime(
            begin != std::string_view::npos,
            "missing VTK XML element");
        const auto end = xml.find('>', begin);
        ::idmesh::detail::require_runtime(
            end != std::string_view::npos,
            "unterminated VTK XML element");
        return xml.substr(begin, end - begin + 1);
    }

    std::string_view xml_element_body(
        std::string_view xml, std::string_view name, bool required)
    {
        const std::string open = "<" + std::string(name);
        const std::string close = "</" + std::string(name) + ">";
        const auto begin = xml.find(open);
        if (begin == std::string_view::npos) {
            if (required)
                throw std::runtime_error("missing VTK XML element");
            return {};
        }
        const auto head_end = xml.find('>', begin);
        const auto end = xml.find(close, head_end);
        ::idmesh::detail::require_runtime(
            head_end != std::string_view::npos
                && end != std::string_view::npos,
            "unterminated VTK XML element");
        return xml.substr(head_end + 1, end - head_end - 1);
    }

    std::vector<XmlAttributes> xml_self_closing_tags(
        std::string_view body, std::string_view name)
    {
        std::vector<XmlAttributes> result;
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
            const auto tag = body.substr(begin, end - begin + 1);
            ::idmesh::detail::require_runtime(
                end > begin && body[end - 1] == '/',
                "expected self-closing VTK XML tag");
            result.push_back(xml_attributes(tag));
            cursor = end + 1;
        }
        return result;
    }

} // namespace idmesh::detail::vtk
