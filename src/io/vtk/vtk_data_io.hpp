#pragma once

#include "idmesh/data.hpp"

#include <bit>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace idmesh::detail::vtk {

    std::string_view legacy_scalar_name(data::ScalarType type);
    data::ScalarType legacy_scalar_type(std::string_view name);

    std::string_view xml_scalar_name(data::ScalarType type);
    data::ScalarType xml_scalar_type(std::string_view name);

    // Numeric conversion shared by legacy VTK and VTU readers.
    std::int64_t signed_integer(
        const data::Array& array, std::size_t tuple,
        std::size_t component = 0);
    std::uint64_t nonnegative_integer(
        const data::Array& array, std::size_t tuple,
        std::size_t component = 0);
    double floating_value(
        const data::Array& array, std::size_t tuple,
        std::size_t component = 0);
    bool read_state_metadata(
        data::State& state, std::string_view name,
        const data::Array& values);

    std::vector<std::byte> interleave(
        const data::ArrayView& array);

    void write_ascii_values(
        std::ostream& out, const data::ArrayView& array);

    data::Array read_ascii_values(
        std::istream& in, data::ScalarType type,
        std::size_t tuples, std::uint32_t components);

    data::Array parse_ascii_values(
        std::string_view text, data::ScalarType type,
        std::size_t tuples, std::uint32_t components);

    std::vector<std::byte> endian_bytes(
        const data::ArrayView& array, std::endian target);

    data::Array array_from_endian_bytes(
        data::ScalarType type, std::size_t tuples,
        std::uint32_t components, std::vector<std::byte> bytes,
        std::endian source);

    std::string base64_encode(
        std::span<const std::byte> bytes);
    std::vector<std::byte> base64_decode(
        std::string_view text);

    std::string xml_escape(std::string_view text);
    std::string xml_unescape(std::string_view text);

    using XmlAttributes = std::map<std::string, std::string>;

    XmlAttributes xml_attributes(std::string_view tag);
    std::string_view xml_required_attribute(
        const XmlAttributes& attributes, std::string_view name);
    std::string_view xml_opening_tag(
        std::string_view xml, std::string_view name);
    std::string_view xml_element_body(
        std::string_view xml, std::string_view name,
        bool required = true);
    std::vector<XmlAttributes> xml_self_closing_tags(
        std::string_view body, std::string_view name);

} // namespace idmesh::detail::vtk
