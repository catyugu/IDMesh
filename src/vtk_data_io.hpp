#pragma once

#include "idmesh/data.hpp"

#include <bit>
#include <cstddef>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace idmesh::detail::vtk {

    std::string_view legacy_scalar_name(data::ScalarType type);
    data::ScalarType legacy_scalar_type(std::string_view name);

    std::string_view xml_scalar_name(data::ScalarType type);
    data::ScalarType xml_scalar_type(std::string_view name);

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

} // namespace idmesh::detail::vtk
