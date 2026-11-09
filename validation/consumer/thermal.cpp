// Standalone installed-SDK consumer: P1 tetrahedral steady conduction on real meshes.
// No IDMesh private headers, solver packages or application-specific mesh adapters.
#include <idmesh/data.hpp>
#include <idmesh/io.hpp>
#include <idmesh/vtk.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace {
using Index = idmesh::Index;
using Point = std::array<double, 3>;
using Clock = std::chrono::steady_clock;

struct Element {
    std::array<Index, 4> nodes;
    std::array<double, 16> matrix;
};

Point subtract(const Point& a, const Point& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Point cross(const Point& a, const Point& b)
{
    return {a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
}

double dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require(bool condition, const std::string& what)
{
    if (!condition)
        throw std::runtime_error(what);
}

struct Options {
    std::string mesh;
    std::string problem = "patch";
    std::string output;
    std::string report;
    bool validate = false;
    int max_iterations = 2000;
    double tolerance = 1e-10;
};

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (++i == argc)
                throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--mesh")
            options.mesh = value();
        else if (arg == "--problem")
            options.problem = value();
        else if (arg == "--output-vtu")
            options.output = value();
        else if (arg == "--report")
            options.report = value();
        else if (arg == "--max-iterations")
            options.max_iterations = std::stoi(value());
        else if (arg == "--tolerance")
            options.tolerance = std::stod(value());
        else if (arg == "--validate-topology")
            options.validate = true;
        else
            throw std::invalid_argument("unknown argument: " + arg);
    }
    if (options.mesh.empty() || (options.problem != "patch" && options.problem != "heat"))
        throw std::invalid_argument("usage: idmesh_thermal --mesh FILE [--problem patch|heat] "
            "[--validate-topology] [--output-vtu FILE] [--report FILE]");
    if (options.max_iterations < 1 || !std::isfinite(options.tolerance)
        || options.tolerance <= 0 || options.tolerance >= 1)
        throw std::invalid_argument("invalid CG iteration limit or tolerance");
    return options;
}

struct Problem {
    std::vector<unsigned char> fixed;
    std::vector<unsigned char> active;
    std::vector<double> boundary_values;
    std::vector<Element> elements;
    std::vector<double> diagonal;
    std::vector<double> rhs;
    std::size_t boundary_facets = 0;
    std::size_t inactive_vertices = 0;
    std::size_t free_dofs = 0;
    std::size_t negative_orientations = 0;
    double minimum_scaled_volume = std::numeric_limits<double>::infinity();
};

Problem assemble(const idmesh::Mesh& mesh, const std::string& name)
{
    require(mesh.dimension() == 3 && mesh.geometric_dimension() == 3,
        "only three-dimensional volume meshes are supported");
    const auto positions = mesh.coordinates();
    const auto n = positions.size();
    require(n == mesh.entity_count(0) && n >= 4, "invalid mesh vertices");
    const auto count = mesh.entity_count(3);
    require(count > 0, "mesh contains no volume cells");
    require(mesh.has_relation(2, 3), "facet-to-cell relation is required");

    Point lower = positions.front(), upper = positions.front();
    for (const auto& p : positions) {
        for (int d = 0; d != 3; ++d) {
            lower[d] = std::min(lower[d], p[d]);
            upper[d] = std::max(upper[d], p[d]);
        }
    }
    const Point extent = subtract(upper, lower);
    for (const double length : extent)
        require(length > 0 && std::isfinite(length), "degenerate mesh bounding box");

    Problem problem;
    problem.fixed.assign(n, 0);
    problem.active.assign(n, 0);
    for (Index c = 0; c < count; ++c) {
        for (const Index v : mesh.entity_vertices(3, c))
            problem.active[v] = 1;
    }
    for (std::size_t v = 0; v < n; ++v)
        if (!problem.active[v]) {
            problem.fixed[v] = 1;
            ++problem.inactive_vertices;
        }
    problem.boundary_values.assign(n, 0.0);
    problem.diagonal.assign(n, 0.0);
    problem.rhs.assign(n, 0.0);

    const auto support = mesh.relation(2, 3);
    std::vector<unsigned char> boundary(n, 0);
    for (Index face = 0; face < mesh.entity_count(2); ++face) {
        const auto adj = support[face].targets;
        require(adj.size() == 1 || adj.size() == 2,
            "disconnected or nonmanifold facet");
        if (adj.size() == 1) {
            ++problem.boundary_facets;
            for (const Index node : mesh.entity_vertices(2, face))
                boundary[node] = 1;
        }
    }
    require(problem.boundary_facets > 0, "missing boundary facets");

    std::size_t cold = 0, hot = 0;
    for (std::size_t v = 0; v < n; ++v) {
        if (!boundary[v] || !problem.active[v])
            continue;
        const auto& p = positions[v];
        if (name == "patch") {
            problem.fixed[v] = 1;
            problem.boundary_values[v] = (p[0] - lower[0]) / extent[0]
                + 0.3 * (p[1] - lower[1]) / extent[1]
                - 0.2 * (p[2] - lower[2]) / extent[2];
        }
        else {
            const double x = (p[0] - lower[0]) / extent[0];
            if (x <= 0.075) {
                problem.fixed[v] = 1;
                ++cold;
            }
            else if (x >= 0.925) {
                problem.fixed[v] = 1;
                problem.boundary_values[v] = 1;
                ++hot;
            }
        }
    }
    if (name == "heat")
        require(cold > 0 && hot > 0, "heat contacts have no boundary vertices");

    problem.free_dofs = n - static_cast<std::size_t>(
        std::count(problem.fixed.begin(), problem.fixed.end(), 1));
    require(problem.free_dofs > 0, "no free DOFs: mesh is too coarse for an end-to-end solve");

    problem.elements.reserve(count);
    for (Index c = 0; c < count; ++c) {
        require(mesh.cell_types()[c] == idmesh::CellType::tetrahedron
                && mesh.cell_order(c) == 1,
            "this P1 example requires first-order tetrahedral volume cells");
        const auto ids = mesh.entity_vertices(3, c);
        require(ids.size() == 4, "invalid tetrahedron node count");
        Element element {};
        std::copy(ids.begin(), ids.end(), element.nodes.begin());
        const Point p0 = positions[ids[0]], p1 = positions[ids[1]],
                    p2 = positions[ids[2]], p3 = positions[ids[3]];
        const auto a = subtract(p1, p0), b = subtract(p2, p0),
                   d = subtract(p3, p0);
        const double det = dot(a, cross(b, d));
        const double h = std::sqrt(std::max({dot(a, a), dot(b, b), dot(d, d)}));
        require(std::isfinite(det) && h > 0, "invalid element geometry");
        const double scaled = std::abs(det) / (h * h * h);
        require(scaled > 1e-15, "degenerate or nearly singular tetrahedron");
        problem.minimum_scaled_volume = std::min(problem.minimum_scaled_volume, scaled);
        if (det < 0)
            ++problem.negative_orientations;
        const double volume = std::abs(det) / 6;
        std::array<Point, 4> gradient {};
        gradient[1] = cross(b, d);
        gradient[2] = cross(d, a);
        gradient[3] = cross(a, b);
        for (int axis = 0; axis != 3; ++axis) {
            for (int i = 1; i != 4; ++i)
                gradient[i][axis] /= det;
            gradient[0][axis] = -(gradient[1][axis] + gradient[2][axis]
                + gradient[3][axis]);
        }
        for (int i = 0; i != 4; ++i) {
            const Index vi = ids[i];
            for (int j = 0; j != 4; ++j) {
                const double kij = volume * dot(gradient[i], gradient[j]);
                element.matrix[i * 4 + j] = kij;
                if (!problem.fixed[vi] && problem.fixed[ids[j]])
                    problem.rhs[vi] -= kij * problem.boundary_values[ids[j]];
            }
            if (!problem.fixed[vi])
                problem.diagonal[vi] += element.matrix[i * 4 + i];
        }
        problem.elements.push_back(element);
    }
    for (std::size_t i = 0; i < n; ++i)
        if (!problem.fixed[i])
            require(problem.diagonal[i] > 0 && std::isfinite(problem.diagonal[i]),
                "free vertex with invalid stiffness diagonal");
    return problem;
}

void multiply(const Problem& model, const std::vector<double>& x,
    std::vector<double>& y, bool include_fixed)
{
    std::fill(y.begin(), y.end(), 0.0);
    for (const auto& cell : model.elements) {
        for (int i = 0; i != 4; ++i) {
            const Index vi = cell.nodes[i];
            if (!include_fixed && model.fixed[vi])
                continue;
            double sum = 0;
            for (int j = 0; j != 4; ++j)
                sum += cell.matrix[i * 4 + j] * x[cell.nodes[j]];
            y[vi] += sum;
        }
    }
}

double dot_free(const Problem& model, const std::vector<double>& a,
    const std::vector<double>& b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!model.fixed[i])
            sum += a[i] * b[i];
    return sum;
}

struct Solution {
    std::vector<double> values;
    int iterations = 0;
    double residual = 0;
    double patch_error = 0;
    double reaction_imbalance = 0;
    double energy = 0;
    double minimum = 0;
    double maximum = 0;
};

Solution solve(const idmesh::Mesh& mesh, const Problem& model,
    const Options& options)
{
    const auto n = mesh.coordinates().size();
    std::vector<double> x(n, 0.0), r = model.rhs, z(n, 0.0),
                        p(n, 0.0), ap(n, 0.0);
    const double norm_b = std::sqrt(dot_free(model, r, r));
    double rz = 0;
    for (std::size_t v = 0; v < n; ++v) {
        if (!model.fixed[v]) {
            z[v] = r[v] / model.diagonal[v];
            p[v] = z[v];
            rz += r[v] * z[v];
        }
    }

    Solution result;
    double residual = norm_b;
    while (norm_b > 0 && residual > options.tolerance * norm_b
        && result.iterations < options.max_iterations) {
        multiply(model, p, ap, false);
        const double denom = dot_free(model, p, ap);
        require(denom > 0 && std::isfinite(denom), "CG operator is not SPD");
        const double alpha = rz / denom;
        for (std::size_t v = 0; v < n; ++v) {
            if (!model.fixed[v]) {
                x[v] += alpha * p[v];
                r[v] -= alpha * ap[v];
            }
        }
        residual = std::sqrt(dot_free(model, r, r));
        ++result.iterations;
        if (residual <= options.tolerance * norm_b)
            break;
        double next_rz = 0;
        for (std::size_t v = 0; v < n; ++v) {
            if (!model.fixed[v]) {
                z[v] = r[v] / model.diagonal[v];
                next_rz += r[v] * z[v];
            }
        }
        require(rz > 0 && std::isfinite(next_rz), "CG preconditioner broke down");
        const double beta = next_rz / rz;
        rz = next_rz;
        for (std::size_t v = 0; v < n; ++v)
            if (!model.fixed[v])
                p[v] = z[v] + beta * p[v];
    }
    require(norm_b == 0 || residual <= options.tolerance * norm_b,
        "CG failed to converge within iteration budget");

    for (std::size_t v = 0; v < n; ++v)
        if (model.fixed[v])
            x[v] = model.boundary_values[v];

    multiply(model, x, ap, true);
    double free_residual_squared = 0, reaction = 0, reaction_abs = 0;
    for (std::size_t v = 0; v < n; ++v) {
        if (!model.fixed[v])
            free_residual_squared += ap[v] * ap[v];
        else {
            reaction += ap[v];
            reaction_abs += std::abs(ap[v]);
        }
    }
    result.residual = std::sqrt(free_residual_squared)
        / std::max(norm_b, 1e-30);
    result.reaction_imbalance = std::abs(reaction)
        / std::max(reaction_abs, 1e-30);
    result.energy = std::inner_product(x.begin(), x.end(), ap.begin(), 0.0);
    result.values = std::move(x);
    const auto bounds = std::minmax_element(result.values.begin(), result.values.end());
    result.minimum = *bounds.first;
    result.maximum = *bounds.second;
    if (options.problem == "patch") {
        Point lower = mesh.coordinates().front(), upper = lower;
        for (const auto& v : mesh.coordinates()) {
            for (int d = 0; d < 3; ++d) {
                lower[d] = std::min(lower[d], v[d]);
                upper[d] = std::max(upper[d], v[d]);
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (!model.active[i])
                continue;
            const auto& v = mesh.coordinates()[i];
            const double exact = (v[0] - lower[0]) / (upper[0] - lower[0])
                + 0.3 * (v[1] - lower[1]) / (upper[1] - lower[1])
                - 0.2 * (v[2] - lower[2]) / (upper[2] - lower[2]);
            result.patch_error = std::max(result.patch_error,
                std::abs(result.values[i] - exact));
        }
    }
    require(std::isfinite(result.energy) && result.energy >= -1e-9,
        "negative energy in diffusion solve");
    require(result.residual <= 1e-7, "true (recomputed) residual is too large");
    require(result.reaction_imbalance <= 1e-6,
        "discrete total heat flux is not conservative");
    if (options.problem == "patch")
        require(result.patch_error <= 1e-6, "linear FE patch test failed");
    else
        require(result.minimum >= -1e-6 && result.maximum <= 1 + 1e-6,
            "steady heat solve violates the discrete maximum principle");
    return result;
}

void export_vtu(const idmesh::Mesh& mesh, const std::vector<double>& values,
    const std::string& filename)
{
    // External meshes may contain point-only features that are not referenced
    // by a volume cell. Convert to a compact volume-domain visualization mesh:
    // the original point order and scalar field remain untouched in the solver.
    const auto n = mesh.entity_count(0);
    std::vector<Index> map(n, idmesh::invalid_index);
    std::vector<double> output_values;
    idmesh::MeshBuilder builder(3, 3);
    for (Index c = 0; c < mesh.entity_count(3); ++c) {
        std::array<Index, 4> mapped {};
        const auto nodes = mesh.entity_vertices(3, c);
        for (std::size_t i = 0; i < 4; ++i) {
            const Index v = nodes[i];
            if (map[v] == idmesh::invalid_index) {
                map[v] = builder.add_vertex(mesh.coordinates()[v]);
                output_values.push_back(values[v]);
            }
            mapped[i] = map[v];
        }
        builder.add_cell(idmesh::CellType::tetrahedron, mapped);
    }
    const auto output_mesh = std::move(builder).finalize(
        {idmesh::TopologyLayout::cell_vertices,
            idmesh::RelationDirection::downward});
    const idmesh::data::MeshDataView field {
        "temperature", idmesh::data::entities(0), idmesh::data::all(),
        idmesh::data::make_scalar_array(std::span<const double>(output_values))
    };
    idmesh::data::StateView state;
    state.fields = std::span<const idmesh::data::MeshDataView>(&field, 1);
    idmesh::vtk::write_vtu(output_mesh, state, filename);

    const auto restored = idmesh::vtk::read_vtu(filename,
        {idmesh::TopologyLayout::cell_vertices, idmesh::RelationDirection::downward});
    require(restored.mesh.entity_count(0) == output_values.size()
            && restored.mesh.entity_count(3) == mesh.entity_count(3),
        "VTU round-trip lost mesh entities");
    bool found = false;
    for (const auto& candidate : restored.state.fields) {
        if (candidate.name != "temperature")
            continue;
        require(candidate.values.type == idmesh::data::ScalarType::float64
                && candidate.values.components == 1
                && candidate.values.tuples == output_values.size()
                && candidate.values.values.size() == output_values.size() * sizeof(double),
            "VTU round-trip changed field schema");
        for (std::size_t i = 0; i < output_values.size(); ++i) {
            double restored_value = 0;
            std::memcpy(&restored_value,
                candidate.values.values.data() + i * sizeof(double), sizeof(double));
            require(std::abs(restored_value - output_values[i]) <= 1e-12,
                "VTU round-trip changed temperature values");
        }
        found = true;
        break;
    }
    require(found, "VTU round-trip lost temperature field");
}

void report(std::ostream& out, const Options& o, const idmesh::Mesh& mesh,
    const Problem& model, const Solution& result,
    double import_s, double assembly_s, double solve_s, double output_s)
{
    std::size_t peak_rss_kb = 0;
#if defined(__linux__)
    rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        peak_rss_kb = static_cast<std::size_t>(usage.ru_maxrss);
#endif
    out << std::setprecision(17)
        << "{\n  \"problem\": \"" << o.problem << "\",\n"
        << "  \"vertices\": " << mesh.entity_count(0) << ",\n"
        << "  \"cells\": " << mesh.entity_count(3) << ",\n"
        << "  \"facets\": " << mesh.entity_count(2) << ",\n"
        << "  \"boundary_facets\": " << model.boundary_facets << ",\n"
        << "  \"free_dofs\": " << model.free_dofs << ",\n"
        << "  \"unused_input_vertices\": " << model.inactive_vertices << ",\n"
        << "  \"negative_orientations\": " << model.negative_orientations << ",\n"
        << "  \"min_scaled_volume\": " << model.minimum_scaled_volume << ",\n"
        << "  \"iterations\": " << result.iterations << ",\n"
        << "  \"relative_residual\": " << result.residual << ",\n"
        << "  \"patch_error_linf\": " << result.patch_error << ",\n"
        << "  \"reaction_imbalance\": " << result.reaction_imbalance << ",\n"
        << "  \"energy\": " << result.energy << ",\n"
        << "  \"temperature_min\": " << result.minimum << ",\n"
        << "  \"temperature_max\": " << result.maximum << ",\n"
        << "  \"import_seconds\": " << import_s << ",\n"
        << "  \"assembly_seconds\": " << assembly_s << ",\n"
        << "  \"solve_seconds\": " << solve_s << ",\n"
        << "  \"vtu_roundtrip_seconds\": " << output_s << ",\n"
        << "  \"peak_rss_kb_linux\": " << peak_rss_kb << "\n}\n";
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse(argc, argv);
        const auto start = Clock::now();
        const auto mesh = idmesh::read_gmsh(options.mesh,
            {idmesh::TopologyLayout::cell_facets,
                idmesh::RelationDirection::bidirectional});
        if (options.validate)
            mesh.validate();
        const double import_s = seconds(start);

        const auto assembly_start = Clock::now();
        const auto model = assemble(mesh, options.problem);
        const double assembly_s = seconds(assembly_start);

        const auto solve_start = Clock::now();
        const auto solution = solve(mesh, model, options);
        const double solve_s = seconds(solve_start);

        double output_s = 0;
        if (!options.output.empty()) {
            const auto output_start = Clock::now();
            export_vtu(mesh, solution.values, options.output);
            output_s = seconds(output_start);
        }
        report(std::cout, options, mesh, model, solution,
            import_s, assembly_s, solve_s, output_s);
        if (!options.report.empty()) {
            std::ofstream file(options.report);
            require(static_cast<bool>(file), "cannot open JSON report");
            report(file, options, mesh, model, solution,
                import_s, assembly_s, solve_s, output_s);
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "IDMesh thermal acceptance failed: " << e.what() << '\n';
        return 1;
    }
}
