#include "idmesh/geometry.hpp"
#include "second_order.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace idmesh {
namespace {

struct Jet {
    double v = 0;
    std::array<double, 3> d {};
    Jet() = default;
    Jet(double x) : v(x) {}
    static Jet variable(double x, int j) {
        Jet result(x);
        result.d[j] = 1;
        return result;
    }
};
Jet operator+(Jet a, Jet b) {
    a.v += b.v;
    for (int j=0;j<3;++j) a.d[j]+=b.d[j];
    return a;
}
Jet operator-(Jet a, Jet b) {
    a.v -= b.v;
    for (int j=0;j<3;++j) a.d[j]-=b.d[j];
    return a;
}
Jet operator-(Jet a) {
    a.v=-a.v;
    for(auto& d:a.d)d=-d;
    return a;
}
Jet operator*(Jet a, Jet b) {
    Jet r;
    r.v=a.v*b.v;
    for(int j=0;j<3;++j)r.d[j]=a.d[j]*b.v+a.v*b.d[j];
    return r;
}
Jet operator/(Jet a, Jet b) {
    if (std::abs(b.v)<1e-14) throw std::invalid_argument("singular reference chart");
    Jet r;
    r.v=a.v/b.v;
    for(int j=0;j<3;++j)r.d[j]=(a.d[j]-r.v*b.d[j])/b.v;
    return r;
}

using Vec = std::array<double, 3>;

Vec midpoint(Vec a, Vec b) {
    for(int j=0;j<3;++j)a[j]=(a[j]+b[j])*0.5;
    return a;
}
Vec center(std::span<const Vec> vertices) {
    Vec c{};
    for(auto p:vertices)for(int j=0;j<3;++j)c[j]+=p[j]/vertices.size();
    return c;
}

std::vector<Vec> vtk_reference_nodes(CellType type, Index count)
{
    using CT=CellType;
    std::vector<Vec> p;
    switch(type) {
    case CT::segment:
        p={{-1,0,0},{1,0,0}};
        if(count==3)p.push_back({0,0,0});
        break;
    case CT::triangle:
        p={{0,0,0},{1,0,0},{0,1,0}};
        if(count==6)p.insert(p.end(),{{.5,0,0},{.5,.5,0},{0,.5,0}});
        break;
    case CT::quadrilateral:
        p={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};
        if(count>4)p.insert(p.end(),{{0,-1,0},{1,0,0},{0,1,0},{-1,0,0}});
        if(count==9)p.push_back({0,0,0});
        break;
    case CT::tetrahedron:
        p={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
        if(count==10) {
            for(auto edge:std::array<std::array<int,2>,6>{{
                {0,1},{1,2},{2,0},{0,3},{1,3},{2,3}}})
                p.push_back(midpoint(p[edge[0]],p[edge[1]]));
        }
        break;
    case CT::hexahedron:
        p={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
           {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
        if(count>8) {
            for(auto edge:std::array<std::array<int,2>,12>{{
                {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}}})
                p.push_back(midpoint(p[edge[0]],p[edge[1]]));
        }
        if(count==27) {
            for(auto face:std::array<std::array<int,4>,6>{{
                {0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7},
                {0,1,2,3},{4,5,6,7}}}) {
                const std::array<Vec,4> vertices{
                    p[face[0]],p[face[1]],p[face[2]],p[face[3]]};
                p.push_back(center(vertices));
            }
            p.push_back({0,0,0});
        }
        break;
    case CT::prism:
        p={{0,0,-1},{1,0,-1},{0,1,-1},
           {0,0,1},{1,0,1},{0,1,1}};
        if(count>6) {
            for(auto edge:std::array<std::array<int,2>,9>{{
                {0,1},{1,2},{2,0},{3,4},{4,5},{5,3},
                {0,3},{1,4},{2,5}}})
                p.push_back(midpoint(p[edge[0]],p[edge[1]]));
        }
        if(count==18)p.insert(p.end(),{{.5,0,0},{.5,.5,0},{0,.5,0}});
        break;
    case CT::pyramid:
        p={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},{0,0,1}};
        if(count>5) {
            for(auto edge:std::array<std::array<int,2>,8>{{
                {0,1},{1,2},{2,3},{3,0},{0,4},{1,4},{2,4},{3,4}}})
                p.push_back(midpoint(p[edge[0]],p[edge[1]]));
        }
        if(count==14)p.push_back({0,0,0});
        break;
    }
    if(p.size()!=count) throw std::invalid_argument("unsupported geometry node count");
    return p;
}

std::array<Jet,3> triangle_linear(Jet x, Jet y)
{
    return {Jet(1)-x-y,x,y};
}
std::array<Jet,6> triangle_quadratic(Jet x, Jet y)
{
    auto l=triangle_linear(x,y);
    return {l[0]*(Jet(2)*l[0]-Jet(1)),
            l[1]*(Jet(2)*l[1]-Jet(1)),
            l[2]*(Jet(2)*l[2]-Jet(1)),
            Jet(4)*l[0]*l[1],
            Jet(4)*l[1]*l[2],
            Jet(4)*l[2]*l[0]};
}
std::array<Jet,3> lagrange2(Jet x) {
    return {x*(x-Jet(1))/Jet(2),Jet(1)-x*x,
            x*(x+Jet(1))/Jet(2)};
}
std::array<Jet,4> quad_linear(Jet x, Jet y) {
    return {(Jet(1)-x)*(Jet(1)-y)/Jet(4),
            (Jet(1)+x)*(Jet(1)-y)/Jet(4),
            (Jet(1)+x)*(Jet(1)+y)/Jet(4),
            (Jet(1)-x)*(Jet(1)+y)/Jet(4)};
}
std::array<Jet,8> quad_serendipity(Jet x, Jet y) {
    const auto v=quad_linear(x,y);
    return {v[0]*(-x-y-Jet(1)),v[1]*(x-y-Jet(1)),
            v[2]*(x+y-Jet(1)),v[3]*(-x+y-Jet(1)),
            (Jet(1)-x*x)*(Jet(1)-y)/Jet(2),
            (Jet(1)-y*y)*(Jet(1)+x)/Jet(2),
            (Jet(1)-x*x)*(Jet(1)+y)/Jet(2),
            (Jet(1)-y*y)*(Jet(1)-x)/Jet(2)};
}
std::array<Jet,9> quad_tensor(Jet x, Jet y) {
    auto u=lagrange2(x),v=lagrange2(y);
    return {u[0]*v[0],u[2]*v[0],u[2]*v[2],u[0]*v[2],
            u[1]*v[0],u[2]*v[1],u[1]*v[2],u[0]*v[1],
            u[1]*v[1]};
}

std::vector<Jet> shape_vtk(CellType type, Index count,
                           std::array<Jet,3> q)
{
    const auto x=q[0],y=q[1],z=q[2];
    std::vector<Jet> result(count);
    switch(type) {
    case CellType::segment:
        if(count==2) return {(Jet(1)-x)/Jet(2),(Jet(1)+x)/Jet(2)};
        else {
            auto b=lagrange2(x);
            return {b[0],b[2],b[1]};
        }
    case CellType::triangle:
        if(count==3) {
            auto b=triangle_linear(x,y);
            return {b.begin(),b.end()};
        } else {
            auto b=triangle_quadratic(x,y);
            return {b.begin(),b.end()};
        }
    case CellType::quadrilateral:
        if(count==4) {auto b=quad_linear(x,y);return {b.begin(),b.end()};}
        if(count==8) {auto b=quad_serendipity(x,y);return {b.begin(),b.end()};}
        {auto b=quad_tensor(x,y);return {b.begin(),b.end()};}
    case CellType::tetrahedron: {
        std::array<Jet,4> b{Jet(1)-x-y-z,x,y,z};
        if(count==4)return {b.begin(),b.end()};
        return {b[0]*(Jet(2)*b[0]-Jet(1)),
                b[1]*(Jet(2)*b[1]-Jet(1)),
                b[2]*(Jet(2)*b[2]-Jet(1)),
                b[3]*(Jet(2)*b[3]-Jet(1)),
                Jet(4)*b[0]*b[1],Jet(4)*b[1]*b[2],
                Jet(4)*b[2]*b[0],Jet(4)*b[0]*b[3],
                Jet(4)*b[1]*b[3],Jet(4)*b[2]*b[3]};
    }
    case CellType::hexahedron: {
        const auto nodes=vtk_reference_nodes(type,count);
        if(count==8 || count==20) {
            for(Index i=0;i<count;++i) {
                const auto a=nodes[i];
                if(i<8) {
                    const auto product=(Jet(1)+a[0]*x)*(Jet(1)+a[1]*y)*
                                       (Jet(1)+a[2]*z)/Jet(8);
                    result[i]=count==8 ? product :
                        product*(a[0]*x+a[1]*y+a[2]*z-Jet(2));
                } else if(a[0]==0) {
                    result[i]=(Jet(1)-x*x)*(Jet(1)+a[1]*y)*(Jet(1)+a[2]*z)/Jet(4);
                } else if(a[1]==0) {
                    result[i]=(Jet(1)-y*y)*(Jet(1)+a[0]*x)*(Jet(1)+a[2]*z)/Jet(4);
                } else {
                    result[i]=(Jet(1)-z*z)*(Jet(1)+a[0]*x)*(Jet(1)+a[1]*y)/Jet(4);
                }
            }
        } else {
            const auto u=lagrange2(x),v=lagrange2(y),w=lagrange2(z);
            const auto idx=[](double p){return p<-.5 ? 0 : p>.5 ? 2 : 1;};
            for(Index i=0;i<count;++i)
                result[i]=u[idx(nodes[i][0])]*v[idx(nodes[i][1])]*w[idx(nodes[i][2])];
        }
        return result;
    }
    case CellType::prism: {
        const auto lin=triangle_linear(x,y);
        if(count==6) {
            for(int i=0;i<3;++i) {
                result[i]=lin[i]*(Jet(1)-z)/Jet(2);
                result[i+3]=lin[i]*(Jet(1)+z)/Jet(2);
            }
        } else {
            const auto tri=triangle_quadratic(x,y);
            if(count==15) {
                const Jet middle=Jet(1)-z*z;
                for(int i=0;i<3;++i) {
                    result[i]=tri[i]*(Jet(1)-z)/Jet(2)-lin[i]*middle/Jet(2);
                    result[i+3]=tri[i]*(Jet(1)+z)/Jet(2)-lin[i]*middle/Jet(2);
                    result[i+12]=lin[i]*middle;
                    result[i+6]=tri[i+3]*(Jet(1)-z)/Jet(2);
                    result[i+9]=tri[i+3]*(Jet(1)+z)/Jet(2);
                }
            } else {
                const auto l=lagrange2(z);
                for(int i=0;i<3;++i) {
                    result[i]=tri[i]*l[0];
                    result[i+3]=tri[i]*l[2];
                    result[i+12]=tri[i]*l[1];
                    result[i+6]=tri[i+3]*l[0];
                    result[i+9]=tri[i+3]*l[2];
                    result[i+15]=tri[i+3]*l[1];
                }
            }
        }
        return result;
    }
    case CellType::pyramid: {
        // Collapsed square coordinates: x=(1-z)u, y=(1-z)v.
        // This standard rational chart has a singular derivative at the apex.
        const Jet u=x/(Jet(1)-z),v=y/(Jet(1)-z);
        if(count==5) {
            const auto base=quad_linear(u,v);
            for(int i=0;i<4;++i)result[i]=(Jet(1)-z)*base[i];
            result[4]=z;
        } else {
            if(count==13) {
                const auto base=quad_serendipity(u,v);
                for(int i=0;i<8;++i)result[i<4?i:i+1]=(Jet(1)-z)*(Jet(1)-Jet(2)*z)*base[i];
            } else {
                const auto base=quad_tensor(u,v);
                for(int i=0;i<4;++i)result[i]=(Jet(1)-z)*(Jet(1)-Jet(2)*z)*base[i];
                for(int i=4;i<8;++i)result[i+1]=(Jet(1)-z)*(Jet(1)-Jet(2)*z)*base[i];
                result[13]=(Jet(1)-z)*(Jet(1)-Jet(2)*z)*base[8];
            }
            const auto c=quad_linear(u,v);
            for(int i=0;i<4;++i)result[9+i]=Jet(4)*z*(Jet(1)-z)*c[i];
            result[4]=z*(Jet(2)*z-Jet(1));
        }
        return result;
    }
    }
    throw std::invalid_argument("unknown reference cell");
}

void check_domain(CellType type,std::array<double,3> p)
{
    constexpr double tol=1e-10;
    const auto x=p[0],y=p[1],z=p[2];
    bool valid=false;
    switch(type) {
    case CellType::segment:valid=std::abs(x)<=1+tol;break;
    case CellType::triangle:valid=x>=-tol && y>=-tol && x+y<=1+tol;break;
    case CellType::quadrilateral:valid=std::abs(x)<=1+tol && std::abs(y)<=1+tol;break;
    case CellType::tetrahedron:valid=x>=-tol&&y>=-tol&&z>=-tol&&x+y+z<=1+tol;break;
    case CellType::hexahedron:valid=std::abs(x)<=1+tol&&std::abs(y)<=1+tol&&std::abs(z)<=1+tol;break;
    case CellType::prism:valid=x>=-tol&&y>=-tol&&x+y<=1+tol&&std::abs(z)<=1+tol;break;
    case CellType::pyramid:
        valid=z>=-tol&&z<1-1e-12&&std::abs(x)<=1-z+tol&&std::abs(y)<=1-z+tol;
        break;
    }
    if(!valid)throw std::invalid_argument("reference coordinate outside element or singular apex");
}
} // namespace

std::vector<std::array<double,3>> reference_geometry_nodes(CellType type,Index count)
{
    auto vtk=vtk_reference_nodes(type,count);
    const auto mapping=detail::vtk_from_gmsh(type,count);
    if(mapping.empty())return vtk;
    std::vector<Vec> gmsh(count);
    for(Index i=0;i<count;++i)gmsh[mapping[i]]=vtk[i];
    return gmsh;
}

GeometryEvaluation evaluate_geometry(const Mesh& mesh, Index cell,
                                      std::array<double,3> reference)
{
    if(cell>=mesh.entity_count(mesh.dimension()))throw std::out_of_range("cell index");
    const auto type=mesh.cell_types()[cell];
    check_domain(type,reference);
    const auto ids=mesh.cell_geometry_nodes(cell);
    const auto perm=detail::vtk_from_gmsh(type,static_cast<Index>(ids.size()));
    const auto vars=std::array<Jet,3>{
        Jet::variable(reference[0],0),Jet::variable(reference[1],1),
        Jet::variable(reference[2],2)};
    const auto shape=shape_vtk(type,static_cast<Index>(ids.size()),vars);
    GeometryEvaluation result;
    for(Index i=0;i<ids.size();++i) {
        const auto position=mesh.geometry_node_position(ids[perm.empty()?i:perm[i]]);
        for(Index a=0;a<mesh.geometric_dimension();++a) {
            result.point[a]+=position[a]*shape[i].v;
            for(Index b=0;b<mesh.dimension();++b)
                result.jacobian[a][b]+=position[a]*shape[i].d[b];
        }
    }
    return result;
}

} // namespace idmesh
