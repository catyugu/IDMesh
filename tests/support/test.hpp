#pragma once

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace idmesh::test {

    class Failure : public std::runtime_error {
    public:
        explicit Failure(std::string message)
            : std::runtime_error(std::move(message))
        {
        }
    };

    inline void check(bool condition, std::string_view message)
    {
        if (!condition)
            throw Failure(std::string(message));
    }

    inline void require(bool condition, std::string_view message)
    {
        check(condition, message);
    }

    template <class Error, class Operation>
    void expect_error(Operation&& operation, std::string_view message)
    {
        try {
            std::forward<Operation>(operation)();
        }
        catch (const Error&) {
            return;
        }
        catch (const std::exception& error) {
            throw Failure(std::string(message)
                + ": wrong exception: " + error.what());
        }
        catch (...) {
            throw Failure(std::string(message)
                + ": wrong non-standard exception");
        }
        throw Failure(std::string(message)
            + ": expected exception was not thrown");
    }

    struct Case {
        std::string_view name;
        void (*function)();
    };

    inline int run(std::initializer_list<Case> cases)
    {
        int failures = 0;
        for (const auto& test : cases) {
            try {
                test.function();
                std::cout << "[pass] " << test.name << '\n';
            }
            catch (const std::exception& error) {
                ++failures;
                std::cerr << "[fail] " << test.name << ": "
                          << error.what() << '\n';
            }
            catch (...) {
                ++failures;
                std::cerr << "[fail] " << test.name
                          << ": non-standard exception\n";
            }
        }
        return failures == 0 ? 0 : 1;
    }

} // namespace idmesh::test

using idmesh::test::check;
using idmesh::test::expect_error;
using idmesh::test::require;
