
#include "doctest.h"
#include <complex>
#include "fulqrum.hpp"
#include <fulqrum.hpp>

typedef std::complex<double> complex;


TEST_CASE("Check operator width") {
    QubitOperator_t op = QubitOperator(19, {{"+XY", {2, 1, 0}, 1.0}, {"ZX", {0, 1}, complex(0,-1)}});
    CHECK(op.width == 19);
}
