
#include <iostream>
#include <metis/metis.hpp>

void vector_example() {
    std::cout << "--- Vector Example ---\n";
    // Symbolic Vector
    metis::SymbolicVector v(3);
    auto y = metis::sym("y");
    // Setting elements
    v(0) = y;
    v(1) = y * y;
    v(2) = 1.0;

    std::cout << "Vector v:\n" << v << "\n";

    // Getting elements
    auto v0 = v(0);
    std::cout << "v(0) = " << v0 << "\n";
}

void set_get_example() {
    std::cout << "--- Set/Get Example ---\n";
    metis::SymbolicMatrix M(2, 2);

    // Setting via operator()
    M(0, 0) = 10.0;
    M(0, 1) = metis::sym("a");
    M(1, 0) = metis::sym("b");
    M(1, 1) = M(0, 1) + M(1, 0);

    std::cout << "Matrix M:\n" << M << "\n";

    // Getting via operator()
    auto val = M(1, 1);
    std::cout << "M(1, 1) extracted: " << val << "\n";

    // Block operations (Eigen style)
    M.row(0) = M.row(1);
    std::cout << "After row copy:\n" << M << "\n\n";
}

int main() {
    // Numeric
    metis::MetisMatrix<double> M_num(2, 2);
    M_num << 1.0, 2.0, 3.0, 4.0;
    std::cout << "Numeric Matrix:\n" << M_num << "\n\n";

    // Symbolic
    auto x = metis::sym("x");
    metis::MetisMatrix<metis::SymbolicScalar> M_sym(2, 2);
    M_sym << x, x + 1, x * 2, metis::sin(x);

    std::cout << "Symbolic Matrix (Expression):\n" << M_sym << "\n\n";

    // Symbolic Constant
    metis::MetisMatrix<metis::SymbolicScalar> M_const(2, 2);
    M_const << 1.0, 2.0, 3.0, 4.0;
    std::cout << "Symbolic Matrix (Constant):\n" << M_const << "\n\n";

    set_get_example();
    vector_example();

    return 0;
}
