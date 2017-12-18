#include "../util/Calculator.h"
#include "../util/Base.h"

int main(int argc, char **argv) {
    array<int, 3> myints = {2,1,2};

    std::sort (myints.begin(),myints.begin()+3);

    std::cout << "The possible permutations with 3 elements:\n";
    do {
        std::cout << myints[0] << ' ' << myints[1] << ' ' << myints[2] << '\n';
    } while ( std::next_permutation(myints.begin(),myints.begin()+3) );

    std::cout << "After loop: " << myints[0] << ' ' << myints[1] << ' ' << myints[2] << '\n';


    // 1: A * B
    double A[] = {1, 0, 2,
                  -1, 3, 1};
    int A_row = 2;
    int A_col = 3;

    double B[] = {3, 1,
                  2, 1,
                  1, 0};
    int B_row = 3;
    int B_col = 2;

    double *result = new double[A_row*B_col];

    Calculator::matrix_product(A, B, result, A_row, A_col, B_row, B_col);

    // should be:
    // 5 1
    // 4 2
    for (int row = 0; row < A_row; row++) {
        for (int col = 0; col < B_col; col++) {
            std::cout << result[row * A_row + col] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----" << std::endl;

    delete[] result;

    double C[] = {1, 0, 2,
                  -1, 3, 1};
    int C_row = 2;
    int C_col = 3;

    double D[] = {3, 2, 1,
                  1, 1, 0};

    int D_row = 2;
    int D_col = 3;

    double *result2 = new double[C_row*D_row];

    Calculator::matrix_product_transpose(C, D, result2, C_row, C_col, D_row, D_col);

    // should be:
    // 5 1
    // 4 2
    for (int row = 0; row < C_row; row++) {
        for (int col = 0; col < D_row; col++) {
            std::cout << result2[row * C_row + col] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----" << std::endl;

    delete[] result2;

}