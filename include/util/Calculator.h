#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "util/Base.h"
#include "util/Parameter.h"

namespace Calculator {

    /**
     * Compute the inner product of two vectors
     * @param A
     * @param B
     * @param size
     * @return
     */
    inline value_type inner_product(value_type *A, value_type *B, const int size) {
        value_type result = 0;
        for (int i = 0; i < size; i++) {
            result += A[i] * B[i];
        }
        return result;
    }

    /**
     * Compute the product of two matrices AB
     * @param A
     * @param B
     * @param result: the space of result matrix should be allocated in advance
     * @param A_row_num
     * @param A_col_num
     * @param B_row_num
     * @param B_col_num
     * @return
     */
    inline void matrix_product(value_type *A, value_type *B, value_type *result, const int A_row_num, const int A_col_num,
                   const int B_row_num, const int B_col_num) {

        // result = A * B, size: A_row * B_col
        for (int i = 0; i < A_row_num; i++) {

            value_type *A_row = A + i * A_col_num;
            value_type *result_row = result + i * B_col_num;

            for (int j = 0; j < B_col_num; j++) {

                result_row[j] = 0;

                for (int k = 0; k < B_row_num; k++) {
                    result_row[j] += A_row[k] * B[k * B_col_num + j];
                }
            }
        }
    }

    /**
     * Compute the product of two matrices AB_T
     * @param A
     * @param B
     * @param result: the space of result matrix should be allocated in advance
     * @param A_row_num
     * @param A_col_num
     * @param B_row_num
     * @param B_col_num
     * @return
     */
    inline void matrix_product_transpose(value_type *A, value_type *B, value_type *result, const int A_row_num, const int A_col_num,
                                     const int B_row_num, const int B_col_num) {

        for (int i = 0; i < A_row_num; i++) {

            value_type *A_row = A + i * A_col_num;
            value_type *result_row = result + i * B_row_num;

            for (int j = 0; j < B_row_num; j++) {

                value_type *B_row = B + j * B_col_num;

                result_row[j] = inner_product(A_row, B_row, A_col_num);
            }
        }
    }

    inline value_type cal_rescal_score(const int rel_id, const int sub_id, const int obj_id, value_type *A,
                     value_type *R, const Parameter *parameter) {

        value_type *R_k = R + rel_id * parameter->dimension * parameter->dimension;

        //Vec p_tmp = prod(row(A, sub_id), R_k);
        value_type *p_tmp = new value_type[parameter->dimension];//TODO: Double check.

        for (int i = 0; i < parameter->dimension; i++) {
            p_tmp[i] = 0;
            for (int j = 0; j < parameter->dimension; ++j) {
                p_tmp[i] += A[sub_id * parameter->dimension + j] * R_k[j * parameter->dimension + i];
            }
        }

        value_type result = inner_product(p_tmp, A + parameter->dimension * obj_id, parameter->dimension);;
        delete[] p_tmp;

        return result;
//        value_type result=0;
//        for(int i=0;i<D;++i){
//            result+=p_tmp[i]*A[D*obj_id+i];
//        }
//        return result;
    }

};
#endif //CALCULATOR_H
