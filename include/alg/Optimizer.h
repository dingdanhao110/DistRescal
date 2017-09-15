//
// Created by dhding on 9/15/17.
//

#ifndef DISTRESCAL_OPTIMIZER_H
#define DISTRESCAL_OPTIMIZER_H

#include "util/Base.h"
#include "util/Parameter.h"
/**
 * Stochastic Gradient Descent
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient (not used)
 * @param d: length of factors/gradient/pre_gradient_square
 */
class sgd {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square, const int d,
                    Parameter *parameter) {

        value_type scale = parameter->step_size;

        for (int i = 0; i < d; i++) {
            factors[i] += scale * gradient[i];
        }

    }
};

/**
 * AdaGrad: Adaptive Gradient Algorithm
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient
 * @param d: length of factors/gradient/pre_gradient_square
 */
class adagrad {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                    const int d, Parameter *parameter) {

        value_type scale = parameter->step_size;

        for (int i = 0; i < d; i++) {
            pre_gradient_square[i] += gradient[i] * gradient[i];
            factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
        }
    }
};

/**
 * AdaDelta: An Adaptive Learning Rate Method
 * @param factors
 * @param gradient
 * @param pre_gradient_square: Square of previous gradient
 * @param d: length of factors/gradient/pre_gradient_square
 * @param weight: scaling value
 */
class adadelta {
public:
    void operator()(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                    const int d, Parameter *parameter) {

        value_type scale = parameter->step_size;

        for (int i = 0; i < d; i++) {
            pre_gradient_square[i] =
                    parameter->Rho * pre_gradient_square[i] + (1 - parameter->Rho) * gradient[i] * gradient[i];
            factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
        }
    }
};
#endif //DISTRESCAL_OPTIMIZER_H
