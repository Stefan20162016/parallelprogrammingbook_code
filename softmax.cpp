// Makefile: 3.2 sec per iteration
// g++-7 softmax.cpp -g -std=c++14 -O3 -fopenmp -o softmax -ftree-vectorize       2.5 sec
// g++-7 softmax.cpp -g -std=c++14 -O3 -fopenmp -o softmax -ftree-vectorize -ffast-math     0.93 sec
// g++-8 softmax.cpp -g -std=c++14 -O3 -fopenmp -o softmax.8 -ftree-vectorize -ffast-math   0.895 sec
// icc -xHost -qopenmp -g -o softmax.icc softmax.cpp -qopt-report=5 -qopt-report-phase=vec  0.51 sec
// icc -fast (== -ipo -O3 -no-prec-div -fp-model fast=2 -static)                            0.435 sec 7.4x speedup

#include "../include/binary_IO.hpp"    // load images
#include "../include/hpc_helpers.hpp"  // timers

#include <cmath>   // std::max
#include <limits>  // numerical limits of data types
#include <vector>  // std::vector

template <
    typename value_t,
    typename index_t>
void softmax_regression(
    value_t *input,
    value_t *output,
    value_t *weights,
    value_t *bias,
    index_t n_input,   // num_features
    index_t n_output){ // num_classes

//#pragma unroll_and_jam(32)
//#pragma vector always
//#pragma nounroll_and_jam
    for (index_t j = 0; j < n_output; j++) {                    // j: num_classes
        value_t accum = value_t(0);
        for (index_t k = 0; k < n_input; k++){                  // k: num_features
            accum   += weights[j * n_input + k] * input[k];
            //output[j] += weights[j*n_input+k]*input[k];         // unroll-and-jam?
        }
        output[j] = accum + bias[j];
        //output[j] += bias[j];
    }

    value_t norm = value_t(0);
    value_t mu = std::numeric_limits<value_t>::lowest();

    // compute mu = max(z_j)
    for (index_t index = 0; index < n_output; index++)
        mu = std::max(mu, output[index]);

    // compute y_j = exp(z_j-mu)
    for (index_t j = 0; j < n_output; j++)
        output[j] = expf(output[j] - mu);
        //output[j] = std::exp(output[j] - mu);

    // compute Z = sum_j z_j
    for (index_t j = 0; j < n_output; j++)
        norm += output[j];

    // compute z_j/Z
    const value_t inv_norm = 1/norm;
    for (index_t j = 0; j < n_output; j++)
        output[j] *= inv_norm;
        
}

template <
    typename value_t,
    typename index_t>
index_t argmax(
    value_t *neurons,
    index_t n_units) {
    index_t arg = 0;
    value_t max = std::numeric_limits<value_t>::lowest();

    for (index_t j = 0; j < n_units; j++) {
        const value_t val = neurons[j];
        if (val > max) {
            arg = j;
            max = val;
        }
    }

    return arg;
}

template <
    typename value_t,
    typename index_t>
value_t accuracy(
    value_t *input,
    value_t *label,
    value_t *weights,
    value_t *bias,
    index_t num_entries,
    index_t num_features,
    index_t num_classes) {
    index_t counter = index_t(0);

#pragma omp parallel for reduction(+: counter)
    for (index_t i = 0; i < num_entries; i++) {
        value_t output[num_classes];
        const uint64_t input_off = i * num_features;
        const uint64_t label_off = i * num_classes;

        softmax_regression(input + input_off, output, weights, bias, num_features, num_classes);

        counter += argmax(output, num_classes) == argmax(label + label_off, num_classes);
    }

    return value_t(counter) / value_t(num_entries);
}

template <
    typename value_t,
    typename index_t>
void train(
    value_t *input, value_t *label, value_t *weights, value_t *bias, index_t num_entries, 
    index_t num_features, index_t num_classes, index_t num_iters = 32,
    value_t epsilon = 0.1f) {
    value_t *grad_bias = new value_t[num_classes];
    value_t *grad_weights = new value_t[num_features * num_classes]; // 28*28 *10 = 7810 "stackable"

#pragma omp parallel
    for (uint64_t index = 0; index < num_iters; index++) {              // ITERATION LOOP
// zero the gradients
#pragma omp single
        for (index_t j = 0; j < num_classes; j++)
            grad_bias[j] = value_t(0);

#pragma omp for collapse(2)
        for (index_t j = 0; j < num_classes; j++)
            for (index_t k = 0; k < num_features; k++)
                grad_weights[j*num_features+k] = value_t(0);

// compute softmax contributions

#pragma omp for reduction(+: grad_bias [0:num_classes]) \
                reduction(+: grad_weights [0:num_classes*num_features])

        for (index_t i = 0; i < num_entries; i++){                     // i: num_entries=55'000
            //value_t *output = new value_t[num_classes];
            value_t output[10]; // stays valid in this scope
            softmax_regression(input+i*num_features, output, weights, bias, num_features, num_classes);

            for (index_t j = 0; j < num_classes; j++){                  // j: num_clases=10
                const value_t lbl_residual = output[j]-label[i*num_classes+j];
                grad_bias[j] += lbl_residual;
                //grad_bias[j] += output[j]-label[i*num_classes+j];

                for (index_t k = 0; k < num_features; k++){             // k: num_features=28*28=784
                    grad_weights[j*num_features+k] += lbl_residual*input[i*num_features+k];
                    //grad_weights[j*num_features+k] += ( output[j]-label[i*num_classes+j] )*input[i*num_features+k];

                }
            }
            //delete[] output;
        }

// adjust bias vector
value_t invnum_entries=1.0/num_entries; // /num_entries
#pragma omp single
        for (index_t j = 0; j < num_classes; j++)
            bias[j] -= epsilon*grad_bias[j] * invnum_entries; // / num_entries;

// adjust weight matrix
#pragma omp for collapse(2)
        for (index_t j = 0; j < num_classes; j++)           // j: classes
            for (index_t k = 0; k < num_features; k++)      // k: features
                weights[j*num_features+k] -= epsilon*grad_weights[j*num_features+k] * invnum_entries; // / num_entries;
    } // END ITERATION LOOP
    delete[] grad_bias;
    delete[] grad_weights;
}

int main() {
    const uint64_t num_features = 28 * 28;
    const uint64_t num_classes = 10;
    const uint64_t num_entries = 65000;

    std::vector<float> input(num_entries * num_features);
    std::vector<float> label(num_entries * num_classes);

    std::vector<float> weights(num_classes * num_features);
    std::vector<float> bias(num_classes);

    load_binary(input.data(), input.size(), "./data/X.bin");
    load_binary(label.data(), label.size(), "./data/Y.bin");
    //load_binary(weights.data(), weights.size(), "./data/A.bin");
    //load_binary(bias.data(), bias.size(), "./data/b.bin");

//    while (true) {
    for(int i=0; i< 40 ; ++i){
        std::cout << "Epoch: " << i+1 << std::endl;
        TIMERSTART(training)
        train(input.data(),
              label.data(),
              weights.data(),
              bias.data(),
              55000UL,
              num_features,
              num_classes);
        TIMERSTOP(training)
    //}
        const uint64_t off_inp = 55000 * num_features;
        const uint64_t off_lbl = 55000 * num_classes;
    
        TIMERSTART(accuracy)
        auto acc = accuracy(input.data() + off_inp,
                            label.data() + off_lbl,
                            weights.data(),
                            bias.data(),
                            10000UL,
                            num_features,
                            num_classes);
        TIMERSTOP(accuracy)

        std::cout << "accuracy_test: " << acc << std::endl;
    }
}
