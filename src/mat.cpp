// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <iostream>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;
typedef double value_type;
typedef double time_type;
typedef int spike_type;

py::array
predict_adaptation(const py::array_t<spike_type> spikes, value_type tau, time_type dt)
{
        auto S = spikes.unchecked<1>();
        const size_t N = spikes.size();
        const value_type A = exp(-dt / tau);

        value_type h = 0.0;
        py::array_t<value_type> Y(N);
        auto Yptr = Y.mutable_unchecked<1>();
        for (size_t i = 0; i < N; ++i) {
                // spikes need to be causal, so we only add the deltas after
                // storing the result of the filter
                h *= A;
                Yptr(i) = h;
                if (S[i]) {
                        h += 1;
                }
        }
        return Y;
}


// py::array
// spgconv(const py::array_t<value_type> spec, const py::array_t<value_type> rf, size_t upsample=1)
// {
//         auto S = spec.unchecked<2>();
//         auto K = rf.unchecked<2>();
//         const size_t nf = S.shape(0);
//         const size_t nt = S.shape(1);
//         const size_t ntau = rf.shape(1);




PYBIND11_MODULE(mat, m) {
        m.doc() = "multi-timescale adaptive threshold neuron model implementation";
        m.def("predict_adaptation", &predict_adaptation);

#ifdef VERSION_INFO
        m.attr("__version__") = py::str(VERSION_INFO);
#else
        m.attr("__version__") = py::str("dev");
#endif

}
