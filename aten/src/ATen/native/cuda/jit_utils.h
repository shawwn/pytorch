#pragma once
#include <string>
#include <ATen/TensorIterator.h>

namespace at{

namespace cuda{

namespace jit {
template<typename inp_calc_t, typename out_calc_t>
TORCH_API std::string generate_code(int64_t N, inp_calc_t ic, out_calc_t oc);



}
}
}
