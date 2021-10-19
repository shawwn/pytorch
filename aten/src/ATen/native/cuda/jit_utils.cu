#include <ATen/cuda/detail/LazyNVRTC.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>
#include <torch/csrc/jit/frontend/code_template.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <ATen/native/cuda/jit_utils.h>

namespace at{

namespace cuda{

namespace jit {

template<typename inp_calc_t, typename out_calc_t>
std::string generate_code(int64_t N, inp_calc_t ic, out_calc_t oc){
    return "foo";
}

// std::string generate_code(
//     const TensorIterator& iter,
//     bool dynamic_casting) {
//   // Constructs kernel code
//   const int nInputs = iter.ninputs();
//   torch::jit::TemplateEnv env;
//   env.s("name", "FooFunctor");
//   env.s("functor", jittable_foo_functor);
//   env.s("index_type", "unsigned int");
//   env.s("nInputs", std::to_string(nInputs));
//   // Identifies scalar type
//   // TODO: there has to be an existing way of doing this (i.e. converting scalar type to string)
//   const auto& common_dtype = iter.common_dtype();
//   std::string common_dtype_string = string_repr(common_dtype);
//   // if (common_dtype == kFloat) {
//   //   common_dtype_string = "float";
//   // } else if (common_dtype == kDouble) {
//   //   common_dtype_string = "double";
//   // }
//   env.s("scalar_type", common_dtype_string);
//   std::stringstream declare_load_arrays;
//   for (int i=0; i < nInputs; i++){
// //TODO these arrays are potentially of the different types, use function traits to determine the types
//     declare_load_arrays << common_dtype_string << " arg" << std::to_string(i) << "[" << std::to_string(thread_work_size) << "];\n";
//   }
//   env.s("declare_load_arrays", declare_load_arrays.str());
//   std::stringstream declare_store_arrays;
//   declare_store_arrays << common_dtype_string << " out" << "[" << std::to_string(thread_work_size) << "];\n";
//   env.s("declare_store_arrays", declare_store_arrays.str());
//   if (!dynamic_casting) {
//     env.s("loader", "LoadWithoutCast");
//     env.s("storer", "StoreWithoutCast");
//   } else {
//     env.s("loader", std::string("LoadWithCast<"+std::to_string(nInputs) + ">"));
//     env.s("storer", "StoreWithCast");
//   }
//   std::stringstream load_inputs;
//   for (int i=0; i < nInputs; i++){
//     auto i_string = std::to_string(i);
//     load_inputs << "arg" << i_string << "[j] = l.load<"
//                 << common_dtype_string << ">(data["
//                 << std::to_string(i + iter.noutputs()) << "], input_offsets["
//                 << i_string << "], " << i_string << ");\n";
//   }
//   env.s("load_inputs", load_inputs.str());
//   std::stringstream store_outputs;
//   store_outputs << "s.store<" << common_dtype_string
//                 << ">(out[j], data[0], output_offsets[0]);\n";
//   env.s("store_outputs", store_outputs.str());
//   std::stringstream functor_args;
//   for (int i=0; i < nInputs - 1; i++){
//     functor_args << "arg" << std::to_string(i) << "[j], ";
//   }
//   functor_args << "arg" << std::to_string(nInputs-1) << "[j]";
//   env.s("args", functor_args.str());
//   static auto cuda_template = at::cuda::detail::load_code_template("/private/home/ngimel/pytorch/aten/src/ATen/native/cuda/code_template.cuh");
//   return  cuda_template.format(env);
// }
}
}
}


// CUfunction jit_pwise_function(
//     JiteratorCache& cache,
//     JiteratorKey key,
//     const std::string& code,
//     const std::string& kernel_name) {

//   // TODO: this lock is could be acquired around the cache updates
//   std::lock_guard<std::mutex> guard{jiterator_mutex};

//   // Compiles the kernel ---

//   // Acquires device and NVRTC properties (for compile arch and occupancy calculations)
//   cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
//   int major, minor;
//   torch::jit::fuser::cuda::getMajorMinor(prop, major, minor);

//   // Creates the NVRTC program
//   nvrtcProgram program;
//   const auto& nvrtc = at::globalContext().getNVRTC();
//   AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
//       &program, code.c_str(), nullptr, 0, nullptr, nullptr));
//   // constructs nvrtc build arguments
//   const std::string compute = "--gpu-architecture=compute_" +
//     std::to_string(major) + std::to_string(minor);
//   const std::vector<const char*> build_args = {
//     "--std=c++14", compute.c_str(), "-default-device"};

//   // compiles and validates result
//   const auto compilation_result =
//         nvrtc.nvrtcCompileProgram(program, build_args.size(), build_args.data());
//   if (compilation_result != NVRTC_SUCCESS) {
//     size_t logsize;
//     AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
//     std::vector<char> log(logsize);
//     AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, log.data()));
//     std::stringstream cu;
//     cu << log.data();
//     throw std::runtime_error(cu.str());
//   }

//   CUmodule module;
//   CUfunction function;
//   std::vector<char> ptx;
//   size_t ptx_size;
//   AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTXSize(program, &ptx_size));
//   ptx.resize(ptx_size);
//   AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTX(program, ptx.data()));
//   AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&module, ptx.data()));
//   AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleGetFunction(&function, module, kernel_name.c_str()));


//   // Updates (or not) the cache and returns the function ---
//   c10::optional<CUfunction> maybe_function = get_jitted_function(cache, key);
//   if (maybe_function) {
//     // Destroys the just compiled but unneccessary program
//     AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcDestroyProgram(&program));
//     return *maybe_function;
//   }

//   store_jitted_function(cache, key, function);
//   return function;
// }

// // TODO: may need/want to initialize CUDA context here (refactor into nvrtc call)
// void launch_jitted_pwise_function(
//     CUfunction function,
//     std::vector<void*>& args,
//     const int nBlocks,
//     const int kBlockSize) {

//   const auto& nvrtc = at::globalContext().getNVRTC();

//   // TODO: seems like this and block calculation should be cached per device
//   // Acquires device and NVRTC properties (for compile arch and occupancy calculations)
//   cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
//   int major, minor;


//   // Launches kernel on current stream
//   auto stream = at::cuda::getCurrentCUDAStream();
//   AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
//     function,
//     nBlocks,
//     1,
//     1,
//     kBlockSize,
//     1,
//     1,
//     0,
//     stream,
//     args.data(),
//     nullptr));
// }

// //launch has to happen in this function because of lifetime of
// //objects going into args vector
// template <typename func_t>
// void prepare_args_and_launch_impl(CUfunction function, TensorIterator iter, func_t f,
// bool needs_dynamic_cast){
//   constexpr int nargs = function_traits<func_t>::arity;
// // Constructs kernel args
//   std::vector<void*> args;
//   args.push_back((void*)&f);
//   // Adds numel arg
//   // NOTE: the intermediate capture is neccessary
//   const int64_t numel = iter.numel();
//   args.push_back((void*)&numel);

//   // Adds data ptrs
//   at::detail::Array<char*, nargs+1> data;
//   for (auto i = decltype(iter.ntensors()){0}; i < iter.ntensors(); i++) {
//     data[i] = (char*)iter.data_ptr(i);
//   }
//   args.push_back((void*)&data);

//   // Addds offset calculators
//   // TODO: maybe combine into one offset calculator?
//   auto input_offset_calculator = make_input_offset_calculator<nargs>(iter);
//   auto output_offset_calculator = make_output_offset_calculator(iter);
//   args.push_back((void*)&input_offset_calculator);
//   args.push_back((void*)&output_offset_calculator);

//   int64_t grid = (numel + block_work_size - 1) / block_work_size;
//   if (needs_dynamic_cast) {
//     at::detail::Array<ScalarType, nargs> dtypes;
//     for (int i = 0; i < iter.ninputs(); i++) {
//       dtypes[i] = iter.tensor(i + iter.noutputs()).scalar_type();
//     }
//     auto loader = memory::LoadWithCast<nargs>(dtypes);
//     auto storer = memory::StoreWithCast(iter.tensor(0).scalar_type());
//     args.push_back((void*)&loader);
//     args.push_back((void*)&storer);
//     launch_jitted_pwise_function(function, args, grid, num_threads);
//     // TORCH_CHECK(false, "dynamic cast not supported yet")
//   } else {
//     auto loader = memory::LoadWithoutCast();
//     auto storer = memory::StoreWithoutCast();
//     args.push_back((void*)&loader);
//     args.push_back((void*)&storer);
//     // need to launch inside the if block because of loader runtime
//     // alternative is to make this function templated on loader and storer types
//     launch_jitted_pwise_function(function, args, grid, num_threads);
//   }
// }
