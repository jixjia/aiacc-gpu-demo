#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "assert.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/providers/providers.h"
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

void* Init_PIX_Env();

struct AiAccModel {
    AiAccModel(void* env, std::string model_path, const int batchsize);
    std::vector<int64_t> get_input_shapes(int index);
    std::vector<int64_t> get_output_shapes(int index);
    void Run();
    void ReloadData(float** input_data, int input_length,
                                float** output_data, int output_length);
    void ReloadData(float** input_data, std::vector<std::vector<int64_t>> input_shapes,
                            float** output_data, std::vector<std::vector<int64_t>> output_shapes);
    ~AiAccModel();

   private:
    int batchsize;
    int input_count;
    int output_count;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    void* input_tensor_;
    void* output_tensor_;
    void* session_;
    void* env;
    void* allocator;
    void* memory_info;
};


void* Init_PIX_Env() { return new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "test"}; }

AiAccModel::AiAccModel(void* env, std::string model_path, const int batchsize)
    : batchsize(batchsize) {
    //just for debug
    if (env == nullptr){
        env = Init_PIX_Env();
    }

    allocator = new Ort::AllocatorWithDefaultOptions();
    memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::SessionOptions session_options;
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options,
        0));
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    session_ = new Ort::Session{*((Ort::Env*)env), model_path.c_str(),
                                session_options};

    Ort::Session* session_st = (Ort::Session*)session_;
    input_count = session_st->GetInputCount();
    output_count = session_st->GetOutputCount();
    input_tensor_ = new std::vector<Ort::Value>;
    // init input tensor
    for (int i = 0; i < input_count; i++) {
        ((std::vector<Ort::Value>*)input_tensor_)
            ->push_back(Ort::Value(nullptr));
    }
    // init output tensor
    output_tensor_ = new std::vector<Ort::Value>;
    for (int i = 0; i < output_count; i++) {
        ((std::vector<Ort::Value>*)output_tensor_)
            ->push_back(Ort::Value(nullptr));
    }
    // get input shape and names
    for (int i_id = 0; i_id < input_count; i_id++) {
        std::string input_name = session_st->GetInputName(i_id,*(Ort::AllocatorWithDefaultOptions*)allocator);
        input_names.push_back(input_name);
        auto in_type_info = session_st->GetInputTypeInfo(i_id);
        auto type_info =
            in_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> i_shape(type_info.GetDimensionsCount());
        type_info.GetDimensions(i_shape.data(), type_info.GetDimensionsCount());
        i_shape[0] = batchsize;
        input_shapes.push_back(i_shape);
        //test input
        std::cout << "input "<< i_id <<" is "<<input_names[i_id]<< std::endl;
        std::cout<< "shape "<<type_info.GetDimensionsCount()<<std::endl;
        for (int dim = 0; dim < type_info.GetDimensionsCount(); dim++) {
            std::cout << "dim[" << dim << "] : " << i_shape[dim] << std::endl;
        }
    }
    // get output shape and names
    for (int o_id = 0; o_id < output_count; o_id++) {
        std::string output_name = session_st->GetOutputName(o_id,*(Ort::AllocatorWithDefaultOptions*)allocator);
        output_names.push_back(output_name);
        auto out_type_info = session_st->GetOutputTypeInfo(o_id);
        auto type_info =
            out_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> o_shape(type_info.GetDimensionsCount());
        type_info.GetDimensions(o_shape.data(), type_info.GetDimensionsCount());
        o_shape[0] = batchsize;
        output_shapes.push_back(o_shape);
        //test output
        std::cout << "output "<< o_id <<" is "<<output_names[o_id]<< std::endl;
        for (int dim = 0; dim < type_info.GetDimensionsCount(); dim++) {
            std::cout << "dim[" << dim << "] : " << o_shape[dim] << std::endl;
        }
    }
};

std::vector<int64_t> AiAccModel::get_input_shapes(int index) {
    return input_shapes[index];
}

std::vector<int64_t> AiAccModel::get_output_shapes(int index) {
    return output_shapes[index];
}

AiAccModel::~AiAccModel() {
    delete (Ort::Session*)session_;
    delete (Ort::AllocatorWithDefaultOptions*)allocator;
    delete (std::vector<Ort::Value>*)input_tensor_;
    delete (std::vector<Ort::Value>*)output_tensor_;
}

void AiAccModel::ReloadData(float** input_data, int input_length,
                            float** output_data, int output_length) {
    assert(input_length == input_count);
    assert(output_length == output_count);
    ReloadData(input_data,input_shapes,output_data,output_shapes);
};

void AiAccModel::ReloadData(float** input_data, std::vector<std::vector<int64_t>> input_shapes,
                            float** output_data, std::vector<std::vector<int64_t>> output_shapes) {
    // Value CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t
    // p_data_element_count, const int64_t* shape, size_t shape_len);
    int input_length = input_shapes.size();
    int output_length = output_shapes.size();
    assert(input_length == input_count);
    assert(output_length == output_count);
    for (int i = 0; i < input_length; i++) {
        int input_size = 1;
        std::vector<int64_t> input_shape_ = input_shapes[i];
        for (auto i_dim : input_shape_) {
            input_size *= i_dim;
        }
        (*(std::vector<Ort::Value>*)input_tensor_)[i] =
            Ort::Value::CreateTensor<float>(
                (OrtMemoryInfo*)memory_info, input_data[i], input_size,
                input_shape_.data(), input_shape_.size());
    }
    for (int i = 0; i < output_length; i++) {
        int output_size = 1;
        std::vector<int64_t> output_shape_ = output_shapes[i];
        for (auto i_dim : output_shape_) {
            output_size *= i_dim;
        }
        (*(std::vector<Ort::Value>*)output_tensor_)[i] =
            Ort::Value::CreateTensor<float>(
                (OrtMemoryInfo*)memory_info, output_data[i], output_size,
                output_shape_.data(), output_shape_.size());
    }
};

void AiAccModel::Run() {
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    for(auto &in_name:input_names){
        input_names_cstr.push_back(in_name.c_str());
    }
    for(auto &output_name:output_names){
        output_names_cstr.push_back(output_name.c_str());
    }

    ((Ort::Session*)session_)
        ->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(),
              ((std::vector<Ort::Value>*)input_tensor_)->data(), input_count,
              output_names_cstr.data(), ((std::vector<Ort::Value>*)output_tensor_)->data(),
              output_count);
};


int main(void){
    float *input_data = new float[1*224*224*3];
    float *output_data = new float[1*1000];
    memset(input_data,0,1*224*224*3*sizeof(float));
    memset(output_data,0,1*1000*sizeof(float));
    AiAccModel resnet(NULL,"./resnet50-v1-7.onnx",1);
    //warm up
    for(int i=0;i<10;i++){
        std::vector<int64_t> input_shape{1,3,224,224};
        std::vector<int64_t> output_shape{1,1000};

        std::vector<std::vector<int64_t>> input_shapes{input_shape};
        std::vector<std::vector<int64_t>> output_shapes{output_shape};
        float *input[1] = {input_data};
        float *output[1] = {output_data};

        resnet.ReloadData(input,input_shapes,output,output_shapes);
        resnet.Run();
    }
    struct  timeval  start, end;
    gettimeofday(&start,NULL);
    for(int i=0;i<1000;i++){
        std::vector<int64_t> input_shape{1,3,224,224};
        std::vector<int64_t> output_shape{1,1000};

        std::vector<std::vector<int64_t>> input_shapes{input_shape};
        std::vector<std::vector<int64_t>> output_shapes{output_shape};
        float *input[1] = {input_data};
        float *output[1] = {output_data};

        resnet.ReloadData(input,input_shapes,output,output_shapes);
        resnet.Run();
    }
    gettimeofday(&end,NULL);
    long timer = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    std::cout<<((double)timer)/1000/1000<< " ms"<<std::endl;


    std::cout<<std::endl;
    for(int i=0;i<100;i++){
        std::cout<<output_data[i]<< " ";
    }
    std::cout<<std::endl;
    return 0;
}
