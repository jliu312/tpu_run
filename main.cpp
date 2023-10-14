// /data/local/tmp

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_plugin.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/core/c/common.h"
#include <random>
#include <chrono>

using tflite::StatefulNnApiDelegate;

#define MODEL_PATH "mobilenet_v2_1.0_224_quant.tflite"
#define NUM_RUNS 110 //>= 20 please

double calculateMean(const size_t* data, size_t dataSize) {
    if (dataSize == 0) {
        return 0.0; // Handle this case as needed
    }
    // Convert size_t to double for arithmetic
    double sum = 0.0;
    for (size_t i = 0; i < dataSize; i++) {
        sum += static_cast<double>(data[i]);
    }
    return sum / dataSize;
}

double calculateStdDev(const size_t* data, size_t dataSize) {
    if (dataSize < 2) {
        return 0.0; // Handle this case as needed
    }
    double mean = calculateMean(data, dataSize);
    double sum = 0.0;
    for (size_t i = 0; i < dataSize; i++) {
        double diff = static_cast<double>(data[i]) - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / (dataSize - 1));
}

int main(void){
  StatefulNnApiDelegate::Options options;
  options.accelerator_name = "google-edgetpu";
  options.use_burst_computation = true;
  options.disallow_nnapi_cpu = true;
  options.execution_preference = StatefulNnApiDelegate::Options::kSustainedSpeed;
  options.execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
  auto delegate = new StatefulNnApiDelegate(options);

  auto model = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH);
  auto resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  tflite::InterpreterBuilder* builder = new tflite::InterpreterBuilder(*model.get(), *resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  (*builder)(&interpreter);
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
      std::cerr << "Failed to apply NNAPI delegate." << std::endl;
      return -1;
    };
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cerr << "Failed to allocate tensors." << std::endl;
      return -2;
    }

  size_t* times = new size_t[NUM_RUNS];

  for(int i = 0; i < NUM_RUNS; i++) {
    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
    uint8_t* randomData = (uint8_t*) malloc(224*224*3);
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0,255);
    for (unsigned int i = 0; i < 224*224*3; i++){
      randomData[i] = static_cast<uint8_t>(dist(rd) & 0xFF);
    }
    input = randomData;

    auto start_time = std::chrono::high_resolution_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Failed to invoke." << std::endl;
      return -3;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    times[i] = duration.count();
    std::cout << "Time in microseconds " << duration.count() << std::endl;
  }
// calculate mean/std w/o first 10 data points
double mean = calculateMean(times + 10, NUM_RUNS-10);
double stdDev = calculateStdDev(times + 10, NUM_RUNS-10);

std::cout << "Mean: " << mean << " Standard Deviation: " << stdDev << " (microseconds)" << std::endl;
  

  delete[] times;
  // uint8_t* outputs = interpreter->typed_output_tensor<uint8_t>(0);
  // size_t outputSize = interpreter->output_tensor(0)->dims->data[1];
  // for(int i = 0; i < outputSize; i++) {
  //   std::cout << unsigned(outputs[i]) << std::endl;
  // }

  return 0;
}