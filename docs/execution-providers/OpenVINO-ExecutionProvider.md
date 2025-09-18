---
title: Intel - OpenVINO™
description: Instructions to execute OpenVINO™ Execution Provider for ONNX Runtime.
parent: Execution Providers
nav_order: 3
redirect_from: /docs/reference/execution-providers/OpenVINO-ExecutionProvider
---

# OpenVINO™ Execution Provider
{: .no_toc }

Accelerate ONNX models on Intel CPUs, GPUs, NPU with Intel OpenVINO™ Execution Provider. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Pre-built packages are published for OpenVINO™ Execution Provider for ONNX Runtime by Intel for each release.
* OpenVINO™ Execution Provider for ONNX Runtime Release page: [Latest v5.8 Release](https://github.com/intel/onnxruntime/releases)
* Python wheels Ubuntu/Windows: [onnxruntime-openvino](https://pypi.org/project/onnxruntime-openvino/)

## Requirements


ONNX Runtime OpenVINO™ Execution Provider is compatible with three latest releases of OpenVINO™.

|ONNX Runtime|OpenVINO™|Notes|
|---|---|---| 
|1.23.0|2025.3|[Details - Placeholder]()|
|1.22.0|2025.1|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.7)|
|1.21.0|2025.0|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.6)|

## Build

For build instructions, please see the [BUILD page](../build/eps.md#openvino).

## Usage

**Python Package Installation**

For Python users, install the onnxruntime-openvino package:
```
pip install onnxruntime-openvino
```

**Set OpenVINO™ Environment Variables**

To use OpenVINO™ Execution Provider with any programming language (Python, C++, C#), you must set up the OpenVINO™ Environment Variables using the full installer package of OpenVINO™.

* **Windows**
```
C:\ <openvino_install_directory>\setupvars.bat
```
* **Linux**
```
$ source <openvino_install_directory>/setupvars.sh
```
**Note for Linux Python Users:** OpenVINO™ Execution Provider installed from PyPi.org comes with prebuilt OpenVINO™ libs and supports flag CXX11_ABI=0. So there is no need to install OpenVINO™ separately. However, if you need to enable CX11_ABI=1 flag, build ONNX Runtime python wheel packages from source. For build instructions, see the [BUILD page](../build/eps.md#openvino).


**Set OpenVINO™ Environment for  C#**

To use csharp api for openvino execution provider create a custom nuget package. Follow the instructions [here](../build/inferencing.md#build-nuget-packages) to install prerequisites for nuget creation. Once prerequisites are installed follow the instructions to [build openvino execution provider](../build/eps.md#openvino) and add an extra flag `--build_nuget` to create nuget packages. Two nuget packages will be created Microsoft.ML.OnnxRuntime.Managed and Intel.ML.OnnxRuntime.Openvino.

# OpenVINO Execution Provider Configuration

## Table of Contents
- [Configuration Options](#configuration-options)
- [Configuration Descriptions](#configuration-descriptions)
- [Examples](#examples)

## Configuration Options

Runtime parameters you set when initializing the OpenVINO Execution Provider to control how inference runs.

**Click on any configuration key below to jump to its detailed description.**

| **Key** | **Type** | **Allowable Values** | **Value Type** | **Description** |
|---------|----------|---------------------|----------------|-----------------|
| [**device_type**](#device_type) | string | CPU, NPU, GPU, GPU.0, GPU.1, HETERO, MULTI, AUTO | string | Choose which hardware device to use for inference |
| [**precision**](#precision) | string | FP32, FP16, ACCURACY | string | Set inference precision level |
| [**num_of_threads**](#num_of_threads--num_streams) | string | Any positive integer > 0 | size_t | Control number of inference threads |
| [**num_streams**](#num_of_threads--num_streams) | string | Any positive integer > 0 | size_t | Set parallel execution streams for throughput |
| [**cache_dir**](#cache_dir) | string | Valid filesystem path | string | Enable model caching by setting cache directory |
| [**load_config**](#load_config) | string | JSON file path | string | Load custom OpenVINO properties from JSON |
| [**enable_qdq_optimizer**](#enable_qdq_optimizer) | string | True/False | boolean | Enable QDQ optimization for NPU |
| [**disable_dynamic_shapes**](#disable_dynamic_shapes--reshape_input) | string | True/False | boolean | Convert dynamic models to static shapes |
| [**model_priority**](#model_priority) | string | LOW, MEDIUM, HIGH, DEFAULT | string | Configure model resource allocation priority |
| [**reshape_input**](#disable_dynamic_shapes--reshape_input) | string | input_name[shape_bounds] | string | Set dynamic shape bounds for NPU models |
| [**layout**](#layout) | string | input_name[layout_format] | string | Specify input/output tensor layout format |

Refer to [Examples](#examples) for usage.

---

## Configuration Descriptions

### device_type
Specifies the target hardware device for inference execution. Supports single devices (CPU, NPU, GPU, GPU.0, GPU.1) and multi-device configurations.

**Valid Device Combinations:**  
- `HETERO:<device1>,<device2>...` - Split execution across devices  
- `MULTI:<device1>,<device2>...` - Parallel execution on devices  
- `AUTO:<device1>,<device2>...` - Automatic device selection  

Minimum two devices required. Example: `HETERO:GPU,CPU`, `AUTO:GPU,NPU,CPU`, `MULTI:GPU,CPU`

**Note:** Deprecated options `CPU_FP32`, `GPU_FP32`, `GPU_FP16`, `NPU_FP16` are no longer supported. Use `device_type` and `precision` separately.

**Auto Device Selection:** Use `AUTO` to automatically select optimal device based on model characteristics. AUTO internally recognizes CPU, integrated GPU, discrete Intel GPUs, and NPU, then assigns inference requests to the best-suited device.

**Heterogeneous Execution:** Use `HETERO` to split network execution across multiple devices, utilizing accelerator power for heavy operations while falling back to CPU for unsupported layers.

**Multi-Device Execution:** Use `MULTI` to run the same model on multiple devices in parallel, improving throughput and performance consistency through load distribution.

### precision
Controls numerical precision during inference, balancing performance and accuracy.

**Device Support:**  
- CPU: FP32  
- GPU: FP32, FP16, ACCURACY  
- NPU: FP16  

**ACCURACY Mode:** Maintains original model precision without conversion, ensuring maximum accuracy. FP16 generally provides 2x better performance on GPU/NPU with minimal accuracy loss.

### num_of_threads & num_streams
**Multi-Threading:** Controls inference thread count for CPU execution (default: 8). OpenVINO EP provides thread-safe inference across all devices.

**Multi-Stream Execution:** Manages parallel inference streams for throughput optimization (default: 1 for latency). Multiple streams improve throughput for batch processing while single stream minimizes latency for real-time applications.

### cache_dir
Enables model caching to dramatically reduce subsequent load times. Supports CPU, NPU, GPU with kernel caching on iGPU/dGPU.

**Benefits:** Saves compiled models and cl_cache files for dynamic shapes, eliminating recompilation overhead. Especially beneficial for complex models 
and frequent application restarts.

### load_config
Loads custom OpenVINO properties from JSON configuration file during runtime.

**JSON Format:**
```json
{
    "DEVICE_KEY": {"PROPERTY": "PROPERTY_VALUE"}
}
```
Validation: Invalid property keys are ignored with warnings. Invalid values cause execution exceptions. Immutable properties are skipped.
Common Properties: PERFORMANCE_HINT, EXECUTION_MODE_HINT, LOG_LEVEL, CACHE_DIR, INFERENCE_PRECISION_HINT.  

### enable_qdq_optimizer
NPU-specific optimization for Quantize-Dequantize operations. Optimizes ORT quantized models by keeping QDQ operations only for supported ops, providing better performance and accuracy.

### disable_dynamic_shapes & reshape_input
**Dynamic Shape Management** : Handles models with variable input dimensions. Option to convert dynamic to static shapes when beneficial for performance.

**NPU Shape Bounds** : Use reshape_input to set dynamic shape bounds specifically for NPU devices (format: input_name[lower..upper] or input_name[fixed_shape]). 
Required for optimal NPU memory management.

### model_priority
Configures resource allocation priority for multi-model deployments:

**HIGH**: Maximum resource allocation

**MEDIUM**: Balanced resource sharing

**LOW**: Minimal allocation, yields to higher priority

**DEFAULT**: System-determined priority

### layout

***Tensor Layout Control:***: Provides explicit control over tensor memory layout for performance optimization. Helps OpenVINO optimize memory access patterns and tensor operations.

***Layout Characters:***: N (Batch), C (Channel), H (Height), W (Width), D (Depth), T (Time), ? (Unknown)

***Format:*** input_name[LAYOUT],output_name[LAYOUT]

## Examples

### [Example 1](#examples)

```python
import onnxruntime as ort

# Multi-device with caching and threading optimization
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{
        'device_type': 'AUTO:GPU,NPU,CPU',
        'precision': 'FP16',
        'num_of_threads': '8',
        'num_streams': '4',
        'cache_dir': './ov_cache'
    }]
)

# Command line equivalent
# onnxruntime_perf_test.exe -e openvino -i "device_type|AUTO:GPU,NPU,CPU precision|FP16 num_of_threads|8 num_streams|4 cache_dir|./ov_cache" model.onnx
```

### Example 2 
```python 
import onnxruntime as ort

# NPU-optimized with custom config and shape management
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{
        'device_type': 'HETERO:NPU,CPU',
        'load_config': 'custom_config.json',
        'enable_qdq_optimizer': 'True',
        'disable_dynamic_shapes': 'True',
        'model_priority': 'HIGH',
        'reshape_input': 'data[1,3,224,224..448]',
        'layout': 'data[NCHW],output[NC]'
    }]
)

# Example custom_config.json
{
    "NPU": {
        "LOG_LEVEL": "LOG_DEBUG",
        "PERFORMANCE_HINT": "THROUGHPUT"
    },
    "CPU": {
        "EXECUTION_MODE_HINT": "ACCURACY"
    }
}

# Command line equivalent
# onnxruntime_perf_test.exe -e openvino -i "device_type|HETERO:NPU,CPU load_config|custom_config.json enable_qdq_optimizer|True disable_dynamic_shapes|True model_priority|HIGH reshape_input|data[1,3,224,224..448] layout|data[NCHW],output[NC]" model.onnx

```

## Configuration Options

OpenVINO™ Execution Provider can be configured with certain options at runtime that control the behavior of the EP. These options can be set as key-value pairs as below:-

### Python API
Key-Value pairs for config options can be set using InferenceSession API as follow:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, providers=['OpenVINOExecutionProvider'], provider_options=[{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that the releases from (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution providers other than the default CPU provider (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.*

### C/C++ API 2.0 
The session configuration options are passed to SessionOptionsAppendExecutionProvider API as shown in an example below for GPU device type:

```
std::unordered_map<std::string, std::string> options;
options[device_type] = "GPU";
options[precision] = "FP32";
options[num_of_threads] = "8";
options[num_streams] = "8";
options[cache_dir] = "";
options[context] = "0x123456ff";
options[enable_qdq_optimizer] = "True";
options[load_config] = "config_path.json";
session_options.AppendExecutionProvider_OpenVINO_V2(options);
```

### C/C++ Legacy API 
Note: This API is no longer officially supported. Users are requested to move to V2 API. 

The session configuration options are passed to SessionOptionsAppendExecutionProvider_OpenVINO() API as shown in an example below for GPU device type:

```
OrtOpenVINOProviderOptions options;
options.device_type = "GPU_FP32";
options.num_of_threads = 8;
options.cache_dir = "";
options.context = 0x123456ff;
options.enable_opencl_throttling = false;
SessionOptions.AppendExecutionProvider_OpenVINO(session_options, &options);
```

### Onnxruntime Graph level Optimization
OpenVINO™ backend performs hardware, dependent as well as independent optimizations on the graph to infer it on the target hardware with best possible performance. In most cases it has been observed that passing the ONNX input graph as it is without explicit optimizations would lead to best possible optimizations at kernel level by OpenVINO™. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime for OpenVINO™ Execution Provider. This can be done using SessionOptions() as shown below:-

* #### Python API
   ```
   options = onnxruntime.SessionOptions()
   options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
   sess = onnxruntime.InferenceSession(<path_to_model_file>, options)
   ```

* #### C/C++ API
   ```
   SessionOptions::SetGraphOptimizationLevel(ORT_DISABLE_ALL);
   ```

## Support Coverage

**ONNX Layers supported using OpenVINO**

The table below shows the ONNX layers supported and validated using OpenVINO™ Execution Provider.The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. Intel Discrete Graphics. For NPU if an op is not supported we fallback to CPU. 

| **ONNX Layers** | **CPU** | **GPU** |
| --- | --- | --- |
| Abs | Yes | Yes |
| Acos | Yes | Yes |
| Acosh | Yes | Yes |
| Add | Yes | Yes |
| And | Yes | Yes |
| ArgMax | Yes | Yes |
| ArgMin | Yes | Yes |
| Asin | Yes | Yes |
| Asinh | Yes | Yes |
| Atan | Yes | Yes |
| Atanh | Yes | Yes |
| AveragePool | Yes | Yes |
| BatchNormalization | Yes | Yes |
| BitShift | Yes | No |
| Ceil | Yes | Yes |
| Celu | Yes | Yes |
| Cast | Yes | Yes |
| Clip | Yes | Yes |
| Concat | Yes | Yes |
| Constant | Yes | Yes |
| ConstantOfShape | Yes | Yes |
| Conv | Yes | Yes |
| ConvInteger | Yes | Yes |
| ConvTranspose | Yes | Yes |
| Cos | Yes | Yes |
| Cosh | Yes | Yes |
| CumSum | Yes | Yes |
| DepthToSpace | Yes | Yes |
| DequantizeLinear | Yes | Yes |
| Div | Yes | Yes |
| Dropout | Yes | Yes |
| Einsum | Yes | Yes |
| Elu | Yes | Yes |
| Equal | Yes | Yes |
| Erf | Yes | Yes |
| Exp | Yes | Yes |
| Expand | Yes | Yes |
| EyeLike | Yes | No |
| Flatten | Yes | Yes |
| Floor | Yes | Yes |
| Gather | Yes | Yes |
| GatherElements | No | No |
| GatherND | Yes | Yes |
| Gemm | Yes | Yes |
| GlobalAveragePool | Yes | Yes |
| GlobalLpPool | Yes | Yes |
| GlobalMaxPool | Yes | Yes |
| Greater | Yes | Yes |
| GreaterOrEqual | Yes | Yes |
| GridSample | Yes | No |
| HardMax | Yes | Yes |
| HardSigmoid | Yes | Yes |
| Identity | Yes | Yes |
| If | Yes | Yes |
| ImageScaler | Yes | Yes |
| InstanceNormalization | Yes | Yes |
| LeakyRelu | Yes | Yes |
| Less | Yes | Yes |
| LessOrEqual | Yes | Yes |
| Log | Yes | Yes |
| LogSoftMax | Yes | Yes |
| Loop | Yes | Yes |
| LRN | Yes | Yes |
| LSTM | Yes | Yes |
| MatMul | Yes | Yes |
| MatMulInteger | Yes | No |
| Max | Yes | Yes |
| MaxPool | Yes | Yes |
| Mean | Yes | Yes |
| MeanVarianceNormalization | Yes | Yes |
| Min | Yes | Yes |
| Mod | Yes | Yes |
| Mul | Yes | Yes |
| Neg | Yes | Yes |
| NonMaxSuppression | Yes | Yes |
| NonZero | Yes | No |
| Not | Yes | Yes |
| OneHot | Yes | Yes |
| Or | Yes | Yes |
| Pad | Yes | Yes |
| Pow | Yes | Yes |
| PRelu | Yes | Yes |
| QuantizeLinear | Yes | Yes |
| QLinearMatMul | Yes | No |
| Range | Yes | Yes |
| Reciprocal | Yes | Yes |
| ReduceL1 | Yes | Yes |
| ReduceL2 | Yes | Yes |
| ReduceLogSum | Yes | Yes |
| ReduceLogSumExp | Yes | Yes |
| ReduceMax | Yes | Yes |
| ReduceMean | Yes | Yes |
| ReduceMin | Yes | Yes |
| ReduceProd | Yes | Yes |
| ReduceSum | Yes | Yes |
| ReduceSumSquare | Yes | Yes |
| Relu | Yes | Yes |
| Reshape | Yes | Yes |
| Resize | Yes | Yes |
| ReverseSequence | Yes | Yes |
| RoiAlign | Yes | Yes |
| Round | Yes | Yes |
| Scatter | Yes | Yes |
| ScatterElements | Yes | Yes |
| ScatterND | Yes | Yes |
| Selu | Yes | Yes |
| Shape | Yes | Yes |
| Shrink | Yes | Yes |
| Sigmoid | Yes | Yes |
| Sign | Yes | Yes |
| Sin | Yes | Yes |
| Sinh | Yes | No |
| SinFloat | No | No |
| Size | Yes | Yes |
| Slice | Yes | Yes |
| Softmax | Yes | Yes |
| Softplus | Yes | Yes |
| Softsign | Yes | Yes |
| SpaceToDepth | Yes | Yes |
| Split | Yes | Yes |
| Sqrt | Yes | Yes |
| Squeeze | Yes | Yes |
| Sub | Yes | Yes |
| Sum | Yes | Yes |
| Softsign | Yes | No |
| Tan | Yes | Yes |
| Tanh | Yes | Yes |
| ThresholdedRelu | Yes | Yes |
| Tile | Yes | Yes |
| TopK | Yes | Yes |
| Transpose | Yes | Yes |
| Unsqueeze | Yes | Yes |
| Upsample | Yes | Yes |
| Where | Yes | Yes |
| Xor | Yes | Yes |


### Topology Support

Below topologies from ONNX open model zoo are fully supported on OpenVINO™ Execution Provider and many more are supported through sub-graph partitioning.
For NPU if model is not supported we fallback to CPU. 

### Image Classification Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| bvlc_alexnet | Yes | Yes |
| bvlc_googlenet | Yes | Yes |
| bvlc_reference_caffenet | Yes | Yes |
| bvlc_reference_rcnn_ilsvrc13 | Yes | Yes |
| emotion ferplus | Yes | Yes |
| densenet121 | Yes | Yes |
| inception_v1 | Yes | Yes |
| inception_v2 | Yes | Yes |
| mobilenetv2 | Yes | Yes |
| resnet18v2 | Yes | Yes |
| resnet34v2 | Yes | Yes |
| resnet101v2 | Yes | Yes |
| resnet152v2 | Yes | Yes |
| resnet50 | Yes | Yes |
| resnet50v2 | Yes | Yes |
| shufflenet | Yes | Yes |
| squeezenet1.1 | Yes | Yes |
| vgg19 | Yes | Yes |
| zfnet512 | Yes | Yes |
| mxnet_arcface | Yes | Yes |


### Image Recognition Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| mnist | Yes | Yes |

### Object Detection Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| tiny_yolov2 | Yes | Yes |
| yolov3 | Yes | Yes |
| tiny_yolov3 | Yes | Yes |
| mask_rcnn | Yes | No |
| faster_rcnn | Yes | No |
| yolov4 | Yes | Yes |
| yolov5 | Yes | Yes |
| yolov7 | Yes | Yes |
| tiny_yolov7 | Yes | Yes |

### Image Manipulation Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| mosaic | Yes | Yes |
| candy | Yes | Yes |
| cgan | Yes | Yes |
| rain_princess | Yes | Yes |
| pointilism | Yes | Yes |
| udnie | Yes | Yes |

### Natural Language Processing Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| bert-squad | Yes | Yes |
| bert-base-cased | Yes | Yes |
| bert-base-chinese | Yes | Yes |
| bert-base-japanese-char | Yes | Yes |
| bert-base-multilingual-cased | Yes | Yes |
| bert-base-uncased | Yes | Yes |
| distilbert-base-cased | Yes | Yes |
| distilbert-base-multilingual-cased | Yes | Yes |
| distilbert-base-uncased | Yes | Yes |
| distilbert-base-uncased-finetuned-sst-2-english | Yes | Yes |
| gpt2 | Yes | Yes |
| roberta-base | Yes | Yes |
| roberta-base-squad2 | Yes | Yes |
| t5-base | Yes | Yes |
| twitter-roberta-base-sentiment | Yes | Yes |
| xlm-roberta-base | Yes | Yes |

### Models Supported on NPU

| **MODEL NAME** | **NPU** |
| --- | --- |
| yolov3 | Yes |
| microsoft_resnet-50 | Yes |
| realesrgan-x4 | Yes |
| timm_inception_v4.tf_in1k | Yes |
| squeezenet1.0-qdq | Yes |
| vgg16 | Yes |
| caffenet-qdq | Yes |
| zfnet512 | Yes |
| shufflenet-v2 | Yes |
| zfnet512-qdq | Yes |
| googlenet | Yes |
| googlenet-qdq | Yes |
| caffenet | Yes |
| bvlcalexnet-qdq | Yes |
| vgg16-qdq | Yes |
| mnist | Yes |
| ResNet101-DUC | Yes |
| shufflenet-v2-qdq | Yes |
| bvlcalexnet | Yes |
| squeezenet1.0 | Yes |

**Note:** We have added support for INT8 models, quantized with Neural Network Compression Framework (NNCF). To know more about NNCF refer [here](https://github.com/openvinotoolkit/nncf).

## OpenVINO™ Execution Provider Samples Tutorials

In order to showcase what you can do with the OpenVINO™ Execution Provider for ONNX Runtime, we have created a few samples that shows how you can get that performance boost you’re looking for with just one additional line of code.

### Python API
[Object detection with tinyYOLOv2 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/tiny_yolo_v2_object_detection)

[Object detection with YOLOv4 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/yolov4_object_detection)

### C/C++ API
[Image classification with Squeezenet in CPP](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP)

### Csharp API
[Object detection with YOLOv3 in C#](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_sharp/OpenVINO_EP/yolov3_object_detection)

## Blogs/Tutorials

### Overview of OpenVINO Execution Provider for ONNX Runtime
[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

### Tutorial on how to use OpenVINO™ Execution Provider for ONNX Runtime Docker Containers
[Docker Containers](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-docker-container.html)

### Tutorial on how to use OpenVINO™ Execution Provider for ONNX Runtime python wheel packages
[Python Pip Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)