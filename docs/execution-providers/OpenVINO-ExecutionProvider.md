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

## Table of Contents
- [Configuration Options](#configuration-options) 
- [Features ](#features ) 
- [Examples](#examples) 
- [Detailed Descriptions](#detailed-descriptions) 


## Configuration Options


Runtime parameters you set when initializing the OpenVINO Execution Provider to control how inference runs.

### Configuration Table

| **Key** | **Type** | **Allowable Values** | **Value Type** | **Description** | **Example** |
|---------|----------|---------------------|----------------|-----------------|-------------|
| **device_type** | string | CPU, NPU, GPU, GPU.0, GPU.1, HETERO, MULTI, AUTO | string | [Choose which hardware device to use](#device_type-config-description) | [Examples](#device_type-config-examples) |
| **precision** | string | FP32, FP16, ACCURACY | string | [Set inference precision level](#precision-config-description) | [Examples](#precision-config-examples) |
| **num_of_threads** | string | Any positive integer > 0 | size_t | [Control number of inference threads](#num_of_threads-config-description) | [Examples](#num_of_threads-config-examples) |
| **num_streams** | string | Any positive integer > 0 | size_t | [Set parallel execution streams](#num_streams-config-description) | [Examples](#num_streams-config-examples) |
| **cache_dir** | string | Valid filesystem path | string | [Enable model caching by setting cache directory](#cache_dir-config-description) | [Examples](#cache_dir-config-examples) |
| **load_config** | string | JSON file path | string | [Load custom OpenVINO properties from JSON](#load_config-config-description) | [Examples](#load_config-config-examples) |
| **enable_qdq_optimizer** | string | True/False | boolean | [Enable QDQ optimization for NPU](#enable_qdq_optimizer-config-description) | [Examples](#enable_qdq_optimizer-config-examples) |
| **disable_dynamic_shapes** | string | True/False | boolean | [Convert dynamic models to static shapes](#disable_dynamic_shapes-config-description) | [Examples](#disable_dynamic_shapes-config-examples) |
| **model_priority** | string | LOW, MEDIUM, HIGH, DEFAULT | string | [Configure model resource allocation priority](#model_priority-config-description) | [Examples](#model_priority-config-examples) |
| **reshape_input** | string | input_name[shape_bounds] | string | [Set dynamic shape bounds for NPU models](#reshape_input-config-description) | [Examples](#reshape_input-config-examples) |
| **layout** | string | input_name[layout_format] | string | [Specify input/output tensor layout format](#layout-config-description) | [Examples](#layout-config-examples) |

Valid Hetero or Multi or Auto Device combinations: `HETERO:<device 1>,<device 2>...`
The `device` can be any of these devices from this list ['CPU','GPU', 'NPU']

A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO, MULTI, or AUTO Device Build.

Example: HETERO:GPU,CPU  AUTO:GPU,CPU  MULTI:GPU,CPU

Deprecated device_type option : CPU_FP32, GPU_FP32, GPU_FP16, NPU_FP16 are no more supported. They will be deprecated in the future release. Kindly upgrade to latest device_type and precision option.

---

## Features 


Built-in capabilities that the OpenVINO EP provides automatically or can be enabled through configuration.


### Features Table

| **Feature** | **Supported Devices** | **Description** | **How to Enable** | **Example** |
|-------------|----------------------|-----------------|-------------------|-------------|
| **Auto Device Selection** | CPU, GPU, NPU | [Automatically selects optimal device for your model](#auto-device-execution-for-openvino-execution-provider) | Set device_type to AUTO | Examples |
| **Model Caching** | CPU, GPU, NPU | [Saves compiled models for faster subsequent loading](#model-caching) | Set cache_dir option | Examples |
| **Multi-Threading** | All devices | [Thread-safe inference with configurable thread count](#multi-threading-for-openvino-execution-provider) | Automatic/configure with num_of_threads | Examples |
| **Multi-Stream Execution** | All devices | [Parallel inference streams for higher throughput](#multi-streams-for-openvino-execution-provider) | Configure with num_streams | Examples |
| **Heterogeneous Execution** | CPU + GPU/NPU | [Split model execution across multiple devices](#heterogeneous-execution-for-openvino-execution-provider) | Set device_type to HETERO | Examples |
| **Multi-Device Execution** | CPU, GPU, NPU | [Run same model on multiple devices in parallel](#multi-device-execution-for-openvino-execution-provider) | Set device_type to MULTI | Examples |
| **INT8 Quantized Models** | CPU, GPU, NPU | [Support for quantized models with better performance](#support-for-int8-quantized-models) | Automatic for quantized models | Examples |
| **External Weights Support** | All devices | [Load models with weights stored in external files](#support-for-weights-saved-in-external-files) | Automatic detection | [Example](#support-for-weights-saved-in-external-files) |
| **Dynamic Shape Management** | All devices | [Handle models with variable input dimensions](#dynamic-shape-management) | Automatic/use reshape_input for NPU | Examples |
| **Tensor Layout Control** | All devices | [Explicit control over tensor memory layout](#tensor-layout-control) | Set layout option | Examples |
| **QDQ Optimization** | NPU | [Optimize quantized models for NPU performance](#enable-qdq-optimizations-passes) | Set enable_qdq_optimizer | Examples |
| **EP-Weight Sharing** | All devices | [Share weights across multiple inference sessions](#openvino-execution-provider-supports-ep-weight-sharing-across-sessions) | Session configuration | Examples |


## Examples

### Configuration Examples


#### device_type Config Examples
```python
Single device
"device_type": "GPU"
"device_type": "NPU"
"device_type": "CPU"
Specific GPU
"device_type": "GPU.1"

Multi-device configurations
"device_type": "HETERO:GPU,CPU"
"device_type": "MULTI:GPU,CPU"
"device_type": "AUTO:GPU,NPU,CPU"

Python API
import onnxruntime as ort
session = ort.InferenceSession(
            "model.onnx",
            providers=['OpenVINOExecutionProvider'],
            provider_options=[{'device_type': 'AUTO:GPU,NPU,CPU'}]
            )

Command line
onnxruntime_perf_test.exe -e openvino -i "device_type|GPU" model.onnx
```

#### precision Config Examples
```python
"precision": "FP32"  
"precision": "FP16"  
"precision": "ACCURACY"  


# Python API

import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'precision': 'FP16'}]
)


```
#### num_of_threads Config Examples
```python
"num_of_threads": "4"   
"num_of_threads": "8"   
"num_of_threads": "16"

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'CPU', 'num_of_threads': '8'}]
)

```

#### num_streams Config Examples
```python
"num_streams": "1"  
"num_streams": "4"  
"num_streams": "8

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'num_streams': '4'}]
)

```

#### cache_dir Config Examples
```python
# Windows
"cache_dir": "C:\\intel\\openvino_cache"

# Linux
"cache_dir": "/tmp/ov_cache" 

# Relative path
"cache_dir": "./model_cache"

import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'cache_dir': './ov_cache'}]
)

```

#### load_config Config Examples
```python
# JSON file path
"load_config": "config.json"
"load_config": "/path/to/openvino_config.json"
"load_config": "C:\\configs\\gpu_config.json"

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'load_config': 'custom_config.json'}]
)

# Example JSON content:
{
    "GPU": {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "EXECUTION_MODE_HINT": "ACCURACY", 
        "CACHE_DIR": "C:\\gpu_cache"
    },
    "NPU": {
        "LOG_LEVEL": "LOG_DEBUG"
    }
}
# Command line usage
onnxruntime_perf_test.exe -e openvino -i "device_type|NPU load_config|config.json" model.onnx

```

#### enable_qdq_optimizer Config Examples
 
```python
"enable_qdq_optimizer": "True"   # Enable QDQ optimization for NPU
"enable_qdq_optimizer": "False"  # Disable QDQ optimization

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'NPU', 'enable_qdq_optimizer': 'True'}]
)
```
#### disable_dynamic_shapes Config Examples
```python
"disable_dynamic_shapes": "True"   # Convert dynamic to static shapes
"disable_dynamic_shapes": "False"  # Keep original dynamic shapes

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'disable_dynamic_shapes': 'True'}]
)

```

#### model_priority Config Examples
```python
"model_priority": "HIGH"     # Highest resource priority
"model_priority": "MEDIUM"   # Medium resource priority  
"model_priority": "LOW"      # Lowest resource priority
"model_priority": "DEFAULT"  # System default priority

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'model_priority': 'HIGH'}]
)
```

#### reshape_input Config Examples
```python
# Command line usage (NPU only)
"reshape_input": "data[1,3,60,80..120]"  # Dynamic height: 80-120
"reshape_input": "input[1,3,224,224]"    # Fixed shape
"reshape_input": "seq[1,10..50,768]"     # Dynamic sequence: 10-50

# Python API
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'NPU', 'reshape_input': 'data[1,3,60,80..120]'}]
)
# Command line
onnxruntime_perf_test.exe -e openvino -i "device_type|NPU reshape_input|data[1,3,60,80..120]" model.onnx

```

#### layout Config Examples
```python 
# Command line usage
"layout": "data_0[NCHW],prob_1[NC]"     # Multiple inputs/outputs
"layout": "input[NHWC]"                 # Single input
"layout": "data[N?HW]"                  # Unknown channel dimension

# Python API  
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'NPU', 'layout': 'data_0[NCHW],output[NC]'}]
)

# Command line
onnxruntime_perf_test.exe -e openvino -i "device_type|NPU layout|data_0[NCHW],prob_1[NC]" model.onnx
```

### Feature Examples

#### auto-device Feature Examples

```python
# Basic AUTO usage
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'AUTO'}]
)

# AUTO with device priority
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'AUTO:GPU,NPU,CPU'}]
)

# Command line
onnxruntime_perf_test.exe -e openvino -i "device_type|AUTO:GPU,CPU" model.onnx

```

#### model-caching Feature Examples
```python
# Enable caching
import onnxruntime as ort
session = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'cache_dir': './ov_cache'}]
)

# First run: compiles and caches model
# Subsequent runs: loads from cache (much faster)
session1 = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'cache_dir': './ov_cache'}]
)  # Slow first time
session2 = ort.InferenceSession(
    "model.onnx",
    providers=['OpenVINOExecutionProvider'],
    provider_options=[{'device_type': 'GPU', 'cache_dir': './ov_cache'}]
)  # Fast second time
```

#### multi-threading Feature Examples
```python

```

#### multi-stream-feature Examples
```python

```
### Detailed Descriptions
#### Configuration Descriptions
#### device_type Config Description
Specifies which hardware device to run inference on. This is the primary configuration that determines execution target.
Available Options:

CPU: Intel CPU execution using OpenVINO CPU plugin
NPU: Neural Processing Unit for AI-optimized inference
GPU: Intel GPU acceleration (integrated or discrete)
GPU.0, GPU.1: Specific GPU device selection in multi-GPU systems
AUTO: Automatic device selection based on model characteristics
HETERO: Heterogeneous execution across multiple devices
MULTI: Multi-device parallel execution

Default Behavior: If not specified, uses the default hardware specified during build time.
#### precision Config Description
Controls the numerical precision used during inference, affecting both performance and accuracy.
Device Support:

CPU: FP32 
GPU: FP32, FP16, ACCURACY
NPU: FP16

ACCURACY Mode: Maintains original model precision without any conversion, ensuring maximum accuracy at potential performance cost.
Performance Considerations: FP16 generally provides 2x better performance on GPU/NPU with minimal accuracy loss.

#### num_of_threads Config Description
Override the default number of inference threads for CPU-based execution.
Default: 8 threads if not specified

#### num_streams Config Description
Controls the number of parallel inference streams for throughput optimization.
Default: 1 stream (latency-optimized)
Use Cases:

Single stream (1): Minimize latency for real-time applications
Multiple streams (2-8): Maximize throughput for batch processing
Optimal count: Usually matches number of CPU cores or GPU execution units

Performance Impact: More streams can improve throughput but may increase memory usage.
#### cache_dir Config Description
Specifies directory path for caching compiled models to improve subsequent load times.
Benefits: Dramatically faster model loading after first compilation
 Reduces initialization overhead significantly E
 specially beneficial for complex models and frequent restarts
Requirements: Directory must be writable by the application
Sufficient disk space for cached models (can be substantial)
Path must be accessible at runtime

Supported Devices: CPU, NPU, GPU 
#### context Config Description

Provides OpenCL context for GPU acceleration when OpenVINO EP is built with OpenCL support.
Usage: Pass cl_context address as void pointer converted to string
Availability: Only when compiled with OpenCL flags enabled
Purpose: Integration with existing OpenCL workflows and shared memory management

#### load_config Config Description
Enables loading custom OpenVINO properties from JSON configuration file during runtime.
JSON Format:

```python
{
    "DEVICE_KEY": {"PROPERTY": "PROPERTY_VALUE"}
}
```
Validation Rules:

Invalid property keys: Ignored with warning logged
Invalid property values: Causes exception during execution
Immutable properties: Skipped with warning logged

Common Properties:

PERFORMANCE_HINT: "THROUGHPUT", "LATENCY"
EXECUTION_MODE_HINT: "ACCURACY", "PERFORMANCE"
LOG_LEVEL: "LOG_DEBUG", "LOG_INFO", "LOG_WARNING"
CACHE_DIR: Custom cache directory path
INFERENCE_PRECISION_HINT: "f32", "f16"

Device-Specific Properties: For setting appropriate `"PROPERTY"`, refer to OpenVINO config options for [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#supported-properties), [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#supported-properties), [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#supported-features-and-properties) and [AUTO](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html#using-auto). 

#### enable_qdq_optimizer Config Description
Enables Quantize-Dequantize (QDQ) optimization specifically for NPU devices.
Target: NPU devices only
Purpose: Optimizes ORT quantized models by keeping QDQ operations only for supported ops
Benefits:

Better performance/accuracy with ORT optimizations disabled
NPU-specific quantization optimizations
Reduced computational overhead for quantized models

#### disable_dynamic_shapes Config Description
Controls whether dynamic input shapes are converted to static shapes at runtime.
Options:

True: Convert dynamic models to static shapes before execution
False: Maintain dynamic shape capabilities (default for most devices)

Use Cases:
\
Models with dynamic batch size or sequence length
Devices that perform significantly better with static shapes
Memory optimization scenarios

#### model_priority Config Description
Configures resource allocation priority when multiple models run simultaneously.
Priority Levels:

HIGH: Maximum resource allocation and priority
MEDIUM: Balanced resource sharing with other models
LOW: Minimal resource allocation, yields to higher priority models
DEFAULT: System-determined priority based on device capabilities

Use Cases: Multi-model deployments, resource-constrained environments.
#### reshape_input Config Description
Allows setting dynamic shape bounds specifically for NPU devices to optimize memory allocation and performance.
Format: input_name[lower_bound..upper_bound] or input_name[fixed_shape]
Device Support: NPU only (other devices handle dynamic shapes automatically)
Purpose: NPU requires shape bounds for optimal memory management with dynamic models
Examples:

data[1,3,224,224..448]: Height can vary from 224 to 448
sequence[1,10..100,768]: Sequence length from 10 to 100
batch[1..8,3,224,224]: Batch size from 1 to 8


#### layout Config Description
Provides explicit control over tensor layout format for inputs and outputs, enabling performance optimizations.
Standard Layout Characters:

N: Batch dimension
C: Channel dimension
H: Height dimension
W: Width dimension
D: Depth dimension
T: Time/sequence dimension
?: Unknown/placeholder dimension

Format: input_name[LAYOUT],output_name[LAYOUT]


### Feature Descriptions

#### Model Caching
OpenVINO™ supports [model caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html).

Model caching feature is supported on CPU, NPU, GPU along with kernel caching on iGPU, dGPU.

This feature enables users to save and load the blob file directly on to the hardware device target and perform inference with improved Inference Latency.

Kernel Caching on iGPU and dGPU:

This feature also allows user to save kernel caching as cl_cache files for models with dynamic input shapes. These cl_cache files can be loaded directly onto the iGPU/dGPU hardware device target and inferencing can be performed.

#### <b> Enabling Model Caching via Runtime options using C++/python API's.</b>

This flow can be enabled by setting the runtime config option 'cache_dir' specifying the path to dump and load the blobs (CPU, NPU, iGPU, dGPU) or cl_cache(iGPU, dGPU) while using the C++/python API'S.

Refer to [Configuration Options](#configuration-options) for more information about using these runtime options.

### Support for INT8 Quantized models

Int8 models are supported on CPU, GPU and NPU.

### Support for Weights saved in external files

OpenVINO™ Execution Provider now  supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations.

See the [OpenVINO™ ONNX Support documentation](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-onnx.html).

Converting and Saving an ONNX Model to External Data:
Use the ONNX API's.[documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md#converting-and-saving-an-onnx-model-to-external-data).

Example:

```python
import onnx
onnx_model = onnx.load("model.onnx") # Your model in memory as ModelProto
onnx.save_model(onnx_model, 'saved_model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='data/weights_data', size_threshold=1024, convert_attribute=False)
```

Note:
1. In the above script, model.onnx is loaded and then gets saved into a file called 'saved_model.onnx' which won't have the weights but this new onnx model now will have the relative path to where the weights file is located. The weights file 'weights_data' will now contain the weights of the model and the weights from the original model gets saved at /data/weights_data.

2. Now, you can use this 'saved_model.onnx' file to infer using your sample. But remember, the weights file location can't be changed. The weights have to be present at /data/weights_data

3. Install the latest ONNX Python package using pip to run these ONNX Python API's successfully.



### Multi-threading for OpenVINO Execution Provider

OpenVINO™ Execution Provider for ONNX Runtime enables thread-safe deep learning inference

### Multi streams for OpenVINO Execution Provider
OpenVINO™ Execution Provider for ONNX Runtime allows multiple stream execution for difference performance requirements part of API 2.0

### Auto-Device Execution for OpenVINO Execution Provider

Use `AUTO:<device 1>,<device 2>..` as the device name to delegate selection of an actual accelerator to OpenVINO™. Auto-device internally recognizes and selects devices from CPU, integrated GPU, discrete Intel GPUs (when available) and NPU (when available) depending on the device capabilities and the characteristic of ONNX models, for example, precisions. Then Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in full system.

For more information on Auto-Device plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Auto Device Plugin](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#automatic-device-selection).

### Heterogeneous Execution for OpenVINO Execution Provider

The heterogeneous execution enables computing for inference on one network on several devices. Purposes to execute networks in heterogeneous mode:

* To utilize accelerator's power and calculate the heaviest parts of the network on the accelerator and execute unsupported layers on fallback devices like the CPU to utilize all available hardware more efficiently during one inference.

For more information on Heterogeneous plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Heterogeneous Plugin](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html).

### Multi-Device Execution for OpenVINO Execution Provider

Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. Potential gains are as follows:

* Improved throughput that multiple devices can deliver (compared to single-device execution)
* More consistent performance, since the devices can now share the inference burden (so that if one device is becoming too busy, another device can take more of the load)

For more information on Multi-Device plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Multi Device Plugin](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#multi-stream-execution).

### Export OpenVINO Compiled Blob 
Export the OpenVINO compiled blob as an ONNX model. Using this ONNX model for subsequent inferences avoids model recompilation and could have a positive impact on Session creation time. This feature is currently enabled for fully supported models only. It complies with the ORT session config keys
```
  Ort::SessionOptions session_options;

      // Enable EP context feature to dump the partitioned graph which includes the EP context into Onnx file.
      // "0": disable. (default)
      // "1": enable.

  session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

      // Flag to specify whether to dump the EP context into single Onnx model or pass bin path.
      // "0": dump the EP context into separate file, keep the file name in the Onnx model.
      // "1": dump the EP context into the Onnx model. (default).

  session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");

      // Specify the file path for the Onnx model which has EP context.
      // Defaults to <actual_model_path>/original_file_name_ctx.onnx if not specified

  session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ".\ov_compiled_epctx.onnx");

  sess = onnxruntime.InferenceSession(<path_to_model_file>, session_options)
```
Refer to [Session Options](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h) for more information about session options.

### Enable QDQ Optimizations Passes
Optimizes ORT quantized models for the NPU device to only keep QDQs for supported ops and optimize for performance and accuracy.Generally this feature will give better performance/accuracy with ORT Optimizations disabled. 
Refer to [Configuration Options](#configuration-options) for more information about using these runtime options.

### Loading Custom JSON OpenVINO Config During Runtime
The `load_config` feature is developed to facilitate loading of OpenVINO EP parameters using a JSON input schema, which mandatorily follows below format -
```
{
    "DEVICE_KEY": {"PROPERTY": "PROPERTY_VALUE"}
}
```
where "DEVICE_KEY" can be CPU, NPU, GPU or AUTO , "PROPERTY" must be a valid entity defined in [OpenVINO™ supported properties](https://github.com/openvinotoolkit/openvino/blob/releases/2025/3/src/inference/include/openvino/runtime/properties.hpp) &  "PROPERTY_VALUE" must be a valid corresponding supported property value passed in as a string. 

If a property is set using an invalid key (i.e., a key that is not recognized as part of the `OpenVINO™ supported properties`), it will be ignored & a warning will be logged against the same. However, if a valid property key is used but assigned an invalid value (e.g., a non-integer where an integer is expected), the OpenVINO™ framework will result in an exception during execution.

The valid properties are of two types viz. Mutable (Read/Write) & Immutable (Read only) these are also governed while setting the same. If an Immutable property is being set, we skip setting the same with a similar warning.

For setting appropriate `"PROPERTY"`, refer to OpenVINO config options for [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#supported-properties), [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#supported-properties), [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#supported-features-and-properties) and [AUTO](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html#using-auto). 

Example:

The usage of this functionality using onnxruntime_perf_test application is as below – 

```
onnxruntime_perf_test.exe -e openvino -m times -r 1 -i "device_type|NPU load_config|test_config.json" model.onnx
``` 
#### Dynamic Shape Management 
Comprehensive handling of models with variable input dimensions across all supported devices.
Device-Specific Handling:

NPU: Requires shape bounds via reshape_input for optimal memory management
CPU/GPU: Automatic dynamic shape handling with runtime optimization
All Devices: Option to convert dynamic to static shapes when beneficial

OpenVINO™ Shape Management:
The reshape method updates input shapes and propagates them through all intermediate layers to outputs. This enables runtime shape modification for different input sizes.
Shape Changing Approaches:

Single input models: Pass new shape directly to reshape method
Multiple inputs: Specify shapes by port, index, or tensor name
Batch modification: Use set_batch method with appropriate layout specification

Performance Considerations:

Static shapes avoid memory and runtime overheads
Dynamic shapes provide flexibility at performance cost
Shape bounds optimization (NPU) balances flexibility and performance

Important: Always set static shapes when input dimensions won't change between inferences for optimal performance.

#### Tensor Layout Control
Enables explicit specification of tensor memory layout for performance optimization currenty supported on CPU.
Layout specification helps OpenVINO optimize memory access patterns and tensor operations based on actual data organization.
Layout Specification Benefits:

Optimized memory access: Improved cache utilization and memory throughput
Better tensor operations: Device-specific operation optimization
Reduced memory copies: Direct operation on optimally laid out data
Hardware-specific optimization: Leverages device-preferred memory layouts

Common Layout Patterns:

NCHW: Batch, Channel, Height, Width 
NHWC: Batch, Height, Width, Channel 
NC: Batch, Channel 
NTD: Batch, Time, Dimension 


### OpenVINO Execution Provider Supports EP-Weight Sharing across sessions
The OpenVINO Execution Provider (OVEP) in ONNX Runtime supports EP-Weight Sharing, enabling models to efficiently share weights across multiple inference sessions. This feature enhances the execution of Large Language Models (LLMs) with prefill and KV cache, reducing memory consumption and improving performance when running multiple inferences.

With EP-Weight Sharing, prefill and KV cache models can now reuse the same set of weights, minimizing redundancy and optimizing inference. Additionally, this ensures that EP Context nodes are still created even when the model undergoes subgraph partitioning. 

These changes enable weight sharing between two models using the session context option: ep.share_ep_contexts.
Refer to [Session Options](https://github.com/microsoft/onnxruntime/blob/5068ab9b190c549b546241aa7ffbe5007868f595/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L319) for more details on configuring this runtime option.



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