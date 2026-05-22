---
title: Intel - OpenVINO™
description: Instructions to execute OpenVINO™ Execution Provider for ONNX Runtime.
parent: Execution Providers
nav_order: 3
redirect_from: /docs/reference/execution-providers/OpenVINO-ExecutionProvider
---

# OpenVINO™ Execution Provider
{: .no_toc }

{: .note }
**Note: The built-in OpenVINO™ Execution Provider in the upstream ONNX Runtime repository is now in legacy mode.** Active development and releases have moved to the Intel-maintained fork at [intel/onnxruntime](https://github.com/intel/onnxruntime). For new projects on Windows, consider [WinML](WinML-ExecutionProvider.md), which provides automatic hardware-aware EP selection including support for Intel devices. 

Accelerate ONNX models on Intel CPUs, GPUs, NPU with Intel OpenVINO™ Execution Provider. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

- Windows 10, version 1809 (build 17763) or later
- Windows 11 24H2+ recommended for full automatic EP selection across all silicon vendors
- Visual Studio 2022 (for C++/C# development)

## OpenVINO™ Execution Provider Usage with WinML

On Windows devices with Intel CPUs, GPUs, or NPUs, WinML automatically routes inference through the **Intel OpenVINO™ EP** when it is the best available option. No additional configuration is required for automatic routing.

> **Note:** The built-in OpenVINO™ EP in upstream `microsoft/onnxruntime` is in legacy mode. Active development continues in [intel/onnxruntime](https://github.com/intel/onnxruntime).

## Install
See the [WinML install section](https://onnxruntime.ai/docs/install/#cccwinml-installs) for WinML-related installation instructions.

### Python
To explicitly use the Intel OpenVINO™ EP from Python, install:

```bash
pip install onnxruntime-openvino
```

### NuGet (C#)

Build a custom NuGet package with the OpenVINO EP following the [build instructions](../build/eps.md#openvino) and add `--build_nuget`. Two packages are produced:
- `Microsoft.ML.OnnxRuntime.Managed`
- `Intel.ML.OnnxRuntime.Openvino`

## Build

For legacy EP build instructions, refer [BUILD page](../build/eps.md#openvino).

## Usage

To explicitly request the OpenVINO™ EP within a WinML-based application, use the V2 API:

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");
Ort::SessionOptions session_options;

// Explicitly use OpenVINO EP targeting CPU
std::unordered_map<std::string, std::string> options;
options["device_type"] = "CPU";   // or "GPU", "NPU", "AUTO"
session_options.AppendExecutionProvider_OpenVINO_V2(options);

Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

### Explicit OpenVINO EP selection — Python

```python
import onnxruntime as ort

options = {
    "device_type": "CPU",   # or "GPU", "NPU", "AUTO:GPU,NPU,CPU"
}
session = ort.InferenceSession(
    "model.onnx",
    providers=[("OpenVINOExecutionProvider", options)]
)

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data})
```

### Supported OpenVINO target devices

| Device string | Hardware |
|---|---|
| `CPU` | Intel CPU (Atom, Core, Xeon) |
| `GPU` | Intel integrated or discrete GPU |
| `GPU.0`, `GPU.1` | Specific GPU when multiple are present |
| `NPU` | Intel Neural Processing Unit |
| `AUTO` | Automatic best-device selection |
| `AUTO:GPU,NPU,CPU` | Prioritized automatic selection |

## Configuration Options

Runtime parameters set during OpenVINO Execution Provider initialization to control the inference flow.


| **Key** | **Type** | **Allowable Values** | **Value Type** | **Description** |
|---------|----------|---------------------|----------------|-----------------|
| [**load_config**](#load_config) | string | JSON string | string | Load and set custom/HW specific OpenVINO properties from JSON |
| [**reshape_input**](#reshape_input) | string | input_name[shape_bounds] | string | Specify upper and lower bound for dynamic shaped inputs for improved performance with NPU |
| [**layout**](#layout) | string | input_name[layout_format] | string | Specify input/output tensor layout format |

## Configuration Descriptions

### `load_config`

**Recommended Configuration Method** for setting OpenVINO runtime properties. Provides direct access to OpenVINO properties through a JSON String during runtime.

#### Overview

`load_config` enables fine-grained control over OpenVINO inference behavior by loading properties from a JSON String. This is the **preferred method** for configuring advanced OpenVINO features, offering:

- Direct access to OpenVINO runtime properties
- Device-specific configuration
- Better compatibility with future OpenVINO releases
- No property name translation required



#### JSON Configuration Format
```json
{
  "DEVICE_NAME": {
    "PROPERTY_KEY": "value"
  }
}
```

`load_config` now supports nested JSON objects up to **8 levels deep** for complex device configurations.

**Maximum Nesting:** 8 levels deep.

**Example: Multi-Level Nested Configuration**
```python
import onnxruntime as ort
import json

# Complex nested configuration for AUTO device
config = {
    "AUTO": {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "DEVICE_PROPERTIES": {
            "CPU": {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "3"
            },
            "GPU": {
                "EXECUTION_MODE_HINT": "ACCURACY",
                "PERFORMANCE_HINT": "LATENCY"
            }
        }
    }
}
```

**Supported Device Names:**
- `"CPU"` - Intel CPU
- `"GPU"` - Intel integrated/discrete GPU
- `"NPU"` - Intel Neural Processing Unit
- `"AUTO"` - Automatic device selection


#### Popular OpenVINO Properties

The following properties are commonly used for optimizing inference performance. For complete property definitions and all possible values, refer to the [OpenVINO properties](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/properties.hpp) header file.
##### Performance & Execution Hints

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `PERFORMANCE_HINT` | `"LATENCY"`, `"THROUGHPUT"` | High-level performance optimization goal |
| `EXECUTION_MODE_HINT` | `"ACCURACY"`, `"PERFORMANCE"` | Accuracy vs performance trade-off |
| `INFERENCE_PRECISION_HINT` | `"f32"`, `"f16"`, `"bf16"` | Explicit inference precision |


**PERFORMANCE_HINT:**
- `"LATENCY"`: Optimizes for low latency
- `"THROUGHPUT"`: Optimizes for high throughput

**EXECUTION_MODE_HINT:**
- `"ACCURACY"`: Maintains model precision, dynamic precision selection
- `"PERFORMANCE"`: Optimizes for speed, may use lower precision

**INFERENCE_PRECISION_HINT:**
- `"f16"`: FP16 precision 
- `"f32"`: FP32 precision - highest accuracy
- `"bf16"`: BF16 precision - balance between f16 and f32

**Important:** Use either `EXECUTION_MODE_HINT` OR `INFERENCE_PRECISION_HINT`, not both. These properties control similar behavior and should not be combined.

**Note:** CPU accepts `"f16"` hint in configuration but will upscale to FP32 during execution, as CPU only supports FP32 precision natively.


##### Threading & Streams

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `NUM_STREAMS` | Positive integer (e.g., `"1"`, `"4"`, `"8"`) | Number of parallel execution streams |
| `INFERENCE_NUM_THREADS` | Integer | Maximum number of inference threads |
| `COMPILATION_NUM_THREADS` | Integer  | Maximum number of compilation threads | 

**NUM_STREAMS:**
- Controls parallel execution streams for throughput optimization
- Higher values increase throughput for batch processing
- Lower values optimize latency for real-time inference

**INFERENCE_NUM_THREADS:**
- Controls CPU thread count for inference execution
- Explicit value: Fixed thread count (e.g., `"4"` limits to 4 threads)

##### Caching Properties

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `CACHE_DIR` | File path string | Model cache directory |
| `CACHE_MODE` | `"OPTIMIZE_SIZE"`, `"OPTIMIZE_SPEED"` | Cache optimization strategy |

**CACHE_MODE:**
- `"OPTIMIZE_SPEED"`: Faster cache creation, larger cache files
- `"OPTIMIZE_SIZE"`: Slower cache creation, smaller cache files
##### Logging Properties

| Property | Valid Values | Description | 
|----------|-------------|-------------|
| `LOG_LEVEL` | `"LOG_NONE"`, `"LOG_ERROR"`, `"LOG_WARNING"`, `"LOG_INFO"`, `"LOG_DEBUG"`, `"LOG_TRACE"` | Logging verbosity level | 

**Note:** `LOG_LEVEL` is not supported on GPU devices.

##### AUTO Device Properties

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `ENABLE_STARTUP_FALLBACK` | `"YES"`, `"NO"` | Enable device fallback during model loading |
| `ENABLE_RUNTIME_FALLBACK` | `"YES"`, `"NO"` | Enable device fallback during inference runtime |
| `DEVICE_PROPERTIES` | Nested JSON string | Device-specific property configuration |

**DEVICE_PROPERTIES Syntax:**

Used to configure properties for individual devices when using AUTO mode.
```json
{
  "AUTO": {
    "DEVICE_PROPERTIES": "{CPU:{PROPERTY:value},GPU:{PROPERTY:value}}"
  }
}
```

#### Property Reference Documentation

For complete property definitions and advanced options, refer to the official OpenVINO properties header:

**[OpenVINO Runtime Properties](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/properties.hpp)**

Property keys used in `load_config` JSON must match the string literal defined in the properties header file.

---
  
### `reshape_input`

**NPU Shape Bounds Configuration**

- Use `reshape_input` to explicitly set dynamic shape bounds for NPU devices.

**Format:**
- Range bounds: `input_name[lower..upper]`
- Fixed shape: `input_name[fixed_shape]`

This configuration is required for optimal NPU memory allocation and management.

---

### `layout`

- Provides explicit control over tensor memory layout for performance optimization. 
- Helps OpenVINO optimize memory access patterns and tensor operations.

**Layout Characters:**

- **N:** Batch dimension
- **C:** Channel dimension
- **H:** Height dimension
- **W:** Width dimension
- **D:** Depth dimension
- **T:** Time dimension
- **?:** Unknown/dynamic dimension

**Format:**

`input_name[LAYOUT],output_name[LAYOUT]`

**Example:**

`input_image[NCHW],output_tensor[NC]`

---

## Examples
### Python
#### Using load_config with JSON string
```python
import onnxruntime as ort
import json

# Create config
config = {
    "AUTO": {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "DEVICE_PROPERTIES": "{GPU:{EXECUTION_MODE_HINT:ACCURACY,PERFORMANCE_HINT:LATENCY}}"
    }
}
# Use config with session
options = {"device_type": "AUTO", "load_config": json.dumps(config)}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])
```

#### Using load_config for CPU
```python
import onnxruntime as ort
import json

# Create CPU config
config = {
    "CPU": {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1"
    }
}
options = {"device_type": "CPU", "load_config": json.dumps(config)}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])
```
#### Using load_config for GPU
```python
import onnxruntime as ort
import json

# Create GPU config with caching
config = {
    "GPU": {
        "EXECUTION_MODE_HINT": "ACCURACY",
        "CACHE_DIR": "./model_cache",
        "PERFORMANCE_HINT": "LATENCY"
    }
}
options = {"device_type": "GPU", "load_config": json.dumps(config)}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])

```

--- 
### Python API
Key-Value pairs for config options can be set using InferenceSession API as follow:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, providers=['OpenVINOExecutionProvider'], provider_options=[{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that the releases from (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution providers other than the default CPU provider (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.*

--- 

# OpenVINO™ Execution Provider Samples & Tutorials

In order to showcase what you can do with the OpenVINO™ Execution Provider for ONNX Runtime, we have created a few samples that show how you can get that performance boost you're looking for with just one additional line of code.

## Samples

Official sample code demonstrating WinML and OpenVINO EP on Windows:

| Sample | Description |
|---|---|
| [WindowsAppSDK-Samples / Samples/WindowsML](https://github.com/microsoft/WindowsAppSDK-Samples/tree/main/Samples/WindowsML) | C++, C#, and Python samples running hardware-accelerated ONNX models on Windows (CPU, GPU, NPU) using WinML |
| [Capture Logs](https://github.com/microsoft/WindowsAppSDK-Samples/tree/main/Samples/WindowsML/capture-logs) | Scripts for capturing Windows ML diagnostic logs |
| [Windows ML Samples (MS Learn)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/samples) | Official Microsoft documentation sample index for Windows ML |
| [AI Dev Gallery](https://github.com/microsoft/ai-dev-gallery) | Collection of open-source samples for on-device AI using Windows Copilot Runtime and ONNX models |

To clone and run the WindowsAppSDK samples:

```bash
git clone https://github.com/microsoft/WindowsAppSDK-Samples.git
cd WindowsAppSDK-Samples/Samples/WindowsML
# Open the solution file in Visual Studio 2022
```

## Additional Resources

- [Windows ML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
- [Windows.AI.MachineLearning API Reference](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)
- [Get Started with ONNX Runtime for Windows](../get-started/with-windows.md)
- [OpenVINO™ Toolkit Documentation](https://docs.openvino.ai/)
- [Intel OpenVINO™ Execution Provider GitHub](https://github.com/intel/onnxruntime)
- [DirectML Execution Provider](DirectML-ExecutionProvider.md) *(sustained engineering)*
- [intel/onnxruntime releases](https://github.com/intel/onnxruntime/releases) — latest Intel OpenVINO EP packages

