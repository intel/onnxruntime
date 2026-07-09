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
pip install onnxruntime-ep-openvino
```

## Build

For legacy EP build instructions, refer [BUILD page](../build/eps.md#openvino).

## Usage

To explicitly request the OpenVINO™ EP within a WinML-based application:

```cpp
#include <onnxruntime_cxx_api.h>

OrtApi const& ortApi = Ort::GetApi();
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");
Ort::SessionOptions session_options;

// 1. Register the EP plugin library
ortApi.RegisterExecutionProviderLibrary(
    env, "OpenVINOExecutionProvider",
    ORT_TSTR("onnxruntime_providers_openvino_plugin.dll"));

// 2. Enumerate available EP devices and find OpenVINO
const OrtEpDevice* const* ep_devices = nullptr;
size_t num_ep_devices;
ortApi.GetEpDevices(env, &ep_devices, &num_ep_devices);

const OrtEpDevice* ov_device = nullptr;
for (size_t i = 0; i < num_ep_devices; i++) {
    if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]),
              "OpenVINOExecutionProvider") == 0) {
        ov_device = ep_devices[i];
        break;
    }
}

// 4. Create session
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
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

## Samples

Official sample code demonstrating WinML on Windows:

| Sample | Description |
|---|---|
| [WindowsAppSDK-Samples / Samples/WindowsML](https://github.com/microsoft/WindowsAppSDK-Samples/tree/main/Samples/WindowsML) | C++, C#, and Python samples running hardware-accelerated ONNX models on Windows (CPU, GPU, NPU) using WinML |

## Additional Resources

- [Windows ML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
- [Windows.AI.MachineLearning API Reference](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)
- [OpenVINO™ Toolkit Documentation](https://docs.openvino.ai/)
