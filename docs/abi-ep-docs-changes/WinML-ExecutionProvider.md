---
title: Windows - WinML
description: Instructions to execute ONNX Runtime on Windows with the WinML Execution Provider, including OpenVINO EP usage
parent: Execution Providers
nav_order: 4
---

# WinML Execution Provider
{: .no_toc }

**WinML is the recommended Windows development path for ONNX Runtime.** WinML automatically selects the best execution provider (EP) based on your customer's hardware — including CPU, GPU, and NPU — without requiring you to configure individual EPs manually. It uses the same ONNX Runtime APIs you're already familiar with.

WinML offers several advantages for Windows developers:

- **Same ONNX Runtime APIs**: WinML uses the same ONNX Runtime C/C++, C#, and Python APIs you already know.
- **Automatic EP selection**: WinML dynamically selects the best available EP (CPU, DirectML GPU, Intel OpenVINO NPU/GPU, Qualcomm QNN, etc.) based on the hardware available on the device.
- **Simplified deployment**: Reduces deployment complexity by bundling all required AI inference dependencies for Windows.
- **Windows 11 24H2+ optimized**: On Windows 11 24H2 and later, WinML provides additional hardware automation and optimization across a broad range of silicon vendors.

For legacy Windows scenarios or specific DirectML requirements, see the [DirectML Execution Provider](DirectML-ExecutionProvider.md) (note: DirectML is in sustained engineering).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

### NuGet (C#/C/C++)

WinML is distributed as a NuGet package:

```bash
dotnet add package Microsoft.AI.MachineLearning
```

See the full [install page](../install/#winml-recommended-for-windows) for nightly builds and additional options.

### Python

```bash
pip install onnxruntime
```

WinML-style automatic EP selection is available on Windows via the standard `onnxruntime` package. To explicitly use the Intel OpenVINO™ EP from Python, install:

```bash
pip install onnxruntime-openvino
```

See [OpenVINO EP Install](OpenVINO-ExecutionProvider.md#install) for details.

## Requirements

- Windows 10, version 1809 (build 17763) or later
- Windows 11 24H2+ recommended for full automatic EP selection across all silicon vendors
- Visual Studio 2022 (for C++/C# development)

## Usage

### C# (WinRT API)

The `Microsoft.AI.MachineLearning` NuGet package exposes the WinRT API:

```csharp
using Windows.AI.MachineLearning;
using Windows.Storage;

// Load the model — WinML selects the best EP automatically
var modelFile = await StorageFile.GetFileFromPathAsync("model.onnx");
var model = await LearningModel.LoadFromStorageFileAsync(modelFile);

// Create a session; WinML picks CPU, GPU, or NPU based on device
var session = new LearningModelSession(model);

// Bind inputs/outputs and evaluate
var binding = new LearningModelBinding(session);
// ... bind tensors ...
var results = await session.EvaluateAsync(binding, "run-1");
```

### C++ (WinRT API)

```cpp
#include <winrt/Windows.AI.MachineLearning.h>

using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Storage;

// Load model — EP selected automatically by WinML
auto modelFile = StorageFile::GetFileFromPathAsync(L"model.onnx").get();
auto model     = LearningModel::LoadFromStorageFileAsync(modelFile).get();
auto session   = LearningModelSession{ model };

LearningModelBinding binding{ session };
// ... bind inputs/outputs ...
session.EvaluateAsync(binding, L"run-1").get();
```

### Python

```python
import onnxruntime as ort

# WinML automatic EP selection on Windows
session = ort.InferenceSession("model.onnx")

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data})
```

## OpenVINO™ Execution Provider Usage with WinML

On Windows devices with Intel CPUs, GPUs, or NPUs, WinML automatically routes inference through the **Intel OpenVINO™ EP** when it is the best available option. No additional configuration is required for automatic routing.

> **Note:** The built-in OpenVINO™ EP in upstream `microsoft/onnxruntime` is in legacy mode. Active development continues in [intel/onnxruntime](https://github.com/intel/onnxruntime). See [OpenVINO™ Execution Provider](OpenVINO-ExecutionProvider.md) for full details.

### Explicit OpenVINO EP selection — C/C++

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
| `HETERO:GPU,CPU` | Heterogeneous split across devices |

For a full list of configuration options (precision, caching, load_config, etc.) see [OpenVINO EP Configuration Options](OpenVINO-ExecutionProvider.md#configuration-options).

## OpenVINO™ Execution Provider Installation

### Python

```bash
pip install onnxruntime-openvino
```

### NuGet (C#)

Build a custom NuGet package with the OpenVINO EP following the [build instructions](../build/eps.md#openvino) and add `--build_nuget`. Two packages are produced:
- `Microsoft.ML.OnnxRuntime.Managed`
- `Intel.ML.OnnxRuntime.Openvino`

### Set OpenVINO™ Environment Variables

Before running any application using the OpenVINO EP, initialize the OpenVINO environment:

**Windows:**
```bat
C:\<openvino_install_directory>\setupvars.bat
```

**Linux:**
```bash
source <openvino_install_directory>/setupvars.sh
```

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
- [OpenVINO™ Execution Provider](OpenVINO-ExecutionProvider.md)
- [DirectML Execution Provider](DirectML-ExecutionProvider.md) *(sustained engineering)*
- [intel/onnxruntime releases](https://github.com/intel/onnxruntime/releases) — latest Intel OpenVINO EP packages
