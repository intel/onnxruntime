// Copyright (C) Intel Corporation
// Licensed under the MIT License
//
// Windows-only test for SxS (Side-by-Side) DLL isolation.
// Verifies that OpenVINO EP loads its own private copy of openvino.dll even
// when another copy is already loaded in the process.

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <psapi.h>

#include <algorithm>
#include <cwctype>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

namespace fs = std::filesystem;

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

static fs::path GetExeDir() {
  wchar_t exe_path[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
  return fs::path(exe_path).parent_path();
}

// Returns full paths of all loaded DLLs whose name contains the given fragment.
static std::vector<std::wstring> FindLoadedModules(std::wstring_view name_fragment_lower) {
  std::vector<std::wstring> result;
  std::vector<HMODULE> modules(1024);
  DWORD cbNeeded = 0;

  // Retry with a larger buffer if the initial allocation is too small.
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (!EnumProcessModulesEx(GetCurrentProcess(), modules.data(),
                              static_cast<DWORD>(modules.size() * sizeof(HMODULE)),
                              &cbNeeded, LIST_MODULES_ALL)) {
      return result;
    }
    if (cbNeeded <= modules.size() * sizeof(HMODULE)) break;
    modules.resize(cbNeeded / sizeof(HMODULE) + 64);
  }

  DWORD count = std::min<DWORD>(cbNeeded / sizeof(HMODULE),
                                static_cast<DWORD>(modules.size()));
  for (DWORD i = 0; i < count; ++i) {
    wchar_t path[MAX_PATH] = {};
    if (!GetModuleFileNameExW(GetCurrentProcess(), modules[i], path, MAX_PATH)) {
      continue;
    }
    std::wstring wpath(path);
    std::wstring lower = wpath;
    for (auto& c : lower) c = static_cast<wchar_t>(std::towlower(c));
    if (lower.find(name_fragment_lower) != std::wstring::npos) {
      result.push_back(wpath);
    }
  }
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
struct SxsIsolationFixture : public ::testing::Test {
  fs::path bin_dir_;
  fs::path temp_dir_;
  fs::path preloaded_ov_;
  HMODULE h_preloaded_ov_ = nullptr;

  void SetUp() override {
    bin_dir_ = GetExeDir();

    if (!fs::exists(bin_dir_ / L"openvino.dll")) {
      GTEST_SKIP() << "openvino.dll not found — run from installed directory";
    }

    // Skip if manifests haven't been embedded (cmake --install)
    if (!fs::exists(bin_dir_ / L"openvino_runtime.manifest")) {
      GTEST_SKIP() << "openvino_runtime.manifest not found — run cmake --install first";
    }

    // Copy openvino.dll to a temp folder and pre-load it.
    // This simulates another app having already loaded
    // its own openvino.dll into this process.
    std::error_code ec;
    temp_dir_ = fs::temp_directory_path(ec) /
                (std::wstring(L"ort_openvino_sxs_test_") + std::to_wstring(GetCurrentProcessId()));
    ASSERT_FALSE(ec) << "Failed to get temp directory: " << ec.message();

    fs::create_directories(temp_dir_, ec);
    ASSERT_FALSE(ec) << "Failed to create temp dir: " << ec.message();

    preloaded_ov_ = temp_dir_ / L"openvino.dll";
    fs::copy_file(bin_dir_ / L"openvino.dll", preloaded_ov_,
                  fs::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << "Failed to copy openvino.dll: " << ec.message();

    // Load into module list without actually initializing the DLL
    h_preloaded_ov_ = LoadLibraryExW(preloaded_ov_.wstring().c_str(), nullptr,
                                     DONT_RESOLVE_DLL_REFERENCES);
    ASSERT_NE(h_preloaded_ov_, nullptr)
        << "Pre-load of openvino.dll failed (error " << GetLastError() << ")";
  }

  void TearDown() override {
    if (h_preloaded_ov_) {
      FreeLibrary(h_preloaded_ov_);
      h_preloaded_ov_ = nullptr;
    }
    if (!temp_dir_.empty()) {
      std::error_code ec;
      fs::remove_all(temp_dir_, ec);
    }
  }
};

// ---------------------------------------------------------------------------
TEST_F(SxsIsolationFixture, SxsLoadsPrivateAssemblyCopy) {
  // At this point we have 1 openvino.dll loaded (the fake pre-loaded one)
  const auto ov_mods_before = FindLoadedModules(L"openvino.dll");
  ASSERT_GE(ov_mods_before.size(), 1u);

  // If a prior test already loaded the bin-dir copy, we can't validate SxS triggers a new load.
  const auto bin_ov = (bin_dir_ / L"openvino.dll").wstring();
  if (std::any_of(ov_mods_before.begin(), ov_mods_before.end(),
                  [&](const auto& p) { return _wcsicmp(p.c_str(), bin_ov.c_str()) == 0; })) {
    GTEST_SKIP() << "openvino.dll already loaded from bin directory; run in a fresh process";
  }

  // Trigger OpenVINO EP load — SxS should cause a SECOND openvino.dll to
  // load from the bin directory, ignoring the already-loaded copy.
  Ort::SessionOptions session_opts;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "CPU";
  session_opts.AppendExecutionProvider_OpenVINO_V2(ov_options);

  // If SxS works, we should now have 2 distinct openvino.dll modules
  auto ov_mods_after = FindLoadedModules(L"openvino.dll");

  ASSERT_GE(ov_mods_after.size(), 2u)
      << "SxS isolation failed: expected >=2 openvino.dll instances, got "
      << ov_mods_after.size();

  // Confirm both paths are present
  auto iequals = [](const std::wstring& a, const std::wstring& b) {
    return _wcsicmp(a.c_str(), b.c_str()) == 0;
  };
  auto contains = [&](const std::wstring& path) {
    return std::any_of(ov_mods_after.begin(), ov_mods_after.end(),
                       [&](const auto& p) { return iequals(p, path); });
  };

  EXPECT_TRUE(contains(preloaded_ov_.wstring()))
      << "Pre-loaded openvino.dll not found in module list";
  EXPECT_TRUE(contains((bin_dir_ / L"openvino.dll").wstring()))
      << "SxS bin-dir openvino.dll not found in module list";
}

#endif  // _WIN32
