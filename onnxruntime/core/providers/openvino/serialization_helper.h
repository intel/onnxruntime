// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <iostream>
#include <vector>

#include "core/common/common.h"

namespace onnxruntime {
namespace openvino_ep {

using byte_iostream = std::basic_iostream<std::byte>;
using byte_istream = std::basic_istream<std::byte>;
using byte_ostream = std::basic_ostream<std::byte>;
using byte_fstream = std::basic_fstream<std::byte>;

//
// Write
//
template <typename T>
byte_iostream& operator<<(byte_iostream& stream, const T& value) {
  // Non-trivial types are unsupported unless they have an explicit specialization
  static_assert(false, "Unsupported");
  return stream;
}

template <typename T>
  requires std::is_trivially_copyable_v<T>
byte_iostream& operator<<(byte_iostream& stream, const T& value) {
  stream.write(reinterpret_cast<const std::byte*>(&value), sizeof(T));
  return stream;
}

template <typename T>
byte_iostream& operator<<(byte_iostream& stream, const std::vector<T>& value) {
  stream << value.size();
  for (const auto& element : value) {
    stream << element;
  }
  return stream;
}

template <>
byte_iostream& operator<<(byte_iostream& stream, const std::string& value);

//
// Read
//
template <typename T>
byte_iostream& operator>>(byte_iostream& stream, T& value) {
  // Non-trivial types are unsupported unless they have an explicit specialization
  static_assert(false, "Unsupported");
  return stream;
}

template <typename T>
  requires std::is_trivially_copyable_v<T>
byte_iostream& operator>>(byte_iostream& stream, T& value) {
  stream.read(reinterpret_cast<std::byte*>(&value), sizeof(T));
  return stream;
}

constexpr size_t MAX_SAFE_DIMENSIONS = 1024;

template <typename T>
byte_iostream& operator>>(byte_iostream& stream, std::vector<T>& value) {
  size_t size;
  stream >> size;

  if (stream.fail()) {
    ORT_THROW("Error: Failed to read size from stream.");
  }

  if (size == 0 || size > MAX_SAFE_DIMENSIONS) {
    ORT_THROW("Invalid size read.");
  }

  try {
    value.resize(size);
  } catch (const std::bad_alloc&) {
    ORT_THROW("Error: Memory allocation failed while resizing container.");
  }

  for (auto& element : value) {
    stream >> element;
  }
  return stream;
}

template <>
byte_iostream& operator>>(byte_iostream& stream, std::string& value);

// Serializable unordered map
template <typename K, typename V>
struct io_unordered_map {
  struct Hash {
    std::size_t operator()(const K& key) const noexcept {
      return std::hash<std::string>()(key.name);
    }
  };

  using Map = std::unordered_map<K, V, Hash>;
  using Key = K;
  using Value = V;

  friend byte_iostream& operator<<(byte_iostream& stream, const io_unordered_map::Map &map) {
    try {
      stream << map.size();

      // Write each key-value pair
      // Put elements in separate lines to facilitate reading
      for (const auto& [key, value] : map) {
        stream << key;
        stream << value;
      }
    } catch (...) {
      ORT_THROW("Error: Failed to write map data.");
    }

    ORT_ENFORCE(stream.good(), "Error: Failed to write map data.");
    return stream;
  };

  friend byte_iostream& operator>>(byte_iostream& stream, io_unordered_map::Map& map) {
    size_t map_size{0};
    try {
      stream >> map_size;

      while (map_size--) {
        K key;
        V value;
        stream >> key;
        stream >> value;
        map.emplace(key, value);
      }
    } catch (...) {
      ORT_THROW("Error: Failed to read map data.");
    }

    return stream;
  };

};

}
}  // namespace onnxruntime