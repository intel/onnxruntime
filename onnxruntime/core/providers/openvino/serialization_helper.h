// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <iostream>
#include <vector>

#include "core/common/common.h"
#include "constants.h"

namespace onnxruntime {
namespace openvino_ep {

//
// Write
//
//

// Scalar
template <typename T>
void write_bytes(std::ostream& stream, const T& value) {
  stream.write(reinterpret_cast<const std::ostream::char_type*>(&value), sizeof(T));
}

// Scalar at offset
template <typename T>
void write_bytes(std::ostream& stream, const T& value, std::streampos pos) {
  stream.seekp(pos);
  write_bytes(stream, value);
}

// Vector
template <typename T>
void write_bytes(std::ostream& stream, const std::vector<T>& value) {
  write_bytes(stream, value.size());
  for (const auto& element : value) {
    write_bytes(stream, element);
  }
}

// String
template <>
void write_bytes(std::ostream& stream, const std::string& value);

//
// Read
//

// Scalar
template <typename T>
void read_bytes(std::istream& stream, T& value) {
  stream.read(reinterpret_cast<std::istream::char_type*>(&value), sizeof(T));
}

// Scalar at offset
template <typename T>
void read_bytes(std::istream& stream, T& value, std::streampos pos) {
  stream.seekg(pos);
  read_bytes<T>(stream, value);
}

// Block read from position
void read_bytes(std::istream& stream, std::streampos pos, std::istream::char_type* data, std::streamsize count);

// Vector
template <typename T>
void read_bytes(std::istream& stream, std::vector<T>& value) {
  size_t size;
  read_bytes(stream, size);

  if (stream.fail()) {
    ORT_THROW("Error: Failed to read size from stream.");
  }

  if (size == 0 || size > constants::max_safe_dimensions) {
    ORT_THROW("Invalid size read.");
  }

  try {
    value.resize(size);
  } catch (const std::bad_alloc&) {
    ORT_THROW("Error: Memory allocation failed while resizing container.");
  }

  for (auto& element : value) {
    read_bytes(stream, element);
  }
}

// String
template <>
void read_bytes(std::istream& stream, std::string& value);

template <typename T>
struct streamable {
  template <typename S>
  friend void write_bytes(S& stream, const T& value);

  template <typename S>
  friend void read_bytes(S& stream, T& value);
};

// Serializable unordered map
template <typename K, typename V, typename... Args>
struct io_unordered_map : std::unordered_map<K, V, Args...>, streamable<io_unordered_map<K, V, Args...>> {
  template <typename S>
  friend void write_bytes(S& stream, const io_unordered_map& map) {
    try {
      write_bytes(stream, map.size());

      // Write each key-value pair
      // Put elements in separate lines to facilitate reading
      for (const auto& [key, value] : map) {
        write_bytes(stream, key);
        write_bytes(stream, value);
      }
    } catch (...) {
      ORT_THROW("Error: Failed to write map data.");
    }

    ORT_ENFORCE(stream.good(), "Error: Failed to write map data.");
  }

  template <typename S>
  friend void read_bytes(S& stream, io_unordered_map& map) {
    size_t map_size{0};
    try {
      read_bytes(stream, map_size);

      while (map_size--) {
        K key;
        V value;
        read_bytes(stream, key);
        read_bytes(stream, value);
        map.emplace(key, value);
      }
    } catch (...) {
      ORT_THROW("Error: Failed to read map data.");
    }
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime