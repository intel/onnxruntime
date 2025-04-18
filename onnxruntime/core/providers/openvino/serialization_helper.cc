// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "serialization_helper.h"
#include "contexts.h"

namespace onnxruntime {
namespace openvino_ep {

template <>
byte_iostream& operator<<(byte_iostream& stream, const std::string& value) {
  stream.write((const std::byte*)value.data(), value.size() * sizeof(std::string::value_type));
  return stream;
}

template <>
byte_iostream& operator>>(byte_iostream& stream, std::string& value) {
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

  stream.read((std::byte*)value.data(), size * sizeof(std::string::value_type));
  return stream;
}

byte_iostream& operator<<(byte_iostream& stream, const weight_map_value& value) {
  stream << value.location;
  stream << value.data_offset;
  stream << value.size;
  stream << value.dimensions;
  stream << value.element_type;
  return stream;
}

byte_iostream& operator>>(byte_iostream& stream, weight_map_value& value) {
  stream >> value.location;
  stream >> value.data_offset;
  stream >> value.size;
  stream >> value.dimensions;
  stream >> value.element_type;
  return stream;
}

bool weight_map_value::operator==(const weight_map_value& other) const {
  return (location == other.location) &&
         (data_offset == other.data_offset) &&
         (size == other.size) &&
         (dimensions == other.dimensions) &&
         (element_type == other.element_type);
}

}  // namespace openvino_ep
}  // namespace onnxruntime