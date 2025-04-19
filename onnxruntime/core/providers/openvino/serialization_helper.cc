// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "serialization_helper.h"
#include "contexts.h"

namespace onnxruntime {
namespace openvino_ep {

//
// Write
//
template <>
void write_bytes(std::ostream& stream, const std::string& value) {
  write_bytes(stream, value.size());
  stream.write(reinterpret_cast<const std::ostream::char_type*>(value.data()), value.size() * sizeof(std::string::value_type));
}

////
//// Read
////
template <>
void read_bytes(std::istream& stream, std::string& value) {
  size_t size{0};
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

  stream.read((std::istream::char_type*)value.data(), size * sizeof(std::string::value_type));
}

void read_bytes(std::istream& stream, std::streampos pos, std::istream::char_type* data, std::streamsize count) {
  stream.seekg(pos);
  stream.read(data, count);
}

//
// Other
//
bool weight_map_value::operator==(const weight_map_value& other) const {
  return (location == other.location) &&
         (data_offset == other.data_offset) &&
         (size == other.size) &&
         (dimensions == other.dimensions) &&
         (element_type == other.element_type);
}

}  // namespace openvino_ep
}  // namespace onnxruntime