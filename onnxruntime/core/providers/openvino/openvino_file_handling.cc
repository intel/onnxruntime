#include <filesystem>
#include <unordered_map>
#include <map>
// #include <memory>
#include <vector>
#include <string>
// #include <string_view>

#include "core/providers/openvino/contexts.h"

using Exception = ov::Exception;

namespace onnxruntime {
namespace openvino_ep {

SharedContext::SharedWeights::WeightsFile::WeightsFile(std::filesystem::path filename) : file_(filename, std::ios::in | std::ios::binary) {
  try {
    file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    weights_size_ = file_.seekg(0, std::ios::end).tellg();
  } catch (std::ifstream::failure& e) {
    ORT_THROW("Error: Failed to open weight file at ", filename.string(), " ", e.what());
  }
}

void SharedContext::SharedWeights::WeightsFile::load_weights(size_t file_offset, void* data, size_t size) {
  ORT_ENFORCE(file_offset < weights_size_ && size <= weights_size_ && (file_offset <= weights_size_ - size), "Error: File offset is out of bounds.");
  file_.seekg(file_offset);
  file_.read(reinterpret_cast<char*>(data), size);
}

//  Utility function to write a string in bin file
void writeString(std::fstream& file, const std::string& str) {
  size_t length = str.size();
  file.write(reinterpret_cast<const char*>(&length), sizeof(length));
  file.write(str.c_str(), length);
}

//  Utility function to write a vector<size_t> in bin file
void writeVector(std::fstream& file, const std::vector<size_t>& vec) {
  size_t length = vec.size();
  file.write(reinterpret_cast<const char*>(&length), sizeof(length));
  file.write(reinterpret_cast<const char*>(vec.data()), length * sizeof(size_t));
}

//  Write the entire metadata map to a binary file
void SharedContext::SharedWeights::Metadata::writeMetadataToBinaryFile(SharedContext& shared_context,
                                                                       const SharedContext::SharedWeights::Metadata::Map& metadata) {
  auto& file = shared_context.shared_weights.shared_bin_file.bin_file_;
  if (!file.is_open()) {
    ORT_THROW("Error opening shared bin file for writing weight as inputs metadata!")
  }

  try {
    size_t metadataSize = metadata.size();
    file.write(reinterpret_cast<const char*>(&metadataSize), sizeof(metadataSize));  // Write map size

    for (const auto& [key, value] : metadata) {
      writeString(file, key.name);
      writeString(file, value.location);
      file.write(reinterpret_cast<const char*>(&value.data_offset), sizeof(value.data_offset));
      file.write(reinterpret_cast<const char*>(&value.size), sizeof(value.size));
      file.write(reinterpret_cast<const char*>(&value.element_type), sizeof(value.element_type));
      writeVector(file, value.dimensions);
    }
  } catch (const Exception& e) {
    ORT_THROW("Error: Failed to write map data.", e.what());
  } catch (...) {
    ORT_THROW("Error: Failed to write map data.");
  }
}

//  Write the entire subgraph metadata map to a binary file
void SharedContext::SharedWeights::SubgraphMetadata::writeSubgraphDataToBinaryFile(SharedContext& shared_context,
                                                                                   const SharedContext::SharedWeights::SubgraphMetadata::Map& subgraph_metadata) {
  auto& file = shared_context.shared_weights.shared_bin_file.bin_file_;
  if (!file.is_open()) {
    ORT_THROW("Error opening shared bin file for writing subgraph metadata!");
  }
  try {
    size_t subgraph_metadataSize = subgraph_metadata.size();
    file.write(reinterpret_cast<const char*>(&subgraph_metadataSize), sizeof(subgraph_metadataSize));  // Write map size

    for (const auto& [key, value] : subgraph_metadata) {
      writeString(file, key.name);
      file.write(reinterpret_cast<const char*>(&value.epctx_offset), sizeof(value.epctx_offset));
      file.write(reinterpret_cast<const char*>(&value.epctx_length), sizeof(value.epctx_length));
    }

  } catch (const Exception& e) {
    ORT_THROW("Error: Failed to write map data.", e.what());
  } catch (...) {
    ORT_THROW("Error: Failed to write map data.");
  }
}

// Utility function to read a string
std::string readString(std::fstream& file) {
  size_t length;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));  // Read string size
  std::string str(length, '\0');
  file.read(&str[0], length);  // Read string content
  return str;
}

// Utility function to read a vector<size_t>
std::vector<size_t> readVector(std::fstream& file) {
  size_t length;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));  // Read vector size
  std::vector<size_t> vec(length);
  file.read(reinterpret_cast<char*>(vec.data()), length * sizeof(size_t));  // Read vector elements
  return vec;
}

// Read the Metadata map from a binary file
void SharedContext::SharedWeights::Metadata::readMetadataFromBinaryFile(SharedContext& shared_context,
                                                                        SharedContext::SharedWeights::Metadata::Map& metadata) {
  auto& file = shared_context.shared_weights.shared_bin_file.bin_file_;
  if (!file) {
    ORT_THROW("Error opening shared bin file for reading weight as input metadata!");
  }

  size_t metadata_mapSize;

  file.read(reinterpret_cast<char*>(&metadata_mapSize), sizeof(metadata_mapSize));  // Read map size

  for (size_t i = 0; i < metadata_mapSize; ++i) {
    SharedContext::SharedWeights::Metadata::Key key;
    SharedContext::SharedWeights::Metadata::Value value;

    key.name = readString(file);        // Read key (name)
    value.location = readString(file);  // Read location
    file.read(reinterpret_cast<char*>(&value.data_offset), sizeof(value.data_offset));
    file.read(reinterpret_cast<char*>(&value.size), sizeof(value.size));
    file.read(reinterpret_cast<char*>(&value.element_type), sizeof(value.element_type));
    value.dimensions = readVector(file);  // Read vector dimensions

    metadata[key] = value;
  }
}

// Read the Subgraph Metadata map from a binary file
void SharedContext::SharedWeights::SubgraphMetadata::readSubgraphDataFromBinaryFile(SharedContext& shared_context,
                                                                                    SharedContext::SharedWeights::SubgraphMetadata::Map& subgraph_metadata) {
  auto& file = shared_context.shared_weights.shared_bin_file.bin_file_;
  if (!file) {
    ORT_THROW("Error opening shared bin file for reading subgraph metadata!");
  }

  size_t subgraph_metadata_mapSize;
  file.read(reinterpret_cast<char*>(&subgraph_metadata_mapSize), sizeof(subgraph_metadata_mapSize));  // Read map size
  for (size_t i = 0; i < subgraph_metadata_mapSize; ++i) {
    SharedContext::SharedWeights::SubgraphMetadata::Key key;
    SharedContext::SharedWeights::SubgraphMetadata::Value value;

    key.name = readString(file);  // Read key (name)
    file.read(reinterpret_cast<char*>(&value.epctx_offset), sizeof(value.epctx_offset));
    file.read(reinterpret_cast<char*>(&value.epctx_length), sizeof(value.epctx_length));
    subgraph_metadata[key] = value;
  }
}

void SharedContext::SharedWeights::SharedBinFile::readBinFile(SharedContext& shared_context_) {
  auto& header = shared_context_.shared_weights.header_;
  auto& footer = shared_context_.shared_weights.footer_;
  auto& subgraph_metadata_map = shared_context_.shared_weights.subgraph_metadata;
  auto& metadata_map = shared_context_.shared_weights.metadata;
  auto& sb = shared_context_.shared_weights.shared_bin_file;
  try {
    if (sb.bin_file_.is_open()) {
      auto header_size = sizeof(SharedContext::SharedWeights::Header);
      if (sb.bin_size_ > header_size) {
        sb.bin_file_.read(reinterpret_cast<char*>(&header), header_size);
      }
      auto footer_size = sizeof(SharedContext::SharedWeights::Footer);
      if (header.footer_offset < sb.bin_size_ && footer_size <= sb.bin_size_ &&
          (header.footer_offset <= sb.bin_size_ - footer_size)) {
        sb.bin_file_.seekp(header.footer_offset, std::ios::beg);
        sb.bin_file_.read(reinterpret_cast<char*>(&footer), footer_size);
      }

      if (footer.subgraph_offset < sb.bin_size_ && footer.subgraph_length <= sb.bin_size_ &&
          (footer.subgraph_offset <= sb.bin_size_ - footer.subgraph_length)) {
        sb.bin_file_.seekp(footer.subgraph_offset, std::ios::beg);
        shared_context_.shared_weights.subgraph_metadata_.readSubgraphDataFromBinaryFile(shared_context_, subgraph_metadata_map);
      }
      if (footer.metadata_offset < sb.bin_size_ && footer.metadata_length <= sb.bin_size_ &&
          (footer.metadata_offset <= sb.bin_size_ - footer.metadata_length)) {
        sb.bin_file_.seekp(footer.metadata_offset, std::ios::beg);
        shared_context_.shared_weights.metadata_.readMetadataFromBinaryFile(shared_context_, metadata_map);
      }
    }
  } catch (std::string msg) {
    ORT_THROW(msg);
  }
}

void SharedContext::SharedWeights::SharedBinFile::dumpBinFile(SharedContext& shared_context_) {
  auto& header = shared_context_.shared_weights.header_;
  auto& footer = shared_context_.shared_weights.footer_;
  auto& subgraph_metadata_map = shared_context_.shared_weights.subgraph_metadata;
  auto& metadata_map = shared_context_.shared_weights.metadata;
  auto& sb = shared_context_.shared_weights.shared_bin_file;
  auto& bin_file = sb.bin_file_;
  try {
    if (bin_file.is_open()) {
      footer.subgraph_offset = static_cast<uint64_t>(bin_file.tellp());
      shared_context_->shared_weights.subgraph_metadata_.writeSubgraphDataToBinaryFile(*shared_context_, subgraph_metadata);
      footer.metadata_offset = static_cast<uint64_t>(bin_file.tellp());
      footer.subgraph_length = static_cast<size_t>(footer.metadata_offset - footer.subgraph_offset);
      shared_context_->shared_weights.metadata_.writeMetadataToBinaryFile(*shared_context_, metadata);
      header.footer_offset = static_cast<uint64_t>(bin_file.tellp());
      footer.metadata_length = static_cast<size_t>(header.footer_offset - footer.metadata_offset);

      // Write footer to the bin file
      bin_file.write(reinterpret_cast<char*>(&footer), sizeof(SharedContext::SharedWeights::Footer));
      // Update header with Footer offset at the end
      bin_file.seekp(0, std::ios::beg);
      bin_file.write(reinterpret_cast<char*>(&header), sizeof(SharedContext::SharedWeights::Header));
      bin_file.close();
    }
  } catch (std::string msg) {
    ORT_THROW(msg);
  }
}
}  // namespace openvino_ep
}  // namespace onnxruntime
