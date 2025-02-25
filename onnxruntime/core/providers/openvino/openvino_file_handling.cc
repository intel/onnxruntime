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
    auto &file = shared_context.shared_weights.shared_bin_file.bin_file_;
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    std::cout << " bin file location in write for metadata map = " << file.tellp() << std::endl;

    try{
        size_t metadataSize = metadata.size();
        file.write(reinterpret_cast<const char*>(&metadataSize), sizeof(metadataSize));  // Write map size
        std::cout << " bin file location after write metadata size = " << file.tellp() << std::endl;

        for (const auto& [key, value] : metadata) {
            writeString(file, key.name);
            writeString(file, value.location);
            file.write(reinterpret_cast<const char*>(&value.data_offset), sizeof(value.data_offset));
            file.write(reinterpret_cast<const char*>(&value.size), sizeof(value.size));
            file.write(reinterpret_cast<const char*>(&value.element_type), sizeof(value.element_type));
            writeVector(file, value.dimensions);
        }
        std::cout << " bin file location after write metadata map = " << file.tellp() << std::endl;
    }  catch (const Exception& e) {
        ORT_THROW("Error: Failed to write map data.", e.what());
    } catch (...) {
        ORT_THROW("Error: Failed to write map data.");
    }
    std::cout << "Map written to binary file successfully!\n";
}

//  Write the entire subgraph metadata map to a binary file
void SharedContext::SharedWeights::SubgraphMetadata::writeSubgraphDataToBinaryFile(SharedContext& shared_context,
                                   const SharedContext::SharedWeights::SubgraphMetadata::Map& subgraph_metadata) {
    auto &file = shared_context.shared_weights.shared_bin_file.bin_file_;
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    try{
        std::cout << " bin file location at write subgraph metadata = " << file.tellp() << std::endl;
        size_t subgraph_metadataSize = subgraph_metadata.size();
        file.write(reinterpret_cast<const char*>(&subgraph_metadataSize), sizeof(subgraph_metadataSize));  // Write map size
        std::cout << " bin file location at write subgraph metadata after metadata size = " << file.tellp() << std::endl;

        for (const auto& [key, value] : subgraph_metadata) {
            writeString(file, key.name);
            file.write(reinterpret_cast<const char*>(&value.epctx_offset), sizeof(value.epctx_offset));
            file.write(reinterpret_cast<const char*>(&value.epctx_length), sizeof(value.epctx_length));
        }
        std::cout << " bin file location after write subgraph metadata = " << file.tellp() << std::endl;

        // std::cout << " File position after writing subgraph metadata = " << file.tellp() << std::end;
    }  catch (const Exception& e) {
        ORT_THROW("Error: Failed to write map data.", e.what());
    } catch (...) {
        ORT_THROW("Error: Failed to write map data.");
    }
    std::cout << "Map written to binary file successfully!\n";
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
    auto &file = shared_context.shared_weights.shared_bin_file.bin_file_;
    if (!file) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    size_t metadata_mapSize;

    file.read(reinterpret_cast<char*>(&metadata_mapSize), sizeof(metadata_mapSize));  // Read map size

    for (size_t i = 0; i < metadata_mapSize; ++i) {
        SharedContext::SharedWeights::Metadata::Key key;
        SharedContext::SharedWeights::Metadata::Value value;

        key.name = readString(file);  // Read key (name)
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
    auto &file = shared_context.shared_weights.shared_bin_file.bin_file_;
    if (!file) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    size_t subgraph_metadata_mapSize;
    file.read(reinterpret_cast<char*>(&subgraph_metadata_mapSize), sizeof(subgraph_metadata_mapSize));  // Read map size
    for (size_t i = 0; i < subgraph_metadata_mapSize; ++i) {
        SharedContext::SharedWeights::SubgraphMetadata::Key key;
        SharedContext::SharedWeights::SubgraphMetadata::Value value;

        key.name = readString(file);  // Read key (name)
        std::cout << " key.name = " << key.name << std::endl;
        file.read(reinterpret_cast<char*>(&value.epctx_offset), sizeof(value.epctx_offset));
        std::cout << "value.epctx_offset = " << value.epctx_offset << std::endl;
        file.read(reinterpret_cast<char*>(&value.epctx_length), sizeof(value.epctx_length));
        std::cout << " value.epctx_length = " << value.epctx_length << std::endl;
        subgraph_metadata[key] = value;
    }
}

void SharedContext::SharedWeights::SharedBinFile::readBinFile(SharedContext& shared_context_) {
    auto &header = shared_context_.shared_weights.header_;
    auto &footer = shared_context_.shared_weights.footer_;
    auto& subgraph_metadata_map = shared_context_.shared_weights.subgraph_metadata;
    auto& metadata_map = shared_context_.shared_weights.metadata;
    auto &sb = shared_context_.shared_weights.shared_bin_file;
    if(sb.bin_file_.is_open()) {
      auto header_size = sizeof(SharedContext::SharedWeights::Header);
      std::cout << " sb.bin_size_ " << sb.bin_size_ << std::endl;
      std::cout << " header_size " << header_size << std::endl;
      if(sb.bin_size_ > header_size){
        sb.bin_file_.read(reinterpret_cast<char*>(&header), header_size);
        std::cout << " Footer offset from header = " << header.footer_offset << std::endl;
      }
      std::cout << " file position after reading header " << sb.bin_file_.tellp() << std::endl;
      auto footer_size = sizeof(SharedContext::SharedWeights::Footer);
      std::cout << " footer_size " << footer_size  << std::endl;
      if(header.footer_offset < sb.bin_size_ && footer_size <= sb.bin_size_ &&
        (header.footer_offset <= sb.bin_size_ - footer_size)) {
        sb.bin_file_.seekp(header.footer_offset, std::ios::beg);
        sb.bin_file_.read(reinterpret_cast<char*>(&footer), footer_size);
        std::cout << " subgraph metadata offset from footer = " << footer.subgraph_offset << std::endl;
        std::cout << " subgraph metadata length from footer = " << footer.subgraph_length << std::endl;
        std::cout << " metadata offset from footer = " << footer.metadata_offset << std::endl;
        std::cout << " metadata length from footer = " << footer.metadata_length << std::endl;
      }
      std::cout << " footer.subgraph_offset = " << footer.subgraph_offset << std::endl;

      if (footer.subgraph_offset < sb.bin_size_ && footer.subgraph_length <= sb.bin_size_ &&
        (footer.subgraph_offset <= sb.bin_size_ - footer.subgraph_length)) {
        std::cout << " inside if for reading subgraph metadata  = " << footer.subgraph_offset << std::endl;
        sb.bin_file_.seekp(footer.subgraph_offset, std::ios::beg);
        shared_context_.shared_weights.subgraph_metadata_.readSubgraphDataFromBinaryFile(shared_context_, subgraph_metadata_map);
        for (const auto& [key, value] : subgraph_metadata_map){
          std::cout << key.name << std::endl;
          std::cout << value.epctx_offset << std::endl;
          std::cout << value.epctx_length << std::endl;
        }
      }
      if (footer.metadata_offset < sb.bin_size_ && footer.metadata_length <= sb.bin_size_ &&
        (footer.metadata_offset <= sb.bin_size_ - footer.metadata_length)) {
        sb.bin_file_.seekp(footer.metadata_offset, std::ios::beg);
        shared_context_.shared_weights.metadata_.readMetadataFromBinaryFile(shared_context_, metadata_map);
        for (const auto& [key, value] : metadata_map){
          std::cout << key.name << std::endl;
          std::cout << value.location << std::endl;
          std::cout << value.data_offset << std::endl;
          std::cout << value.element_type << std::endl;
          std::cout << value.size << std::endl;
          for (const auto& dim : value.dimensions) {
            std::cout << dim << ", ";
          }
          std::cout << std::endl;
        }
      // exit(1);
      }
    //   // After reading the Subgraph map and metadata map move the file ptr to start of subgraph metadata map
    //   // so that epctx blobs that are not already exisiting in the bin file gets written.
    //   // Once all the epctx blobs are written, subgraph map and metadata map are written to the bin file along with updating header and footer.
    //   if(!subgraph_metadata_map.empty()){
    //     sb.bin_file_.seekp(footer.subgraph_offset, std::ios::beg);
    //     std::cout << " After setting bin file offset to the start of subgraph offset = " << sb.bin_file_.tellp() << std::endl;
    //   }
    }

}
}
}
