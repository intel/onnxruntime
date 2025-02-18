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
    try{
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
        size_t subgraph_metadataSize = subgraph_metadata.size();
        file.write(reinterpret_cast<const char*>(&subgraph_metadataSize), sizeof(subgraph_metadataSize));  // Write map size

        for (const auto& [key, value] : subgraph_metadata) {
            writeString(file, key.name);
            file.write(reinterpret_cast<const char*>(&value.epctx_offset), sizeof(value.epctx_offset));
            file.write(reinterpret_cast<const char*>(&value.epctx_length), sizeof(value.epctx_length));
        }
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

// Read the entire map from a binary file
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

    // file.close();
    std::cout << "Map read from binary file successfully!\n";
}

// Read the entire map from a binary file
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
        file.read(reinterpret_cast<char*>(&value.epctx_offset), sizeof(value.epctx_offset));
        file.read(reinterpret_cast<char*>(&value.epctx_length), sizeof(value.epctx_length));

        subgraph_metadata[key] = value;
    }

    // file.close();
    std::cout << "Map read from binary file successfully!\n";
}
// std::ostream& operator<<(std::ostream& stream, const SharedContext::SharedWeights::Metadata::Map& metadata) {
//   try {
//     stream << metadata.size();

//     // Write each key-value pair
//     // Put elements in separate lines to facilitate reading
//     for (const auto& [key, value] : metadata) {
//       stream << std::endl
//              << key.name;
//       stream << std::endl
//              << value.location;
//       stream << std::endl
//              << value.data_offset;
//       stream << std::endl
//              << value.size;
//       stream << std::endl
//              << value.dimensions.size();
//       for (const auto& dim : value.dimensions) {
//         stream << std::endl
//                << dim;
//       }
//       stream << std::endl
//              << value.element_type;
//     }
//   } catch (const Exception& e) {
//     ORT_THROW("Error: Failed to write map data.", e.what());
//   } catch (...) {
//     ORT_THROW("Error: Failed to write map data.");
//   }

//   ORT_ENFORCE(stream.good(), "Error: Failed to write map data.");
//   return stream;
// }

// std::ostream& operator<<(std::ostream& stream,
//                          const SharedContext::SharedWeights::SubgraphMetadata::Map& subgraph_metadata) {
//   try {
//     stream << subgraph_metadata.size();

//     // Write each key-value pair
//     // Put elements in separate lines to facilitate reading
//     for (const auto& [key, value] : subgraph_metadata) {
//       stream << std::endl
//              << key.name;
//       stream << std::endl
//              << value.epctx_offset;
//       stream << std::endl
//              << value.epctx_length;
//     }
//   } catch (const Exception& e) {
//     ORT_THROW("Error: Failed to write subgraph map data.", e.what());
//   } catch (...) {
//     ORT_THROW("Error: Failed to write subgraph map data.");
//   }
//   ORT_ENFORCE(stream.good(), "Error: Failed to write subgraph map data.");
//   return stream;
// }

// std::istream& operator>>(std::istream& stream, SharedContext::SharedWeights::Metadata::Map& metadata) {
//   size_t map_size{0};
//   try {
//     stream >> map_size;

//     while (!stream.eof()) {
//       SharedContext::SharedWeights::Metadata::Key key;
//       SharedContext::SharedWeights::Metadata::Value value;
//       stream >> key.name;
//       stream >> value.location;
//       stream >> value.data_offset;
//       stream >> value.size;
//       size_t num_dimensions;
//       stream >> num_dimensions;

//       if (stream.fail()) {
//         ORT_THROW("Error: Failed to read num_dimensions from stream.");
//       }

//       constexpr size_t MAX_SAFE_DIMENSIONS = 1024;

//       size_t safe_num_dimensions = num_dimensions;

//       if(num_dimensions == 0 || safe_num_dimensions > MAX_SAFE_DIMENSIONS) {
//          ORT_THROW("Invalid number of dimensions provided.");
//       }
//       try {
//           value.dimensions.resize(safe_num_dimensions);
//       } catch (const std::bad_alloc&) {
//           ORT_THROW("Error: Memory allocation failed while resizing dimensions.");
//       }

//       for (auto& dim : value.dimensions) {
//         stream >> dim;
//       }
//       stream >> value.element_type;
//       metadata.emplace(key, value);
//     }
//   } catch (const Exception& e) {
//     ORT_THROW("Error: Failed to read map data.", e.what());
//   } catch (...) {
//     ORT_THROW("Error: Failed to read map data.");
//   }

//   ORT_ENFORCE(metadata.size() == map_size, "Error: Inconsistent map data.");

//   return stream;
// }
// std::istream& operator>>(std::istream& stream, SharedContext::SharedWeights::SubgraphMetadata::Map& subgraph_metadata) {
//   size_t map_size{0};
//   try {
//     stream >> map_size;

//     while (!stream.eof()) {
//       SharedContext::SharedWeights::SubgraphMetadata::Key key;
//       SharedContext::SharedWeights::SubgraphMetadata::Value value;
//       stream >> key.name;
//       stream >> value.epctx_offset;
//       stream >> value.epctx_length;

//       subgraph_metadata.emplace(key, value);
//     }
//   } catch (const Exception& e) {
//     ORT_THROW("Error: Failed to read map data.", e.what());
//   } catch (...) {
//     ORT_THROW("Error: Failed to read map data.");
//   }

//   ORT_ENFORCE(subgraph_metadata.size() == map_size, "Error: Inconsistent map data.");

//   return stream;
// }

}
}
