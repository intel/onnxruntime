// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

template <typename Key, typename Resource>
class SharedResourceManager {
 public:
    // Create or return the active resource for the given key.
    // If the resource for 'key' does not exist it will be constructed with
    // the provided constructor arguments.
    template <typename... Args>
    std::shared_ptr<Resource> GetActiveResourceOrCreate(const Key& key, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_resource_) {
            return active_resource_;
        }
        auto [it, inserted] = resources_.try_emplace(key, nullptr);
        if (inserted) {
            it->second = std::make_shared<Resource>(std::forward<Args>(args)...);
        }
        active_resource_ = it->second;
        return it->second;
    }

    // Get or create a resource for 'key' and make it the active resource.
    // If the resource for 'key' does not exist it will be constructed with
    // the provided constructor arguments.
    template <typename... Args>
    std::shared_ptr<Resource> GetOrCreateResource(const Key& key, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = resources_.try_emplace(key, nullptr);
        if (inserted) {
            it->second = std::make_shared<Resource>(std::forward<Args>(args)...);
        }
        return it->second;
    }

    void ClearActiveResource() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_resource_ = nullptr;
    }

 private:
    mutable std::mutex mutex_;
    std::unordered_map<Key, std::shared_ptr<Resource>> resources_;
    std::shared_ptr<Resource> active_resource_;
};
