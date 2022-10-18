#pragma once

#include "model.h"

namespace ctranslate2 {
  namespace models {

    class ModelFactory {
    public:
      static ModelFactory& get_instance() {
        static ModelFactory factory;
        return factory;
      }

      template <typename Model, typename... Args>
      void register_model(const std::string& name, Args&&... args) {
        Builder builder = [args...]() { return std::make_shared<Model>(args...); };
        _registry.emplace(name, std::move(builder));
      }

      std::shared_ptr<Model> create(const std::string& name) const {
        auto it = _registry.find(name);
        if (it == _registry.end())
          throw std::invalid_argument("Unknown model " + name);
        return it->second();
      }

    private:
      ModelFactory() = default;

      using Builder = std::function<std::shared_ptr<models::Model>(void)>;
      std::unordered_map<std::string, Builder> _registry;
    };


    template <typename Model>
    class RegisterModel {
    public:
      template <typename... Args>
      RegisterModel(const std::string& name, Args&&... args) {
        ModelFactory::get_instance().register_model<Model>(name, std::forward<Args>(args)...);
      }
    };

  }
}
