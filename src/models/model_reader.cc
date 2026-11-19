#include "ctranslate2/models/model_reader.h"

#include "ctranslate2/filesystem.h"

namespace ctranslate2 {
  namespace models {

    std::unique_ptr<std::istream> ModelReader::get_required_file(const std::string& filename,
                                                                 const bool binary) {
      std::unique_ptr<std::istream> file = get_file(filename, binary);
      if (!file)
        throw std::runtime_error("Unable to open file '" + filename
                                 + "' in model '" + get_model_id() + "'");
      return file;
    }


    ModelFileReader::ModelFileReader(std::string model_dir)
      : _model_dir(std::move(model_dir))
    {
    }

    std::string ModelFileReader::get_model_id() const {
      return _model_dir;
    }

    std::unique_ptr<std::istream> ModelFileReader::get_file(const std::string& filename,
                                                            const bool binary) {
      const std::string path = _model_dir + "/" + filename;
      const std::ios_base::openmode mode = binary ? std::ios_base::binary : std::ios_base::in;
      auto stream = std::make_unique<std::ifstream>(open_file_read(path, mode, /*check=*/false));
      if (!stream || !(*stream))
        return nullptr;
      return stream;
    }


    struct membuf : std::streambuf {
      membuf(const char* base, size_t size) {
        char* p = const_cast<char*>(base);
        setg(p, p, p + size);
      }

      std::streampos seekoff(std::streamoff off,
                             std::ios_base::seekdir dir,
                             std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in))
          return std::streampos(-1);

        char* begin = eback();
        char* end = egptr();
        std::streamoff base = 0;
        if (dir == std::ios_base::beg)
          base = 0;
        else if (dir == std::ios_base::cur)
          base = gptr() - begin;
        else if (dir == std::ios_base::end)
          base = end - begin;
        else
          return std::streampos(-1);

        const std::streamoff next = base + off;
        if (next < 0 || next > end - begin)
          return std::streampos(-1);

        setg(begin, begin + next, end);
        return next;
      }

      std::streampos seekpos(std::streampos pos, std::ios_base::openmode which) override {
        return seekoff(static_cast<std::streamoff>(pos), std::ios_base::beg, which);
      }
    };

    struct imemstream : virtual membuf, std::istream {
      imemstream(const char* base, size_t size)
        : membuf(base, size)
        , std::istream(static_cast<std::streambuf*>(this))
      {
      }
    };


    ModelMemoryReader::ModelMemoryReader(std::string model_name)
      : _model_name(std::move(model_name))
    {
    }

    void ModelMemoryReader::register_file(std::string filename, std::string content) {
      _files.emplace(std::move(filename), std::move(content));
    }

    std::string ModelMemoryReader::get_model_id() const {
      return _model_name;
    }

    std::unique_ptr<std::istream> ModelMemoryReader::get_file(const std::string& filename,
                                                              const bool) {
      const auto it = _files.find(filename);

      if (it == _files.end())
        return nullptr;

      const auto& content = it->second;
      return std::make_unique<imemstream>(content.data(), content.size());
    }


    std::shared_ptr<Vocabulary>
    load_vocabulary(ModelReader& model_reader,
                    const std::string& filename,
                    VocabularyInfo vocab_info) {
      std::unique_ptr<std::istream> file;

      file = model_reader.get_file(filename + ".json");
      if (file)
        return std::make_shared<Vocabulary>(Vocabulary::from_json_file(*file, std::move(vocab_info)));

      file = model_reader.get_file(filename + ".txt");
      if (file)
        return std::make_shared<Vocabulary>(Vocabulary::from_text_file(*file, std::move(vocab_info)));

      return nullptr;
    }

  }
}
