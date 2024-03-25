#include <iostream>
#include <filesystem>
#include <ctranslate2/translator.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/logging.h>

void execute_translation(ctranslate2::Translator &translator, const std::vector<std::vector<std::string>>& batch, const std::string& msg) {
    std::cout << "input data: " << std::endl;
    for (const auto &input: batch) {
        for (const auto &word: input) {
            std::cout << word << ' ';
        }
        std::cout << "\n";
    }

    std::cout << "Start: " << msg << " examples\n";
    // const auto start{std::chrono::steady_clock::now()};
    const std::vector <ctranslate2::TranslationResult> results = translator.translate_batch(batch);
    // const auto end{std::chrono::steady_clock::now()};
    // std::cout << "End: " << msg << " examples. time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    std::cout << "output: " << std::endl;
    for (const auto &token: results[0].output())
        std::cout << token << ' ';
    std::cout << std::endl;
}

int main(int, char* argv[]) {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "current path: " << cwd << std::endl;
    const std::string input_data_path = argv[1];
    std::cout << "input data path: " << std::filesystem::absolute(input_data_path) << std::endl;

    if(!std::filesystem::exists(input_data_path)) {
        std::cout << input_data_path << " does not exist" << std::endl;
    }

    ctranslate2::set_log_level(ctranslate2::LogLevel::Info);

    const auto device = ctranslate2::str_to_device("auto");
    ctranslate2::Translator translator(input_data_path, device);
    const std::vector <std::vector<std::string>> batch = {{"▁H", "ello", "▁world", "!"}};
    execute_translation(translator, batch, "Warmup");
    execute_translation(translator, batch, "Query");

    return 0;
}

