#include "ctranslate2/encoder.h"

namespace ctranslate2 {

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(std::vector<std::vector<std::string>> tokens) {
    return post<EncoderForwardOutput>(
      [tokens = std::move(tokens)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(tokens);
      });
  }

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(std::vector<std::vector<size_t>> ids) {
    return post<EncoderForwardOutput>(
      [ids = std::move(ids)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(ids);
      });
  }

  std::future<EncoderForwardOutput>
  Encoder::forward_batch_async(StorageView ids, StorageView lengths) {
    return post<EncoderForwardOutput>(
      [ids = std::move(ids), lengths = std::move(lengths)]
      (models::SequenceEncoderReplica& encoder) {
        return encoder.forward(ids, lengths);
      });
  }

}
