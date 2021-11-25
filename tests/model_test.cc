#include <ctranslate2/models/sequence_to_sequence.h>

#include "test_utils.h"

TEST(ModelTest, ContainsModel) {
  ASSERT_TRUE(models::contains_model(default_model_dir()));
}

TEST(ModelTest, LoadReplicas) {
  const auto replicas = models::load_replicas(default_model_dir(),
                                              Device::CPU,
                                              {0, 0},
                                              ComputeType::DEFAULT);

  // The replicas should use the same model weights but be different model instances.
  ASSERT_EQ(replicas.size(), 2);
  EXPECT_NE(replicas[0], replicas[1]);

  const std::string weight_name = "decoder/projection/weight";
  EXPECT_EQ(replicas[0]->get_variable(weight_name).buffer(),
            replicas[1]->get_variable(weight_name).buffer());

  for (const auto replica : replicas) {
    const auto* seq2seq = dynamic_cast<const models::SequenceToSequenceModel*>(replica.get());
    ASSERT_NE(seq2seq, nullptr);

    const auto results = seq2seq->sample({{"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}});
    EXPECT_EQ(results[0].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  }
}
