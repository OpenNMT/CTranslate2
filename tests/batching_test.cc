#include <ctranslate2/batch_reader.h>
#include <ctranslate2/utils.h>

#include "test_utils.h"

TEST(BatchingTest, RebatchInput) {
  const std::vector<std::vector<std::string>> source = {
    {"a", "b"},
    {"a", "b", "c"},
    {"a"},
    {},
    {"a", "b", "c", "d"},
    {"a", "b", "c", "d", "e"}
  };
  const std::vector<std::vector<std::string>> target = {
    {"1"},
    {"2"},
    {"3"},
    {"4"},
    {"5"},
    {"6"}
  };
  const std::vector<std::vector<size_t>> expected_batches = {
    {5, 4},
    {1, 0},
    {2, 3}
  };

  const auto batches = rebatch_input(load_examples({source, target}), 2, BatchType::Examples);
  ASSERT_EQ(batches.size(), expected_batches.size());

  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    EXPECT_EQ(batch.get_stream(0), index_vector(source, expected_batches[i]));
    EXPECT_EQ(batch.get_stream(1), index_vector(target, expected_batches[i]));
    EXPECT_EQ(batch.example_index, expected_batches[i]);
  }
}

TEST(BatchingTest, BatchReaderGetNext_Examples) {
  const std::vector<std::vector<std::string>> examples = {
    {"a", "b"},
    {"a", "b", "c"},
    {"a"},
    {"a", "b", "c", "d"}
  };
  const std::vector<std::vector<size_t>> expected_batches = {{0, 1}, {2, 3}};

  VectorReader reader(examples);

  for (const auto& expected_batch : expected_batches) {
    auto batch = reader.get_next(2, BatchType::Examples, true);
    ASSERT_EQ(batch.size(), expected_batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      EXPECT_EQ(batch[i].streams[0], examples[expected_batch[i]]);
    }
  }
}

TEST(BatchingTest, BatchReaderGetNext_TokensFixed) {
  const std::vector<std::vector<std::string>> source = {
    {"a", "b", "c", "d"},
    {"a", "b", "c", "d", "e"},
    {"a"},
    {"a", "b", "c"},
    {"a", "b"}
  };
  const std::vector<std::vector<std::string>> target = {
    {"1"},
    {"2"},
    {"3"},
    {"4"},
    {"5"}
  };

  const std::vector<std::vector<size_t>> expected_batches = {{1}, {0}, {3, 4}, {2}};

  const auto batches = rebatch_input(load_examples({source, target}), 6, BatchType::Tokens);
  ASSERT_EQ(batches.size(), expected_batches.size());

  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    EXPECT_EQ(batch.get_stream(0), index_vector(source, expected_batches[i]));
    EXPECT_EQ(batch.get_stream(1), index_vector(target, expected_batches[i]));
    EXPECT_EQ(batch.example_index, expected_batches[i]);
  }
}

TEST(BatchingTest, BatchReaderGetNext_TokensDynamic) {
  const std::vector<std::vector<std::string>> examples = {
    {"a", "b"},
    {"a", "b", "c"},
    {"a"},
    {"a", "b", "c", "d"},
    {"a", "b", "c", "d", "e"}
  };

  const std::vector<std::vector<size_t>> expected_batches = {{0, 1, 2}, {3}, {4}};

  VectorReader reader(examples);

  for (const auto& expected_batch : expected_batches) {
    auto batch = reader.get_next(6, BatchType::Tokens, false);
    ASSERT_EQ(batch.size(), expected_batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      EXPECT_EQ(batch[i].streams[0], examples[expected_batch[i]]);
    }
  }
}

TEST(BatchingTest, BatchReaderGetNext_TokensFixed2) {
  const std::vector<std::vector<std::string>> source = {
    {"a", "b", "c", "d", "e"},
    {"a", "b"},
    {"a"}
  };
  const std::vector<std::vector<std::string>> target = {
    {"1"},
    {"2"},
    {"3"}
  };

  const std::vector<std::vector<size_t>> expected_batches = {{0}, {1, 2}};
  const auto batches = rebatch_input(load_examples({source, target}), 8, BatchType::Tokens);
  ASSERT_EQ(batches.size(), expected_batches.size());

  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    EXPECT_EQ(batch.get_stream(0), index_vector(source, expected_batches[i]));
    EXPECT_EQ(batch.get_stream(1), index_vector(target, expected_batches[i]));
    EXPECT_EQ(batch.example_index, expected_batches[i]);
  }
}

TEST(BatchingTest, BatchReaderGetNext_TokensDynamic2) {
  const std::vector<std::vector<std::string>> source = {
    {"a", "b", "c", "d", "e"},
    {"a", "b"},
    {"a"}
  };

  const std::vector<std::vector<size_t>> expected_batches = {{0, 1, 2}};
  VectorReader reader(source);

  for (const auto& expected_batch : expected_batches) {
    auto batch = reader.get_next(8, BatchType::Tokens, false);
    ASSERT_EQ(batch.size(), expected_batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      EXPECT_EQ(batch[i].streams[0], source[expected_batch[i]]);
    }
  }
}