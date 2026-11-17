#include <ctranslate2/decoding.h>

#include "test_utils.h"

TEST(DecodingTest, DisableTokens) {
  StorageView input({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  StorageView expected({2, 5}, std::vector<float>{1, 0, 0, 4, 5, 6, 7, 0, 9, 0});

  DisableTokens disable_tokens(input, 0);
  disable_tokens.add(2);
  disable_tokens.add(0, 1);
  disable_tokens.add(1, 4);
  disable_tokens.apply();

  expect_storage_eq(input, expected);
}

TEST(DecodingTest, DisableTokensRejectsOutOfRangeIds) {
  const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  StorageView input({2, 5}, values);
  StorageView expected({2, 5}, values);

  DisableTokens disable_tokens(input, 0);
  EXPECT_THROW(disable_tokens.add(5), std::invalid_argument);
  EXPECT_THROW(disable_tokens.add(1 << 20), std::invalid_argument);
  EXPECT_THROW(disable_tokens.add(-1), std::invalid_argument);
  EXPECT_THROW(disable_tokens.add(static_cast<dim_t>(static_cast<size_t>(-1))),
               std::invalid_argument);
  EXPECT_THROW(disable_tokens.add(1, 5), std::invalid_argument);
  disable_tokens.apply();

  expect_storage_eq(input, expected);
}
