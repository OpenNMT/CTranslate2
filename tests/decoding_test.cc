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

TEST(BiasDecodingTest, BiasTokens) {
  StorageView input({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  StorageView expected({2, 5}, std::vector<float>{1, 3, 6, 4, 5, 6, 7, 48, 27, 10});
  BiasTokens bias_tokens(input);

  bias_tokens.add(2, 2.0);
  bias_tokens.add(0, 1, 1.5);
  bias_tokens.add(1, 3, 3.0);
  bias_tokens.add(1, 2, 3.0);
  bias_tokens.apply();

  expect_storage_eq(input, expected);
}
