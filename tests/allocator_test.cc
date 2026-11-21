#include <cstdint>

#include <gtest/gtest.h>

#include "ctranslate2/allocator.h"
#include "cpu/backend.h"

using namespace ctranslate2;

TEST(AllocatorTest, FollowsMklBackend) {
  EXPECT_EQ(cpu::allocator_uses_mkl(), cpu::mayiuse_mkl());
}

TEST(AllocatorTest, ReturnsAlignedMemory) {
  auto& allocator = get_allocator<Device::CPU>();
  void* ptr = allocator.allocate(128);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0);
  allocator.free(ptr);
}
