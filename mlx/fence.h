// Copyright © 2024 Apple Inc.

// TODO
#include "mlx/backend/metal/device.h"

namespace mlx::core {

/* A fence to be used for synchronizing work between streams.
 *
 * Calls to `update_gpu` should be paired with calls to `wait`. This ensures
 * that the array passed to `update_gpu` is computed and visible to the CPU
 * after the call to `wait` returns.
 *
 * Calls to `update` should be paired with calls to `wait_gpu`. This ensures
 * that the array passed to `wait_gpu` will not be read by the GPU until the CPU
 * has called `update`.
 *
 * The fence supports slow (default) and fast mode. Fast mode requires setting
 * the environment variable `MLX_METAL_FAST_SYNCH=1`. Fast mode also requires
 * Metal 3.2+ (macOS 15+, iOS 18+).
 */
class Fence {
 public:
  Fence();

  void update(Stream stream, const array& x);
  void wait(Stream stream, array& x);

 private:
  std::shared_ptr<void> fence_;
  uint32_t count_{0};
  bool use_fast_;
  std::atomic_uint* cpu_value();
};

} // namespace mlx::core
