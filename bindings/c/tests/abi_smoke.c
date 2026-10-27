// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

#include "tikv_client_async_c.h"
#include "tikv_client_metrics_c.h"

#include <assert.h>
#include <string.h>

/* Run through the public C headers and the built library without a cluster.
 * The RawKV example is also compiled and linked in CI to cover its API symbols.
 */
int main(void) {
  tikv_async_client_t *client = NULL;
  tikv_error_t *error = NULL;

  /* Invalid endpoints fail before connecting to PD. Check enum return values,
   * output pointers, and Rust-owned error strings across the C ABI.
   */
  tikv_status_t status = tikv_async_client_new_with_grpc_max_decoding_message_size(
      "", 0, false, 1024, &client, &error);
  assert(status == TIKV_STATUS_INVALID_ARGUMENT);
  assert(client == NULL);
  assert(error != NULL);
  const char *message = tikv_error_message(error);
  assert(message != NULL);
  assert(strstr(message, "at least one endpoint") != NULL);
  tikv_error_destroy(error);
  error = NULL;

  /* This fresh process has no RPC metrics. Check the C array layout and empty
   * view contract, and release the snapshot using its public destroy function.
   * Populated samples and histogram values are covered by the Rust tests.
   */
  tikv_metrics_snapshot_t *snapshot = NULL;
  tikv_metric_samples_t samples = {NULL, 123};
  status = tikv_metrics_snapshot_collect(&snapshot, &samples, &error);
  assert(status == TIKV_STATUS_OK);
  assert(error == NULL);
  assert(snapshot != NULL);
  assert(samples.ptr == NULL);
  assert(samples.len == 0);
  tikv_metrics_snapshot_destroy(snapshot);
  return 0;
}
