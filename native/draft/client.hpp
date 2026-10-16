#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One serial client per inference worker. Transport failures return zero draft
// tokens; protocol decisions and request timing are recorded in audit_path.
void *sf_draft_create(const char *directory, const char *url, int vocab_size, int max_length,
                      long timeout_ms, const char *audit_path);
const char *sf_draft_last_error(void);
// Returns 1 and writes a NUL-terminated 64-digit SHA256, or returns 0 on error.
int sf_draft_tokenizer_fingerprint(const char *directory, char output[65]);
int sf_draft_propose(void *handle, const int32_t *prefix, size_t length, size_t k, int32_t *output,
                     size_t capacity);
// Last proposal's JSON audit record, valid until the next call on this handle.
const char *sf_draft_last_result(void *handle);
void sf_draft_destroy(void *handle);

#ifdef __cplusplus
}
#endif
