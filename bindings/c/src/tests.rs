// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

use super::*;
use crate::client::build_config;
use crate::raw::{build_write_batch_runs, slice_mutations_to_owned};

fn synthetic_request(kind: RequestKind, output: RequestOutput) -> *mut tikv_async_request_t {
    Arc::into_raw(Arc::new(RequestState {
        kind,
        completion: RefCell::new(CompletionState {
            complete: true,
            completion_status: Some(tikv_status_t::TIKV_STATUS_OK),
            error_message: None,
            output: Some(output),
            output_taken: false,
        }),
        callback: None,
        userdata: ptr::null_mut(),
    })) as *mut tikv_async_request_t
}

#[test]
fn ffi_client_grpc_decoding_limit_is_configurable() {
    // Zero preserves the Rust default; explicit and unlimited FFI limits must
    // override it. Avoid pinning the upstream client's default size here.
    let default_limit = Config::default().grpc_max_decoding_message_size;
    assert_eq!(
        build_config(0, false, 0).grpc_max_decoding_message_size,
        default_limit
    );
    assert_eq!(
        build_config(0, false, 32 * 1024 * 1024).grpc_max_decoding_message_size,
        32 * 1024 * 1024
    );
    assert_eq!(
        build_config(0, false, usize::MAX).grpc_max_decoding_message_size,
        usize::MAX
    );
}

#[test]
fn metrics_snapshot_returns_default_registry_array() {
    // Check the exported array and borrowed strings against the default registry.
    // Family filtering and histogram values are covered with isolated registries
    // in metrics::tests, so one populated counter suffices at the FFI boundary.
    let counter = prometheus::IntCounterVec::new(
        prometheus::Opts::new("tikv_request_total", "test tikv request total"),
        &["type"],
    )
    .expect("counter vec");
    prometheus::default_registry()
        .register(Box::new(counter.clone()))
        .expect("register counter");
    counter.with_label_values(&["raw_get"]).inc_by(3);

    let mut error = ptr::null_mut();
    let mut snapshot = ptr::null_mut();
    let mut samples = tikv_metric_samples_t {
        ptr: ptr::null(),
        len: 0,
    };
    // SAFETY: all output slots are disjoint, writable local variables.
    let status = unsafe { tikv_metrics_snapshot_collect(&mut snapshot, &mut samples, &mut error) };
    assert_eq!(status, tikv_status_t::TIKV_STATUS_OK);
    assert!(error.is_null());
    assert!(!snapshot.is_null());
    assert!(!samples.ptr.is_null());
    assert_eq!(samples.len, 1);
    let sample = unsafe { &*samples.ptr };
    assert_eq!(
        unsafe { CStr::from_ptr(sample.name) }.to_bytes(),
        b"tikv_request_total"
    );
    assert_eq!(
        unsafe { CStr::from_ptr(sample.type_label) }.to_bytes(),
        b"raw_get"
    );
    assert_eq!(
        sample.kind,
        tikv_metric_sample_kind_t::TIKV_METRIC_SAMPLE_COUNTER
    );
    assert_eq!(sample.value, 3.0);

    // SAFETY: no sample references are used after releasing this snapshot once.
    unsafe { tikv_metrics_snapshot_destroy(snapshot) };
}

#[test]
fn metrics_snapshot_collect_rejects_missing_samples_output() {
    // Failure must clear whichever output slot was supplied, so callers cannot
    // accidentally retain a stale pointer after an invalid collection call.
    let mut error = ptr::null_mut();
    let mut snapshot = std::ptr::NonNull::<tikv_metrics_snapshot_t>::dangling().as_ptr();
    // SAFETY: only the slots are accessed; the dangling sentinel is overwritten
    // without dereferencing it. A null output is rejected by the API.
    let status =
        unsafe { tikv_metrics_snapshot_collect(&mut snapshot, ptr::null_mut(), &mut error) };
    assert_eq!(status, tikv_status_t::TIKV_STATUS_INVALID_ARGUMENT);
    assert!(snapshot.is_null());
    assert!(!error.is_null());
    // SAFETY: this error was just returned by the library and is released once.
    unsafe { tikv_error_destroy(error) };

    let mut samples = tikv_metric_samples_t {
        ptr: std::ptr::NonNull::<tikv_metric_sample_t>::dangling().as_ptr(),
        len: 1,
    };
    // SAFETY: writable slots are disjoint; the sample sentinel is never read.
    let status =
        unsafe { tikv_metrics_snapshot_collect(ptr::null_mut(), &mut samples, &mut error) };
    assert_eq!(status, tikv_status_t::TIKV_STATUS_INVALID_ARGUMENT);
    assert!(samples.ptr.is_null());
    assert_eq!(samples.len, 0);
    assert!(!error.is_null());
    // SAFETY: this is the new error from the second call, released once.
    unsafe { tikv_error_destroy(error) };
}

#[test]
fn write_batch_runs_preserve_put_delete_phase_order() {
    // Coalescing must not move a mutation across a put/delete phase boundary.
    let runs = build_write_batch_runs(vec![
        OwnedMutation::Put(KvPair::new(b"a".to_vec(), b"v1".to_vec())),
        OwnedMutation::Put(KvPair::new(b"b".to_vec(), b"v2".to_vec())),
        OwnedMutation::Delete(Key::from(b"c".to_vec())),
        OwnedMutation::Delete(Key::from(b"d".to_vec())),
        OwnedMutation::Put(KvPair::new(b"e".to_vec(), b"v3".to_vec())),
    ]);

    assert_eq!(
        runs,
        vec![
            WriteBatchRun::Put(vec![
                KvPair::new(b"a".to_vec(), b"v1".to_vec()),
                KvPair::new(b"b".to_vec(), b"v2".to_vec()),
            ]),
            WriteBatchRun::Delete(vec![Key::from(b"c".to_vec()), Key::from(b"d".to_vec())]),
            WriteBatchRun::Put(vec![KvPair::new(b"e".to_vec(), b"v3".to_vec())]),
        ]
    );
}

#[test]
fn write_batch_splits_repeated_put_keys() {
    // Repeated keys need separate RPC runs to preserve last-write-wins ordering.
    let runs = build_write_batch_runs(vec![
        OwnedMutation::Put(KvPair::new(b"a".to_vec(), b"v1".to_vec())),
        OwnedMutation::Put(KvPair::new(b"b".to_vec(), b"v2".to_vec())),
        OwnedMutation::Put(KvPair::new(b"a".to_vec(), b"v3".to_vec())),
    ]);

    assert_eq!(
        runs,
        vec![
            WriteBatchRun::Put(vec![
                KvPair::new(b"a".to_vec(), b"v1".to_vec()),
                KvPair::new(b"b".to_vec(), b"v2".to_vec()),
            ]),
            WriteBatchRun::Put(vec![KvPair::new(b"a".to_vec(), b"v3".to_vec())]),
        ]
    );
}

#[test]
fn ffi_mutations_copy_inputs_and_ignore_delete_value() {
    // Async submission must own its inputs, and DELETE must not dereference
    // the unused value slice even if that slice is invalid.
    let mut key = b"put-key".to_vec();
    let mut value = b"put-value".to_vec();
    let mutations = [
        tikv_async_raw_mutation_t {
            op: TIKV_ASYNC_RAW_MUTATION_OP_PUT,
            key: tikv_slice_t {
                ptr: key.as_ptr(),
                len: key.len(),
            },
            value: tikv_slice_t {
                ptr: value.as_ptr(),
                len: value.len(),
            },
        },
        tikv_async_raw_mutation_t {
            op: TIKV_ASYNC_RAW_MUTATION_OP_DELETE,
            key: tikv_slice_t {
                ptr: key.as_ptr(),
                len: key.len(),
            },
            value: tikv_slice_t {
                ptr: ptr::null(),
                len: usize::MAX,
            },
        },
    ];
    let owned = slice_mutations_to_owned(tikv_async_raw_mutations_t {
        ptr: mutations.as_ptr(),
        len: mutations.len(),
    })
    .unwrap();
    key.fill(0);
    value.fill(0);
    assert_eq!(
        owned,
        vec![
            OwnedMutation::Put(KvPair::new(b"put-key".to_vec(), b"put-value".to_vec())),
            OwnedMutation::Delete(Key::from(b"put-key".to_vec())),
        ]
    );
}

#[test]
fn request_batch_get_result_transfers_pairs_once() {
    // Batch results use a separate accessor and nested C allocations; verify
    // their contents, single transfer, and matching destruction helpers.
    let request = synthetic_request(
        RequestKind::BatchGet,
        RequestOutput::Pairs(vec![KvPair::new(b"k1".to_vec(), b"v1".to_vec())]),
    );

    let mut error = ptr::null_mut();
    let mut out_pairs = empty_kv_pairs();
    // SAFETY: the synthetic request is complete and exclusively accessed here;
    // output slots are disjoint, writable locals.
    let status =
        unsafe { tikv_async_request_take_batch_get_result(request, &mut out_pairs, &mut error) };
    assert_eq!(status, tikv_status_t::TIKV_STATUS_OK);
    assert!(error.is_null());
    assert_eq!(out_pairs.len, 1);
    let key = unsafe { slice::from_raw_parts((*out_pairs.ptr).key.ptr, (*out_pairs.ptr).key.len) };
    let value =
        unsafe { slice::from_raw_parts((*out_pairs.ptr).value.ptr, (*out_pairs.ptr).value.len) };
    assert_eq!(key, b"k1");
    assert_eq!(value, b"v1");
    // SAFETY: all borrowed pair slices are no longer used; free the array once.
    unsafe { tikv_kv_pairs_destroy(out_pairs) };

    // SAFETY: the request remains live and exclusively accessed on this thread.
    let status =
        unsafe { tikv_async_request_take_batch_get_result(request, &mut out_pairs, &mut error) };
    assert_eq!(status, tikv_status_t::TIKV_STATUS_INVALID_ARGUMENT);
    assert!(!error.is_null());
    // SAFETY: both handles are library-owned and released exactly once.
    unsafe {
        tikv_error_destroy(error);
        tikv_async_request_destroy(request);
    }
}
