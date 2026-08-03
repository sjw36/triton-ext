// RUN: triton-opt --split-input-file %s -mega-hoist-first-access | FileCheck %s

// A load of a tensor first accessed *after* the barrier is hoisted to just
// before it; a load of a tensor already accessed *before* the barrier is not.
// CHECK-LABEL: @hoist_first_access
tt.func @hoist_first_access(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>,
                            %np: i32, %sense: i32,
                            %x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
  // %y is written before the barrier, so its post-barrier load stays put.
  %v = tt.load %y : !tt.ptr<f32>
  tt.store %y, %v : !tt.ptr<f32>

  // %x is first accessed after the barrier: its load is prefetched above it.
  // CHECK: %[[XV:.*]] = tt.load %arg4 : !tt.ptr<f32>
  // CHECK-NEXT: mega.bulk_sync
  mega.bulk_sync %arrival, %release, %np, %sense : !tt.ptr<i32>, !tt.ptr<i32>

  // CHECK: tt.load %arg5 : !tt.ptr<f32>
  %x0 = tt.load %x : !tt.ptr<f32>
  %y1 = tt.load %y : !tt.ptr<f32>
  %sum = arith.addf %x0, %y1 : f32
  tt.store %x, %sum : !tt.ptr<f32>
  tt.return
}

// -----

// The address arithmetic that a hoistable load depends on is moved with it.
// CHECK-LABEL: @hoist_with_address
tt.func @hoist_with_address(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>,
                            %np: i32, %sense: i32,
                            %x: !tt.ptr<f32>, %off: i32) {
  // CHECK: %[[P:.*]] = tt.addptr %arg4, %arg5
  // CHECK-NEXT: %[[V:.*]] = tt.load %[[P]]
  // CHECK-NEXT: mega.bulk_sync
  mega.bulk_sync %arrival, %release, %np, %sense : !tt.ptr<i32>, !tt.ptr<i32>
  %p = tt.addptr %x, %off : !tt.ptr<f32>, i32
  %v = tt.load %p : !tt.ptr<f32>
  tt.store %p, %v : !tt.ptr<f32>
  tt.return
}

// -----

// TMA descriptor loads are hoisted on the same first-access rule: %dx is first
// accessed after the barrier (hoisted), while %dy was written before it (kept).
// CHECK-LABEL: @hoist_descriptor_load
tt.func @hoist_descriptor_load(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>,
                               %np: i32, %sense: i32,
                               %dx: !tt.tensordesc<64x64xf32>,
                               %dy: !tt.tensordesc<64x64xf32>, %i: i32) {
  %v = tt.descriptor_load %dy[%i, %i] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
  tt.descriptor_store %dy[%i, %i], %v : !tt.tensordesc<64x64xf32>, tensor<64x64xf32>

  // CHECK: tt.descriptor_load %arg4[{{.*}}]
  // CHECK-NEXT: mega.bulk_sync
  mega.bulk_sync %arrival, %release, %np, %sense : !tt.ptr<i32>, !tt.ptr<i32>

  // CHECK: tt.descriptor_load %arg5[{{.*}}]
  %a = tt.descriptor_load %dx[%i, %i] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
  %b = tt.descriptor_load %dy[%i, %i] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
  tt.return
}

// -----

// A descriptor load aliases an ordinary pointer store through its base, so the
// pre-barrier store makes it NOT a first access: it must stay below the barrier.
// CHECK-LABEL: @descriptor_alias_not_hoisted
tt.func @descriptor_alias_not_hoisted(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>,
                                      %np: i32, %sense: i32, %base: !tt.ptr<f32>,
                                      %sh: i32, %st: i64, %i: i32) {
  %z = tt.load %base : !tt.ptr<f32>
  tt.store %base, %z : !tt.ptr<f32>
  // CHECK: mega.bulk_sync
  // CHECK: tt.make_tensor_descriptor
  // CHECK-NEXT: tt.descriptor_load
  mega.bulk_sync %arrival, %release, %np, %sense : !tt.ptr<i32>, !tt.ptr<i32>
  %d = tt.make_tensor_descriptor %base, [%sh, %sh], [%st, %st] : <f32>, <64x64xf32>
  %r = tt.descriptor_load %d[%i, %i] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
  tt.return
}
