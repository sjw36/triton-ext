// RUN: triton-opt --split-input-file %s -mega-bulk-sync-lowering | FileCheck %s

// CHECK-LABEL: @grid_barrier
// CHECK-SAME:  (%[[ARR:.*]]: !tt.ptr<i32>, %[[REL:.*]]: !tt.ptr<i32>, %[[NP:.*]]: i32, %[[SENSE:.*]]: i32)
tt.func @grid_barrier(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>, %num_programs: i32, %sense: i32) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // tl.debug_barrier()
  // CHECK: ttg.barrier
  // last = tl.atomic_add(arrival_ptr, 1, sem="acq_rel")
  // CHECK: %[[LAST:.*]] = tt.atomic_rmw add, acq_rel, gpu, %[[ARR]], %[[C1]] : (!tt.ptr<i32>, i32) -> i32
  // last == num_programs - 1
  // CHECK: %[[LASTIDX:.*]] = arith.subi %[[NP]], %[[C1]] : i32
  // CHECK: %[[ISLAST:.*]] = arith.cmpi eq, %[[LAST]], %[[LASTIDX]] : i32
  // CHECK: scf.if %[[ISLAST]] {
  // CHECK:   tt.atomic_rmw exch, release, gpu, %[[ARR]], %[[C0]] : (!tt.ptr<i32>, i32) -> i32
  // CHECK:   tt.atomic_rmw exch, release, gpu, %[[REL]], %[[SENSE]] : (!tt.ptr<i32>, i32) -> i32
  // CHECK: } else {
  // CHECK:   scf.while : () -> () {
  // CHECK:     %[[CUR:.*]] = tt.atomic_rmw add, acquire, gpu, %[[REL]], %[[C0]] : (!tt.ptr<i32>, i32) -> i32
  // CHECK:     %[[NE:.*]] = arith.cmpi ne, %[[CUR]], %[[SENSE]] : i32
  // CHECK:     scf.condition(%[[NE]])
  // CHECK:   } do {
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK: }
  // tl.debug_barrier()
  // CHECK: ttg.barrier
  // CHECK-NOT: mega.bulk_sync
  mega.bulk_sync %arrival, %release, %num_programs, %sense : !tt.ptr<i32>, !tt.ptr<i32>
  tt.return
}
