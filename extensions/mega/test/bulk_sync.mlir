// RUN: mega.py %s | %filecheck %s

// This test checks that the `mega.bulk_sync` op is correctly parsed and printed
// and that it is not canonicalized away (it carries side effects).
tt.func @mega_bulk_sync(%arrival: !tt.ptr<i32>, %release: !tt.ptr<i32>, %num_programs: i32, %sense: i32) {
  // CHECK: mega.bulk_sync %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<i32>, !tt.ptr<i32>
  mega.bulk_sync %arrival, %release, %num_programs, %sense : !tt.ptr<i32>, !tt.ptr<i32>
  tt.return
}
