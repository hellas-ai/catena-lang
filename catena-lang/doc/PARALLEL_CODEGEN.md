# Parallel GPU Code Generation

This note describes how the parallel operations introduced with the first GPU
implementation are lowered to CUDA or HIP. It focuses on the procedure rather
than the exact generated variable names.

At a high level, codegen chooses a concrete GPU interpretation from the type
of the context:

```text
Hex parallel program
        |
        v
closure conversion and specialization
        |
        v
lower dependent types to runtime C types
        |
        v
inspect context type
   +----+------------------+
   |                       |
grid host              block worker
   |                       |
kernel launch          inline cooperative work
```

## 1. Prepare the task

The source task passed to `parallel.materialize` is initially a closure:

```text
task : (worker, index) => f32
```

Closure conversion replaces `parallel.materialize` with
`parallel.materializec`. The operation then receives:

```text
context
writable buffer
captured environment
task function
```

The generated code therefore does not need a runtime closure. It calls an
ordinary generated function and passes the captured values explicitly.

## 2. Lower parallel types

Most topology information exists only to type-check the Hex program. For
example:

```text
worker<
    gpu.block.2d<tile, tile>,
    workers,
    groups,
    Ix tile * Ix tile,
    storage-layout>
```

lowers to the concrete backend type:

```cpp
catena_gpu_block_worker_t
```

The GPU backend currently gives runtime representations to these generic
parallel types:

```text
host<gpu.grid.2d, ...>   -> catena_gpu_grid_host_t
worker<gpu.grid.2d, ...> -> catena_gpu_grid_worker_t
worker<gpu.block.2d, ...> -> catena_gpu_block_worker_t
shared<layout>           -> uint64_t
available<layout>        -> catena_gpu_shared_available_t
```

The dependent shared layout erases to its required byte count:

```text
shared<slot<A, f32, slot<B, f32, empty>>>
    -> uint64_t
```

Buffer modifiers such as `writable` and `pending` lower to the underlying
buffer pointer. They constrain the Hex program but do not require different C
representations.

## 3. Build the host context

`gpu.grid.2d` preserves the runtime grid and block dimensions:

```text
gpu.grid.2d(grid-x, grid-y, block-x, block-y)
```

The program constructs the shared layout independently:

```text
empty  = parallel.shared.empty()     # 0 bytes
b-slot = gpu.shared(tile-count, empty)
layout = gpu.shared(tile-count, b-slot)
```

Each `gpu.shared` adds `length * sizeof(float)` to the runtime byte count.
`parallel.host` combines the launch dimensions and shared size into:

```cpp
typedef struct {
    catena_gpu_launch_t launch;
    uint64_t shared_bytes;
} catena_gpu_grid_host_t;
```

`parallel.host` creates only the host context. A worker is created later when
materialization launches the task.

### Why shared memory is dynamically sized

CUDA permits a statically sized shared array only when its size is a C++
compile-time constant:

```cpp
__shared__ float tile[256];
```

That is not generally true for Catena. A Hex program can describe a slot using
metavariables such as `tile-size * tile-size`, while the corresponding runtime
dimension is supplied to the compiled function by its caller. Code generation
knows the relationship between the slot length and the topology, but it does
not necessarily know a concrete integer with which to declare a fixed C++
array.

Using a fixed maximum would also be misleading: it would waste shared memory
for smaller tiles, impose an arbitrary capacity on every kernel, and would not
make the requested layout type-safe.

The GPU lowering therefore uses CUDA/HIP dynamic shared memory:

```cpp
__global__ void kernel(...) {
    extern __shared__ unsigned char shared[];
    // All workers in this block see the same arena.
}

kernel<<<grid, block, shared_bytes>>>(...);
```

The host computes `shared_bytes` by evaluating the `gpu.shared` layout
builders before launch. Inside the kernel, `parallel.allocate` divides that
one arena into typed, non-overlapping slots using byte offsets.

This is called dynamic shared allocation because its size is selected at
launch time. It is not device-heap allocation: the kernel does not call
`malloc`, and every block receives its own shared-memory arena managed by the
GPU runtime. The requested total must still fit the device's per-block shared
memory limit; this hardware limit is currently checked by CUDA/HIP rather than
encoded in the Hex types.

## 4. Allocate global output storage

For a grid host:

```text
parallel.host.allocate(host, count)
    -> (host, writable-buffer)
```

lowers to a CUDA or HIP device allocation:

```cpp
cudaMalloc(&buffer, count * sizeof(float));
```

The host value is preserved for the following materialization and await.

## 5. Generate a kernel for grid materialization

When the context of `parallel.materializec` is a grid host, codegen emits both:

- an auxiliary GPU kernel; and
- a launch of that kernel from the enclosing host function.

The generated kernel has this shape:

```cpp
__global__ void kernel(float *out, grid_host context, environment...) {
    extern __shared__ unsigned char shared[];

    uint64_t global_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t global_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t index = global_y * total_width + global_x;

    grid_worker worker = {
        context.launch,
        index,
        shared
    };

    float value = task(environment..., worker, index);
    out[index] = value;
}
```

The launch uses the topology and shared-memory size stored in the host:

```cpp
kernel<<<context.launch.grid_dim,
         context.launch.block_dim,
         context.shared_bytes>>>(out, context, environment...);
```

This is where the backend creates the grid worker required by the task.

## 6. Derive the current block worker

Inside the task, `gpu.current-block` maps the grid worker to coordinates local
to its CUDA block.

`parallel.worker.index` first expands the flat grid-worker index into global X
and Y coordinates. The GPU definition then calculates:

```text
local-x = global-x % block-x
local-y = global-y % block-y
```

Finally, generic `parallel.worker` constructs a block worker and its unique
initial storage capability. The worker's flattened index is:

```text
local-index = local-y * block-x + local-x
```

The launch metadata and shared-memory base pointer are preserved from the
parent worker. At the same time, the initial allocator cursor is created:

```cpp
available = {
    .base = child_worker.shared,
    .offset = 0
};
```

## 7. Split the dynamic shared arena into slots

The allocator cursor is returned with the newly constructed block worker.
Each `parallel.allocate` returns a pointer at the current offset and advances
that cursor:

```cpp
A_tile = (float *)(available.base + available.offset);
available.offset += A_count * sizeof(float);

B_tile = (float *)(available.base + available.offset);
available.offset += B_count * sizeof(float);
```

At the Hex level, the changing type of `available` enforces the slot order:

```text
available<A, B>
    -> allocate A
available<B>
    -> allocate B
available<empty>
    -> parallel.shared.finish
```

`parallel.shared.finish` emits no C++ code. It proves that the declared layout
was exhausted without allocating an undeclared slot.

## 8. Lower block materialization to cooperative writes

When the `parallel.materializec` context is a block worker, codegen does not
launch another kernel. The current GPU thread calls the producer once using
its block-local index:

```cpp
uint64_t index = block_worker.index;
float value = task(environment..., block_worker, index);
tile[index] = value;
```

Consequently:

```text
materialize(block-worker, A-tile, load-A)
materialize(block-worker, B-tile, load-B)
```

makes every thread in the block write one A element and one B element into
shared memory.

## 9. Lower synchronization from the context

The generic synchronization operations preserve their runtime inputs and
outputs. The context determines the actual synchronization instruction:

```text
grid host    -> cudaDeviceSynchronize() or hipDeviceSynchronize()
block worker -> __syncthreads()
```

In tiled matmul:

```text
parallel.await(block-worker, pending-A, pending-B)
```

emits one block barrier for both cooperative loads. After the dot product:

```text
parallel.release(block-worker)
```

emits another block barrier, preventing a faster worker from overwriting the
shared slots for the next K tile while another worker is still reading them.

The backend emits this second barrier because the Hex program contains
`parallel.release`. The current buffer types do not force the program to
include it.

## 10. Place generated functions

Generated functions are classified as either:

```text
host-only
host-and-device
```

A function that launches a grid kernel or allocates global GPU memory must be
host-only. Functions called by the kernel, including indexing helpers, tile
loaders, and the cell computation, must be available on the device.

Host-only placement is propagated through the call graph: a function that
calls a host-only function also becomes host-only.

## Complete tiled-matmul procedure

The resulting procedure is:

```text
host:
    compute grid and block dimensions
    compute dynamic shared-memory bytes
    allocate global C
    launch one grid kernel
    wait for the kernel

each kernel worker:
    compute its output index
    derive its block-local worker
    obtain distinct A and B slots in the block's shared arena

    for every K tile:
        cooperatively load A and B
        block barrier
        compute one partial dot product
        block barrier before reusing the slots

    write one element of C
```

The generic parallel code understands the flattened interfaces of contexts,
buffers, environments, and task functions. GPU-specific lowering decides that
these operations mean kernel launches, CUDA/HIP indices, dynamic shared
memory, pointer offsets, and block barriers.

The generic decoding is implemented in `src/codegen/ops/parallel/mod.rs`. The
CUDA/HIP interpretation is implemented in
`src/codegen/ops/parallel/gpu.rs`.
