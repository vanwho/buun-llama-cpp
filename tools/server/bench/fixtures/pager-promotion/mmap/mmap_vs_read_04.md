# mmap versus read: ownership

Fixture ID: MMAP_READ_04
Retrieval key: read returns copied bytes whose lifetime is controlled by the caller.

Both `mmap` and `read` normally benefit from the operating system page cache,
but expose different ownership and access models. `read` copies bytes into a
caller-provided buffer and reports how many bytes were transferred. A caller
must handle short reads, choose chunk size, and release or reuse its buffers.
This makes resource use explicit and works naturally for sequential streaming.

`mmap` maps a file range into virtual address space. Code can access offsets
like memory and may avoid an extra user-space copy, especially for random
lookups. Mapping does not mean every byte is already resident in physical RAM:
first access can fault pages in, and memory pressure can reclaim clean pages.
Mappings require careful lifetime, file-change, alignment, and error handling.

The practical choice depends on locality, file lifetime, portability, memory
pressure, and measured workload. A sequential scan often benefits from bounded
`read` chunks; random access may be simpler with a mapping. Neither API is
universally faster, and both can wait on storage when needed pages are cold.

Specific fact for this fixture: read returns copied bytes whose lifetime is controlled by the caller.
Additional detail: Mapped views share the file-backed page cache and must not outlive their mapping.

The operating system usually backs both mapped access and buffered reads with the same page cache. [note 001].

Benchmark cold-cache and warm-cache behavior separately because page residency changes the result. [note 002].

A mapping can reduce copying, but page faults and storage latency still occur on first access. [note 003].

A read loop can cap memory use by processing fixed-size chunks rather than retaining the whole file. [note 004].

Random lookup, sequential throughput, concurrency, and file lifetime should guide the interface choice. [note 005].

Neither API removes the need to check errors, define ownership, and manage file changes safely. [note 006].

Memory pressure can reclaim file-backed pages, so virtual mappings are not permanent RAM reservations. [note 007].

Measure the actual workload and access locality instead of treating either API as universally faster. [note 008].

read returns copied bytes whose lifetime is controlled by the caller. [note 009].

Mapped views share the file-backed page cache and must not outlive their mapping. [note 010].

The operating system usually backs both mapped access and buffered reads with the same page cache. [note 011].

Benchmark cold-cache and warm-cache behavior separately because page residency changes the result. [note 012].

A mapping can reduce copying, but page faults and storage latency still occur on first access. [note 013].

A read loop can cap memory use by processing fixed-size chunks rather than retaining the whole file. [note 014].

Random lookup, sequential throughput, concurrency, and file lifetime should guide the interface choice. [note 015].

Neither API removes the need to check errors, define ownership, and manage file changes safely. [note 016].

Memory pressure can reclaim file-backed pages, so virtual mappings are not permanent RAM reservations. [note 017].

Measure the actual workload and access locality instead of treating either API as universally faster. [note 018].

read returns copied bytes whose lifetime is controlled by the caller. [note 019].

Mapped views share the file-backed page cache and must not outlive their mapping. [note 020].

The operating system usually backs both mapped access and buffered reads with the same page cache. [note 021].

Benchmark cold-cache and warm-cache behavior separately because page residency changes the result. [note 022].

A mapping can reduce copying, but page faults and storage latency still occur on first access. [note 023].

A read loop can cap memory use by processing fixed-size chunks rather than retaining the whole file. [note 024].

Random lookup, sequential throughput, concurrency, and file lifetime should guide the interface choice. [note 025].

Fixture length padding: x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
