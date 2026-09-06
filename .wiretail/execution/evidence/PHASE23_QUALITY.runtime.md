# Task 23-05 runtime occupancy capture

Captured locally on 2026-09-06 while preserving all existing server
processes. No endpoint was contacted and no process was changed.

Command:

```text
ps -fp 3761038
```

Output:

```text
UID          PID     PPID C STIME TTY          TIME CMD
ninja    3761038        1 0 07:55 ?        00:00:07 /srv/ai/paged-kv/results/20-07-runtime-bundle-20260905T220000Z/bin/llama-server -m /srv/ai/models/text/current.gguf --alias qwen38-fast-turbo4-mtp ... --port 8080 ... --kv-pager selective --kv-page-size 256 --kv-hot-pages 4 ...
```

Command:

```text
ss -ltnp | rg ':8080|:8092'
```

Output:

```text
LISTEN 0 512 0.0.0.0:8080 users:("llama-server",pid=3761038,fd=16)
LISTEN 0 512 172.18.0.1:8092 users:("llama-server",pid=6334,fd=3)
```

The ellipsis removes unrelated command-line flags only; no authentication
material was captured. Port 8080 is a historical selected-reference
candidate, while port 8092 belongs to another server and was not touched.
