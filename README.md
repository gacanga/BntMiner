# BntMiner — NVIDIA GPU miner for Blocknet (BNT), Argon2id 2 GiB

BntMiner mines [Blocknet](https://blocknetcrypto.com/) (BNT, the privacy coin of `blocknetprivacy`) on NVIDIA GPUs.
The proof of work is **Argon2id v1.3 with 2 GiB of memory, 1 pass, 1 lane**, and the miner talks to the
[bntpool.com](https://bntpool.com/) stratum. The CUDA engine was written from scratch for Pascal cards (P104-100,
GTX 1070/1080, sm_61).

- **No dev fee.**
- Open source, Apache-2.0.
- Windows x64 release built with CUDA 12.9 (sm_61 code + PTX, so newer GPUs run it through the driver's JIT).
- Checked against the Rust `argon2` crate called exactly like Blocknet core, at several memory sizes and at the real
  2 GiB parameters, on the CPU and on every GPU (`test.bat`).

Not to be confused with the older Blocknet (BLOCK) project or with Bancor (BNT).

Version française : [README.fr.md](README.fr.md).

## Quick start (Windows)

1. Get a Blocknet address with the official wallet
   ([blocknet-gui](https://github.com/blocknetprivacy/blocknet-gui/releases/latest) or the
   [CLI](https://github.com/blocknetprivacy/blocknet/releases/latest)). The node does not need to be synced to mine on
   the pool.
2. Download the zip from [Releases](https://github.com/gacanga/BntMiner/releases) and extract it.
3. Open `bntminer.conf` and replace `YOUR_BLOCKNET_ADDRESS` with your address (a blocknet.id shortname such as
   `@name` also works); `rig1` is the worker name, change it if you like.
4. Double-click `start_bnt.bat`. It restarts the miner 10 s after any exit and logs to `bntminer.log`; close the window
   to stop.

`test.bat` runs the correctness tests on every GPU, `bench.bat` a 120 s H/s benchmark (stop mining first).

Requirements: Windows 10/11 x64, an NVIDIA driver with CUDA 12 support (R525 or newer; R580 is the last branch for
Pascal), **2 GiB of free GPU memory per lane** (an 8 GB card runs 3 lanes). Nothing else needs to be installed.

## Pool

| Pool | URL | Fee |
| --- | --- | --- |
| bntpool.com | `stratum+tcp://bntpool.com:3333` | 2.5 %, PPLNS (6 h window), minimum payment 0.1 BNT, 60 confirmations |

Protocol (JSON lines, the one of the reference miner [seine](https://github.com/zpalmtree/seine); checked against the
live pool):

```
-> {"id":1,"method":"login","params":{"address","worker","protocol_version":2,"capabilities":[...]}}
<- {"id":1,"status":"ok","result":{"protocol_version":2,"capabilities":[...],"required_capabilities":["submit_claimed_hash"]}}
<- {"method":"job","params":{"job_id","header_base","target","difficulty","network_target","height","nonce_start","nonce_end"}}
-> {"id":n,"method":"submit","params":{"job_id","nonce","claimed_hash","miner_version"}}
<- {"id":n,"status":"ok","result":{"accepted":true,"difficulty":d,"status":"provisional","verified":false}}
```

`header_base` is the 100-byte block header without the nonce; the pool gives every job its own nonce range. A share
is valid when `Argon2id(password = nonce as u64 little-endian, salt = header_base) <= target` (big-endian bytes). The
pool checks shares asynchronously ("provisional", then "verified").
Your shares and balance: `https://bntpool.com/api/miner/<address>` and `.../balance?include_pending_estimate=true`.

## Configuration

The miner reads `bntminer.conf` next to the executable (or one folder above it), or the file given with
`--config`. Each line is `key = value`; `#` and `;` start a comment. Every key is also a command-line option
(`--key value`); command-line values override the file, and `-o` on the command line replaces the pools of the file.

| Option / key | Default | Meaning |
| --- | --- | --- |
| `-o`, `--pool`, `--url` / `pool` | — | Pool URL, repeatable (failover order). `stratum+tcp://` = plain TCP, `stratum+ssl://` or `stratum+tls://` = TLS |
| `-u`, `--user`, `--wallet`, `--address` / `user` | — | Blocknet address or blocknet.id shortname |
| `--worker`, `--rig` / `worker` | `rig` | Worker name |
| `--gpus`, `--devices`, `-d` / `gpus` | `all` | `all` or a list such as `0,1,2` (PCI bus order, as nvidia-smi) |
| `--lanes` / `lanes` | `0` | Hashes in parallel per GPU (2 GiB each); `0` = as many as the free memory allows |
| `--reserve-mb` / `reserve-mb` | `160` | GPU memory left free (MB) |
| `--chunk` / `chunk` | `32768` | 1 KiB blocks per kernel launch (the run can stop between two launches) |
| `--stats` / `stats` | `60` | Seconds between summary lines (minimum 10) |
| `--log` / `log` | — | Append the log to this file |
| `--cafile` / `cafile` | `ca-bundle.crt` next to the exe | CA bundle for TLS pools |
| `--config`, `-c` | `bntminer.conf` | Configuration file |
| `--args-file FILE` | — | Read more arguments from a file (whitespace separated) |
| `--test` | | Correctness tests (CPU, then every selected GPU), exit code 0 when all pass |
| `--bench [s]` | 120 s | H/s benchmark on the selected GPUs |
| `--profile` | | Time of the Argon2i and Argon2d halves of one hash, for 1 .. N lanes |
| `--help`, `-h` | | Usage |

Exit codes used by the restart loop: 2 = bad argument or missing pool/user, 3 = CUDA error, 4 = a GPU run did not
return for 60 s (unstable clocks?), 5 = test/benchmark timeout.

## Performance (measured)

NVIDIA P104-100 8 GB, bntpool:

| Setup | H/s |
| --- | --- |
| 1 × P104-100, 1 lane | ≈ 0.48 |
| 1 × P104-100, 3 lanes | ≈ 1.5 (benchmark 1.50, pool 1.48–1.65) |
| 2 × P104-100 | ≈ 3.2 |

A P104-100 draws only about 66 W on this algorithm. With a single lane (one Argon2 chain) per warp the speed is set
by the latency of the compression function, so it grows almost linearly with the number of lanes, which is limited
by the GPU memory (2 GiB each).

## How it works

- PoW (Blocknet core, `crypto-rs/src/pow.rs`): Argon2id v0x13, m = 2,097,152 KiB (2^21 blocks of 1 KiB), t = 1,
  p = 1, 32-byte tag. With one lane and one pass, blocks 0 .. m/2-1 use Argon2i addressing (indices independent of
  the data: a fixed 4 MB table computed once on the host) and blocks m/2 .. m-1 use Argon2d addressing (J1 = low 32
  bits of the previous block). Reference index of block i: area = i − 1, x = J1²>>32, ref = area − 1 − (area·x>>32).
- The host computes H0 and the first two blocks (BLAKE2b, H'), the GPU fills the 2^21 blocks, the host computes the
  final tag H'(32, last block) and compares it with the target.
- CUDA engine (`gpu/bnt_gpu.cu`): one warp of 32 threads per lane. Every thread keeps 4 of the 128 words of the
  current block in registers (words 2t, 2t+1, 64+2t, 65+2t, read and written as two coalesced 16-byte accesses). The
  BlaMka permutation runs in padded shared memory (row step, then column step, each thread computing one of the 32
  quarter-rounds, diagonals by warp shuffles). In the Argon2i half the reference block of i+1 is loaded while block i
  is computed. Kernel launches of 32,768 blocks keep the Windows display watchdog away and let a new block stop the
  run early.
- `src/argon_ref.h`: header-only reference (BLAKE2b, H', H0, G, address table, full CPU hash), shared by the DLL,
  the miner and the tests.

## Building from source (Windows)

Prerequisites, as expected by the scripts:

- CUDA Toolkit **12.9** (`CUDA_PATH_V12_9`, default `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`);
  CUDA 13 cannot target Pascal.
- Visual Studio 2022 (or Build Tools) with the C++ x64 tools (found with `vswhere`).
- MSYS2 in `C:\msys64` with the MINGW64 packages `mingw-w64-x86_64-gcc` and `mingw-w64-x86_64-openssl`.

| Script | What it does |
| --- | --- |
| `build_all.bat` | Builds `dev\bgpu.dll` (nvcc + MSVC, `windows\build_gpu_dll.bat`) and `dev\bntminer.exe` (MSYS2 g++, `windows/build_host.sh`), then runs the tests. `build_all.bat nobuild` = tests only |
| `run_test.bat` | `dev\bntminer.exe --test`, log `logs\test.log`; extra arguments in `run\test_args.txt` |
| `run_bench.bat` | 120 s benchmark of `dev\`, log `logs\bench.log`; `run_bench.bat profile` = Argon2i / Argon2d halves (`logs\profile.log`); extra arguments in `run\bench_args.txt` |
| `deploy.bat` | Copies `dev\` to `bin\`, only if the last `logs\test.log` says `TEST RESULT: ALL OK` |
| `bnt_pool.bat` | Mines with `bin\bntminer.exe` and `bntminer.conf`, restart loop, log `logs\pool.log`; extra arguments in `run\pool_args.txt` |
| `stop_bnt.bat` | Stops the restart loop and `bntminer.exe` |

Build options (environment variables): `BM_NVCC_EXTRA` (or `run\nvcc_extra.txt`) adds nvcc flags, `BM_HOST_ARCH`
replaces `-march=native`. For a portable release build use `set BM_HOST_ARCH=-march=x86-64 -mtune=generic` before
`build_all.bat`. `bgpu.dll` uses the dynamic Microsoft C runtime (`/MD`): the release zip ships its DLLs. The zip
contains `bntminer.exe`, `bgpu.dll`, `cudart64_12.dll`, the VC++ runtime DLLs, `ca-bundle.crt` and the files of
[`dist/`](dist).

## Tests

`bntminer.exe --test`:

1. CPU: BLAKE2b known answer, then 16 Argon2id vectors (m = 8, 64, 1024 and 65,536 KiB, 4 nonces each) of
   `src/bnt_vectors.h`;
2. every selected GPU: the same vectors with 4 kernel launch sizes (1, 5, 333 blocks and the default), which covers
   the prefetch and the launch boundaries, then 3 vectors of the real 2 GiB PoW computed in parallel (one lane each).

The vectors come from `tools/bntref` (Rust, `argon2 = "0.5"`, the same call as Blocknet core):

```sh
cd tools/bntref && cargo build --release
for m in 8 64 1024 65536; do ./target/release/bntref $m 4; done > ../../tests/vectors.txt
./target/release/bntref 2097152 3 >> ../../tests/vectors.txt
cd ../.. && python3 tools/gen_vectors.py      # -> src/bnt_vectors.h
```

Linux checks without a GPU:

```sh
g++ -O2 -std=c++17 -Isrc tests/cloud_test.cpp -o cloud_test && ./cloud_test tests/vectors.txt full
g++ -O2 -std=gnu++17 -Isrc -Igpu -o bntminer_stub src/main.cpp src/tls_conn.cpp tests/bg_stub.cpp -lssl -lcrypto -pthread
./bntminer_stub --test
python3 tests/fake_pool.py &  ./bntminer_stub -o stratum+tcp://127.0.0.1:13333 -u TEST --lanes 3
```

`tests/cloud_test.cpp` checks the reference code and a CPU emulation of the kernel's thread mapping (layout, padded
shared memory, shuffles) against the vectors; `tests/bg_stub.cpp` is a CPU stand-in for `bgpu.dll`;
`tests/fake_pool.py` is a minimal bntpool-like server. `tools/nvrtc_check.py` compiles the kernel for sm_61 with NVRTC
and prints the ptxas register / spill report.

## Credits and licence

- CUDA engine, reference code, host code and stratum client: gacanga. The stratum client (`src/json.h`,
  `src/tls_conn.*`) comes from FiloMiner / NullMiner (Apache-2.0).
- The pool protocol follows [seine](https://github.com/zpalmtree/seine) (BSD-3-Clause); no seine code is included.
- The Rust [argon2](https://github.com/RustCrypto/password-hashes) crate and
  [Blocknet core](https://github.com/blocknetprivacy/core) were used as correctness references for the tests only;
  they are not linked into the miner.

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). No warranty: you mine at your own risk, and
mining uses electricity. Blocknet describes itself as experimental; it plans a network upgrade (around 2027-02-16)
with a different proof of work, after which this miner will need an update.
