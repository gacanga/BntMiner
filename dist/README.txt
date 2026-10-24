BntMiner - NVIDIA GPU miner for Blocknet (BNT, blocknetcrypto.com), Argon2id 2 GiB
Windows x64, CUDA 12.9 build (native code for Pascal sm_61: P104-100, GTX 1070/1080)

Open source (Apache-2.0), NO developer fee.
Pool: bntpool.com (stratum+tcp://bntpool.com:3333).
Not to be confused with the older Blocknet (BLOCK) project or with Bancor (BNT).

QUICK START
  1. Open bntminer.conf and replace YOUR_BLOCKNET_ADDRESS with your Blocknet address
     (or a blocknet.id shortname such as @name); "rig1" is the worker name, you can change it.
  2. Double-click start_bnt.bat. It restarts the miner automatically and
     writes bntminer.log; close the window to stop.

Other scripts
  test.bat   correctness tests (CPU, then every GPU against reference hashes)
  bench.bat  120 s H/s benchmark (stop mining first)

Requirements
  - Windows 10/11 x64
  - NVIDIA driver with CUDA 12 support (R525 or newer; R580 is the last branch
    for Pascal). Nothing else to install.
  - 2 GiB of free GPU memory per hash in parallel ("lane"): an 8 GB card runs 3 lanes
  - Newer GPUs run the PTX code through the driver's JIT (works, not tuned).

Measured: P104-100: ~1.5 H/s per card (3 lanes), about 66 W.

Files: bntminer.exe, bgpu.dll, cudart64_12.dll (NVIDIA CUDA runtime), vcruntime140.dll,
vcruntime140_1.dll, msvcp140.dll (Microsoft Visual C++ runtime), ca-bundle.crt
(certificates for TLS pools), bntminer.conf, start_bnt.bat, test.bat, bench.bat, LICENSE, NOTICE.
Command line: bntminer.exe --help

Source code, documentation, releases: https://github.com/gacanga/BntMiner
