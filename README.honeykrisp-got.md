# What this fork is

Upstream Mesa, plus the Honeykrisp (Asahi) Vulkan driver changes made while
getting *Ghost of Tsushima DIRECTOR'S CUT* to run on an M1 Max under Fedora
Asahi Remix. The measurement harness, the evidence and the written record live
in a separate repository: <https://github.com/aquarat/got-bringup>.

**Read that repository before trusting any performance number here.** Three
measurement hazards — the game's dynamic resolution, and two missing shader
cache keys — silently corrupted results during this work and produced two
conclusions that had to be retracted.

## The branches

| branch | what |
|---|---|
| `local-deploy` | **the one that matters.** Everything below, plus the profiler and the measurement flags. This is what got-bringup pins and packages. |
| `main` | plain upstream, the rebase base |
| `cdm-*`, `tess-*`, `latency-sched`, ... | single-hypothesis branches kept for the record; most were measured and rejected |

## The two changes worth 4.1x on compute time

Ablated at pinned render resolution, one arm per change:

| fix | fps | compute |
|---|---|---|
| independent compute dispatches allowed to overlap (weak CDM barrier) | +34.5% | 2.17x |
| shader constant tables kept out of per-invocation scratch | +30.1% | 1.88x |

The first is the interesting one. Honeykrisp emitted `AGX_BARRIER_ALL` after
every dispatch, which sets all 23 bits of the barrier word under a comment
admitting the bits are not understood. The game's hottest shader dispatches 214
invocations at a time, 414 times a frame — about 10% of the GPU, with 29 cores
idle — so the only cover available for its memory latency is *other dispatches*,
which that barrier prevented. Per-bit cost was measured rather than guessed: the
default is now mask `0x80`, which measures 2.891 ms against a 2.892 ms floor at
no barrier at all.

`HK_PERFTEST=nooverlap` restores the old behaviour, and
`AGX_CDM_BARRIER_MASK=0x1fffff` the original barrier.

## Before trusting this after a rebase

The weak barrier is sound only because hk ends the compute control stream at
every Vulkan synchronisation point, so two dispatches sharing a stream are
guaranteed independent. Nothing enforces that. If a rebase brings in a
synchronisation primitive that does not end the stream, the barrier becomes
unsound **and nothing will fail loudly**. got-bringup's `STATE.md` has the full
argument, the conformance results (15,307 tests, no regression) and the list of
what to re-check.

## Building just the driver

    meson setup build --prefix=/usr --libdir=lib64 --buildtype=release \
      -Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms=x11,wayland \
      -Dopengl=false -Dglx=disabled -Degl=disabled -Dgbm=disabled \
      -Dvideo-codecs= -Dtools= -Dbuild-tests=false
    ninja -C build

`libvulkan_asahi.so` is a self-contained ICD and is the whole of what gets
deployed. `.github/workflows/asahi-vulkan-build.yml` does exactly this on every
push; got-bringup turns the same configuration into an installable RPM.
