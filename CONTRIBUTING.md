# Contributing

This is a research fork of [`IAZero/wisardpkg`](https://github.com/IAZero/wisardpkg)
maintained at [`github.com/muanlartins/wisardpkg`](https://github.com/muanlartins/wisardpkg).

## Branch policy

This fork uses a **single canonical branch named `main`**. All work happens on
`main`; there are no long-lived feature branches.

- **`main`** — the one source of truth. New features, fixes, and documentation
  land here. The old `develop` / `master` / `f4rm` / `feature-*` branches have
  been deleted; the 2019 base64-compaction experiment is preserved only as the
  `archive/base64-compaction` tag.
- **`performance`** — the *one* exception: a dedicated long-lived branch for the
  in-progress C++ reimplementation aimed at performance. It is allowed to live
  alongside `main` until the reimplementation is ready to merge.

Short-lived topic branches for a single change are fine as long as they are
merged back into `main` and deleted promptly. Do not let them accumulate.

Upstream (`IAZero/wisardpkg`) remains the `origin` remote and develops on its
own `develop` branch; this fork's `main` tracks the `fork` remote.

### Commits

- Imperative, concise commit messages (e.g., "Add H3 hashing to BloomFilter").
- One logical change per commit.
- Git operations are managed by the maintainer — do not commit, push, or amend
  without an explicit ask.

## Building

```bash
pip install .                      # build + install the C++ core
pip install ".[torch]"             # also installs torch + numpy → enables DWN, ULEEN
pip install --no-build-isolation . # if pybind11 is already in the venv
make install                       # equivalent to pip install .
make geninclude                    # regenerate the standalone C++ header
make clean                         # remove compiled .so files from test/
```

All `.cc` files are `#include`'d into a single compilation unit via
`src/wisardpkg.h` — they are not independently compiled. The C++ extension
builds as `wisardpkg._native`, and `wisardpkg/__init__.py` re-exports its
symbols so `import wisardpkg as wp` is unchanged.

## Testing

```bash
make unittest                # compile the C++ extension, then run test/testset.py
python3 test/test_dwn.py     # DWN tests (require torch)
python3 test/test_uleen.py   # ULEEN tests (require torch)
```

Run the existing tests before and after any change. If you change behavior,
update or add the corresponding test under `test/`. The torch-gated port tests
(`test_dwn.py`, `test_uleen.py`) are run separately and only when `torch` is
installed.

Reproducibility drivers for the Python ports live under `scripts/sweeps/`;
`scripts/verify_against_papers.py` tabulates measured-vs-published accuracy
deltas (flagging gaps over 5 percentage points).

## Documentation conventions

- **`docs/`** is the upstream documentation site — leave its structure intact
  and mirror upstream changes there.
- **`claude-docs/`** is this fork's in-depth developer reference. When you add
  or change a feature, update the matching `claude-docs/*.md` page **and** the
  model table / source-location pointers in `README.md`. New binarizations,
  models, or mappings should get a section in their respective page, not just a
  passing mention.
- **`CLAUDE.md`** is the orientation file for automated tooling; keep its
  version, branch, and feature summary in sync with the code.

When in doubt, read two or three analogous features first and match the
surrounding conventions rather than appending alongside them.
