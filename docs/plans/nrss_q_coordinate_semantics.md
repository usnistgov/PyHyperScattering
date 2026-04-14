# NRSS q-coordinate semantics unification plan

## Purpose

This note records the implementation and validation plan for unifying q-coordinate semantics between:

- NRSS simulation outputs
- PyHyperScattering reduction workflows
- downstream analytical form-factor comparisons

It is intentionally stored outside the Sphinx tree so it can be used as a dev-facing design and execution note without changing docs tooling.

## Current diagnosis

The mismatch described in `/homes/deand/dev/NRSS/test-reports/nrss_q_coordinate_reduction_issue_draft.md` is real and is visible in current code:

- `WPIntegrator` in PyHyperScattering is a warp-polar remesher over existing `qx/qy` coordinates.
  - It finds the image center from `qx=0` and `qy=0`.
  - It runs `warp_polar`.
  - It labels the radial axis with `sqrt(qx^2 + qy^2)`.
  - This is `q_perp`, not a detector-geometry-corrected `|q|`.
- NRSS and `cyrsoxsLoader` currently export FFT-style `qx/qy` detector-plane coordinates derived from `PhysSize` and array shape.
- The modern NRSS 3D backend already uses detector-aware projection geometry internally during intensity formation, but that geometry is not preserved in the exported xarray metadata.
- As a result:
  - 2D NRSS reduced with `WPIntegrator` is semantically fine.
  - 3D NRSS reduced with `WPIntegrator` carries a radial axis that is semantically wrong for comparison to geometry-aware experiment reduction or pure analytical `I(|q|)`.

## Repository state at time of writing

### PyHyperScattering

- Repo: `/homes/deand/dev/PyHyperScattering`
- Active branch:
  - `221-unify-q-coordinate-semantics-between-nrss-simulation-outputs-and-pyhyperscattering-reduction-workflows`
- Worktree state:
  - clean
- Important finding:
  - this branch is currently identical to `main`
  - there is no implementation on the branch yet

### NRSS

- Repo: `/homes/deand/dev/NRSS`
- Active branch:
  - `backend/cupy-cyrsoxs-clone`
- Worktree state:
  - dirty
  - modified: `src/NRSS/backends/cupy_rsoxs.py`
  - several deleted design/spec markdown files in repo root
- Consequence:
  - phase 1 should avoid requiring NRSS code changes
  - local NRSS should initially be used only as a consumer/validation target

## Key code anchors

### PyHyperScattering

- `WPIntegrator`
  - `/homes/deand/dev/PyHyperScattering/src/PyHyperScattering/WPIntegrator.py`
- `cyrsoxsLoader`
  - `/homes/deand/dev/PyHyperScattering/src/PyHyperScattering/cyrsoxsLoader.py`
- `PFGeneralIntegrator`
  - `/homes/deand/dev/PyHyperScattering/src/PyHyperScattering/PFGeneralIntegrator.py`
- Existing tests
  - `/homes/deand/dev/PyHyperScattering/tests/test_WPintegrator.py`
  - `/homes/deand/dev/PyHyperScattering/tests/test_pf_general_integrator_basic.py`

### NRSS

- legacy xarray export
  - `/homes/deand/dev/NRSS/src/NRSS/morphology.py`
- cupy backend xarray export
  - `/homes/deand/dev/NRSS/src/NRSS/backends/cupy_rsoxs.py`
- maintained sphere form-factor validation
  - `/homes/deand/dev/NRSS/tests/validation/test_analytical_sphere_form_factor.py`
- existing plot-writing validation pattern
  - `/homes/deand/dev/NRSS/tests/validation/test_sphere_contrast_scaling.py`
  - `/homes/deand/dev/NRSS/tests/validation/test_sphere_orientational_contrast_scaling.py`

## High-level decision

Do **not** force the first implementation through `PFGeneralIntegrator` or synthetic pyFAI metadata.

Reason:

- `PFGeneralIntegrator` is built around raw detector images with `pix_x/pix_y` plus pyFAI-style geometry metadata.
- NRSS currently provides `energy/qy/qx` arrays, not raw detector pixel stacks with a stable calibration contract.
- inventing fake pyFAI geometry in phase 1 adds risk and does not solve the metadata problem

Therefore:

- build a new PyHyperScattering-side `NRSSIntegrator` first
- keep it native to NRSS-style `qx/qy` arrays
- use explicit semantics and attrs
- validate with local NRSS before any NRSS metadata bridge is merged

## Planned end state

### For 2D NRSS output

- reduction remains a radial remesh over detector-plane reciprocal coordinates
- the semantics should be explicit:
  - radial axis is `q_perp`
- the result may still use a dimension named `q` for compatibility, but must carry metadata that says it is `q_perp`

### For 3D NRSS output

- reduction should still use the simulated detector-plane image intensity
- but the radial coordinate should be corrected from detector-plane `q_perp` to full `|q|`
- this should match the geometry already used inside the 3D NRSS detector-projection backend

### Migration target

- short term:
  - keep detector-space `WPIntegrator`-based validations where they already exist
  - add `NRSSIntegrator` validations against pure analytical form factors
- long term:
  - replace detector-space analytical comparison tests in NRSS with `NRSSIntegrator` vs pure analytical `I(|q|)` tests

## Flow 1: PyHyperScattering implementation first

This is the required first step.

### Scope

Implement a new `NRSSIntegrator` class in PyHyperScattering.

Suggested file:

- `/homes/deand/dev/PyHyperScattering/src/PyHyperScattering/NRSSIntegrator.py`

Expected exports also need to be wired through whatever import surface currently exposes integrators, for example:

- `src/PyHyperScattering/integrate.py`
- package `__init__` if needed by current conventions

### Required behavior

#### Input contract

`NRSSIntegrator` should accept an xarray with NRSS-like coordinates:

- spatial dims:
  - `qy`, `qx`
- optional stack dims:
  - `energy`
  - any additional grouping/index coords that current integrators already tolerate

It should also accept explicit metadata kwargs for cases where attrs are missing:

- `phys_size_nm`
- `shape_zyx` or at least `z_dim`
- `energy_ev` or an energy coord
- `projection_mode` or equivalent semantic hint

It should first look in `img.attrs`, then use explicit kwargs as fallback.

### Semantic mode detection

The integrator must distinguish at least two modes:

#### Mode A: reciprocal-plane / 2D semantics

Use this when:

- `z_dim == 1`
- or attrs explicitly say the output is reciprocal-plane / 2D

Behavior:

- reduce exactly like `WPIntegrator`
- radial axis semantics = `q_perp`

#### Mode B: detector-aware / 3D semantics

Use this when:

- `z_dim > 1`
- or attrs explicitly say detector-aware / 3D projection

Behavior:

- use the same detector-plane intensity image
- keep the same angular remesh approach
- but relabel the radial axis using corrected `|q|`

### Core geometry formula for 3D mode

This formula should match the logic already present in NRSS cupy backend detector-projection geometry.

Definitions:

- `q_perp = sqrt(qx^2 + qy^2)`
- `k = 2*pi / lambda`
- with energy in eV:
  - `lambda_nm = 1239.84197 / energy_ev`
  - `k = 2*pi / lambda_nm`

Detector-aware projection z component:

```text
qz = -k + sqrt(k^2 - q_perp^2)
```

Corrected magnitude:

```text
q = sqrt(q_perp^2 + qz^2)
```

Constraints:

- values are only valid where `k^2 - q_perp^2 >= 0`
- if the detector-plane image already contains invalid edge regions as NaN, preserve that masking behavior

### Output contract

Return a `chi/q` reduced xarray compatible with existing downstream usage.

Recommended output shape:

- `chi`, `q`
- plus any stacked axes preserved the same way `WPIntegrator` preserves them

Required metadata:

- `radial_semantics`
  - `"q_perp"` for 2D mode
  - `"q_abs_detector_corrected"` for 3D mode
- `source_integrator`
  - `"NRSSIntegrator"`
- `nrss_semantic_mode`
  - `"2d_reciprocal_plane"` or `"3d_detector_aware"`
- `phys_size_nm`
- `z_dim`
- `energy_ev` if scalar

Recommended auxiliary coordinate for 3D mode:

- keep output dim name `q` for compatibility
- also attach a `q_perp` coordinate or attr summary when possible

### Compatibility rule

Do **not** change `WPIntegrator` behavior in phase 1.

Only add a clarifying docstring or warning so users understand:

- `WPIntegrator` radial output is `q_perp`
- it is still correct for 2D NRSS reciprocal-plane output
- it is not the geometry-correct reduction for 3D detector-aware NRSS output

### PyHyperScattering tests to add

Add a new test module:

- `/homes/deand/dev/PyHyperScattering/tests/test_NRSSIntegrator.py`

At minimum include:

1. `test_nrss_integrator_2d_matches_wp_semantics`
   - build a synthetic `qx/qy` image
   - set `z_dim=1`
   - verify reduced radial axis equals `WPIntegrator` radial axis
   - verify attrs say `radial_semantics == "q_perp"`

2. `test_nrss_integrator_3d_corrects_radial_axis`
   - build a synthetic `qx/qy` image with an energy
   - set `z_dim > 1`
   - verify output `q` differs from `sqrt(qx^2 + qy^2)` at higher q
   - verify corrected values match the explicit formula above

3. `test_nrss_integrator_falls_back_to_explicit_kwargs`
   - no attrs on the xarray
   - pass explicit `phys_size_nm`, `z_dim`, and `energy_ev`
   - verify reduction succeeds

4. `test_nrss_integrator_preserves_stack_axis`
   - use an `energy` stack or another grouping coord
   - verify output preserves group axis

5. `test_wp_integrator_semantics_note`
   - if behavior is not changed, at least assert the warning/doc metadata path if one is added

### PyHyperScattering docs to update

Because this plan is outside Sphinx, no docs build change is required now.

But after implementation, update at least:

- `/homes/deand/dev/PyHyperScattering/docs/source/getting_started/integration.rst`

Add short text clarifying:

- `WPIntegrator` is a `qx/qy -> chi/q_perp` polar remesher
- `NRSSIntegrator` is the correct reducer for NRSS detector-aware 3D outputs

## Flow 2: NRSS validation and migration

This phase starts after `NRSSIntegrator` exists and has passing unit tests in PyHyperScattering.

### Immediate goal

Use local NRSS as the validation target for `NRSSIntegrator` without requiring NRSS metadata changes first.

### Environment notes

Per local instructions:

- for joint NRSS + PyHyperScattering work use environment:
  - `nrss-pyhyper-dev`
- prefer `mamba` for environment work
- do not fire more than one env command at once

### First validation target

Base the dev-only validation style on:

- `/homes/deand/dev/NRSS/tests/validation/test_analytical_sphere_form_factor.py`

This file already provides:

- sphere morphology construction
- analytical sphere form-factor comparison
- backend runtime path helpers
- optional plot writing to `test-reports/`

Existing test-report plot pattern is also visible in:

- `/homes/deand/dev/NRSS/tests/validation/test_sphere_contrast_scaling.py`
- `/homes/deand/dev/NRSS/tests/validation/test_sphere_orientational_contrast_scaling.py`

They use:

- `PLOT_DIR = REPO_ROOT / "test-reports" / "<name>-dev"`
- `WRITE_VALIDATION_PLOTS = os.environ.get("NRSS_WRITE_VALIDATION_PLOTS", "").strip() == "1"`

Follow that exact pattern.

### NRSS dev-only tests to add

These should initially be dev validation tests, not immediately promoted to maintained reference parity until the semantics settle.

Suggested new test module:

- `/homes/deand/dev/NRSS/tests/validation/test_nrss_integrator_sphere_form_factor.py`

If the team prefers to keep it explicitly provisional, use a clearer name such as:

- `/homes/deand/dev/NRSS/tests/validation/test_nrss_integrator_sphere_form_factor_dev.py`

The test should:

1. build or reuse the same sphere morphology style as `test_analytical_sphere_form_factor.py`
2. run NRSS backend(s) to produce detector-plane xarray output
3. reduce that output twice:
   - once with `WPIntegrator`
   - once with `NRSSIntegrator`
4. compare both reductions to the same pure analytical sphere form factor
5. write plots when `NRSS_WRITE_VALIDATION_PLOTS=1`

### Required visualizations

Each validation run should generate a comparison figure showing at minimum:

1. detector image in `qx/qy`
2. `WPIntegrator` reduced curve vs detector-space expectation if still useful
3. `NRSSIntegrator` reduced curve vs pure analytical `I(|q|)`
4. residual or log-error plot
5. minima alignment markers if available

The important expected result is:

- `NRSSIntegrator` should align with the pure analytical form factor
- `WPIntegrator` should visibly diverge in the 3D case at higher q

### Suggested plot/report directory

Use a new dev report directory such as:

- `/homes/deand/dev/NRSS/test-reports/nrss-integrator-sphere-dev`

### Suggested scenario matrix

At minimum:

#### 2D scenario

- `z_dim == 1`
- expected:
  - `WPIntegrator` and `NRSSIntegrator` should agree
  - both should align with the 2D-appropriate analytical interpretation

#### 3D scenario

- `z_dim > 1`
- expected:
  - `WPIntegrator` follows detector-plane `q_perp`
  - `NRSSIntegrator` follows analytical `|q|`
  - `NRSSIntegrator` should provide the physically correct comparison target

Possible concrete settings:

- reuse sphere diameters already maintained:
  - `70 nm`
  - `128 nm`
- start with one energy:
  - `285.0 eV`
- reuse `PhysSize` and shape conventions already established in the analytical sphere test where practical

### Threshold strategy

For the first NRSS dev-only validation, do not start with aggressive fixed thresholds.

Instead:

1. generate plots
2. measure:
   - RMS log error
   - P95 absolute log error
   - minima position error
   - integrated intensity agreement over the chosen q window
3. inspect results
4. then set thresholds appropriate for maintained migration

### Long-term NRSS migration

After `NRSSIntegrator` is validated:

1. update NRSS maintained analytical sphere tests so the primary analytical comparison path is:
   - `NRSS output -> NRSSIntegrator -> pure analytical I(|q|)`
2. demote or remove detector-space `WPIntegrator` analytical checks where they are only preserving the old semantics
3. keep a small compatibility test proving that:
   - `WPIntegrator` still behaves as detector-plane `q_perp`
   - it remains correct for 2D reciprocal-plane reductions

## Metadata bridge to add later in NRSS

This is intentionally **not** part of flow 1.

After PyHyperScattering-side reduction is working and validated, NRSS should export enough attrs for automatic mode detection.

### Suggested attrs on exported xarray

- `phys_size_nm`
- `num_zyx`
- `z_dim`
- `nrss_output_semantics`
  - `"2d_reciprocal_plane"`
  - `"3d_detector_aware"`
- `backend`
- `backend_options`
- `energy_ev` if scalar or rely on energy coord

### Likely files to patch later

- `/homes/deand/dev/NRSS/src/NRSS/morphology.py`
- `/homes/deand/dev/NRSS/src/NRSS/backends/cupy_rsoxs.py`

### Important caution

Do not mix the metadata-bridge change into the first PyHyperScattering implementation unless absolutely necessary.

Reason:

- PyHyper should first be able to operate on current local NRSS output via explicit kwargs
- that keeps the semantic correction testable before any cross-repo contract change lands

## Concrete execution order

### Step 1

In PyHyperScattering:

- add `NRSSIntegrator`
- add unit tests
- keep `WPIntegrator` behavior unchanged

### Step 2

Smoke-test `NRSSIntegrator` on synthetic xarrays in PyHyperScattering only.

### Step 3

Use local NRSS output directly with `NRSSIntegrator` by passing explicit metadata kwargs.

### Step 4

Add NRSS dev-only sphere-form-factor validations with optional plot writing.

### Step 5

Confirm expected semantic split:

- 2D:
  - `WPIntegrator == NRSSIntegrator`
- 3D:
  - `NRSSIntegrator` matches pure analytical form factor better than `WPIntegrator`

### Step 6

Only after the above succeeds, add the NRSS xarray metadata bridge.

### Step 7

Promote the NRSS analytical comparison migration:

- replace detector-space analytical tests with `NRSSIntegrator` / pure analytical checks where appropriate

## Resume checklist for a fresh context

If resuming from a new session, do this in order:

1. read:
   - `/homes/deand/dev/PyHyperScattering/docs/plans/nrss_q_coordinate_semantics.md`
   - `/homes/deand/dev/NRSS/test-reports/nrss_q_coordinate_reduction_issue_draft.md`
2. confirm branch state:
   - PyHyperScattering branch `221-unify-q-coordinate-semantics-between-nrss-simulation-outputs-and-pyhyperscattering-reduction-workflows`
   - NRSS branch `backend/cupy-cyrsoxs-clone`
3. inspect current code anchors:
   - `WPIntegrator.py`
   - `cyrsoxsLoader.py`
   - `morphology.py`
   - `cupy_rsoxs.py`
4. implement `NRSSIntegrator` in PyHyperScattering first
5. add `test_NRSSIntegrator.py`
6. run PyHyperScattering tests
7. switch to `nrss-pyhyper-dev`
8. validate with local NRSS sphere outputs
9. add NRSS dev-only visualization tests
10. only then discuss NRSS metadata export changes

## Commands likely to be useful

These are not authoritative, just a restart aid.

### PyHyperScattering tests

```bash
pytest /homes/deand/dev/PyHyperScattering/tests/test_WPintegrator.py
pytest /homes/deand/dev/PyHyperScattering/tests/test_NRSSIntegrator.py
```

### NRSS analytical sphere reference

```bash
pytest /homes/deand/dev/NRSS/tests/validation/test_analytical_sphere_form_factor.py -k sphere
```

### NRSS dev validation with plots

```bash
NRSS_WRITE_VALIDATION_PLOTS=1 pytest /homes/deand/dev/NRSS/tests/validation/test_nrss_integrator_sphere_form_factor.py
```

## Non-goals for the first implementation

- rewriting `WPIntegrator`
- replacing PyHyperScattering reduction with synthetic pyFAI geometry
- requiring NRSS metadata contract changes before testing
- migrating all NRSS validation files at once
- touching unrelated dirty NRSS worktree changes

## Success criteria

The work is successful when all of the following are true:

1. PyHyperScattering exposes a working `NRSSIntegrator`.
2. For 2D NRSS output, `NRSSIntegrator` reproduces `WPIntegrator` semantics.
3. For 3D NRSS output, `NRSSIntegrator` emits corrected radial `q` semantics.
4. Local NRSS sphere validation shows `NRSSIntegrator` aligns with pure analytical form factor better than `WPIntegrator`.
5. Dev-only NRSS plots clearly show the semantic difference and support eventual migration of maintained tests.

