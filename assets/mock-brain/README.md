# Particle brain — SPAR

The approved v1 particle brain, packaged for use in a webapp. No npm packages, model calls, remote assets, or build step are required. Rendering uses Canvas 2D with a projected 3D particle cloud. The default mapping is cerebellum = Core Beliefs, left cerebral hemisphere = A1, and right cerebral hemisphere = A2. Each has an independent deviation input.

## Files

- `particle-brain.js`: reusable ES module exporting `createParticleBrain`.
- `particle-brain.css`: required styles; the module loads this sibling file automatically into its shadow root.
- `index.html`: interactive demo using the reusable module; serve over HTTP.
- `results.html`, `results.css`, `results-viewer.js`: Hasbro-style local replay layout with the brain on the left and chat on the right.
- `results-adapter.js`: maps saved turn records to region highlights.
- `standalone.html`: self-contained SPAR preview with the same controller API exposed as `window.brain`; open directly as a local file.
- `particle-brain.d.ts`: TypeScript declarations for the module API.

## Preview

From the `belief-revision-llm` repository's root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/assets/mock-brain/`. The demo exposes `window.brain` for trying the API in the browser console. Opening the module demo with `file://` can fail because browser module imports require HTTP; use `standalone.html` for a double-click preview.

## Mount in a webapp

Copy this directory to your app's public/static directory, preserving the relative location of the JS and CSS files. For example, placing it at `public/assets/mock-brain/` exposes `/assets/mock-brain/particle-brain.js`.

```html
<div id="model-activity"></div>
<script type="module">
  import { createParticleBrain } from '/assets/mock-brain/particle-brain.js';

  const brain = createParticleBrain(document.querySelector('#model-activity'), {
    mode: 'spar',
    controls: false,
  });

  // Start a looping simulated front-to-back wave while waiting for an answer.
  brain.setState({ generating: true, progress: null, deviations: [0, 0, 0] });

  // If the backend provides meaningful progress, pass a normalized fraction.
  brain.setState({ progress: 0.45, deviations: [0.2, 0.7, 0.1] });

  // On completion, cancellation, or failure, stop the wave.
  brain.setState({ generating: false, progress: null });

  // On route teardown/component unmount:
  // brain.destroy();
</script>
```

`controls: false` hides demo controls and captions; rotation by dragging remains available. Multiple instances are isolated using shadow DOM. The module appends its own element without replacing the container's existing content. Call it only in a browser after the container mounts; there is no server-rendering entry point.

For React, initialize inside an effect and return `() => brain.destroy()` from that effect. Keep the returned controller in a ref and call `setState` when application signals change. For other frameworks use the equivalent mount/unmount lifecycle. Do not recreate the controller on every token.

## API

`createParticleBrain(container, options)` returns a controller. Options accept `controls` (default `true`) and any `setState` field below.

| State field | Values | Meaning |
| --- | --- | --- |
| `mode` | `'hasbro'` or `'spar'` | Defaults to SPAR. |
| `generating` | boolean | Start/stop activity. When driven through the API, runs until explicitly stopped. |
| `progress` | number 0–1, or `null` | Explicit front-to-back wave position. `null` loops a simulated nine-second sweep during API-controlled generation. |
| `deviation` | number 0–1 | Core Beliefs deviation (cerebellum in SPAR); shorthand for `deviations[0]`. |
| `deviations` | `[core, a1, a2]`, each 0–1 | Three independent deviation signals. If both forms are passed, `deviation` overrides the core entry. |
| `activity` | `[core, a1, a2]`, each 0–1 | Independent blue highlights; separate from red deviation. |
| `paused` | boolean | Freeze automatic motion and simulated time. Defaults to the OS reduced-motion preference. Direct rotation and state updates remain possible. |

Numeric deviation/progress values are clamped to 0–1; nonnumeric values become zero. Deviation values are visual intensities, not raw activations or calibrated standard deviations: normalize upstream. The demo status uses `> 0.35` as an illustrative alert threshold; red intensity changes continuously.

Controller methods:

- `setState(patch)`: update only supplied fields.
- `getState()`: return a snapshot, including a copied deviations array and target view angles. `progress: null` indicates simulated progress, not a measured value.
- `setView({ yaw, pitch })`: angles in radians; pitch is clamped to −1.4…1.4. Default view: `{ yaw: -0.60, pitch: 0.27 }`; rear: `{ yaw: -Math.PI / 2, pitch: 0.27 }`.
- `refreshTheme()`: re-read CSS colors after app-specific theme changes. Document-root theme changes are observed automatically.
- `destroy()`: cancel animation, disconnect observers, remove the media-query listener, and remove the mounted element. Safe to call repeatedly. Destroy before removing its parent from the document.
- `element`: mounted host element. It emits `brainchange` with a state snapshot for manual slider, project, pause, and view interactions; programmatic `setState` does not emit it.

The demo's Generate button is deliberately a single nine-second sweep. The API's `generating: true` uses persistent activity until the app stops it. Neither performs inference.

## Mapping and appearance

Hasbro mode lights a cortical region for deviation and leaves the cerebellum inactive. SPAR maps both cerebellar halves to Core Beliefs, the anatomical left cerebral hemisphere to A1, and the right to A2. Core deviation colors the cerebellum; A1/A2 deviations color their respective large hemispheres. Anatomical sides rotate with the object; they do not mean screen-left/screen-right. The three signals stay independent.

Floating A1 and A2 labels retain connector lines to their respective regions;
left/right descriptions remain in the controls and region key.

The approved geometry includes the wider frontal profile, a soft center seam, sculpted folds, a tucked-in cerebellum, and two-axis dragging. It is an artistic metaphor, not an anatomical or measured network visualization.

Override styles on the container or controller's host element:

```css
#model-activity {
  --brain-height: 360px;
  --brain-blue: #65b5ff;
  --brain-red: #ff627d;
  --brain-text: #e4edf9;
  --brain-muted: #99a9bc;
}
```

Without overrides, colors follow the browser's light/dark preference. The background is transparent. Height defaults to 440px, or 330px on narrow viewports. Keep a positive container width. Touch drags inside the canvas rotate the brain and therefore do not scroll the page.

## Integration limits

The renderer uses about 13,700 particles and sorts them per frame. Profile on target mobile devices before using many instances at once. It requires a modern browser with ES modules, Canvas 2D, shadow DOM, ResizeObserver, and CSS `light-dark()`. A strict CSP must allow the module and stylesheet from their serving origin. Use the ES module for application integration; the standalone file embeds a snapshot of that module and its CSS.

No application credentials, server connection, or telemetry are included. Wire your own generation lifecycle and normalized deviation signals. Hasbro and SPAR repository copies do not automatically synchronize. Update this module for production integration; the standalone snapshot does not rebuild automatically from it.

## Replay current experiment results

Open `http://localhost:8000/assets/mock-brain/results.html` after starting the local server above. Select `dialogues_full_crossed.jsonl`, choose a conversation, and step through its turns. The desktop layout keeps the brain and belief details on the left and a scrolling chat transcript on the right; narrow screens stack them. Chat history appears through the selected turn, and moving backward hides subsequent exchanges. The live NDIF connection is explicitly marked as disconnected. No upload, inference, or database access occurs. The JSONL must use `storage.build_conversation_record`'s current format: one conversation per line with a nonempty `rows` array. The viewer also displays the user/assistant text, core claim, displayed auxiliary texts, and recorded assistant probe scores. Missing scores remain “not measured.”

`results-adapter.js` exports `turnToBrainState(row)`, returning `{patch, reasons}`. Apply a recorded turn directly with:

```js
import { turnToBrainState } from './results-adapter.js';
brain.setState(turnToBrainState(row).patch);
```

Highlight mapping:

- `active_auxiliaries`: a soft blue highlight on the named cerebral hemisphere.
- `inject_evidence` plus `evidence_role` beginning with `core_`: bright Core Beliefs/cerebellum highlight.
- Injected `evidence_targets`: bright highlights on targeted regions.
- `probe_ran_after_turn` plus `changed_node` of `C`, `A1`, or `A2`: a bright highlight on the newly reported changed region. Carried-forward probe fields do not retrigger it. `source` and `none` have no dedicated region.

Evidence targets and auxiliary IDs already use displayed/counterbalanced A1/A2 in the stored results; do not remap them again. Every turn clears the previous turn's highlights. Belief scores are not deviations: replay keeps red deviation at zero. Supply independent measured/normalized deviations via the API if desired.

This is replay of completed results, not a live connection to the Python runner. The runner currently saves results after completed conversations/cells; a future live transport can use the same adapter and `activity` API without changing the visual mapping. The viewer reads the selected file into memory; use a subset for very large result files. No current result file is bundled with the viewer.

Run the adapter's offline regression checks from the repo root:

```sh
node assets/mock-brain/results-adapter.test.mjs
```

Use **Load latest local run** in the results viewer to read the repository-root
`dialogues_full_crossed.jsonl` from the local server. **Open results** still
accepts other saved JSONL files. Loading does not launch inference or upload data.

**Auto-play · 3s** advances the selected conversation every three seconds and
loops from its final turn to its first. **Pause replay**, manual navigation,
loading results, or selecting another conversation stops playback. Brain motion
has its own pause control. Replay data currently has no deviation measurements;
red intensity stays zero, while recorded belief changes use blue highlights.
