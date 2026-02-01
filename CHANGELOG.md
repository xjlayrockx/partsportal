# Parts Viewer — Development Log

## Session: 2026-01-31

### Features Added

#### Manual Card Thumbnails + Model Name Display
- **Thumbnail**: Manual library cards now show a first-page screenshot thumbnail (`page_001.png`) with `object-fit: contain` so the full page is visible
- **Model Name**: Cards display a cleaned-up model name instead of the raw PDF filename. Extracted by stripping `.pdf` extension and `_web` suffix
- **Editable Name**: Hover a card to reveal a pencil icon — click to inline-edit the name. Saved to manifest via `PUT /api/manual/<id>/rename`
- **Backward Compatible**: Old manifests without `model_name` derive it from filename on-the-fly

#### All Pages Navigable in Sidebar
- `pair_sections()` in `pipeline.py` now includes every PDF page as a section, not just diagram pages
- Cover, TOC, and other non-diagram pages appear in the sidebar with a `Page N` subtitle
- Table pages already consumed by a diagram group are not duplicated
- Sections sorted by page number to follow PDF order
- Added `page_type` field to each section (`diagram`, `table`, `other`)

#### Section Title Fix
- Changed `^\d+\s*` regex to `^\d+\s+` in `extract_section_title()` to preserve measurements like `54" Deck` (the `"` after digits prevents stripping)

#### Editable Section Titles
- Pencil icon on each sidebar item for inline rename
- `PUT /api/manual/<id>/section/<idx>/rename` endpoint persists to manifest
- Enter saves, Escape cancels, click-away saves

#### Removable Pages
- `×` button on each sidebar item to delete a section
- `DELETE /api/manual/<id>/section/<idx>` endpoint removes from manifest
- Adjusts current view if the active section is deleted

#### 3D View — Hide Sidebar + Show All Parts
- Switching to 3D hides the sidebar, giving the 3D viewer more space
- Parts list shows all unique parts from every section (deduplicated by part number)
- Switching back to 2D restores sidebar and current section's parts

#### 3D Part Selection — Ghost Mode + Fly-To
- Clicking a part in the parts list while in 3D mode:
  1. Selects the part in the 3D scene (cyan highlight)
  2. Auto-enables ghost mode (rest of model goes translucent at 10% opacity)
  3. Camera flies to the selected part

#### "View in 3D" Button (2D → 3D with Section Filter)
- `3D →` button in the 2D zoom controls bar
- Switches to 3D and hides all parts NOT in the current section's parts list
- Uses `isolatePartNumbers3D()` which adds non-matching parts to `hiddenPartNumbers`
- "Show All" button restores everything

#### 3D Part Selection → Parts List Sync
- Clicking a part on the 3D model highlights and scrolls the matching row to the center of the parts panel
- Uses direct container scroll calculation for reliability with nested scroll containers

---

### Bug Fixes

#### Raycasting Through Hidden Parts
- `raycaster.intersectObjects()` was hitting invisible mesh geometry, blocking clicks on parts underneath
- Fix: Filter raycast results with `hit.object.visible` so hidden meshes are skipped

#### `glb-indexed` Event Overwrote 3D Parts List
- When the GLB model finished loading, the `glb-indexed` handler re-rendered the table using only the current section's parts, overwriting the full combined list
- Fix: Handler now checks `viewMode` and calls `renderAllPartsTable()` when in 3D mode

#### `isolatePartNumbers` Null Check Bug
- `if (!pn && !keepPNs.has(pn))` — `pn` is null, so `keepPNs.has(null)` is always false, hiding ALL meshes without a part number
- Fix: Changed to `if (!pn)` — meshes without a part number are correctly hidden when isolating

#### Duplicate Parts in 3D List
- Same part appearing in multiple sections was listed multiple times
- Fix: Deduplicate by normalized part number before rendering

---

### Critical Performance & Stability Fixes

#### Three.js Memory Leaks
- **Problem**: Materials, geometries, and textures were never disposed when switching GLB models
- **Fix**: Added `disposeMaterial()` helper that disposes all texture maps and the material itself. Old model is fully traversed on removal — every mesh's geometry and materials are disposed

#### Animation Loop Never Cancelled
- **Problem**: `requestAnimationFrame` loop kept running when switching to 2D view. Multiple loops accumulated
- **Fix**: Extracted into `startRenderLoop()` / `stopRenderLoop()`. `switchViewMode` stops the loop when leaving 3D, starts it when entering

#### Manifest File Race Conditions
- **Problem**: Rename, section rename, and section delete all read/write `manifest.json` without locking. Concurrent requests could corrupt data
- **Fix**: Added per-manual `threading.Lock` system (`_get_manifest_lock`). All manifest-mutating endpoints acquire the lock before read/write

#### YOLO on Non-Diagram Pages
- **Problem**: After `pair_sections` was changed to include all pages, YOLO detection ran on covers, TOC, and table pages wastefully
- **Fix**: `detect_callouts()` checks `page_type` and skips non-diagram sections

---

### Files Modified
- `pipeline.py` — `_derive_model_name()`, `pair_sections()`, `extract_section_title()`, `detect_callouts()`
- `app.py` — `_derive_model_name()`, `/api/manuals`, `/api/manual/<id>/rename`, `/api/manual/<id>/section/<idx>/rename`, `/api/manual/<id>/section/<idx>` DELETE, manifest locking
- `templates/index.html` — Card thumbnails, editable names, sidebar rename/delete, 3D view changes, render loop management, resource disposal, raycast fix, parts list dedup

---

### Known Issues (Not Yet Fixed)
- **Race condition on GLB load**: Rapidly clicking "3D" can trigger multiple parallel GLB loads
- **ResizeObserver never disconnected**: Created on each `initThreeScene()` call
- **Pan offset not reset on section switch**: Switching sections keeps old pan position
- **Processing states dict grows unbounded**: Every upload adds an entry never cleaned up
- **PDF opened multiple times in pipeline**: Could pass a single `fitz.Document` through
- **No input length validation on renames**: User could set very long names
- **Tooltip can render off-screen**: No viewport bounds checking
- **Zoom resets on every section switch**: `zoomFit()` called in `renderSection()`
- **Processing poll retries forever**: No max retry limit on network errors
