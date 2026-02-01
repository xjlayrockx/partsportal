# 3D Viewer Module — Progress Notes

## What's Built

### Core 3D Viewer
- Three.js scene with OrbitControls, ACES filmic tone mapping
- GLB model loading with DRACO compression support
- Automatic part indexing — walks mesh parent hierarchy to resolve part numbers (`XXX-XXXX-XX` format from Group node names)
- Raycasting for click/hover/double-click on meshes
- Fly-to-part camera animation from table row clicks
- Part info overlay (PN, description, instance count)

### Selection & Highlighting
- **Click** selects a part — bright cyan (`#00e5ff`) color + emissive glow
- **Hover** shows soft cyan preview glow (emissive 0.3)
- Color-based highlighting works on all material types (MeshBasic, MeshStandard, etc.)
- Drag detection prevents orbit/pan from triggering selection (4px threshold)

### Hide Mode (Toggle)
- Click **Hide** button to enter hide mode (button highlights orange, cursor becomes crosshair)
- Hover any part to preview, click to hide it instantly
- Click **Hide** again to exit mode, resume normal selection
- Works on all parts including 11 meshes with no part number (bumper plate, hex nuts) — these are tracked via `hiddenMeshes` Set by parent group
- **Hidden indicator** pill shows count, clickable to Show All

### Isolate
- Click a part then **Isolate** — hides everything except that part
- **Double-click** a part to isolate directly

### Show All / Reset
- **Show All** restores all hidden parts (both PN-based and direct mesh hides)
- **Reset** clears everything: selection, hover, ghost, explode, hide mode, hidden parts, filter

### Ghost Mode
- Toggle to fade all non-selected parts to 10% opacity

### Explode Mode
- Toggle to spread parts outward from model center

### Parts Filter
- Text input above parts table: "Search by part number or description..."
- Matches against part number, description, and item number (case-insensitive)
- Shows result count badge (`12/45`)
- Clear button (×)
- 3D sync: non-matching parts ghost to 10% opacity, matching parts get subtle emissive glow
- Debounced at 150ms

### Table Integration
- Table rows highlight on 3D selection and vice versa
- Hidden parts show strikethrough + "HIDDEN" badge in table
- Clicking a hidden part's row shows toast notification
- 3D mesh indicator dots (green = has mesh, dimmed = no 3D)
- Filter + hidden states coexist correctly

## Architecture

### State Variables (Module Script)
| Variable | Type | Purpose |
|----------|------|---------|
| `partIndex` | `{pn: [mesh...]}` | Part number → mesh array |
| `originalMaterials` | `Map<mesh, {colorHex, emissiveHex, ...}>` | Original material properties for reset |
| `meshToPn` | `Map<mesh, string>` | Cached mesh → PN lookup (built once at load) |
| `selectedPartNumber` | `string\|null` | Currently selected PN |
| `hoveredPN` | `string\|null` | Currently hovered PN |
| `hideModeOn` | `boolean` | Hide mode toggle |
| `hiddenPartNumbers` | `Set<string>` | Hidden PNs |
| `hiddenMeshes` | `Set<mesh>` | Directly hidden meshes (no PN) |
| `filterMatchPNs` | `Set\|null` | Active filter matches (null = no filter) |
| `ghostModeOn` | `boolean` | Ghost mode toggle |
| `explodeModeOn` | `boolean` | Explode mode toggle |

### Central Material Resolver: `applyAllMaterials()`
Single function that loops all meshes once, applying the highest-priority rule:

1. **Hidden** → `mesh.visible = false`
2. **Selected** → cyan color + emissive (0x00e5ff, 0.6)
3. **Hovered** → soft cyan emissive (0x00e5ff, 0.3)
4. **Filtered out** → ghost (opacity 0.1)
5. **Filter match** → subtle glow (emissive 0x444444, 0.3)
6. **Ghost non-selected** → ghost (opacity 0.1)
7. **Normal** → restore original properties

Materials are mutated in-place (no cloning per call). `needsUpdate` set when transparency state changes.

### Event Flow
```
Canvas click/hover → raycast → extractPartNumber (walks parents) → update state → applyAllMaterials()
                                                                 → dispatch CustomEvent → table updates
```

### Custom Events
- `glb-indexed` — model loaded, carries `partNumbers` Set
- `3d-part-selected` — part selected/deselected, carries `partNumber`
- `3d-hidden-changed` — hidden set changed, carries `hiddenPartNumbers`
- `3d-reset` — full reset occurred

### Window Exposures
`initThreeScene`, `loadGLBModel`, `flyToMeshes3D`, `selectPart3D`, `clearSelection3D`, `resetCamera3D`, `toggleGhostMode3D`, `toggleExplode3D`, `hidePart3D` (toggleHideMode), `isolatePart3D`, `showAllParts3D`, `setFilterMatches3D`

## GLB Model Notes
- File: `BRBXT7240EVG_web.glb` (20.4 MB, DRACO compressed)
- 2,803 nodes total: 1,769 mesh nodes, 962 group nodes with PNs, 72 empty nodes
- Part numbers are on **parent Group nodes** (e.g., `"216-1051-00:1"`), not mesh nodes (named `"Solid1"`, `"Solid2"`, etc.)
- 11 meshes have no PN in ancestry: 1 bumper plate + 10 hex flange nuts

## Known Issues / Future Work
- Hover preview in table → 3D (bidirectional hover highlighting)
- Quick-filter chips: "Has 3D", "Hidden", "All"
- Part count per section showing 3D coverage
- Keyboard shortcut (Ctrl+F) to focus parts filter
- `esc()` helper creates throwaway DOM element per call — could cache
- `showPartInfo()` does linear scan for description — could build PN→desc index once
