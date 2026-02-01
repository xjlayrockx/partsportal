#!/usr/bin/env python3
"""
Parts Manual Viewer — Flask backend

Upload PDF parts manuals, process with YOLO + OCR + table extraction,
and serve an interactive web viewer.
"""

import os
import re
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory)
from werkzeug.utils import secure_filename

from pipeline import ProcessingState, process_manual

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Directories
BASE_DIR = Path('/app/data')
MANUALS_DIR = BASE_DIR / 'manuals'
UPLOAD_DIR = BASE_DIR / 'uploads'
MODELS_DIR = Path('/models')
GLB_DIR = BASE_DIR / 'glb_models'

for d in [MANUALS_DIR, UPLOAD_DIR, GLB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Processing state tracking (thread-safe via GIL for simple dict access)
processing_states: dict[str, ProcessingState] = {}
processing_lock = threading.Lock()

# Per-manual locks for manifest.json read/write
_manifest_locks: dict[str, threading.Lock] = {}
_manifest_locks_lock = threading.Lock()


def _get_manifest_lock(manual_id: str) -> threading.Lock:
    """Get or create a per-manual lock for manifest operations."""
    with _manifest_locks_lock:
        if manual_id not in _manifest_locks:
            _manifest_locks[manual_id] = threading.Lock()
        return _manifest_locks[manual_id]


def _derive_model_name(filename):
    """Derive a display-friendly model name from the PDF filename."""
    stem = Path(filename).stem
    stem = re.sub(r'_web$', '', stem, flags=re.IGNORECASE)
    return stem


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum upload size is 500 MB.'}), 413


# ── Pages ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── Upload & Processing ──────────────────────────────────────────────

@app.route('/api/upload-manual', methods=['POST'])
def upload_manual():
    """Upload a PDF and start background processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are accepted'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manual_id = f"{timestamp}_{Path(filename).stem}"

    # Save uploaded PDF
    pdf_path = UPLOAD_DIR / f"{manual_id}.pdf"
    file.save(str(pdf_path))

    # Create processing state
    state = ProcessingState()
    with processing_lock:
        processing_states[manual_id] = state

    # Start background processing
    model_path = str(MODELS_DIR / 'best.pt')
    thread = threading.Thread(
        target=_run_pipeline,
        args=(pdf_path, manual_id, MANUALS_DIR, model_path, state),
        daemon=True,
    )
    thread.start()

    return jsonify({
        'manual_id': manual_id,
        'message': 'Processing started',
    })


def _run_pipeline(pdf_path: Path, manual_id: str, output_dir: Path,
                  model_path: str, state: ProcessingState):
    """Background thread: run the full pipeline."""
    try:
        process_manual(pdf_path, manual_id, output_dir,
                       model_path=model_path, state=state)
    except Exception as e:
        state.error = str(e)
        state.done = True


@app.route('/api/processing-status/<manual_id>')
def processing_status(manual_id):
    """Poll processing progress."""
    with processing_lock:
        state = processing_states.get(manual_id)

    if state is None:
        # Check if already processed
        manifest = MANUALS_DIR / manual_id / 'manifest.json'
        if manifest.exists():
            return jsonify({'done': True, 'progress': 100,
                            'message': 'Complete'})
        return jsonify({'error': 'Unknown manual ID'}), 404

    return jsonify(state.to_dict())


# ── Manual Data ───────────────────────────────────────────────────────

@app.route('/api/manuals')
def list_manuals():
    """List all processed manuals."""
    manuals = []
    if MANUALS_DIR.exists():
        for d in sorted(MANUALS_DIR.iterdir(), reverse=True):
            manifest_path = d / 'manifest.json'
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    manuals.append({
                        'manual_id': manifest['manual_id'],
                        'filename': manifest['filename'],
                        'model_name': manifest.get('model_name') or _derive_model_name(manifest['filename']),
                        'total_pages': manifest.get('total_pages', 0),
                        'section_count': len(manifest.get('sections', [])),
                        'total_callouts': manifest.get('processing_log', {}).get(
                            'total_callouts_detected', 0),
                        'total_matched': manifest.get('processing_log', {}).get(
                            'total_matched', 0),
                        'has_thumbnail': (MANUALS_DIR / manifest['manual_id'] / 'pages' / 'page_001.png').exists(),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    return jsonify(manuals)


@app.route('/api/manual/<manual_id>')
def get_manual(manual_id):
    """Get the full manifest for a manual."""
    manifest_path = MANUALS_DIR / manual_id / 'manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': 'Manual not found'}), 404
    with open(manifest_path) as f:
        return jsonify(json.load(f))


@app.route('/api/manual/<manual_id>/rename', methods=['PUT'])
def rename_manual(manual_id):
    """Update the display name of a manual."""
    manifest_path = MANUALS_DIR / manual_id / 'manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': 'Manual not found'}), 404
    data = request.get_json()
    new_name = (data or {}).get('model_name', '').strip()
    if not new_name:
        return jsonify({'error': 'Name cannot be empty'}), 400
    with _get_manifest_lock(manual_id):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest['model_name'] = new_name
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    return jsonify({'model_name': new_name})


@app.route('/api/manual/<manual_id>/section/<int:section_idx>/rename', methods=['PUT'])
def rename_section(manual_id, section_idx):
    """Update the title of a section within a manual."""
    manifest_path = MANUALS_DIR / manual_id / 'manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': 'Manual not found'}), 404
    data = request.get_json()
    new_title = (data or {}).get('title', '').strip()
    if not new_title:
        return jsonify({'error': 'Title cannot be empty'}), 400
    with _get_manifest_lock(manual_id):
        with open(manifest_path) as f:
            manifest = json.load(f)
        sections = manifest.get('sections', [])
        if section_idx < 0 or section_idx >= len(sections):
            return jsonify({'error': 'Section not found'}), 404
        sections[section_idx]['title'] = new_title
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    return jsonify({'title': new_title})


@app.route('/api/manual/<manual_id>/section/<int:section_idx>', methods=['DELETE'])
def delete_section(manual_id, section_idx):
    """Remove a section (page) from a manual."""
    manifest_path = MANUALS_DIR / manual_id / 'manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': 'Manual not found'}), 404
    with _get_manifest_lock(manual_id):
        with open(manifest_path) as f:
            manifest = json.load(f)
        sections = manifest.get('sections', [])
        if section_idx < 0 or section_idx >= len(sections):
            return jsonify({'error': 'Section not found'}), 404
        sections.pop(section_idx)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    return jsonify({'ok': True, 'remaining': len(sections)})


@app.route('/api/manual/<manual_id>/image/<filename>')
def get_image(manual_id, filename):
    """Serve a page image."""
    pages_dir = MANUALS_DIR / manual_id / 'pages'
    filename = secure_filename(filename)
    if not (pages_dir / filename).exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(str(pages_dir), filename)


@app.route('/api/manual/<manual_id>', methods=['DELETE'])
def delete_manual(manual_id):
    """Delete a processed manual."""
    manual_dir = MANUALS_DIR / manual_id
    if manual_dir.exists():
        shutil.rmtree(str(manual_dir))
    # Also remove uploaded PDF
    pdf_path = UPLOAD_DIR / f"{manual_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    # Clean up state
    with processing_lock:
        processing_states.pop(manual_id, None)
    return jsonify({'message': 'Deleted'})


# ── GLB Models ────────────────────────────────────────────────────────

@app.route('/api/glb')
def list_glb():
    """List available GLB model files."""
    files = []
    if GLB_DIR.exists():
        for f in sorted(GLB_DIR.iterdir()):
            if f.suffix.lower() == '.glb' and f.is_file():
                files.append({
                    'filename': f.name,
                    'size_mb': round(f.stat().st_size / (1024 * 1024), 1),
                })
    return jsonify(files)


@app.route('/api/glb/<filename>')
def serve_glb(filename):
    """Serve a GLB model file."""
    filename = secure_filename(filename)
    if not (GLB_DIR / filename).exists():
        return jsonify({'error': 'GLB file not found'}), 404
    return send_from_directory(str(GLB_DIR), filename,
                               mimetype='model/gltf-binary')


# ── Search ────────────────────────────────────────────────────────────

@app.route('/api/search')
def search():
    """Search part numbers and descriptions across all manuals."""
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify([])

    results = []
    if MANUALS_DIR.exists():
        for d in MANUALS_DIR.iterdir():
            manifest_path = d / 'manifest.json'
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            manual_id = manifest['manual_id']
            manual_name = manifest['filename']

            for sec in manifest.get('sections', []):
                for part in sec.get('parts_table', []):
                    pn = part.get('part_number', '')
                    desc = part.get('description', '')
                    if query in pn.lower() or query in desc.lower():
                        results.append({
                            'manual_id': manual_id,
                            'manual_name': manual_name,
                            'section_title': sec.get('title', ''),
                            'section_index': manifest['sections'].index(sec),
                            'item': part['item'],
                            'part_number': pn,
                            'description': desc,
                            'quantity': part.get('quantity', 1),
                        })

    # Limit results
    return jsonify(results[:100])


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
