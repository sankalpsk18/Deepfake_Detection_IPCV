# Prompt Plan for Fixing the Deepfake Detection Project with Nemotron (OpenRouter)

## Rate-limit strategy

OpenRouter's free models have tight limits, typically around 20 requests/min and about 50 requests/day (more if you've added credits). These limits change, so check OpenRouter's current docs.

- **Group issues by file**, so one prompt fixes several issues at once (about 12 prompts instead of 34).
- **Paste only the files each prompt needs.** Don't paste the whole repo, because free models have limited context and output.
- **Ask for the complete updated file** (or a diff) so you can paste it back without hand-merging.
- **Start a fresh chat per phase** with the preamble below, since long chats waste tokens.
- **Commit to git and run the tests after every phase.** If a phase breaks something, you've only lost one request.
- **Wait about 3 seconds between requests**, and if you get a 429 error, wait a minute and retry.

---

## Prompt 0: Reusable preamble

Paste this at the start of every new chat, followed by the phase prompt.

```
You are a senior Python/FastAPI engineer helping harden a deepfake detection
project (FastAPI backend, ONNX Runtime models, React/Vite frontend) for production.
Rules:
- Only change what I ask. Do not rename public functions or change API response
  shapes unless I say so.
- Return the COMPLETE updated file(s) in code blocks, labelled with the file path.
- After the code, give a max-5-line summary of what changed.
- If something is ambiguous, make the safest assumption and state it. Don't ask questions.
```

---

## Phase 1: Safe cleanup (low risk, quick wins)

### Prompt 1: `visualization.py` (issues 1, 20, 21, 22)

```
Here is backend/app/utils/visualization.py: [PASTE FILE]

Fix these issues:
1. Remove every print() call. Change every logger.info("DEBUG: ...") to logger.debug(...).
2. GradCAM.__init__ creates a tempfile.NamedTemporaryFile(delete=False) ONNX file
   that is never deleted. Remove the leak: either avoid the temp file entirely, or
   delete it in a try/finally or on close()/__del__.
3. Stop accessing the private attribute self.session._sess._model_path. Pass the
   model path into GradCAM explicitly instead.
4. generate_gradcam() creates a new GradCAM per call. Cache one instance per model
   path (module-level dict with a threading.Lock, or functools.lru_cache on a factory).
Keep the public function signatures unchanged.
```

### Prompt 2: Small fixes across files (issues 7, 15, 18)

```
Make these small fixes. Files: [PASTE schemas/detection.py, models/efficientnet.py,
models/mesonet.py, tests/test_single_model_ensemble.py]

1. detection.py: replace datetime.utcnow with a lambda using datetime.now(timezone.utc).
2. efficientnet.py and mesonet.py: remove the artificial time.sleep() calls in the
   fallback prediction paths.
3. test_single_model_ensemble.py: remove the duplicate import of settings.
Return all four files in full.
```

---

## Phase 2: Model layer refactor (do before touching routes)

### Prompt 3: BaseDetector (issues 19, 14)

```
Here are efficientnet.py, mesonet.py and detector.py: [PASTE ALL THREE]

Refactor:
1. Create models/base.py with a BaseDetector class that holds the shared logic:
   _resolve_model_path, _bytes_to_image, _softmax, _normalize_binary_probabilities,
   _load_model and the common predict() flow. Subclasses override only
   model-specific preprocessing and details.
2. Make EfficientNetDetector and MesoNetDetector subclass BaseDetector.
3. In detector.py, when ensemble detection fails, do NOT return
   prediction="unknown". Raise a custom DetectionError(Exception) defined in
   models/exceptions.py.
Return base.py, exceptions.py and the three updated files in full.
```

### Prompt 4: Single-model honesty (issues 9, 10)

```
Files: [PASTE detector.py, config.py, api/routes.py (the part that builds the response)]

Both models currently point at the same ONNX file, so the "ensemble" is really one
model. Make the system honest about it:
1. Add an `ensemble_active: bool` and `models_used: list[str]` field to the detection
   response schema, set from whether the two model paths differ.
2. Remove the hardcoded accuracy=0.92. Load it from an optional
   models/model_metadata.json (keys: accuracy, dataset), and return null if the
   file is missing.
Return the changed files in full.
```

> **Note:** No prompt can give you a real MesoNet model. You have to train or download one.

---

## Phase 3: API layer

### Prompt 5: `main.py` + config (issues 2, 6, 23)

```
Files: [PASTE main.py, config.py, .env.example]

1. Restrict CORS: allow_methods=["GET","POST","OPTIONS"],
   allow_headers=["Content-Type","X-API-Key"].
2. Store the detector on app.state in the lifespan handler. Remove the global
   _detector duplication between main.py and routes.py (routes should use a
   FastAPI dependency that reads request.app.state.detector).
3. In config.py, default DEBUG=False and ENVIRONMENT="production". Add a startup
   validator that raises if ENVIRONMENT=="production" and required settings
   (model paths, allowed origins) are missing.
Return all files in full.
```

### Prompt 6: `routes.py` (issues 4, 5, 12, 13)

```
File: [PASTE routes.py] (assume the app.state detector dependency from the previous change)

1. Replace detail=f"Detection failed: {str(e)}" with a generic message; log the
   real exception with logger.exception().
2. Replace the _heatmap_cache dict with a small class: true LRU (OrderedDict +
   move_to_end), TTL expiry (e.g. 10 min), a lock for thread safety, and max size
   from settings.
3. Sanitize the upload filename: use os.path.basename, strip non-[A-Za-z0-9._-]
   characters, cap the length at 100, and never use it to build file paths.
4. Add a background cleanup function that deletes files in settings.TEMP_DIR older
   than 1 hour, run at startup and every 30 minutes via an asyncio task in lifespan.
   Give me the lifespan snippet separately.
```

### Prompt 7: Rate limiting (issue 3)

```
Add rate limiting to the FastAPI app using slowapi: 10 requests/min per IP on
POST /detect, 60/min on other endpoints. Make limits configurable via settings.
Handle RateLimitExceeded with a JSON 429 response. Show the changes to main.py,
routes.py (decorator + request: Request parameter) and requirements.txt.
Files: [PASTE main.py and routes.py]
```

### Prompt 8: Preprocessing (issue 8)

```
File: [PASTE preprocessing.py]
preprocess_image() runs detect_and_crop_face() twice (once inside
preprocess_for_efficientnet and again at the redundant call near line 268).
Refactor so face detection runs exactly once and its result (face_detected,
cropped image) is passed down. Keep public signatures backward compatible.
```

---

## Phase 4: Tests (now that the code is stable)

### Prompt 9: Fix and extend tests (issues 16, 17)

```
Files: [PASTE tests/, config.py, routes.py, schemas/detection.py]

1. Fix the test asserting MODEL_OUTPUT_ORDER == "real_first": config says
   "fake_first", so update the test to match the config.
2. Write pytest integration tests using FastAPI TestClient for: GET /health;
   POST /detect with a valid small JPEG; a non-image file (expect 400/415); an
   oversized file (expect 413); an empty file; GET /heatmap/{id} with an unknown id
   (expect 404); the rate limit (expect 429 after the limit). Mock the detector
   with a fixture so the tests don't need real ONNX models.
Return a conftest.py and test_api.py in full.
```

---

## Phase 5: Docker and frontend

### Prompt 10: Docker + CI (issues 11, 29, 34)

```
Project layout: backend/ (FastAPI, requirements.txt), frontend/ (Vite + React).
Create:
1. frontend/Dockerfile (multi-stage: node build -> nginx serving dist) and
   frontend/nginx.conf (SPA fallback, gzip, proxy /api to backend:8000).
2. A backend Dockerfile if missing (python slim, non-root user).
3. .gitignore covering __pycache__, .pytest_cache, node_modules, .venv, .env,
   *.zip, dist.
4. .github/workflows/ci.yml running backend pytest and frontend `npm run build`.
Here is my docker-compose.yml: [PASTE]. Fix it if needed.
```

### Prompt 11: Frontend (issues 24, 31)

```
Files: [PASTE App.jsx/main.jsx and the component with the GitHub pill link]
1. Add a reusable ErrorBoundary class component with a friendly fallback UI and a
   "Reload" button, and wrap <App /> in it.
2. Replace the https://github.com link with a VITE_REPO_URL env variable.
3. Add SEO meta tags (title, description, og:*) to index.html.
```

Also delete the 91-byte root `package-lock.json` yourself (issue 33).

---

## Phase 6: Optional features (only if you have requests left)

Do these one per prompt, in this order.

### Prompt 12: API key auth (issue 25)

```
Add optional API-key auth via X-API-Key header, keys from env var API_KEYS
(comma-separated), as a FastAPI dependency applied to /detect.
```

### Prompt 13: Structured logging and request IDs (issues 26, 32)

```
Add JSON logging and a middleware that attaches a correlation ID to every request
and log line, plus a /metrics Prometheus endpoint.
```

### Prompt 14: Batch endpoint (issue 27)

```
Implement POST /detect/batch using the existing BatchDetectionRequest/Response
schemas, max 10 images, processed sequentially.
```

### Not prompt-able
- **Video support (issue 28):** treat it as a separate project, not a quick prompt.
- **Dependency lockfile (issue 30):** do this yourself with `pip-compile`.

---

## Order summary

| Order | Prompt | Why this position |
|---|---|---|
| 1 | 1, 2 | Independent, low-risk cleanup |
| 2 | 3, 4 | Models refactor first, since the routes depend on it |
| 3 | 5, 6, 7, 8 | API layer builds on the new structure |
| 4 | 9 | Tests last, so they target final code |
| 5 | 10, 11 | Deployment and frontend are independent |
| 6 | 12 to 14 | Nice-to-haves |

Total is 11 core prompts, which fits comfortably in a free-tier daily limit. Always run `pytest` and the app after each phase before moving on. Nemotron can introduce subtle bugs, so review the diffs rather than pasting blindly.
