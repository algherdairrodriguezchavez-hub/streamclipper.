# Changelog
## Unreleased
### Added
- Facecam and gameplay crops per clip: `--cam-fit standard|tight|close` frames the face closer inside the detected cam overlay; `--game-anchor side|center` keeps the gameplay crop in the middle of the screen, clear of the streamer and the cam overlay; `--cam-track auto` cuts a clip into parts with their own crops when the cam moves, changes size or turns off inside it (rendered with the reframe composite).
- `--streamer-profile FILE`: cam positions seen per streamer (detected or corrected), used when a clip's cam is not detected and a face is found inside a known position; it also keeps the streamer's crop settings.
- `--preview-only` writes `layout_preview/*.jpg` (crops on a pixel grid + a mock of the vertical clip) and `layout_overrides.json`; `--layout-overrides FILE` renders the clips marked `"auto": false` with their own boxes, matched on start time.
- Colab automatic mode (`notebooks/StreamClipper_Automatico.ipynb`, Spanish, 3 cells): paste a link and get the clips. Every link gets its own folder `MyDrive/StreamClipper/proyectos/<date>_<channel>_<video id>/`, so a new link never reuses or overwrites another link's clips; `proyectos.json` + `historial.csv` list every run. The same link again continues its folder, or makes a new version (`_v2`) with or without the saved transcript. A transcript made earlier in another folder for the same video is copied (never moved). The streamer profile defaults to the channel name; `MyDrive/privado/cookies.txt` is used automatically. New module `streamclipper.autorun` (`process_link`, `prepare_project`, `history_html`).
- YouTube from Colab: `yt-dlp[default]>=2026.10.26` (with the `yt-dlp-ejs` challenge solver) and a new `youtube` extra (Deno + the `bgutil-ytdlp-pot-provider` plugin). `streamclipper.ytaccess.setup()` builds and starts the bgutil PO Token server (tag 2.0.2, with Deno) on 127.0.0.1:4416; `ytaccess.diagnose(url, cookies)` checks the cookies file (names and expiry only, never values) and makes two real 10 s downloads (without and with cookies) to tell apart invalid cookies, missing JavaScript runtime or PO token, and a blocked IP. The automatic notebook runs it when a download is refused (new cell 4). A Deno next to Python but not on PATH is passed with `--js-runtimes`. No player client is forced.
- Colab: cell 6 has `cam_fit`, `game_anchor`, `cam_track`, `streamer` and the clip length (`min_dur`, `target_dur`, `max_dur`); new cell 6b previews the crops and corrects one clip from a form. `run_pipeline` takes the same options.
- Long clips: `--max-dur` over 60 raises the 59.5 s Shorts cap to `max_dur` (at most 180 s, the Shorts/Reels limit); `run_pipeline(min_pause=...)` exposes how long a silence must be to be cut (999 keeps the gameplay between lines, which made clips ~18 s before).
- `--visual-rank off|qwen-3b|qwen-7b` (new `visualrank` extra): a local vision-language model (Qwen2.5-VL, on the GPU; 7B in 4 bit) watches 12 frames and reads the transcript of the best 24 candidates and rates them; the final score blends audio 40% and vision 60%. Ratings are cached in `analysis/vision_scores.json`; any model failure keeps the audio ranking. For links, only a 360p copy of each candidate window is downloaded.
- Hook styles `clean` (white text on a rounded dark card), `clean-light` (black text on a white card) and `minimal` (white text, thin outline).
- Automatic notebook: clips 60-75-90 s by default, `pausas` (mantener/solo_largas/cortar), `ver_video` (off/qwen-3b/qwen-7b, default qwen-3b) and the `clean` hook by default.
- `--always-split`: every clip with a face keeps the facecam on top and the gameplay below. A centred gameplay crop that does not fit tries the side; when no crop clear of the streamer exists (big cams: the streamer's body box covers most of the frame), a full-height crop away from the cam is used instead of a face-only clip; a big cam with no overlay border is split too (face crop on top, the picture below).
- `--caption-position seam`: captions and the hook title centred on the line between the facecam and the gameplay; the title shows first, alone, then the captions (the export gate does not expect captions under the title and allows seam captions over the bottom 110 px of the cam panel).
- Faster renders: the final MP4 is encoded with NVENC (`h264_nvenc`, cq 19) when the GPU has it (Colab T4), libx264 otherwise; `STREAMCLIPPER_ENCODER=cpu` forces the CPU. `--visual-rank` rates 12 candidates with 8 frames each and reads the frames straight from the link's 360p stream (no window downloads when that works).
- Automatic notebook: `siempre_dividido` (on), `caption_position = seam`, `hud = off` by default (the GTA 6 HUD profile is experimental and scans the whole stream).
- `--game-follow action` (new module `streamclipper.action`): the gameplay crop moves smoothly to where the motion is (motion-weighted centre, at most 12% of the width per second, a dead zone against wobble, still scenes keep the crop), kept clear of the cam overlay and the streamer; written as reframe keyframes.
- `--cam-fit box`: the whole detected cam overlay, trimmed only to the cam panel's aspect. The overlay detection now also finds the outer border of a floating cam (not touching the frame edge) when the strip outside it is moving gameplay.
- Automatic notebook: `cam_fit = box` and `seguir_accion` (on) by default.
- GTA 6 vision judging (`--visual-rank`, v2): the model sees 10 frames (7 spread, 3 around the reaction) with their seconds, the transcript and the voice peak, and answers in JSON: event type, routine or not, and 0-10 for action, surprise, humor, reaction, spectacle, curiosity and story, the frame of the strongest instant, how strong a cold open it makes and a short Spanish title. Final score = 30% audio + 70% vision (routine driving, menus and chatting count 0.6x); rated clips under 50 and repeats of the same event within 3 minutes are dropped, so there are fewer, better clips. The model's title becomes the hook headline.
- Cold open from the vision pick: the clip opens with 2-3 s that build into the strongest instant (stopping just after it), then goes back and reaches it again with its context. It is used only when the model rates the moment as a strong hook and not routine, and dropped when the moment is not in the rest of the clip after trimming. `--rewind-label TEXT` shows a short label ("UNOS SEGUNDOS ANTES...") as the clip goes back.
- `--motion-scan` (new module `streamclipper.motionscan`): one pass over the stream's keyframes (one every 2 s, 160 px; links use a 144p video-only download, or the HUD's 360p copy) finds stretches that move far more than the stream usually does around them (a chase, a crash) even when the streamer stays quiet. They become candidates (at half the score bar) and up to 6 of the 20 clips the vision model judges are kept for them. Cached in `analysis/motion_scan.json`.
- Hook style `gta` (Bangers, tilted, last word yellow, pink swoosh; GTA-like look, no game logo) and caption style `gta-pop` (big Anton, yellow active word, pink hype words).
- `--cam-fit box` with an overlay of another shape (a 4:3 or square cam): the whole cam is scaled to fit the cam panel and the sides are filled with a blurred copy of the cam itself, never gameplay, so the head is never cut.
- Automatic notebook: `hook_style = gta`, `caption_style = gta-pop`, `texto_regreso` ("UNOS SEGUNDOS ANTES...") and `buscar_movimiento` (on, used when `ver_video` is on).
### Changed
- Sharper facecam: `tight`/`close` cam crops never enlarge the face more than 2.5x (the crop widens instead), crops enlarged over 1.2x get a light unsharp after the lanczos scale, panels are encoded at CRF 15 and the final render at CRF 18 (maxrate 12M, the export gate's limit) instead of 20.
### Fixed
- Resume: a clip is reused only when its window is the same (`_clip_key`); a new discovery with other lengths no longer reuses another window's MP4 for the same rank.
## 5.3.0 (2026-11-19)
### Added
- Game HUD detection: `--hud auto|off|gta5|gta6|generic` and `--hud-profile FILE` on `discover-clips` and `clip-stream`. The GTA V profile finds WASTED, BUSTED, MISSION PASSED, MISSION FAILED and 3+ wanted stars (test videos: precision 1.00, recall 0.71); `gta6` is experimental; `generic` looks for big text, flashes and kill-feed bursts. Events add a `hud` score component and reasons such as "WASTED on screen". OCR uses a PaddleOCR ONNX model downloaded once (SHA-256 checked).
- `streamclipper calibrate-hud` writes a HUD profile and a contact sheet from a game video (for GTA 6 after release).
- MediaPipe face tracking (BlazeFace full range, bundled) with tiling for small facecams, smoothing with a dead zone and active-speaker choice for 2+ faces; OpenCV Haar stays as the fallback (`vision-haar` extra). `doctor` shows the face detector and warns about two OpenCV packages.
- Editing effects, all off by default: `--zoom light|medium|strong` (punch-in on the payoff and hype words, faces kept in frame), `--remove-fillers`, `--enhance-speech light|strong` (ffmpeg `arnndn` with a bundled RNNoise model), `--brand-kit FILE` (logo, handle, font, colours, intro/outro card). In the dashboard (Home Advanced drawer, editor Effects card, Settings Brand kit) and the Colab notebook.
- Publishing with your own credentials: `streamclipper publish youtube|instagram|pack|schedule` (YouTube Shorts private by default, scheduled uploads, Instagram Reels, TikTok publish pack, a 7-day schedule, `--dry-run`); dashboard Publish dialog, Schedule week and Settings > Publishing. Tested only against a mocked API.
- Export gate row `caption_coverage`: share of spoken words on screen and the longest uncaptioned stretch of speech (FAIL below 97% or above 0.6 s).
- Selection check on 8 unseen public streams with a blind, pre-registered tune/holdout split; `tools/eval_selection.py --split tune|holdout|all`, `--bar`, `--weights`, `--before`. The numbers are in the guide (Selection quality); no threshold changed.
- Colab notebook: installs the `vision` extra; cell 6 has `hud`, `zoom`, `remove_fillers` and `enhance_speech`.
### Fixed
- Captions: very short and zero-length words were dropped or joined badly; shouted speech that Whisper's voice filter skipped is recovered with a second pass per clip window; captions no longer count small background faces as faces to avoid.
- Speech enhancement output is delayed back into sync (ffmpeg `arnndn` adds 10 ms of latency).
- Colab (found on a real T4 run): Whisper fell back to the CPU because the image has no CUDA 12 cuBLAS; the notebook now installs the `gpu` extra on a GPU runtime and StreamClipper preloads its CUDA 12 libraries on Linux (T4: 58.8x realtime). Running cell 7 before cells 5-6 prints which cell to run instead of a NameError. Cell 2 prints the pip and doctor output.
- Resume: re-running `clip-stream` (or Colab cell 7) with the same inputs keeps the clips that are already made instead of making `clip_01_2` copies.
- Captions: several words that Whisper puts at the same instant now share one caption instead of getting a 0 s caption (found by the Linux check).
- Loudness: the AAC overshoot check no longer lowers the limiter ceiling below -6 dB; a hard-limited live clip stayed at -14.4 LUFS before.
- `doctor` says when MediaPipe is installed but cannot start (missing libEGL on headless Linux) and gives the apt fix.
## 5.2.1 (2026-11-18)
### Fixed
- Editor preview: in the first 2 s the hook banner and the caption no longer draw on top of each other; the preview places the hook and moves the caption below it with the same rules as the render.
- Editor toolbar: grouped controls, icon-only buttons with tooltips under 1280 px and a More menu (Merge, Delete keyframe, Reset crop), so it fits on two rows at 1024 px. Home: a shorter link placeholder on phones.
- Auto-reframe: two quick clicks (or two tabs) start one face-tracking run; the second request gets "already running" (HTTP 409).
- Auto-reframe with a facecam overlay in a corner: segments where the face is in the cam box are set to Split (facecam on top, gameplay below) instead of a full-height pan with the face low; a full-layout crop puts the face in the upper third where the source allows.
- Jobs API: `render`, `drafts` and the delete `confirm` flag are parsed strictly (true/false, 1/0, yes/no); `"render": "false"` no longer counts as true, and other values give HTTP 400.
## 5.2.0 (2026-11-18)
### Added
- Premium dashboard: left sidebar (bottom bar on phones), Home with one-click **Get clips** and an Advanced drawer, a first-run set-up check (doctor in the background), Projects grid with thumbnails, source icons, progress rings and clip counts, vertical clip cards with hover preview and score breakdown, a Styles page with Set as default, skeleton loaders, toasts, confirm dialogs, keyboard shortcuts (`/`, `?`) and Lucide icons (ISC).
- Reframe editor: Split at playhead into segments with their own layout (full, split, square frame, caption punch, speaker) and crop windows, pan keyframes with smooth or linear easing, cut/smooth joins, a live canvas preview drawn with the render maths, a timeline with thumbnails, waveform, segment blocks, keyframe diamonds and word chips, undo/redo (Ctrl+Z / Ctrl+Shift+Z), caption drag with safe-zone snapping and a face warning, style tiles, and editor shortcuts (Space, J/K/L, S, [, ], Delete).
- Auto-reframe (face tracking in the background, keyframes about once a second) and Reset to auto-detected.
- Versions with Compare: the original and an edited MP4 side by side, played and seeked together, with both audit tables.
- Timeline thumbnail sprite (`GET .../clips/<key>/thumbs`, one ffmpeg call, cached).
- Segmented edits render as one ffmpeg graph (frame-exact trim per segment + concat, continuous source audio) through the normal export gate; CapCut drafts get one video segment per editor segment (`reframe_segments` and `caption_anchor` in the clip spec).
### Changed
- Dashboard routes: `#/home`, `#/projects`, `#/project/<id>`, `#/edit/<job>/<key>`, `#/live`, `#/styles`, `#/settings`; the old `#/new`, `#/jobs` and `#/job/<id>` links redirect.
- Default `--prefix` is now `clip` (was GTA6_Viral): `clip-stream` without `--prefix` writes `clip_01.mp4`, the same names as `render` and the dashboard (B07).
### Fixed
- B01: in `--lan` mode the editor serves only videos inside the workspace.
- B02: a bad share `days` value no longer gives a 500 (it is parsed once; default 7 days).
- B03: the README screenshot shows the dashboard UI (Projects page) instead of only video frames.
- B04: Settings and Options no longer wait on the CapCut/Ollama checks (background, cached).
- B05: a slow page can no longer replace the page you moved to.
- B06: the editor caches its source probe, so reopening a clip runs no ffprobe.
## 5.1.0 (2026-11-17)

### Added

- `streamclipper dashboard`: a local web dashboard (standard library only, no new dependency): new jobs from a link, an upload or a file, a job queue with live progress, cancel and resume, a clips library (player, scores, copy buttons, audit table, re-render, Open in CapCut, posted/rejected, delete to trash), a live page and settings. Light and dark themes.
- Dashboard clip editor: trim with word-snapped handles, facecam/gameplay crop boxes with a 9:16 preview, caption word fixes, styles, hook text and post text; it renders through `render --spec` and the export gate.
- `--lan`: the dashboard on your network with an access token, and view-only share links (expiry, revoke).
- Colab: `use_google_drive` (work without Drive, zip download at the end), opt-in `share_on_drive` Drive links, and result cards with copy buttons.
- `<name>.gate.json` next to every rendered MP4 (the export gate table); `drive_link` column in clips.csv.

### Changed

- Licence changed from MIT to AGPL-3.0-or-later.
- README shortened; the full documentation moved to docs/GUIDE.md.

## 5.0.1 (2026-11-17)
### Fixed
- Full layout (full-cam, and the corner-cam fallback when the game screen is black): captions and the hook go to the zone of the frame that has no face (face detection on the panel), instead of a fixed low zone that could cover the chin and mouth. New export-gate row `captions_off_face` (every layout): FAIL when a caption covers a detected face on 2+ sampled frames.
- `live`: render and probe subprocesses start in their own process group, so Ctrl+C (or Ctrl+Break) stops the ingest and the clip being rendered and the queued ones still finish. Window files (`windows/live_*.mp4` and the copy cut from each) are deleted once their clip is made; failed or skipped clips keep theirs. New `--max-disk-gb` (default 20) warns when the output folder grows past it.
- Colab live cell: the rolling buffer, windows and panels stay on the Colab disk (`/content/streamclipper_live`); only finished clips, thumbnails, `clips.json`/`clips.csv`, `index.html` and `live_summary.json` are copied to Drive, after every clip.
- Hook headlines: a literal line is used only when it is complete (no sentence end inside, no dangling first or last word, about 3 content words or a whole short exclamation such as 'BRO WHAT IS THAT?!'); otherwise a category template.
- The labelled eval fixtures are now `eval_stream_a.json` / `eval_stream_b.json` without video id, title or transcript text (word codes keep what the quality bar reads).
- `test_live_fake`: the 40 s time budget is a warning, not a failure.

## 5.0.0 (2026-11-17)

### Added
- `streamclipper live`: clips Twitch, YouTube Live and Kick streams while they run (streamlink ingest into a rolling buffer on disk, Whisper on every new piece, the VOD scoring and quality bar, background renders, Twitch and YouTube chat, `--buffer-min`, `--max-clips-per-hour`, `--stop-after`). Colab cell 10 runs it.
- Podcast mode (beta): `--preset podcast-9x16`, `--content talk` (strong lines, laughter, amounts, an LLM prompt for self-contained stories) and the `speaker` layout (per camera shot: a crop that follows the face, two people stacked, B-roll shown whole). New gate row `faces_whole`.
- Face-based layout detection (OpenCV, extra `vision`): per-clip facecam detection, blank-camera clips skipped, the gameplay crop keeps the streamer out.
- Precision-first quality bar with a labelled eval (`tools/eval_selection.py`, `tests/fixtures/eval/`); plain-English titles, descriptions and hashtags without an LLM; unique hook headlines per run.
- `tools/check_sync.py` (caption timing against Whisper on the MP4) and `tools/docker/Dockerfile` (Linux test image).
- `--caption-style auto` (the new default) picks the caption style per clip from its category; the chosen style is in `clips.json`, `clips.csv` and `index.html`, with a copyable re-render command per clip. `render --spec <name>.clip.json --caption-style NAME` re-renders one MP4 in another style.
- `streamclipper styles --preview` renders a sample of every caption and hook style plus `styles_gallery.png` (shown in the README and in the Colab options cell).

### Changed
- Clips are at most 60 s by default (`--max-dur 60`), rendered clips at most 59.5 s.
- `render --clips` leaves `clips.json` as it is and writes `<stem>_rendered.json/.csv/.html`.
- The resume hash covers only the speech-recognition settings, so changing `--max-dur` or `--llm` no longer re-runs Whisper.
- `streamlink` is a core dependency.

### Fixed
- Story windows end after the reaction and never cut inside a word; gain staging avoids audible limiter pumping; dark game scenes are no longer taken for a blank camera card; slice downloads retry after a network drop.

## 4.0.0 (2026-11-17)

First public release as the `streamclipper` package.

### Added
- Installable package with the `streamclipper` command (`pip install -e ".[asr]"`, extras `gpu` and `dev`). `scripts/capcut_studio.py` still runs the same CLI.
- Long streams: audio-only download, full-stream transcription in 10-minute chunks with faster-whisper on CUDA (float16, batched) with automatic CPU fallback, and resume after an interruption (`state.json`, `--fresh`). Flags `--asr`, `--whisper-model auto`, `--device`.
- Every-clip mode: `--min-score` and `--max-clips` (0 = all), non-overlapping and ranked, each clip with a 0-100 score, a hook/story/emotion/pacing/engagement breakdown, a reason and a category (funny, rage, clutch, reaction, fail, hot take, reveal, wholesome, chaos).
- `clips.csv` and a static `index.html` report with previews, thumbnails and file links.
- Chat replay signal from YouTube live chat and Twitch rechat (`--chat`, or fetched with `--url`).
- Optional LLM (`--llm ollama|gemini`) for story segments, hooks, titles, descriptions and hashtags.
- English stream-slang triggers ("no way", "let's go", "clip that", "W", "GG" ...); Hindi as a second list.
- 15 caption styles (`--caption-style`, `streamclipper styles`), 3 hook banner styles (`--hook-style`), caption options (`--caption-case`, `--words-per-caption`, `--caption-position`, `--highlight-color`, `--caption-font`, `--emoji`, `--profanity-filter`), and new OFL/Apache fonts.
- `square-frame-9x16` preset and a 1080x1920 thumbnail per clip.
- Google Colab notebook (`notebooks/StreamClipper_Colab.ipynb`) driven by `streamclipper.colab.run_pipeline`, with Drive output, resume, YouTube bot-check help and an optional local Ollama on the T4.
- Portable CapCut drafts (`--portable-drafts`, automatic without CapCut) and `streamclipper import-draft --zip` to bring drafts from Colab into CapCut Desktop.
- MIT licence, CI (ruff + pytest on Ubuntu and Windows), issue and PR templates, CONTRIBUTING.md, unit tests.

### Changed
- `--top` is now an alias of `--max-clips`; `discover-clips` and `render` default to all clips above the score, `clip-stream` still makes 3 drafts by default.
- Re-runs never overwrite MP4s, plans or sidecars: they get the `_2` name, like drafts.
- `root_meta_info.json` backups: one first backup plus the newest 5 timestamped ones.
- Presets, fonts and templates moved into the package (`src/streamclipper/data/`); `env_detected.json` moved to the user cache folder.
- The SFX of unverified licence were replaced by SFX generated with ffmpeg (CC0).

### Fixed
- The CapCut-open guard now only blocks writes into CapCut's own draft folder, so test roots set through `CAPCUT_DRAFT_ROOT` work while CapCut is open.
