# captura_v3 damage annotation in Label Studio

Step-by-step setup and annotation protocol for the impact-damage boxes of the captura_v3 dataset. The capture
procedure is in [`docs/capture/guia-lote-1.md`](../capture/guia-lote-1.md).

| | |
| --- | --- |
| Tool | Label Studio from the repository `.venv`, running locally |
| Task | One capture: its RGB and NIR photo side by side |
| First pass | **+12 h** (the bruise is most visible): 200 tasks = 2 lots × 25 mangos × 4 faces |
| You produce | At most **one damage box in total** (on whichever image shows the bruise best), plus two required answers: damage visible in RGB? in NIR? |
| Config | [`configs/label_studio/captura_v3_damage.xml`](../../configs/label_studio/captura_v3_damage.xml) |
| Tasks file | `data/label_studio_captura_v3/tasks_12h.json` (`data/label_studio*/` is git-ignored) |

## Summary

| Step | What you do | Where |
| --- | --- | --- |
| 1 | Generate the tasks file (already done for +12 h) | `scripts/build_label_studio_tasks.py` |
| 2 | Start Label Studio with local-files serving | terminal |
| 3 | Create the project and paste the config | Label Studio UI |
| 4 | Add a Local files source storage (do **not** sync it) | Project settings |
| 5 | Import the tasks file | Data Manager |
| 6 | Annotate | Label stream |

## Setup

Run every command from the repository root of the checkout that holds `data/cache/captura_v3/`.

### 1. Generate the tasks

```bash
.venv/bin/python scripts/build_label_studio_tasks.py \
    --data-root data/cache/captura_v3 \
    --document-root "$PWD/data" \
    --moments 12 \
    --out data/label_studio_captura_v3/tasks_12h.json
```

It prints tasks per lot and moment, dropped duplicates, missing slots and how many RGB photos are
white-balance corrected, and it fails if an image is missing or outside the document root.

| Option | Meaning |
| --- | --- |
| `--moments` | Hours after the impact and/or `sano`, comma separated: `12`, `12,10`, `sano` (default `12`) |
| `--jornadas` | Lot numbers or jornada ids, e.g. `1` or `lote2-20261025` (default: both) |
| `--document-root` | Folder Label Studio serves files from; must contain `--data-root` and match step 2. Pass it explicitly; if omitted it defaults to the folder above the single `cache/` directory of `--data-root`, and the script asks for it when that is ambiguous |

Rule for duplicates: when one face has two photos at the same moment, the single in-tolerance one is kept.
Anything else is an error that names the slot.

### 2. Start Label Studio

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/data" \
.venv/bin/label-studio start
```

The document root is `data/` because task image URLs look like `/data/local-files/?d=cache/captura_v3/news/...`.
If Label Studio was already running without these variables, stop it and start it again.

### 3. Create the project

1. **Create Project**, name it `captura_v3 damage` and skip the Data Import tab.
2. **Labeling Setup** → **Custom template** → **Code**, paste the whole content of
   `configs/label_studio/captura_v3_damage.xml` and **Save**. A copy sits next to the tasks file.

### 4. Authorize the images

**Settings** → **Cloud Storage** → **Add Source Storage**:

- Storage type: **Local files**
- Absolute local path: `<repository>/data/cache/captura_v3/news`
- Leave everything else as is and click **Add Storage**.

Do **not** click **Sync Storage**. The storage only authorizes Label Studio to serve the files under that
folder; the tasks come from the JSON.

### 5. Import the tasks

**Data Manager** → **Import** → upload `tasks_12h.json`. You should see 200 tasks, ordered by lot, mango, face.

## Annotation protocol

An illustrative task header (made-up values, not a real task):

```text
Lot 1 · mango 12 · face 3 · +12 h · hit zone on fruit: superior · hit 14:05 (12.0 h ago)
Flags: timing out of tolerance (38.9 min off) | QC NIR: lighting/camera changed (edge 18 vs 30, -40%)
```

1. **Read the header.** The hit time tells you how old the bruise is. The hit zone is where the operator aimed the
   tube on the fruit while holding that face up; it is **not** a direction in the image. Measured on 94 faces at
   +12 h (from the 2026-10-26 registration/zone analysis of the +12 h photos, not in the repository yet),
   `superior`/`inferior`/`izquierda`/`derecha` show no consistent image side, and some bruises sit on the
   opposite side of the label. Only `centro` is a usable cue (the middle of the visible face). Search the whole face
   and draw what you see, never where the label points.
2. **Compare both images.** Zoom each one, and use its brightness and contrast sliders; NIR often needs them.
3. **Draw ONE box in total** on the image where the damage is most visible. Press `1` (RGB) or `2` (NIR), or
   click `damage` under that image, then drag a box around the whole bruise, inside the mango. The other modality
   is derived later through the RGB↔NIR homography, so do not draw it twice. Label Studio will not stop you from
   adding a second box on the other image (its limit is one per image): don't. If a task ends up with a box on
   each image, the planned export rule keeps the one on the image marked as showing the damage best. To redo a
   box, delete it first.
4. **Answer both questions**, whatever image you drew on: *visible in RGB?* and *visible in NIR?* with
   `yes`, `no` or `unsure`. Both are required.
5. **Nothing visible in either image:** draw no box, answer `no` to both (or `unsure` if you doubt it).
6. Add a note only when something is odd, then **Submit**.

Annotate the **+12 h** moment first, for all 200 faces. Earlier moments come later, possibly with boxes projected
back from +12 h as pre-annotations; the task format stays the same, only `--moments` changes.

## Known caveats

| What | What it means for you |
| --- | --- |
| `WB-corrected RGB` flag | 57 lot-1 +4 h RGB photos were white-balance corrected after capture and look slightly olive. The task points to the corrected file. None are in the +12 h set. |
| `QC NIR/RGB: lighting/camera changed` | The frame border is darker or brighter than the reference. In the +12 h set, all 100 lot-1 NIR photos carry it (17-47% darker) and 40 lot-1 RGB photos (15-19% brighter). They are usable: adjust brightness and contrast. |
| 1-px lines in lot 2 NIR | 11 lot-2 NIR photos have 1-pixel sensor lines (source: the lot-2 capture summary `jornadas/lote2-20261025/resumen_lote2.md` in the OCI bucket). They are an artifact, not damage. |
| `relabeled (was mango N face M)` | The photo was assigned to a different mango/face after capture. Trust the header, not the original label. Lot 2 at +2 h has a block of these (mangos 33-43), plus one dropped duplicate (mango 33 face 1) and one missing photo (mango 34 face 3). |
| Lot 1 +12 h retakes | Five photos were taken 33-68 min late (`timing out of tolerance`), so the bruise is a little older. Two of them (mango 3, faces 2 and 3) were shot in swapped order and are flagged `relabeled`. |

## If something fails

| Symptom | What to do |
| --- | --- |
| Images do not load (403) | Label Studio was started without the two `LABEL_STUDIO_LOCAL_FILES_*` variables: restart it as in step 2. |
| Images do not load (404) | No Local files storage covers the folder: redo step 4 and check the path ends in `captura_v3/news`. |
| `Submit` stays disabled | Both visibility questions are required. |
| Cannot draw a second box on the same image | By design (limit of one per image). Delete the existing box first. The protocol is ONE box in total: Label Studio would accept one more on the other image, but don't add it. |
| The generator prints `error:` | It names the missing image, the file outside the document root, or the slot with several in-tolerance photos. |

## Next

Projecting boxes to earlier moments and the export step (Label Studio JSON → one box per capture on both
modalities, using the lot-1 homography `matriz_homografia_lote1_20261024_centros.npy`; lot 2 has no homography
yet) come later.
