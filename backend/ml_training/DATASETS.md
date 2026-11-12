# Mirage datasets

Raw data lives in `backend/ml_training/data/<folder>/`. It is **never committed** (git-ignored)
and **never shipped with the app**. Models learn from it offline; the running app only loads
small trained files from `ml_engine/weights/`.

Mirage gives a therapist supporting signals. It does **not** diagnose anything.

## Order of work (finish one dataset before starting the next)

| #   | Dataset      | Folder         | Becomes                            | Why this position                                                                 |
| --- | ------------ | -------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| 1   | HaGRID       | `hagrid`       | hand gesture -> **responsiveness** | No access request, clear labels, split by person: the best first end-to-end slice |
| 2   | MMASD        | `mmasd`        | body movement -> **arousal**       | Public repository, skeleton data only (no faces)                                  |
| 3   | Saliency4ASD | `saliency4asd` | gaze pattern -> **attention**      | Small, but it needs careful design (see notes)                                    |
| 4   | LIRIS-CSE    | `liris_cse`    | face emotion -> **distress**       | Access must be requested (**send the request now**; approval takes time)          |

## Gates every dataset passes, in order

- [ ] **G1 Access**: request or download permission obtained
- [ ] **G2 Licence**: read the usage terms; confirm a research/prototype use is allowed
- [ ] **G3 Placed**: files are in `data/<folder>/`
- [ ] **G4 Inspected**: `python -m ml_training.inspect_datasets` ran and the layout is understood
- [ ] **G5 Prepared**: a prepare script turned it into training-ready files in `data/<folder>/_prepared/`
- [ ] **G6 Trained**: a training script produced the file in `ml_engine/weights/`
- [ ] **G7 Evaluated**: scored on people the model never saw (see split rules below)
- [ ] **G8 Integrated**: a runtime module returns a `Signal` and has unit tests

`python -m ml_training.status` shows G3-G6 for all four and prints the next step.

## The four datasets

### HaGRID (hand gestures)

- Adults (aged 18-65), photographed 0.5-4 m from the camera. **No children.**
- Version 1: 18 gesture classes plus "no gesture"; annotations include hand boxes and 21 landmarks.
- Use the small subsample (about 100 images per gesture), not the 700 GB set.
- Licence: a variant of CC BY-SA 4.0 with its own licence file. Read it.
- **Split by `user_id`** so the same person is never in both training and validation.
- Limit: static poses; a wave is movement over time, so a wave needs several frames.

### MMASD (body movement)

- 32 children with autism in play-therapy sessions; 2D/3D skeletons, optical flow, clinician scores, 11 activity classes.
- Privacy-preserving: no raw video, no faces.
- Licence: check the repository before using it.
- **Split by child.**
- Limit: group therapy on a floor, not a screen. Use it for movement statistics and reference
  ranges, not for diagnosis or predicting clinical scores.

### Saliency4ASD (gaze)

- 300 natural images; eye movements from 14 children with autism and 14 typically developing children.
- Comes with a usage agreement file on Zenodo. Read it.
- **Split by participant.**
- Limit: gaze on photographs, not on a character's face. It cannot teach "looked at the eyes".
  Use it to design and sanity-check gaze features (fixation rate, saccade size, spread).
  Do **not** use it to classify autism versus typical development.
- Use each child's **own baseline** (the first minute of their session) rather than population norms.

### LIRIS-CSE (emotion)

- 12 children aged about 6-12, spontaneous expressions of six emotions (about 26,000 frames).
- Access by request; research use only.
- **Leave one child out** when evaluating: frames from the same child look almost identical.
- Limit: **no neutral class**, so a calm face might be read as sad or fearful. Needs a confidence
  threshold and "must persist for several seconds" before acting.
- The children are not described as autistic; accuracy on autistic children's expressions may be lower.

## How the four combine (runtime)
