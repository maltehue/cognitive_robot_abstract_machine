# Recorded presentation without Internet

Measured rehearsal: [2026-09-16 validation](offline_validation_2026_09_16.md)
records 21.4 seconds to the interactive scene and 33.1 seconds to the additionally
opened Transport recording and plan, under application network isolation.

## Start on the prepared laptop

From the repository root:

```bash
./scripts/start_cramera_offline.sh
```

The command uses the repository's installed Python environment and the presentation
at `~/.local/share/cramera/offline-demo`. It opens the chapter player on port 8713.
Pass `--no-browser` when opening the local URL yourself, or `--port 8714` if the
default port is occupied. `CRAMERA_OFFLINE_BUNDLE` selects a different bundle.
Stop the owned server with Ctrl-C.

Press **Start** for the complete recorded tour. The chapter controls also let a
presenter pause, repeat a chapter, or move forward manually. Recorded execution
chapters finish their trajectory before advancing. Scene questions are answered
from the saved semantic snapshot; the plan tree shows the recorded execution.

## Storyboard

1. Semantic world model: objects, handles and supporting surfaces.
2. The two annotated handles highlighted in the scene.
3. Free placement regions for the milk carton.
4. Tracy's successful collision-enabled semantic Transport.
5. Its action hierarchy and the explanation of replaceable policy steps; no
   learned policy is executed.
6. PR2's continuous GCS navigation, turning toward travel and avoiding the obstacle.

These are separate reference scenes. They do not depict a shared multi-robot world
or the Siemens lab. The lab and external robots must be supplied before recording
the final Siemens storyboard. The current tour covers the available single-robot
demonstrations and their explanatory chapters.

## Network isolation

The launcher starts only the recording viewer in a Linux network namespace with no
external network route. A localhost listening socket is opened before isolation
and passed to that process, so the host browser can display it. The host's network
settings remain unchanged. Page resources and connections are restricted to their
own origin through Content Security Policy; microphone and camera are disabled.
Presets and typed queries remain available locally. No robot, scaffold or live
bridge is required, and this server rejects plan execution and recording mutations.

This application isolation provides a repeatable offline check while the rest of
the laptop remains connected. A whole-laptop airplane-mode rehearsal is a separate
physical setting; it should use the same command and recorded tour.

## Prepare or refresh the recording bundle

Prepare once while all recordings are available. The command checks referenced
assets, including mesh materials and textures, and copies the required recordings
out of temporary storage. It refuses to overwrite an existing destination.

```bash
.venv/bin/python -m cramera.offline ~/.local/share/cramera/offline-demo \
  --prepare-from /path/to/recorded/scenes --check-only
```

The packaged storyboard expects `semantic_query_rehearsal`,
`cramera_collision_avoidance_transport`, and `cramera_navigation_facing`.
To use different recordings, pass `--storyboard /path/to/storyboard.json`, following
the schema in `web/offline_storyboard.json`. Export a new destination to replace a
presentation deliberately; existing bundles are retained.

Check a prepared bundle without opening the viewer:

```bash
.venv/bin/python -m cramera.offline ~/.local/share/cramera/offline-demo --check-only
```

The scene assets are self-contained. The CRAM checkout and installed Python
environment are still required; this is not a standalone application installer.
