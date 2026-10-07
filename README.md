# Project 44: piano music in, sheet music out, and new music back

Two projects that form one system. Each folder is self-contained with its own README, setup and data.

| Folder | Half | What it does | Start with |
|---|---|---|---|
| [`musicML/`](musicML/) | reading | a piano performance (audio or MIDI) becomes an engraved score (MusicXML, PDF); evaluation harness, training stack, research on tuplet decoding | [`musicML/README.md`](musicML/README.md), then [`musicML/AGENTS.md`](musicML/AGENTS.md) |
| [`music-cwt/`](music-cwt/) | writing | a few bars of score become more score in the same style (compound-word transformer over kern notation; code by Antoine, GitHub Toitoine1) | [`music-cwt/HANDOFF_NOTES.md`](music-cwt/HANDOFF_NOTES.md), then [`music-cwt/README.md`](music-cwt/README.md) |

The two halves share one idea: a note is a bundle of parallel attributes, not a single symbol, and a
score is something you can check like code. The reading half can also manufacture training data for
the writing half from recordings.

## Working here

- Large files (model checkpoints, datasets, caches) are not in git. Each project fetches them from a
  release of this repository with its own script (`musicML/scripts/fetch_assets.sh`), which verifies
  every file against recorded checksums.
- Stage files by path; never commit checkpoints, data packs, caches or outputs. Each project's
  `.gitignore` already excludes them.
- The MAESTRO and ASAP material used by `musicML` is CC BY-NC-SA: research use inside this private
  repository only.
