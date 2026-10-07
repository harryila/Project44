# Provenance and reproducibility boundary

This supplement supports two levels of checking.

1. The four standalone audit commands reproduce the headline and tabulated experimental
   arithmetic from the packaged machine-readable records. They require only Python 3.11 or
   later, except that figure regeneration also requires Matplotlib.
2. A full transcription rerun additionally requires ASAP v1.1, the released checkpoint,
   MUSTER and its pinned dependencies, the phase-conditioned duration prior, the frozen
   experiment source, and the decoded candidate streams. Those larger or upstream assets
   are not all redistributed here. The package therefore verifies the reported record but
   is not a self-contained rerun environment for candidate generation or MUSTER.

## Frozen external and working-tree identities

```text
8cba199e15931975542010a7ea2ff94a6fc9cbee  ASAP v1.1-based dataset snapshot
115432bda16ca16e0fec2e9465788f2ba369971f  upstream model repository base
4ef1165edda1f719068c4c699bd8ab2076e4d7ec  MUSTER wrapper and evaluator snapshot
7b8ec6e3da365b97443fb67a8f0b37d63997e93c152d665d43cb2011245db638  released checkpoint
666bf3264d42d982766e8a9c14b6b07579bbe27d46ffc132c6391eafbdf4798f  epoch-13 replication checkpoint
7e0b8c2543362e75f70f42b46d93832db983c4f41692320a3771b0cee47710e4  phase-conditioned duration prior

86a12e6a887cf01d0aa38a8060d4d94de990a2f1be831d8c160e093bc6ef8a10  model generation source
fa0c2e9963d0e690d97d64ed14eea814cf83d8d023af5f0c58d78f588947d924  inference utility source
8a3cd1f7e6d7ef61b767aae8bcf747610c65ed8a6ef241f07871ff51806bb96e  tokenizer source

cb9556b057abd85906eafcf66be3a6ef3dd84e911c2a9a3d32c67241b61d230d  routed selector
18fd0c62482dbfd4e806277e23155f75e98e3df480aca4e0c6b4dd0b44fc40ac  decode lever sweep
65e183aace68c385e8a1f5cba58a892942afae9de94c77150b07d5636ed4e0b3  guarded selector
2c7fbf435bbe5eca5035298a71cc0bf44956e280adbea050a6257d295d077fb2  likelihood and timing selector
370cf01a2e529b0e314c65a4550033e47231cfdd551ed5a771308e966d2776f4  local repair
eace77dca21dd49b14f92e72be3af89f88c3209eeee0b8bf3bcc0600ca8197dd  whole-sequence verifier
c17d5d95dd856ec1600b77a47ae72f11c99ef7fba7cbe9e4eaffdf0bada39d8b  confirmatory runner
3de575199ce1e4e7cbcd447b1ff5265b1ead998d63e84e242dcc99ead23dbe1f  oracle and prefix runner
```

The frozen experiment-source and prior file hashes above are also embedded in the oracle
record. The checkpoints, prior, and full experiment source are not in this repository.
The released checkpoint can be obtained from the cited upstream project. The second
checkpoint is a separate training run and is used only for the hidden-tail replication.

Public upstream locations:

```text
https://github.com/TimFelixBeyer/asap-dataset
https://github.com/TimFelixBeyer/MIDI2ScoreTransformer
https://github.com/TimFelixBeyer/amtevaluation.github.io
```

## Performance identities

Every analysis uses one aligned performance per score. Tail decomposition and the four-piece
local/development diagnostics choose the performance with the fewest tokenized input notes.
The frozen confirmation runner chooses the first aligned row for each score in the pinned
dataset metadata. This latter rule was part of the frozen runner before selection; it is not
the same as shortest-performance selection.

```text
Score          Diagnostic performance     Confirmatory performance
Bach           Shi05M.mid                  Shi05M.mid
Beethoven      Hou02M.mid                  Hou02M.mid
Brahms         Shilyaev03.mid              Shilyaev03.mid
Chopin         Sladek04M.mid               BuiJL04M.mid
Debussy        ParkJH13M.mid               Kleisen11M.mid
Haydn          Song05M.mid                 Song05M.mid
Liszt          LeungM08M.mid               development piece
Mozart         TET01.mid                   development piece
Prokofiev      Teo05.mid                   Colafelice11.mid
Rachmaninoff   WuuE07M.mid                 ChenGuang12M.mid
Ravel          GarritsonL05M.mid           development piece
Schubert       Teo06M.mid                  Duepree08M.mid
Schumann       ParkS15M.mid                Min09M.mid
Scriabin       Shi08M.mid                  development piece
```

The score-relative directories are recorded in the machine-readable piece keys. The public
dataset snapshot and the rules above determine each full MIDI path without a local pathname.

## Stream identity

The ground-truth-blind selection record stores one SHA-256 identity over the concatenated
offset, duration, and keep streams for each locked baseline and selection on all ten
confirmatory pieces. Phase-two evaluation regenerated those rhythmic streams and aborted on
any mismatch. The oracle record stores
candidate stream hashes for 93 of 256 sampled rows. The other 163 rows were tied to their
locked phase-one records by deterministic reconstruction checks over seed, candidate index,
coverage, rhythmic counts, and timing evidence. Since raw candidate streams are not in this
archive, the latter identities cannot be rehashed by a reader without a full rerun.

## External license boundary

The ASAP snapshot includes a CC BY-NC-SA 4.0 license. The public MIDI2ScoreTransformer and
MUSTER repository snapshots used by the experiments did not declare standalone repository
licenses. They are cited and identified here, but their source and checkpoint files are not
redistributed in this archive. The license in `LICENSE.md` applies only to the original audit
code, figure builder, and derived numerical records packaged here.

## Chronology

The internal record places the V10.1 confirmation addendum before selection, the locked
selection record before ground-truth evaluation, and the oracle analysis after evaluation.
There is no external immutable timestamp. The writeup states this limitation and labels the
oracle, guard sensitivity, and subset calculations as post-selection secondary analyses.

During the final audit, the rollout and uniform-boost prevalence denominators were found to
include decoder slots rejected by the model keep stream. The fixed diagnostics were rerun in
CPU FP32 with the score-detokenization rule `pad probability > 0.5`. The superseded unmasked
records remain under `artifacts/`. Teacher-forced ranks, MUSTER values, candidate pools,
routes, and selector outcomes did not change.

`MANIFEST.sha256`, created by the packager, gives the full SHA-256 identity of every other
file in the archive. It intentionally omits itself.
