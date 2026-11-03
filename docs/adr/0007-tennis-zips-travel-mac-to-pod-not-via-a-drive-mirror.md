# Tennis dataset zips travel Mac→pod directly, not via a Drive mirror

`arg-bolts` reaches a pod through a Google Drive mirror and `gdown`, with the
file id in `FODCV_DATASET_GDRIVE_ID`. The tennis datasets do not follow that
pattern. They are fetched once on the Mac with the operator's Roboflow link and
copied to the pod with `rsync`.

The reason is the credential. A Roboflow download URL carries `?key=` — an
account credential — and fetching on the pod would put it in the shell history
and environment of a rented machine. The existing rule that gated credentials
live in the environment "and nowhere else… not in a log that gets pushed" is
protecting against precisely that machine. The zip is needed on the Mac anyway
for the Mac Gate, so fetching it twice buys nothing, and copying it directly
avoids Drive's large-file interstitial — the failure mode that makes `curl`
save an HTML page which passes as a zip until a SHA check hours later.

## Consequences

- The `sha256` of both zips, the Roboflow **version number** and the download
  date are recorded in the repo. The hash is not a secret; the link is. Without
  the version pin this is unreproducible, because regenerating a Roboflow
  version with different augmentation yields different bytes from the same URL.
- The two URLs stay in the operator's shell as `ROBOFLOW_URL_TENNIS_NEG` and
  `ROBOFLOW_URL_TENNIS_HOLDOUT`, and are never committed.
- If a pod dies mid-run the zip is re-uploaded rather than re-fetched, which is
  slower on a poor uplink. If that becomes painful, switch to the Drive mirror.
