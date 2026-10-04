# Anonymous review preparation

The review branch is prepared by
`training/bc_task_vlm/robotalk_hf/anonymize_review.py`.

## Scope

- Public dataset `main` is not edited.
- The branch retains the original trajectory and Parquet contents.
- README loading examples use a local downloaded directory.
- Dataset metadata omits the source repository identifier.
- Every media archive is rebuilt with zero uid/gid/mtime and empty owner names.
- Every archived image is compared byte-for-byte with its source after rebuilding.
- Raw JSON, Parquet values/schema metadata, image metadata, and archive paths
  are scanned for known personal/institutional identifiers and email addresses.
- The old `preview.html` is omitted from the branch. The separate explorer Space
  and its visibility are not modified.
- Existing licensing text is preserved, not replaced with a new license.

## Verification

`audit.json` records the preparation checks; `upload_receipt.json` records the
last uploaded commit and gains `verified: true` only after checking uploaded
replacement hashes, the file inventory, and the unchanged main revision.

The sanitized README's exact local `load_dataset` instructions were tested for
both configurations: 7,950 trajectory rows and 66,557 tick rows.

## Submission

Wait for the upload receipt to contain `verified: true`. Submit the
`anonymous-review` branch URL to anonymous-hf.com using the author's own login.
Share the resulting anonymous proxy URL with reviewers, not the HF URL.
Inspect the proxy while logged out before submission; expiry and service
availability must cover the review period. This process removes direct identity
metadata, but cannot prevent matching this release to the existing public data.

This directory contains internal provenance and is **not** itself uploaded.
Only the `upload/` subtree supplies replacement files for the branch.
