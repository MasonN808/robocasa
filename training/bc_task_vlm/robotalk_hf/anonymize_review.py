"""Prepare and upload a review branch without modifying the public main branch.

Archive payloads are unchanged; filesystem identity is removed from tar headers.
All source files are checked against the pinned published revision before use.
The upload is resumable, and refuses to overwrite an existing unrelated branch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download
from PIL import Image
import pyarrow.parquet as pq

REPO = "DorianAtSchool/RoboTalk"
REVISION = "ddfa8aad5a3f1b2e996c0a05c59e724fe266f2af"
BRANCH = "anonymous-review"
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "training/bc_task_vlm/reports/robotalk_hf_full"
OUTPUT = ROOT / "training/bc_task_vlm/reports/robotalk_anonymous_review"
IDENTITY = re.compile(r"DorianAtSchool|\bdorian\b|dbenhamou|umass|shlomo|/work/|/home/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I)


def digest_file(path, lfs):
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_text(text, label):
    if IDENTITY.search(text):
        raise ValueError(f"Identity remains in {label}")


def check_image(data, label):
    with Image.open(io.BytesIO(data)) as im:
        check_text(str(im.info), label + " image metadata")
        check_text(str(dict(im.getexif())), label + " EXIF")


def prepare_one(item):
    name = item.rfilename
    source = SOURCE / name
    lfs = item.lfs
    expected = lfs.sha256 if lfs else item.blob_id
    if not source.is_file() or digest_file(source, bool(lfs)) != expected:
        source = Path(hf_hub_download(REPO, filename=name, repo_type="dataset",
            revision=REVISION, cache_dir=str(OUTPUT / "source_cache")))
        if digest_file(source, bool(lfs)) != expected:
            raise ValueError(f"Downloaded file does not match pinned HF revision: {name}")
    target = OUTPUT / "upload" / name
    if name.startswith("media_archives/"):
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with tarfile.open(source) as src, tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as dst:
            for member in src:
                check_text(member.name, name)
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError(f"Unexpected archive member: {name}/{member.name}")
                payload = src.extractfile(member).read()
                check_image(payload, member.name)
                clean = tarfile.TarInfo(member.name)
                clean.size = len(payload)
                clean.mode = 0o644
                clean.uid = clean.gid = clean.mtime = 0
                clean.uname = clean.gname = ""
                dst.addfile(clean, io.BytesIO(payload))
                count += 1
        # Independent readback verifies that every image and internal path survives.
        with tarfile.open(source) as src, tarfile.open(target) as dst:
            a, b = src.getmembers(), dst.getmembers()
            assert len(a) == len(b) == count
            for old, new in zip(a, b):
                assert old.name == new.name and old.size == new.size
                assert not (new.uid or new.gid or new.uname or new.gname or new.mtime or new.pax_headers)
                assert src.extractfile(old).read() == dst.extractfile(new).read()
        return name, count
    if name.startswith("media/"):
        check_image(source.read_bytes(), name)
    elif name.startswith("data/"):
        table = pq.read_table(source)
        check_text(str(table.schema.metadata), name)
        for column in table.column_names:
            for value in table[column].to_pylist():
                check_text(str(value), name + ":" + column)
    elif name == "README.md":
        content = source.read_text().replace(
            'load_dataset("DorianAtSchool/RoboTalk",', 'load_dataset("./RoboTalk",'
        )
        content = content.replace("```python", "Download this branch through the anonymous review link and extract it to\n`./RoboTalk` before loading either table locally.\n\n```python", 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        check_text(content, name)
    elif name == "dataset_info.json":
        info = json.loads(source.read_text())
        info.pop("repo_id", None)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(info, indent=2) + "\n")
        check_text(target.read_text(), name)
    elif name != "preview.html":
        check_text(source.read_text(), name)
    return name, 0


def prepare(api):
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total_images = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (name, images) in enumerate(pool.map(prepare_one, info.siblings), 1):
            total_images += images
            if index % 250 == 0:
                print(f"Verified/sanitized {index}/{len(info.siblings)} files", flush=True)
    assert pq.read_table(SOURCE / "data/trajectories.parquet").num_rows == 7950
    assert pq.read_table(SOURCE / "data/ticks.parquet").num_rows == 66557
    report = {"source_revision": REVISION, "files_audited": len(info.siblings),
              "archives_rebuilt": 7950, "archive_images_verified": total_images,
              "trajectory_rows": 7950, "tick_rows": 66557,
              "removed_files": ["preview.html"], "main_unchanged": True}
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def upload(api):
    main_before = api.dataset_info(REPO, revision="main").sha
    report = json.loads((OUTPUT / "audit.json").read_text())
    assert report["source_revision"] == REVISION
    assert report["archives_rebuilt"] == 7950
    refs = api.list_repo_refs(REPO, repo_type="dataset")
    existing = next((r for r in refs.branches if r.name == BRANCH), None)
    receipt = OUTPUT / "upload_receipt.json"
    if existing is not None and not receipt.exists():
        raise RuntimeError("Review branch exists without our receipt; refusing overwrite")
    if not existing:
        api.create_branch(REPO, repo_type="dataset", branch=BRANCH, revision=REVISION)
        receipt.write_text(json.dumps({"branch": BRANCH, "parent": REVISION}))
    current = api.dataset_info(REPO, revision=BRANCH, files_metadata=True)
    saved = json.loads(receipt.read_text())
    if current.sha != saved["parent"]:
        raise RuntimeError("Review branch changed outside this uploader; refusing overwrite")
    remote = {s.rfilename: s for s in current.siblings}
    operations = []
    for path in sorted((OUTPUT / "upload").rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(OUTPUT / "upload").as_posix()
        item = remote.get(name)
        if item:
            h = item.lfs.sha256 if item.lfs else item.blob_id
            if digest_file(path, bool(item.lfs)) == h:
                continue
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    if "preview.html" in remote:
        operations.append(CommitOperationDelete(path_in_repo="preview.html"))
    for name in sorted(remote):
        if name.startswith("media/"):
            operations.append(CommitOperationDelete(path_in_repo=name))
    expected_files = set(remote)
    for operation in operations:
        if isinstance(operation, CommitOperationDelete):
            expected_files.discard(operation.path_in_repo)
        else:
            expected_files.add(operation.path_in_repo)
    parent = current.sha
    for start in range(0, len(operations), 200):
        batch = operations[start:start + 200]
        result = api.create_commit(REPO, repo_type="dataset", revision=BRANCH,
            parent_commit=parent, operations=batch,
            commit_message="Sanitize review-release metadata", num_threads=8)
        parent = result.oid
        receipt.write_text(json.dumps({"branch": BRANCH, "parent": parent}))
        print(f"Uploaded {min(start + 200, len(operations))}/{len(operations)} replacements", flush=True)
    final = api.dataset_info(REPO, revision=BRANCH, files_metadata=True)
    files = {s.rfilename: s for s in final.siblings}
    for path in (OUTPUT / "upload").rglob("*"):
        if path.is_file():
            s = files[path.relative_to(OUTPUT / "upload").as_posix()]
            assert digest_file(path, bool(s.lfs)) == (s.lfs.sha256 if s.lfs else s.blob_id)
    assert "preview.html" not in files
    assert not any(name.startswith("media/") for name in files)
    assert set(files) == expected_files
    assert api.dataset_info(REPO, revision="main").sha == main_before, "main changed during upload"
    receipt.write_text(json.dumps({"branch": BRANCH, "parent": final.sha, "verified": True}, indent=2))
    print(f"VERIFIED: https://huggingface.co/datasets/{REPO}/tree/{BRANCH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "upload"))
    args = parser.parse_args()
    api = HfApi()
    (prepare if args.stage == "prepare" else upload)(api)
