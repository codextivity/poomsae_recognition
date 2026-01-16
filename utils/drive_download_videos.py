import os
import io
import re
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Optional

from tqdm import tqdm
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete token.json and re-authenticate.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

VIDEO_MIME_PREFIX = "video/"
DEFAULT_CHUNK_SIZE = 1024 * 1024 * 8  # 8 MB chunks


def authenticate() -> "googleapiclient.discovery.Resource":
    creds = None
    token_path = Path("token.json")

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            cred_path = Path("credentials.json")
            if not cred_path.exists():
                print("Missing credentials.json. Put your OAuth client file in this folder.", file=sys.stderr)
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return name or "unnamed"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_children(service, folder_id: str) -> List[Dict]:
    # Fetch both files and folders
    q = f"'{folder_id}' in parents and trashed = false"
    items: List[Dict] = []
    page_token = None

    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return items


def walk_folder(service, folder_id: str, rel_path: Path = Path(".")) -> Iterable[Dict]:
    """Yield dicts for all items in folder tree with an added 'rel_path'."""
    for item in list_children(service, folder_id):
        item_rel = rel_path / safe_filename(item["name"])
        if item["mimeType"] == "application/vnd.google-apps.folder":
            yield from walk_folder(service, item["id"], item_rel)
        else:
            item["rel_path"] = item_rel
            yield item


def download_file(service, file_id: str, out_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    """
    Downloads a single Drive file in original quality.
    Returns True if downloaded (or already present), False if skipped due to access/not found.
    """
    ensure_dir(out_path.parent)

    # Try to read metadata (size) to support skipping existing files
    try:
        meta = service.files().get(
            fileId=file_id,
            fields="size, name",
            supportsAllDrives=True
        ).execute()
    except HttpError as e:
        status = getattr(e.resp, "status", None)
        if status in (403, 404):
            print(f"SKIP (no access / not found): {file_id}")
            return False
        raise  # unexpected error -> bubble up

    remote_size = meta.get("size")

    # Skip if already exists with same size
    if out_path.exists() and remote_size is not None:
        try:
            if out_path.stat().st_size == int(remote_size):
                return True
        except Exception:
            pass

    # Download media
    try:
        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.FileIO(out_path, mode="wb")
        downloader = MediaIoBaseDownload(fh, req, chunksize=chunk_size)

        done = False
        pbar = tqdm(total=int(remote_size) if remote_size else None,
                    unit="B", unit_scale=True, desc=out_path.name)

        try:
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pbar.n = int(status.resumable_progress)
                    pbar.refresh()
        finally:
            pbar.close()
            fh.close()

        return True

    except HttpError as e:
        status = getattr(e.resp, "status", None)
        if status in (403, 404):
            print(f"SKIP during download (no access / not found): {file_id}")
            # Remove partial file if any
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            return False
        raise



def main():
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python drive_download_videos.py <FOLDER_ID> <OUTPUT_DIR>\n\n"
            "Example:\n"
            "  python drive_download_videos.py 1AbCDeFgHiJkLmNoPqRsTuVwXyZ downloads\n"
        )
        sys.exit(2)

    folder_id = sys.argv[1]
    out_dir = Path(sys.argv[2])

    service = authenticate()

    # Traverse and download only videos
    items = list(walk_folder(service, folder_id))
    video_items = [it for it in items if it.get("mimeType", "").startswith(VIDEO_MIME_PREFIX)]
    # video_items = [
    #     it for it in items
    #     if it.get("mimeType", "").startswith(VIDEO_MIME_PREFIX)
    #        and "front" in it.get("name", "").lower()
    # ]

    print(f"Found {len(video_items)} video file(s) in folder tree.")
    ensure_dir(out_dir)

    for it in video_items:
        rel_path: Path = it["rel_path"]
        out_path = out_dir / rel_path
        out_path = out_path.with_name(safe_filename(out_path.name))  # extra safety
        download_file(service, it["id"], out_path)

    print("Done.")


if __name__ == "__main__":
    main()
#python drive_download_videos.py https://drive.google.com/drive/folders/1TgFepC-Fa5EkqNyUhfkXeqrHSwop9K23 "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\videos"