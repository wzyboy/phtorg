#!/usr/bin/env python

import csv
import hashlib
import logging
import dataclasses
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections.abc import Iterable

import pytz
from tqdm import tqdm
from PIL import Image
from PIL import ExifTags
from pillow_heif import register_heif_opener
from pymediainfo import MediaInfo
from tabulate import tabulate
from dateutil.parser import isoparse

from phtorg import constants


register_heif_opener()
log = logging.getLogger(__name__)


@dataclasses.dataclass(order=True)
class PhotoInfo:
    path: Path
    datetime: datetime | None
    datetime_source: str | None
    errors: list[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        return f'{self.path} @ {self.datetime} ({self.datetime_source})'

    @staticmethod
    def header() -> list[str]:
        return ['src', 'errors']

    def row(self) -> dict:
        return {
            'src': str(self.path),
            'errors': '; '.join(self.errors),
        }


@dataclasses.dataclass(order=True)
class RenameTask:
    photo_info: PhotoInfo
    destination: Path

    def __repr__(self) -> str:
        return f'{self.photo_info} -> {self.destination}'

    @staticmethod
    def header() -> list[str]:
        return ['src', 'datetime', 'datetime_source', 'dst']

    def row(self) -> dict:
        return {
            'src': str(self.photo_info.path),
            'datetime': str(self.photo_info.datetime),
            'datetime_source': self.photo_info.datetime_source,
            'dst': str(self.destination),
        }


class PhotoOrganizer:

    pillow_exts = {'.jpg', '.jpeg', '.heic'}
    mediainfo_exts = {'.mov', '.mp4', '.m4v'}
    screenshot_exts = {'.png', '.gif', '.bmp', '.webp'}
    allowed_exts = pillow_exts | mediainfo_exts | screenshot_exts
    timezone = pytz.timezone('America/Vancouver')

    def __init__(self, src_dir: Path, dst_dir: Path) -> None:
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.rename_tasks: list[RenameTask] = []
        self.skipped_items: list[PhotoInfo] = []

    def get_info(self, photo: Path) -> PhotoInfo:
        ext = photo.suffix.lower()
        if ext in self.pillow_exts:
            info = self.get_info_from_pillow(photo)
        elif ext in self.mediainfo_exts:
            info = self.get_info_from_mediainfo(photo)
        elif self.is_screenshot(photo):
            info = self.get_info_from_file(photo)
        else:
            raise RuntimeError(f'Unexpected extension: {photo}')

        # Validation
        if info.datetime is not None:
            tzinfo = info.datetime.tzinfo
            assert tzinfo is not None, 'timezone does not exist'
            assert str(tzinfo) == str(self.timezone), 'timezone does not match'

        return info

    def parse_timestamp(self, ts: int | float) -> datetime:
        '''Parse Unix timestamp into an aware datetime'''
        return datetime.fromtimestamp(ts, tz=self.timezone)

    def is_screenshot(self, photo: Path) -> bool:
        return photo.suffix.lower() in self.screenshot_exts or photo.parent.name == 'Screenshots'

    def get_info_from_file(self, photo: Path) -> PhotoInfo:
        dt = self.parse_timestamp(photo.stat().st_mtime)
        return PhotoInfo(photo, dt, 'mtime')

    def get_info_from_pillow(self, photo: Path) -> PhotoInfo:
        image = Image.open(photo)
        _exif1 = image.getexif()
        _exif2 = _exif1.get_ifd(0x8769)
        _exif = dict(_exif1) | _exif2
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in _exif.items()
            if k in ExifTags.TAGS and type(v) is not bytes
        }

        # If photo does not have EXIF at all, use mtime
        if not exif:
            info = self.get_info_from_file(photo)
            error = 'File is EXIF-compatible but no EXIF found'
            info.errors.append(error)
            return info

        # Extract dt from EXIF
        _exif_dt = exif.get('DateTimeOriginal') or exif.get('DateTimeDigitized') or exif.get('DateTime')
        if _exif_dt is None:
            info = self.get_info_from_file(photo)
            error = f'EXIF exists but no datetime found: {exif}'
            info.errors.append(error)
            return info
        else:
            # Some software appends non-ASCII bytes like '下午'
            # 'DateTime': '2018:12:25 18:19:37ä¸\x8bå\x8d\x88'
            _exif_dt = _exif_dt[:19].replace(':', '-', 2)
            exif_dt = self.timezone.localize(isoparse(_exif_dt))
            return PhotoInfo(photo, exif_dt, 'EXIF')

    def get_info_from_mediainfo(self, photo: Path) -> PhotoInfo:
        mediainfo = MediaInfo.parse(photo)
        general_track = mediainfo.general_tracks[0]  # type: ignore
        if dt_str := general_track.comapplequicktimecreationdate:
            # com.apple.quicktime.creationdate         : 2018-10-08T21:24:34-0700
            dt = isoparse(dt_str)
        elif dt_str := general_track.encoded_date or general_track.tagged_date:
            assert dt_str.startswith('UTC') or dt_str.endswith('UTC'), 'encoded_date/tagged_date should have UTC marking'
            dt_str = dt_str.removeprefix('UTC').removesuffix('UTC').strip()
            dt = isoparse(dt_str)
        else:
            info = self.get_info_from_file(photo)
            error = f'Cannot extract datetime from MediaInfo: {mediainfo}'
            info.errors.append(error)
            return info

        # If dt is aware, convert to local dt
        if dt.tzinfo:
            local_dt = dt.astimezone(self.timezone)
        # If dt is naive, assume it's UTC
        else:
            local_dt = pytz.utc.localize(dt).astimezone(self.timezone)
        return PhotoInfo(photo, local_dt, 'MediaInfo')

    def start(self):
        if self.src_dir.is_file():
            photo_paths = [self.src_dir]
        else:
            photo_paths = self.src_dir.rglob('*.*')

        self._prepare_rename_tasks(photo_paths)
        self.rename_tasks = sorted(self.rename_tasks)
        self.skipped_items = sorted(self.skipped_items)
        log.info(f'Collected {len(self.rename_tasks)} rename tasks.')
        log.info(f'Collected {len(self.skipped_items)} skipped items.')
        self._confirm_rename()

    @staticmethod
    def get_deterministic_filename(photo: Path, dt: datetime, prefix: str = constants.DEFAULT_PREFIX) -> str:
        timestamp = dt.strftime('%Y%m%d_%H%M%S')
        # Generate a Git-like hash (first 7 chars of SHA-1)
        with open(photo, 'rb') as f:
            hash_obj = hashlib.sha1()
            while chunk := f.read(1024 * 1024 * 10):  # 10 MiB
                hash_obj.update(chunk)
        h = hash_obj.hexdigest()[:7]
        fn = f'{prefix}{timestamp}_{h}{photo.suffix.lower()}'
        return fn

    def _get_rename_task(self, photo: Path) -> RenameTask:
        info = self.get_info(photo)
        assert info.datetime is not None, 'cannot rename without datetime'

        # Compute filename
        if self.is_screenshot(photo):
            prefix = constants.SCREENSHOT_PREFIX
        else:
            prefix = constants.DEFAULT_PREFIX
        fn = self.get_deterministic_filename(photo, info.datetime, prefix)

        full_path = self.dst_dir / str(info.datetime.year) / fn
        rename_task = RenameTask(info, full_path)
        return rename_task

    def _prepare_rename_tasks(self, photo_paths: Iterable[Path]) -> None:
        # Prime the generator so that we can see progress in tqdm
        photos = sorted(p for p in photo_paths if p.suffix.lower() in self.allowed_exts)

        tpe = concurrent.futures.ThreadPoolExecutor()
        futures_map = {
            tpe.submit(self._get_rename_task, photo): photo
            for photo in photos
        }
        pending = set(futures_map.keys())
        pbar = tqdm(total=len(futures_map))
        try:
            while pending:
                # Wait for a few to complete
                done_now, pending = concurrent.futures.wait(pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done_now:
                    pbar.update(1)
                    try:
                        rename_task = future.result()
                    except Exception as e:
                        photo = futures_map[future]
                        log.error(f'{photo}: {e!r}')
                        info = PhotoInfo(photo, None, None, [repr(e)])
                        self.skipped_items.append(info)
                        continue

                    # Validate
                    if rename_task.destination.exists():
                        # Allow idempotent operations: don't rename a file
                        # if its filename is already what we want
                        if rename_task.destination.samefile(rename_task.photo_info.path):
                            continue
                        info = rename_task.photo_info
                        info.errors.append(f'Destination already exists: {rename_task.destination}')
                        self.skipped_items.append(info)
                    else:
                        self.rename_tasks.append(rename_task)

        except KeyboardInterrupt:
            log.warning('KeyboardInterrupt')
            tpe.shutdown(wait=False, cancel_futures=True)
        finally:
            pbar.close()

    def _confirm_rename(self) -> None:
        print('Rename the files, preview the tasks, save the tasks in CSV, or abort?')
        try:
            resp = input('(R)ename/(p)review/(s)ave/(a)bort? ').lower()
        except KeyboardInterrupt:
            return

        if resp == 'r':
            self._do_rename()
        elif resp == 'p':
            self._preview_tasks()
            self._confirm_rename()
        elif resp == 's':
            self._save_tasks()
            self._confirm_rename()
        elif resp == 'a':
            return
        else:
            self._confirm_rename()

    def _do_rename(self) -> None:
        for task in tqdm(self.rename_tasks):
            task.destination.parent.mkdir(parents=True, exist_ok=True)
            task.photo_info.path.rename(task.destination)

    def _preview_tasks(self) -> None:
        print(f'Rename ({len(self.rename_tasks)}):')
        print(tabulate([t.row() for t in self.rename_tasks], headers='keys'))
        print(f'Skip ({len(self.skipped_items)}):')
        print(tabulate([i.row() for i in self.skipped_items], headers='keys'))

    def _save_tasks(self) -> None:
        with open('rename_tasks.csv', 'w', encoding='utf-8') as f:
            rename_tasks_csv = csv.DictWriter(f, fieldnames=RenameTask.header())
            rename_tasks_csv.writeheader()
            rename_tasks_csv.writerows(t.row() for t in self.rename_tasks)
        with open('skipped_items.csv', 'w', encoding='utf-8') as f:
            skipped_items_csv = csv.DictWriter(f, fieldnames=PhotoInfo.header())
            skipped_items_csv.writeheader()
            skipped_items_csv.writerows(i.row() for i in self.skipped_items)
        log.info('Preview of operations written to `rename_tasks.csv` and `skipped_items.csv`')
