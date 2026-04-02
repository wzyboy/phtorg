#!/usr/bin/env python

import re
import io
import csv
import hashlib
import logging
import dataclasses
from pathlib import Path
from datetime import datetime
from datetime import UTC
from zoneinfo import ZoneInfo
from collections.abc import Iterable

import click
from tqdm import tqdm
from PIL import Image
from PIL import ExifTags
from pillow_heif import register_heif_opener
from pymediainfo import MediaInfo
from tabulate import tabulate
from dateutil.parser import isoparse

from phtorg import constants
from phtorg.tpe import tpe_submit


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

    @classmethod
    def no_datetime(cls, path: Path, error: str):
        return cls(path, None, None, [error])


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

    pillow_exts = {'.jpg', '.jpeg', '.heic', '.dng'}
    mediainfo_exts = {'.mov', '.mp4', '.m4v'}
    screenshot_exts = {'.png', '.gif', '.bmp', '.webp'}
    allowed_exts = pillow_exts | mediainfo_exts | screenshot_exts
    allow_mtime = False

    def __init__(self, src_dir: Path, dst_dir: Path, timezone_name: str) -> None:
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.rename_tasks: list[RenameTask] = []
        self.skipped_items: list[PhotoInfo] = []
        self.timezone = ZoneInfo(timezone_name)

    def get_info(self, photo: Path) -> PhotoInfo:
        ext = photo.suffix.lower()
        if ext in self.pillow_exts:
            info = self.get_info_from_pillow(photo)
        elif ext in self.mediainfo_exts:
            info = self.get_info_from_mediainfo(photo)
        elif ext in self.screenshot_exts:
            info = PhotoInfo.no_datetime(photo, 'Datetime extraction is skipped for this type of file')
        else:
            raise RuntimeError(f'Unexpected extension: {photo}')

        if info.datetime is None:
            if self.allow_mtime:
                info = self.get_info_from_file(info.path)
            else:
                raise Exception('Cannot determine datetime from EXIF/MediaInfo. Fallback to mtime is not allowed.')
        else:
            # Validate timezone
            tzinfo = info.datetime.tzinfo
            assert tzinfo is not None, 'timezone does not exist'
            assert str(tzinfo) == str(self.timezone), 'timezone does not match'

        return info

    def iter_photo(self) -> Iterable[Path]:
        if self.src_dir.is_file():
            yield self.src_dir
        else:
            for p in self.src_dir.rglob('*.*'):
                if p.suffix.lower() in self.allowed_exts:
                    yield p

    def parse_timestamp(self, ts: int | float) -> datetime:
        '''Parse Unix timestamp into an aware datetime'''
        return datetime.fromtimestamp(ts, tz=self.timezone)

    def get_info_from_file(self, photo: Path) -> PhotoInfo:
        dt = self.parse_timestamp(photo.stat().st_mtime)
        return PhotoInfo(photo, dt, 'mtime')

    def get_info_from_pillow(self, photo: Path) -> PhotoInfo:
        with Image.open(photo) as image:
            _exif1 = image.getexif()
            _exif2 = _exif1.get_ifd(0x8769)
        _exif = dict(_exif1) | _exif2
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in _exif.items()
            if k in ExifTags.TAGS and type(v) is not bytes
        }

        # No EXIF at all
        if not exif:
            return PhotoInfo.no_datetime(photo, 'File is EXIF-compatible but no EXIF found')

        # Extract datetime and offset from EXIF
        # EXIF 2.31 (July 2016) introduced "OffsetTime", "OffsetTimeOriginal" and "OffsetTimeDigitized".
        # They are formatted as seven ASCII characters (including the null terminator) denoting
        # the hours and minutes of the offset, like +01:00 or -01:00.
        _exif_dt = exif.get('DateTimeOriginal') or exif.get('DateTimeDigitized') or exif.get('DateTime')
        _exif_time_offset = exif.get('OffsetTimeOriginal') or exif.get('OffsetTimeDigitized') or exif.get('OffsetTime')
        # If not conform to standard, treat it as garbage.
        if _exif_time_offset is not None and not re.match(r'[+-]\d\d\:\d\d', _exif_time_offset):
            _exif_time_offset = ''

        # No datetime in EXIF
        if _exif_dt is None:
            return PhotoInfo.no_datetime(photo, 'EXIF exists but no datetime found')

        # Parse datetime string
        # Some software appends non-ASCII bytes like '下午'
        # 'DateTime': '2018:12:25 18:19:37ä¸\x8bå\x8d\x88'
        dt_str = _exif_dt[:19].replace(':', '-', 2)
        if _exif_time_offset:
            dt_str += _exif_time_offset
            dt = isoparse(dt_str).astimezone(self.timezone)
        else:
            dt = isoparse(dt_str).replace(tzinfo=self.timezone)
        return PhotoInfo(photo, dt, 'EXIF')

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
            return PhotoInfo.no_datetime(photo, 'Cannot extract datetime from MediaInfo')

        # If dt is aware, convert to local dt
        if dt.tzinfo:
            local_dt = dt.astimezone(self.timezone)
        # If dt is naive, assume it's UTC
        else:
            local_dt = dt.replace(tzinfo=UTC).astimezone(self.timezone)
        return PhotoInfo(photo, local_dt, 'MediaInfo')

    def start(self):
        self._prepare_rename_tasks(self.iter_photo())
        self.rename_tasks = sorted(self.rename_tasks)
        self.skipped_items = sorted(self.skipped_items)
        log.info(f'Collected {len(self.rename_tasks)} rename tasks.')
        log.info(f'Collected {len(self.skipped_items)} skipped items.')
        self._confirm_rename()

    @staticmethod
    def get_deterministic_filename(photo: Path, dt: datetime, prefix: str = constants.DEFAULT_PREFIX) -> str:
        timestamp = dt.strftime(constants.DATETIME_FMT)
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
        assert info.datetime is not None

        # Compute filename
        fn = self.get_deterministic_filename(photo, info.datetime)

        full_path = self.dst_dir / str(info.datetime.year) / fn
        rename_task = RenameTask(info, full_path)
        return rename_task

    def _prepare_rename_tasks(self, photos: Iterable[Path]) -> None:
        completed, failed = tpe_submit(self._get_rename_task, sorted(photos))
        for photo, task in completed:
            # Validate
            if task.destination.exists():
                # Allow idempotent operations: don't rename a file
                # if its filename is already what we want
                if task.destination.samefile(task.photo_info.path):
                    continue
                info = task.photo_info
                info.errors.append(f'Destination already exists: {task.destination}')
                self.skipped_items.append(info)
            else:
                self.rename_tasks.append(task)
        for photo, exception in failed:
            info = PhotoInfo.no_datetime(photo, repr(exception))
            self.skipped_items.append(info)

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
        text = io.StringIO()
        text.write(f'Rename ({len(self.rename_tasks)}):\n')
        text.write(tabulate([t.row() for t in self.rename_tasks], headers='keys'))
        text.write('\n\n')
        text.write(f'Skip ({len(self.skipped_items)}):\n')
        text.write(tabulate([i.row() for i in self.skipped_items], headers='keys'))
        text.write('\n\n')
        click.echo_via_pager(text.getvalue())

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
