"""File utilities: reading, writing, hashing, zipping, and inspecting files of any type."""
import os
import json
import hashlib
import zipfile
import mimetypes
import shutil


def get_file_info(path: str) -> dict:
    """Return basic metadata about a file: size, extension, mime type, last modified time."""
    stat = os.stat(path)
    mime_type, _ = mimetypes.guess_type(path)
    return {
        "size_bytes": stat.st_size,
        "extension": os.path.splitext(path)[1],
        "mime_type": mime_type or "unknown",
        "modified_time": stat.st_mtime,
    }


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read and return the full contents of a text file."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write text content to a file, overwriting it if it exists."""
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def read_json_file(path: str) -> object:
    """Read and parse a JSON file, returning the resulting Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, data: object, indent: int = 2) -> None:
    """Write a Python object to a file as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_yaml_file(path: str) -> object:
    """Read and parse a YAML file, returning the resulting Python object."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml_file(path: str, data: object) -> None:
    """Write a Python object to a file as YAML."""
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def compute_file_hash(path: str, algorithm: str = "sha256") -> str:
    """Compute the hex digest hash of a file's contents (works for any file type)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_files(file_paths: list[str], output_zip_path: str) -> None:
    """Compress a list of files into a single .zip archive."""
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            zf.write(path, arcname=os.path.basename(path))


def unzip_file(zip_path: str, output_dir: str) -> list[str]:
    """Extract a .zip archive into output_dir, returning the list of extracted file names."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
        return zf.namelist()


def list_files_in_directory(directory: str, extension: str | None = None) -> list[str]:
    """List all files in a directory, optionally filtered by extension (e.g. '.txt')."""
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return files


def get_file_extension(path: str) -> str:
    """Return the file extension of a path, including the leading dot (e.g. '.pdf')."""
    return os.path.splitext(path)[1]


def copy_file(src: str, dst: str) -> None:
    """Copy a file from src to dst."""
    shutil.copy2(src, dst)


def delete_file(path: str) -> None:
    """Delete a file. Raises FileNotFoundError if it doesn't exist."""
    os.remove(path)