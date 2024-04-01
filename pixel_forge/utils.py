from typing import Optional, List
import os
import requests
from tqdm import tqdm


def load_list(data_path: str, filename: Optional[str] = None) -> List[str]:
    """Load a list of strings from a file."""
    if filename is not None:
        data_path = os.path.join(data_path, filename)
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items


def prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0


def truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text


def download_file(url: str, filepath: str, chunk_size: int = 4 * 1024 * 1024, quiet: bool = False):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        return

    file_size = int(r.headers.get("Content-Length", 0))
    filename = url.split("/")[-1]
    progress = tqdm(total=file_size, unit="B", unit_scale=True, desc=filename, disable=quiet)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()
