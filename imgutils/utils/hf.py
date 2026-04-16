"""
Hugging Face Hub utilities with cache-first download behaviour.

When a file has already been downloaded, :func:`hf_hub_download` will serve it
directly from the local cache without making any network requests.  Only when
the file is not yet cached will it fall back to the normal download path.
"""
from huggingface_hub import hf_hub_download as _hf_hub_download

__all__ = ['hf_hub_download']


def hf_hub_download(repo_id, filename, **kwargs):
    """
    Download a file from a Hugging Face Hub repository, preferring the local cache.

    Attempts to load the file with ``local_files_only=True`` first.  If the file
    is not present in the cache (e.g. on the very first run), it falls back to
    the regular download path so the file is fetched from the Hub and stored in
    the cache for future calls.

    All keyword arguments are forwarded to :func:`huggingface_hub.hf_hub_download`.
    The ``local_files_only`` keyword is accepted but ignored — it is managed
    internally by this wrapper.

    :param repo_id: The repository ID on Hugging Face Hub (e.g. ``"deepghs/imgutils-models"``).
    :param filename: Path to the file inside the repository.
    :return: Local path to the cached file.
    :rtype: str
    """
    kwargs.pop('local_files_only', None)
    try:
        return _hf_hub_download(repo_id, filename, local_files_only=True, **kwargs)
    except Exception:
        return _hf_hub_download(repo_id, filename, **kwargs)
