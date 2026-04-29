"""
Runtime patches for twikit==2.3.3 (ryanstoic fork).

Twitter has been pruning the `legacy` block from its user-profile JSON
(observed missing on 2026-04-29: pinned_tweet_ids_str, withheld_in_countries).
twikit's User.__init__ does ~27 hard `legacy[key]` subscripts and any
missing key crashes the constructor, killing the realtime tracker.

The bot only consumes `tweet.created_at_datetime` from twikit; the User
object is purely transient. We seed every legacy field that User.__init__
touches with a sensible default before delegating to the original
__init__, so the constructor can never KeyError on a missing legacy key.

Idempotent: re-importing has no extra effect. Apply by importing this
module BEFORE any `from twikit import ...`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "__polymath_legacy_seed_patched__"

# All legacy[...] keys that twikit's User.__init__ subscripts directly,
# enumerated from twikit/user.py + twikit/guest/user.py at v2.3.3.
# Keep `entities` shaped as a nested dict because line 102 chains
# legacy['entities']['description']['urls'].
_LEGACY_DEFAULTS: dict = {
    "created_at": "",
    "name": "",
    "screen_name": "",
    "profile_image_url_https": "",
    "location": "",
    "description": "",
    "entities": {"description": {"urls": []}, "url": {"urls": []}},
    "pinned_tweet_ids_str": [],
    "verified": False,
    "possibly_sensitive": False,
    "can_dm": False,
    "can_media_tag": False,
    "want_retweets": False,
    "default_profile": False,
    "default_profile_image": False,
    "has_custom_timelines": False,
    "followers_count": 0,
    "fast_followers_count": 0,
    "normal_followers_count": 0,
    "friends_count": 0,
    "favourites_count": 0,
    "listed_count": 0,
    "media_count": 0,
    "statuses_count": 0,
    "is_translator": False,
    "translator_type": "",
    "profile_interstitial_type": "",
    "protected": False,
    "withheld_in_countries": [],
}


def _ensure_entities(entities) -> dict:
    """legacy['entities']['description']['urls'] and ['url']['urls'] are
    chained subscripts in twikit; ensure both nested paths resolve to a list."""
    if not isinstance(entities, dict):
        entities = {}
    description = entities.get("description")
    if not isinstance(description, dict):
        description = {}
    description.setdefault("urls", [])
    url = entities.get("url")
    if not isinstance(url, dict):
        url = {}
    url.setdefault("urls", [])
    return {**entities, "description": description, "url": url}


def _seed_legacy(data: dict) -> dict:
    """Return a shallow copy of `data` with `legacy` augmented to contain
    every key User.__init__ requires, defaulting any missing one."""
    legacy = data.get("legacy")
    if not isinstance(legacy, dict):
        return data
    seeded = dict(legacy)
    for k, v in _LEGACY_DEFAULTS.items():
        seeded.setdefault(k, v)
    seeded["entities"] = _ensure_entities(seeded.get("entities"))
    return {**data, "legacy": seeded}


def _patch(target_cls) -> None:
    if getattr(target_cls, _PATCHED_FLAG, False):
        return
    original_init = target_cls.__init__

    def __init__(self, client, data, *args, **kwargs):
        if isinstance(data, dict):
            data = _seed_legacy(data)
        original_init(self, client, data, *args, **kwargs)

    target_cls.__init__ = __init__
    setattr(target_cls, _PATCHED_FLAG, True)


def _apply() -> None:
    try:
        import twikit.user as _twikit_user
        _patch(_twikit_user.User)
    except Exception as e:
        logger.warning(f"twikit.user patch failed (non-fatal): {e}")

    try:
        import twikit.guest.user as _twikit_guest_user
        _patch(_twikit_guest_user.User)
    except Exception as e:
        logger.warning(f"twikit.guest.user patch failed (non-fatal): {e}")


_apply()
