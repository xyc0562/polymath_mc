"""Verify src.utils.twikit_patches actually fixes the pinned_tweet_ids_str KeyError."""

from __future__ import annotations

from unittest.mock import MagicMock


def _legacy_payload_without_pinned_tweet_ids() -> dict:
    """Realistic legacy payload from Twitter's response, minus pinned_tweet_ids_str."""
    return {
        "created_at": "Fri Mar 16 14:00:00 +0000 2007",
        "name": "Test",
        "screen_name": "test",
        "profile_image_url_https": "",
        "location": "",
        "description": "",
        "entities": {"description": {"urls": []}, "url": {"urls": []}},
        "verified": False,
        "possibly_sensitive": False,
        "can_dm": False,
        "can_media_tag": False,
        "want_retweets": False,
        "default_profile": True,
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


def test_user_init_no_longer_crashes_without_pinned_tweet_ids():
    # Apply patch (idempotent — safe even if another test imported twikit first)
    from src.utils import twikit_patches  # noqa: F401
    import twikit.user

    data = {
        "rest_id": "44196397",
        "is_blue_verified": False,
        "profile_image_shape": None,
        "legacy": _legacy_payload_without_pinned_tweet_ids(),
    }

    user = twikit.user.User(client=MagicMock(), data=data)
    assert user.pinned_tweet_ids == []
    assert user.id == "44196397"
    assert user.screen_name == "test"


def test_user_init_preserves_pinned_tweet_ids_when_present():
    from src.utils import twikit_patches  # noqa: F401
    import twikit.user

    legacy = _legacy_payload_without_pinned_tweet_ids()
    legacy["pinned_tweet_ids_str"] = ["1234567890"]

    data = {
        "rest_id": "44196397",
        "is_blue_verified": False,
        "profile_image_shape": None,
        "legacy": legacy,
    }

    user = twikit.user.User(client=MagicMock(), data=data)
    assert user.pinned_tweet_ids == ["1234567890"]


def test_patch_is_idempotent():
    from src.utils import twikit_patches  # noqa: F401
    import twikit.user

    init_after_first = twikit.user.User.__init__

    # Re-apply
    twikit_patches._apply()
    init_after_second = twikit.user.User.__init__

    assert init_after_first is init_after_second


def test_user_init_survives_partial_entities():
    """If Twitter returns entities={'description': {}} (no 'urls'), the chained
    subscript legacy['entities']['description']['urls'] would KeyError without
    deep-merging."""
    from src.utils import twikit_patches  # noqa: F401
    import twikit.user

    legacy = _legacy_payload_without_pinned_tweet_ids()
    legacy["pinned_tweet_ids_str"] = []
    legacy["entities"] = {"description": {}}  # missing nested 'urls'

    user = twikit.user.User(
        client=MagicMock(),
        data={"rest_id": "1", "is_blue_verified": False, "profile_image_shape": None, "legacy": legacy},
    )
    assert user.description_urls == []
    assert user.urls == []


def test_user_init_survives_dropping_any_seeded_legacy_field():
    """Future-proof: if Twitter drops more fields, the patch still saves us
    as long as the field is in _LEGACY_DEFAULTS."""
    from src.utils import twikit_patches
    import twikit.user

    base = _legacy_payload_without_pinned_tweet_ids()
    base["pinned_tweet_ids_str"] = []  # restore it for the baseline
    fields_to_test = list(twikit_patches._LEGACY_DEFAULTS.keys())

    failures = []
    for missing_field in fields_to_test:
        legacy = {k: v for k, v in base.items() if k != missing_field}
        data = {
            "rest_id": "1",
            "is_blue_verified": False,
            "profile_image_shape": None,
            "legacy": legacy,
        }
        try:
            twikit.user.User(client=MagicMock(), data=data)
        except Exception as e:
            failures.append(f"{missing_field}: {type(e).__name__}: {e}")

    assert not failures, f"User crashed when these fields were missing:\n  " + "\n  ".join(failures)
