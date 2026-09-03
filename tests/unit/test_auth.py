"""`W4-04` — kho API key, chỗ ép tenant, và token bucket.

Nhóm 2 là nhóm quan trọng nhất của cả hạng mục: nó đóng lỗ mà `W2-06` **ghi ra
là không đóng được** — không có chỗ nào trong `rag_core` phân biệt được "không
lọc tenant" là đúng (eval chạy toàn corpus) với "serving quên truyền".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_core.retrieval.filters import MetadataFilter
from rag_core.schemas import Language
from serving.core.auth import (
    ADMIN_SCOPE,
    ApiKeyStore,
    CrossTenantError,
    Principal,
    digest_of,
    key_hint,
    mint,
    tenant_filter,
)
from serving.core.ratelimit import RateLimiter

# ---------------------------------------------------------------------------
# 1. Kho key
# ---------------------------------------------------------------------------


def test_minting_never_writes_the_raw_key(tmp_path: Path) -> None:
    """⭐⭐ Tính chất trung tâm của kho key.

    Ghi key thô ra đĩa nghĩa là ai đọc được file cấu hình thì gọi được API —
    và file ấy đi qua backup, qua image container, qua `docker cp`. Trên đĩa chỉ
    được có digest; mất key thì cấp key mới, không có đường đọc lại.
    """
    store = tmp_path / "keys.json"
    raw = mint(store, tenant_id="acme")
    text = store.read_text(encoding="utf-8")
    assert raw not in text
    assert digest_of(raw) in text


def test_a_minted_key_actually_opens_the_door(tmp_path: Path) -> None:
    store = tmp_path / "keys.json"
    raw = mint(store, tenant_id="acme", scopes=[ADMIN_SCOPE], rate_limit_per_minute=5)
    principal = ApiKeyStore.load(store).lookup(raw)
    assert principal is not None
    assert (principal.tenant_id, principal.is_admin, principal.rate_limit_per_minute) == (
        "acme",
        True,
        5,
    )


def test_minting_a_second_key_does_not_revoke_the_first(tmp_path: Path) -> None:
    """Ghi đè cả file là cách xoay vòng key biến thành một sự cố: cấp key mới cho
    khách hàng B làm khách hàng A mất quyền, và không gì báo cho tới lần gọi kế."""
    store = tmp_path / "keys.json"
    first = mint(store, tenant_id="acme")
    second = mint(store, tenant_id="globex")
    keys = ApiKeyStore.load(store)
    assert keys.lookup(first) is not None
    assert keys.lookup(second) is not None


def test_a_missing_store_has_no_keys_rather_than_all_keys(tmp_path: Path) -> None:
    """⭐ Hướng hỏng của cấu hình thiếu. Mở hết biến một lỗi triển khai (quên mount
    volume) thành một API công khai, và mọi request đều thành công nên không gì báo."""
    assert len(ApiKeyStore.load(tmp_path / "không-có.json")) == 0


def test_a_wrong_key_returns_none_not_an_exception() -> None:
    keys = ApiKeyStore.from_mapping({digest_of("đúng"): {"tenant_id": "acme"}})
    assert keys.lookup("sai") is None


def test_the_log_hint_is_not_enough_to_reuse() -> None:
    """Log lỗi xác thực phải nói được **key nào** trong lúc xoay vòng, mà không
    đưa luôn key vào log — nơi nó sẽ sống lâu hơn chính key đó."""
    raw = "rag_" + "S" * 40
    hint = key_hint(raw)
    assert hint.startswith("rag_SSS")
    assert len(hint) < 12
    assert raw not in hint


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Chỗ ép tenant mà `W2-06` không ép được
# ---------------------------------------------------------------------------


ACME = Principal(tenant_id="acme", key_id="acme-1")


def test_a_request_without_a_filter_is_still_scoped_to_its_tenant() -> None:
    """⭐⭐ Chính là lỗ `W2-06` để lại: `retrieve(query)` không filter thấy **tất
    cả**, và `rag_core` không phân biệt được thế là đúng hay là rò."""
    assert tenant_filter(ACME).tenant_id == "acme"


def test_asking_for_another_tenant_is_refused_not_silently_rewritten() -> None:
    """⭐⭐ Ghi đè lặng lẽ an toàn về **dữ liệu** nhưng sai về **thông tin**: nó biến
    một request sai thành một kết quả rỗng hợp lệ, và người gọi kết luận "tenant
    kia không có tài liệu" — đúng chế độ hỏng im lặng mà `MetadataFilter` sinh ra
    để chặn."""
    with pytest.raises(CrossTenantError):
        tenant_filter(ACME, MetadataFilter(tenant_id="globex"))


def test_the_refusal_does_not_reveal_whether_that_tenant_exists() -> None:
    """Cùng một lỗi cho tenant có thật và tenant bịa, nếu không thì endpoint thành
    máy đếm khách hàng."""
    with pytest.raises(CrossTenantError) as real:
        tenant_filter(ACME, MetadataFilter(tenant_id="globex"))
    with pytest.raises(CrossTenantError) as fake:
        tenant_filter(ACME, MetadataFilter(tenant_id="không-tồn-tại-đâu"))
    assert str(real.value) == str(fake.value)
    assert "globex" not in str(real.value)


def test_naming_your_own_tenant_is_allowed() -> None:
    assert tenant_filter(ACME, MetadataFilter(tenant_id="acme")).tenant_id == "acme"


def test_the_other_filter_fields_survive() -> None:
    """Chúng chỉ **thu hẹp** thêm, nên giữ nguyên là đúng — và mất chúng sẽ là một
    lỗi đúng-nhưng-chậm rất khó thấy."""
    narrowed = tenant_filter(ACME, MetadataFilter(lang=Language.VI, doc_id="d1"))
    assert (narrowed.tenant_id, narrowed.lang, narrowed.doc_id) == ("acme", Language.VI, "d1")


def test_the_result_cannot_be_widened_afterwards() -> None:
    """`W2-06` đặt `frozen` trên `MetadataFilter` và gọi đó là một quyết định bảo
    mật. Đây là chỗ quyết định ấy được dùng: kiểm xong rồi mà filter còn sửa được
    thì phép kiểm chỉ là trang trí."""
    scoped = tenant_filter(ACME)
    with pytest.raises(ValidationError):
        scoped.tenant_id = "globex"


# ---------------------------------------------------------------------------
# 3. Token bucket
# ---------------------------------------------------------------------------


def test_the_first_burst_is_allowed_up_to_the_limit() -> None:
    limiter = RateLimiter()
    assert [limiter.check("acme", 3).allowed for _ in range(4)] == [True, True, True, False]


def test_remaining_counts_down() -> None:
    limiter = RateLimiter()
    assert [limiter.check("acme", 3).remaining for _ in range(3)] == [2, 1, 0]


def test_retry_after_is_at_least_one_second() -> None:
    """⭐ `int()`/`round()` cho **0** với mọi khoảng chờ dưới một giây, và một client
    lịch sự đọc `Retry-After: 0` sẽ thử lại ngay — nhận 429 tiếp, thử lại ngay.
    Header sinh ra để giảm tải lại thành một vòng lặp nóng."""
    limiter = RateLimiter()
    # 600/phút = 10 token/giây, tức chờ thật là 0,1 giây — chỗ mà làm tròn xuống sai.
    for _ in range(600):
        limiter.check("acme", 600)
    assert limiter.check("acme", 600).retry_after_s == 1


def test_two_tenants_do_not_share_a_bucket() -> None:
    limiter = RateLimiter()
    for _ in range(4):
        limiter.check("acme", 3)
    assert limiter.check("globex", 3).allowed is True


def test_the_bucket_refills_over_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _freeze(monkeypatch)
    limiter = RateLimiter()
    for _ in range(61):
        limiter.check("acme", 60)
    assert limiter.check("acme", 60).allowed is False
    clock[0] += 1.0  # 60/phút = 1 token/giây
    assert limiter.check("acme", 60).allowed is True


def test_a_long_silence_does_not_bank_more_than_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không có trần thì một tenant im một ngày rồi bắn 86.400 request trong một
    giây — hợp lệ theo sổ sách, và đủ để làm sập chính mình."""
    clock = _freeze(monkeypatch)
    limiter = RateLimiter()
    limiter.check("acme", 60)
    clock[0] += 86_400
    allowed = sum(limiter.check("acme", 60).allowed for _ in range(200))
    assert allowed == 60


def test_idle_buckets_are_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bucket sinh theo tenant và không tự mất đi. Với tenant sinh động (mỗi khách
    một tenant) đó là một chỗ rò bộ nhớ tăng đều — và bucket đã đầy không mang
    thông tin gì, nó tương đương một tenant chưa từng gọi."""
    clock = _freeze(monkeypatch)
    limiter = RateLimiter()
    for i in range(1100):
        limiter.check(f"tenant-{i}", 60)
    clock[0] += 120
    limiter.check("người-mới", 60)
    assert len(limiter.buckets) < 1100


def test_a_zero_limit_is_a_configuration_error_not_a_total_block() -> None:
    """`rate_limit_per_minute: 0` trong file key đọc như "không giới hạn" với ít
    nhất một nửa số người, và như "chặn hết" với nửa kia. Hai cách đọc trái ngược
    nhau về cùng một dòng cấu hình, nên nó phải nổ."""
    with pytest.raises(ValueError, match="≥ 1"):
        RateLimiter().check("acme", 0)


def test_the_store_round_trips_through_json(tmp_path: Path) -> None:
    """Kho key đi qua đĩa dưới dạng JSON, nên `scopes` là list ở đó và phải thành
    `frozenset` ở đây — lệch chỗ này làm `is_admin` luôn sai một cách im lặng."""
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps({digest_of("k"): {"tenant_id": "ops", "scopes": [ADMIN_SCOPE]}}),
        encoding="utf-8",
    )
    principal = ApiKeyStore.load(path).lookup("k")
    assert principal is not None and principal.is_admin


def _freeze(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Đồng hồ điều khiển được, thay **tham chiếu module** chứ không vá
    `time.monotonic` toàn cục — vá toàn cục thì mọi thứ khác đang chạy trong
    cùng tiến trình pytest cũng thấy đồng hồ đứng yên."""
    clock = [1000.0]

    class _Clock:
        @staticmethod
        def monotonic() -> float:
            return clock[0]

    monkeypatch.setattr("serving.core.ratelimit.time", _Clock)
    return clock
