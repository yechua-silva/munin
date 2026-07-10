from __future__ import annotations

from munin.rocm_patch import apply_nms_patch


class TestRocmPatch:
    """Tests for NMS monkey-patch."""

    def test_patch_is_idempotent(self) -> None:
        """Calling apply_nms_patch() twice should not crash."""
        apply_nms_patch()
        apply_nms_patch()  # Should not raise

    def test_patch_function_exists(self) -> None:
        """apply_nms_patch should be callable."""
        assert callable(apply_nms_patch)

    def test_patch_module_importable(self) -> None:
        """Module should be importable without errors."""
        import munin.rocm_patch as rp
        assert hasattr(rp, 'apply_nms_patch')
