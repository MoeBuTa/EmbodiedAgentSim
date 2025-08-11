from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_hm3d_eqa_dataset():
    try:
        from easim.datasets.eqa.hm3d_eqa import (  # noqa: F401 isort:skip
            HM3DDatasetV1,
        )
    except ImportError as e:
        hm3deqa_import_error = e

        @registry.register_dataset(name="HM3DEQA-v1")
        class HM3DDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise hm3deqa_import_error
