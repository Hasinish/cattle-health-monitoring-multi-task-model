# -*- coding: utf-8 -*-
"""DEPRECATED: Profile-split into modal_export_bcs.py and modal_export_behavior.py.

In Modal, App objects and volume bindings (modal.Volume.from_name) are evaluated
at module import time under the active profile specified by --profile.
Because BCS lives on tigerwood697 ('sciencedb-perception-cache') and Behavior lives
on tigerwood693 ('behavior-perception-cache'), combining them into one module causes
a Volume NotFoundError when Modal tries to resolve both volumes in a single profile.

Use:
- scripts/modal_export_bcs.py on tigerwood697
- scripts/modal_export_behavior.py on tigerwood693
"""

from scripts.modal_export_bcs import (
    prepare_bcs_export_remote,
    cleanup_bcs_export_remote,
)
from scripts.modal_export_behavior import (
    prepare_behavior_export_remote,
    cleanup_behavior_export_remote,
)

__all__ = [
    "prepare_bcs_export_remote",
    "cleanup_bcs_export_remote",
    "prepare_behavior_export_remote",
    "cleanup_behavior_export_remote",
]
