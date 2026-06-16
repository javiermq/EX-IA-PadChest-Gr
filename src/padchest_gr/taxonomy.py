from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PadChestGRCategory:
    name: str
    column: str


PADCHEST_GR_CATEGORIES: tuple[PadChestGRCategory, ...] = (
    PadChestGRCategory("Hyperinflated lung", "hyperinflated_lung"),
    PadChestGRCategory("Hypoexpansion", "hypoexpansion"),
    PadChestGRCategory("Electrical device", "electrical_device"),
    PadChestGRCategory("Cardiomegaly", "cardiomegaly"),
    PadChestGRCategory("Interstitial pattern", "interstitial_pattern"),
    PadChestGRCategory("Scoliosis", "scoliosis"),
    PadChestGRCategory("NSG tube", "nsg_tube"),
    PadChestGRCategory("Alveolar pattern", "alveolar_pattern"),
    PadChestGRCategory("Osteopenia", "osteopenia"),
    PadChestGRCategory("Aortic elongation", "aortic_elongation"),
    PadChestGRCategory("Vertebral degenerative changes", "vertebral_degenerative_changes"),
    PadChestGRCategory("Hiatal hernia", "hiatal_hernia"),
    PadChestGRCategory("Central venous catheter", "central_venous_catheter"),
    PadChestGRCategory("Hemidiaphragm elevation", "hemidiaphragm_elevation"),
    PadChestGRCategory("Vascular hilar enlargement", "vascular_hilar_enlargement"),
    PadChestGRCategory("Pleural effusion", "pleural_effusion"),
    PadChestGRCategory("Bronchiectasis", "bronchiectasis"),
    PadChestGRCategory("Atelectasis", "atelectasis"),
    PadChestGRCategory("Goiter", "goiter"),
    PadChestGRCategory("Pleural thickening", "pleural_thickening"),
    PadChestGRCategory("Fracture", "fracture"),
    PadChestGRCategory("Endotracheal tube", "endotracheal_tube"),
    PadChestGRCategory("Nodule", "nodule"),
    PadChestGRCategory("Aortic atheromatosis", "aortic_atheromatosis"),
)

PADCHEST_GR_CATEGORY_SET = {category.name for category in PADCHEST_GR_CATEGORIES}
PADCHEST_GR_CATEGORY_COLUMNS = [category.column for category in PADCHEST_GR_CATEGORIES]
PADCHEST_GR_COLUMN_TO_NAME = {category.column: category.name for category in PADCHEST_GR_CATEGORIES}
PADCHEST_GR_NAME_TO_COLUMN = {category.name: category.column for category in PADCHEST_GR_CATEGORIES}

OTHER_CATEGORY = PadChestGRCategory("Other", "other")


def category_token(category_id: str) -> str:
    return f"<finding_{category_id}>"
