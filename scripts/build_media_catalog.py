"""Rebuild named medium vectors from the supplied recipe table; retain the trained vocabulary."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "Casamino acids": "Casamino acid",
    "CuSO4": "Copper sulfate",
    "Meat extract": "Meat_extract",
    "SMMa": "SMM",
    "Soya peptone": "Sooya peptone",
    "Tryptic soy broth": "Trypsin_soy_broth",
}


def read_recipes(path):
    recipes = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"Expected five tab-separated fields: {row}")
            name, kind, count, strains, text = row
            name = name.strip()
            if name in recipes:
                raise ValueError(f"Duplicate medium: {name}")
            composition = {}
            for part in filter(None, (s.strip() for s in text.split(";"))):
                part = re.sub(r"(\d+(?:\.\d+)?)\s*×\s*10\s*(-\d+)", r"\1e\2", part)
                match = re.fullmatch(r"(.+?)\s+([\d.]+(?:e[-+]?\d+)?)", part)
                if not match:
                    raise ValueError(f"Cannot parse {name}: {part}")
                ingredient, amount = match.group(1).strip(), float(match.group(2))
                if ingredient in composition:
                    raise ValueError(f"Duplicate ingredient in {name}: {ingredient}")
                composition[ingredient] = amount
            if len(composition) != int(count):
                raise ValueError(f"Component count mismatch: {name}")
            recipes[name] = {"type": kind.strip(), "n_components": int(count),
                             "composition": composition or None}
    return recipes


def main():
    path = ROOT / "configs/media_catalog.json"
    catalog = json.loads(path.read_text())
    components = catalog["component_list"]
    if len(components) != 159 or len(set(components)) != 159:
        raise ValueError("Expected the frozen 159-component vocabulary")
    if not set(ALIASES.values()) <= set(components):
        raise ValueError("Alias target missing from model vocabulary")
    source = ROOT / "configs/media_recipes.tsv"
    recipes = read_recipes(source)
    media, undefined = {}, {}
    for name, entry in recipes.items():
        if entry["composition"] is None:
            undefined[name] = "Composition unspecified in the supplied table; provide an explicit composition or vector."
            continue
        mapped = {}
        for ingredient, amount in entry["composition"].items():
            target = ALIASES.get(ingredient, ingredient)
            if target not in components or target in mapped:
                raise ValueError(f"Unmapped or duplicate component in {name}: {ingredient}")
            mapped[target] = amount
        media[name] = [mapped.get(c, 0.0) for c in components]
    catalog.update(media=media, ambiguous_names={}, undefined_media=undefined,
                   component_aliases=ALIASES,
                   source="User-supplied recipe table (2026-11-27): configs/media_recipes.tsv; named recipes replace historical defaults; trained component order retained.",
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   units="Numeric amounts are preserved from the supplied recipe table. No unit conversion is performed; use the source recipe units.",
                   notes=["SMMa is interpreted as SMM with a footnote marker; the base mix is not expanded.",
                          "Only the six explicit component_aliases are resolved; other distinct trained vocabulary labels remain distinct.",
                          "BHI and HIB have unspecified recipes and require explicit composition or vector inputs.",
                          "Named recipe updates do not retrain the model or modify its reference medium vectors."])
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "configs/media_recipes.json").write_text(json.dumps(recipes, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(media)} named recipes and {len(undefined)} unspecified media; vector dimension={len(components)}")


if __name__ == "__main__":
    main()
