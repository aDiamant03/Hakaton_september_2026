from pathlib import Path

import create_region_maps as maps


ROOT = Path(__file__).resolve().parents[1]
maps.REGIONS_CSV = ROOT / "working_data" / "preanalysis" / "region_hypothesis.csv"
maps.OUT = ROOT / "figures" / "preanalysis"
maps.OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    outputs = [
        maps.render_map(
            "fine_did_per_client_month",
            "Сезонно скорректированное изменение штрафов",
            "Разность разностей: изменение разрыва 2026–2025 после мая; синий — снижение штрафов",
            "map_region_fine_did.png",
            "diverging",
            lambda value: f"{value:+.3f}",
        ),
        maps.render_map(
            "price_change_pct",
            "Изменение робастной цены топлива после мая",
            "Средневзвешенная цена положительных покупок 50–130 ₽/л: июнь–август к апрелю–маю",
            "map_region_price_change.png",
            "price",
            lambda value: f"{value:+.1%}",
        ),
    ]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
