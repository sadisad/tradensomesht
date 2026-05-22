"""Quick verification: load each per-pair config + confirm MT5 symbol availability."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config import load_config
from bot.broker_mt5 import MT5Client


def main() -> None:
    configs = [
        "config.local.gbpusd.yaml",
        "config.local.usdjpy.yaml",
        "config.local.usdcad.yaml",
    ]
    for cfg_name in configs:
        cfg = load_config(cfg_name)
        print(
            f"-- {cfg_name} -> symbol={cfg['trading']['symbol']} "
            f"magic={cfg['broker']['magic']} risk={cfg['risk']['risk_per_trade_pct']}% "
            f"model={cfg['ml']['model_path']} log={cfg['logging']['file']}"
        )

    print("\nMT5 sanity check:")
    client = MT5Client(load_config("config.local.gbpusd.yaml")["broker"])
    client.connect()
    try:
        for s in ["GBPUSD", "USDJPY", "USDCAD"]:
            df = client.get_rates(s, "M5", 2)
            last = df.index[-1]
            bid = client.current_price(s, "sell")
            ask = client.current_price(s, "buy")
            spread_pts = client.current_spread_points(s)
            print(
                f"  {s:8s} last_bar={last} bid={bid:.5f} ask={ask:.5f} "
                f"spread={spread_pts:.1f}pts"
            )
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
