# Krypto-Bot-Backtests

Grid, Momentum-Rotation, Signal-Setups, Funding Farming/Arbitrage und Copy Trading – in Python nachgebaut und mit echten historischen Marktdaten getestet, nach Gebühren, Slippage und Funding.

Dieses Repository enthält Code und Reports zur Artikelserie **„Krypto-Trading-Bots im Backtest“** auf MyDeFi:

- Deutsch: https://mydefi.xyz/learn/de/trading-bots-im-test
- Englisch: https://mydefi.xyz/learn/en/trading-bots-backtested

[English version of this README](README.md)

> **Keine Anlageberatung.** Backtests beruhen auf Annahmen (Gebühren, Ausführung, Datenquelle) und sagen nichts Sicheres über die Zukunft aus. Nutzung auf eigene Verantwortung.

## Ergebnisse auf einen Blick

| Bot | Getestet | Ergebnis nach Kosten | Zuverlässig? |
|---|---|---|---|
| **Grid-Bot** (Liquidity Range / V3) | BTC, ETH, 2 × 6 Monate, 141 Einstellungen in 570 Läufen | Median −4 % pro 6 Monate, 0 von 120 Sweep-Einstellungen in allen vier Läufen positiv | nein |
| **Momentum-Rotation** (long/short) | 10–650 Coins, 2 × 12 Monate, Ausführung 1 Tag nach dem Ranking | Vorjahr +69 % bis +163 %, letzte 12 Monate −50 % bis +11 % (2× Hebel), Drawdowns 44–57 % | nein, phasenabhängig |
| **Signal-Setups** (ATH, EMA, MACD+RSI, Volume Profile, Breakout) | 21 Coins, 24 Monate, 4.315 Trades | alle 5 negativ (−22 R bis −141 R), keines besser als Zufall (bester p-Wert 0,27) | nein |
| **Funding-Arbitrage** (dynamisch) | 25 Coins, 3 Börsen, 2 × 12 Monate | −11,7 % bis +0,5 % p. a. | nein |
| **Funding Farming** (statisch, Spot-Hedge) | BTC/ETH und 20 große Coins | +0,3 % bis +10,7 % p. a., BTC/ETH auf Hyperliquid +3,6 % bis +9,3 %, Drawdown des Kontos ≤ 1 % | ja, mit Börsenrisiko |
| **Funding Farming** (zwei Perps) | Short Hyperliquid, Long Binance | −6,5 % / +3,4 % p. a. (Liquidationen im Crash vom 10.10.2025) | nein |
| **Copy Trading** (Hyperliquid) | 186 Konten, 487 Beobachtungen, 90/90 Tage | Median der zeitgewichteten Rendite −12 % bis −26 % in den 90 Tagen nach der Auswahl | nein |

Anbieter-Gebühren (Abo, Gebühr pro Ausführung, Gewinnbeteiligung) sind nur dort enthalten, wo es angegeben ist. Sie verschlechtern alle Ergebnisse zusätzlich.

**Stand der Rotations-Läufe:** Die Läufe, die die aktuelle Hyperliquid-Coin-Liste brauchen (`--universe hyperliquid`, `top20vol180`, `top40vol365`), enthalten noch Ergebnisse vor den letzten Korrekturen (Ausführungsverzögerung, Listing-Daten). Sie werden mit den Befehlen unten neu gerechnet, sobald die Hyperliquid-API erreichbar ist.

## Installation

```bash
git clone https://github.com/mydefi-xyz/crypto-bot-backtests.git
cd crypto-bot-backtests
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Getestet mit Python 3.14, pandas 3.1, numpy 2.5, requests 2.32 und plotly 7.1 (Python ≥ 3.11 sollte reichen). Kein API-Key nötig: Alle Daten kommen von öffentlichen Endpunkten und werden in `data/` zwischengespeichert (nicht im Repo, rund 150 MB). Der erste Lauf eines Skripts lädt die Daten herunter und dauert entsprechend länger.

## Skripte und Aufrufe

Die vollständige Befehls- und Parameterliste steht in der [englischen README](README.md#scripts). Kurzüberblick:

| Thema | Befehl | Ausgabe in `reports/…` |
|---|---|---|
| Grid-Bot | `python run.py --end 2026-11-06T10:31`, `python sweep.py`, `python sweep.py --end 2026-05-10T10:31`, `python sweep_exit.py`, `python explore_check.py` | `liquidity/` |
| Momentum-Rotation | `python rotation.py`, `python rotation.py --rebalance 72`, `python rotation_xs.py --universe hyperliquid` und die Varianten mit `--tag` | `rotation/` |
| Signal-Setups | `python signals.py` | `signals/` (Methode: [reports/signals/README.md](reports/signals/README.md)) |
| Funding | `python funding_download.py` (einmalig), `python funding.py` | `funding/` |
| Copy Trading | `python copytrade.py --n 800` (`--offline` nur Cache) | `copytrade/` |

Ausgabedateien enthalten einen Tag (`--tag`), damit sich verschiedene Läufe nicht überschreiben. HTML-Reports laden Plotly per CDN; GitHub zeigt HTML nicht direkt an (Datei herunterladen oder `https://htmlpreview.github.io/?<raw-URL>` nutzen).

## Wichtigste Annahmen und Grenzen

- **Kurse:** Binance-Kerzen als Ersatz für DEX-Kurse (Abweichung meist < 0,1 %).
- **Grid-Ausführung:** Limit-Order gilt erst als gefüllt, wenn der Kurs die Stufe durchkreuzt. Pfad innerhalb der Kerze: grün O→L→H→C, rot O→H→L→C. Stopps, Verlustlimit und frühes Schließen werden am Extremkurs der Minute geprüft und zum schlechteren Kurs aus Auslöse- und Extremkurs plus Slippage gefüllt.
- **Gebühren:** Presets je Börse (Basis-Tier, Extended Maker 0 % / Taker 0,025 %), Slippage 0,03–0,05 % auf Market- und Stop-Orders. Anbieter-Gebühr 0,02 % pro Fill nur bei Grid-Variante 1 und in den Rotations-Kosten.
- **Rotation:** Tagesschlusskurse, Ausführung eine Kerze nach dem Ranking, pauschale Kosten je Seite. Coin-Listen und Varianten wurden auf denselben zwei Jahren gewählt, auf denen getestet wurde – das Vorjahr ist kein echter Out-of-Sample-Zeitraum. Delistete Binance-Coins sind nur enthalten, wenn sie noch mit Status `SETTLING` geführt werden.
- **Signale:** Coin-Auswahl nach heutiger Größe (Survivorship), Kurse und Funding von Binance. Trades laufen teils parallel; die Signifikanz wird über Wochensummen berechnet.
- **Funding:** Basis (Preisunterschied Spot/Perp bzw. zwischen Börsen) und Margin-Transfers zwischen Börsen sind nicht modelliert, die Margin wird täglich geprüft.
- **Copy Trading:** nur Konten, die heute noch im Leaderboard stehen (Survivorship) → Ergebnisse eher zu optimistisch. Portfolio-Historie wöchentlich; Kopierkosten aus dem Lebenszeit-Umsatz geschätzt.

## Lizenz

[MIT](LICENSE) © MyDeFi
