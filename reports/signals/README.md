# Signal-Setups – Reports

Erzeugt mit `python signals.py` (Zeitraum 2024-11-06 bis 2026-11-06, Binance-USDT-M-Kerzen, 21 Coins).

| Datei | Inhalt |
|---|---|
| `signals_report.html` | kumulierte R-Kurven, Zufalls-Vergleich (Verteilung der 50 Läufe), alle Kennzahlen |
| `signals_results.csv` | Kennzahlen je Setup, Coin-Auswahl (21 Coins / Auswahl-Coins) und Zeitraum |
| `signals_random_runs.csv` | jeder Zufallslauf einzeln: Trades, Ø R, Summe R |
| `signals_trades.csv` | alle Trades der Setups (Einstieg, Ausstieg, Richtung, R, Funding in R, Ausstiegsgrund) |

## Methode

- **Kosten:** 0,15 % pro Trade (Taker 0,045 % + Slippage 0,03 % je Seite) plus Binance-Funding während der Haltedauer (Long zahlt bei positiver Rate). Zeitpunkte innerhalb einer Kerze werden für das Funding mit der Kerzenmitte angenähert.
- **Reihenfolge in der Kerze:** Sind Stop und Ziel in derselben Kerze erreichbar, zählt der Stop. Bei Limit-Einstieg (EMA pullback) wird die 4h-Kerze, in der die Order ausgeführt wird, mit 1h-Kerzen aufgelöst. In der 1h-Kerze des Fills zählt der Stop, sobald er erreichbar ist, das Ziel nur, wenn es sicher nach dem Fill erreicht wurde (Ziel liegt auf der vom Eröffnungskurs abgewandten Seite oder die Kerze schließt jenseits des Ziels). Sonst läuft der Trade weiter. Dieselbe Regel gilt für die Zufallsläufe.
- **Zufalls-Vergleich:** exakt dieselben Trades wie das Setup (gleicher Einstiegskurs, gleiche Kerze, gleiche Stop- und Zielabstände), nur die Richtung per Münzwurf, gespiegelt um den tatsächlichen Einstieg. 50 Läufe, eigener reproduzierbarer Zufallsstrom je Setup und Coin (`default_rng([1, setup, coin])`).
- **Empirischer p-Wert:** (1 + Läufe mit Ø R ≥ Setup) / 51. Kleinster möglicher Wert 0,02.
- **t-Wert (Wochen):** R je Kalenderwoche summiert (inkl. Wochen ohne Trade), t-Wert über die Wochensummen. Das berücksichtigt, dass viele Trades gleichzeitig auf korrelierten Coins laufen. Der naive t-Wert je Trade steht nur zum Vergleich in der Tabelle.
- **Zinseszins:** Jeder Trade riskiert 1 % des realisierten Kontostands beim Einstieg, überlappende Trades sind erlaubt.

## Grenzen

- **Survivorship:** Die 21 Coins sind nach heutiger Größe gewählt. Coins, die im Zeitraum abgestürzt oder delistet wurden, fehlen. Das macht das Ergebnis eher zu freundlich.
- Kurse und Funding von Binance, nicht von der Börse, auf der ein Signal-Bot handeln würde.
- Regeln aus verbreiteten Setup-Beschreibungen nachgebaut, nicht optimiert. Schwellen weichen je nach Anbieter ab.
