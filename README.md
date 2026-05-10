# NeuroBridge: Systematic Intelligence for Alternative Data Trading

NeuroBridge is a private, high-performance orchestration layer designed to bridge the gap between real-time market data and local language intelligence (LLMs). While traditional quant funds excel at processing structured numerical data, NeuroBridge focuses on the **Intelligence Axis**—converting unstructured language data (filings, news, transcripts) into actionable alpha signals at medium-frequency.

## 🚀 The Core Edge: Intelligence Over Latency
Instead of competing in the microsecond latency wars (HFT), NeuroBridge leverages the reasoning capabilities of local 3B and 7B models to process alternative data sources faster and more accurately than human analysts.

- **Unstructured Signal Extraction**: Real-time parsing of SEC EDGAR filings and earnings transcripts using a local 7B Analyst model.
- **Microstructure Denoising**: Kalman Filter and Wavelet-based noise reduction on raw exchange feeds.
- **Private & Air-Gapped**: All intelligence processing happens locally (via Ollama), ensuring proprietary strategies and research never leak to cloud providers.
- **Multi-Modal Signal Fusion**: Combining price action, macro indicators (FRED), and sentiment signals into a unified ensemble.

## 🏗️ Architecture
- **Neuro/Finance/Data**: Connectors for IEX, SEC EDGAR, FRED, and Binance Websockets.
- **Neuro/Finance/Signals**: Sentiment extraction, regime detection, and filing anomaly triggers.
- **Neuro/Finance/Backtest**: Vectorized backtesting engine with walk-forward validation.
- **Neuro/Router**: A dual-layer routing system that triages signals based on complexity before escalation.

## 🛠️ Getting Started
```bash
# Initialize the finance environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the NeuroBridge Finance Orchestrator
python3 neuro/finance/main.py
```

## 📊 Roadmap
- [x] Kalman Filter Noise Reduction
- [x] SEC EDGAR real-time ingestor
- [x] Binance Trade Websocket integration
- [x] 7B Analyst Sentiment Engine
- [ ] Portfolio Optimization (Markowitz/Kelly)
- [ ] Satellite Imagery Signal Pipeline (AIS/MODIS)
- [ ] Geopolitical Event Tracking (GDELT/ACLED)

## ⚖️ License
MIT License.
