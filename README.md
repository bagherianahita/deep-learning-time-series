# Deep Learning Time Series (CNN-LSTM)

**Hybrid CNN-LSTM model** for multivariate time-series forecasting — applied to drilling / well-log sensor data (DTCO, ROP, GR, etc.).

![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

---

## Architecture

```
┌─────────────┐   window    ┌─────────────┐   conv    ┌─────────────┐
│ Sensor logs │ ──────────► │  CNN layers │ ────────► │  LSTM       │
│ (multivariate)│           │  feature    │           │  temporal   │
└─────────────┘             └─────────────┘           └──────┬──────┘
                                                             ▼
                                                    ┌─────────────┐
                                                    │  Forecast   │
                                                    └─────────────┘
```

---

## Quick start (employers — ~2 min, synthetic data)

```bash
pip install -r requirements.txt
python demo.py
```

---

## License

MIT — see [LICENSE](LICENSE).
