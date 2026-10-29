Assignment 8 - Self-organizing maps and neighborhood functions

FILES
  assignment8_report.pdf       English comparison and conclusions
  assignment8.py              Complete MiniSom experiment and plotting code
  requirements.txt            Tested package versions (Python 3.12)
  figures/                    Four Fig.3-style plots and comparison figures
  results/                    Actual metrics, five-seed runs, Iris data
  models/                     Trained prototype weights and BMU assignments

INSTALL AND RUN (from this folder)
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python assignment8.py

Use --output-dir another_folder to preserve the supplied result files.

DESIGN
  Iris data, MinMax scaling, 8x8 rectangular map, PCA initialization.
  Learning rate 0.5, initial sigma 1.5, 5000 random updates.
  Linear learning-rate decay to zero; linear sigma decay to one.
  Euclidean BMU distance. Primary seed 123; robustness seeds 123-127.
  Only neighborhood_function changes: gaussian, bubble, mexican_hat, triangle.
  Initial prototypes and training sample order match across functions.
  Correct 0-based Iris labels and one common deterministic plotting jitter.
  Class labels never enter training. Purity is a descriptive diagnostic,
  not held-out classification accuracy.

NUMERICAL NOTE
  Under these fixed settings Mexican-hat unused weights diverge dramatically.
  This observed failure is retained rather than replaced with retuned settings.
  Magnitudes and occupancy are recorded; a low topographic error is misleading
  when the map degenerates to two occupied cells.
  A common positive rescaling prevents overflow in U-matrix visualization;
  it cancels in U-matrix normalization and does not change trained weights.
  BMUs are cross-checked with overflow-safe Euclidean distances.
  MiniSom's noninteger sigma warning is documented: sigma=1.5 is deliberately
  preserved for bubble and triangle to change only neighborhood choice.

SOURCES
  DataCamp, Self-Organizing Maps: An Intuitive Guide with Python Examples:
  https://www.datacamp.com/tutorial/self-organizing-maps
  MiniSom implementation: https://github.com/JustGlowing/minisom
  Iris dataset: https://archive.ics.uci.edu/dataset/53/iris
