# Artificial-Intelligence — Codes & Slides

Welcome — this repository collects lecture slides and Python notebooks used for an introductory Machine Learning / Python for ML course. It includes slide decks (organized by module) and hands‑on Jupyter notebooks with sample datasets and exercises.

## Repository overview

Top-level folders
- Machine-Learning/  
  - Slide decks, lecture materials and supporting files organized by module (e.g., `Module 01`, `Module 02`, ...). Use these for theory, slides and lecture notes.
- Python-for-ML/  
  - Jupyter notebooks and datasets used for in‑class labs and practice exercises.

What you'll find in Python-for-ML (selection)
- Notebooks:
  - `Module_1.ipynb`
  - `Module_2.ipynb`
  - `Module_3_List.ipynb`
  - `Module_3_string.ipynb`
  - `Module_5.ipynb`
  - `Module_6.ipynb`
  - `Module_7.ipynb`
  - `Module_8.ipynb`
  - `Module_10.ipynb`
  - `Module_11.ipynb`
  - `Module_12.ipynb`
  - `Module_14.ipynb`
  - `Module_15.ipynb`
  - `Module 16 (2).ipynb` (alternate/updated Module 16)
- CSV datasets:
  - `Practice Day (1).csv`
  - `enrollment_data.csv`
  - `final-employee-ds.csv`
  - `sns_data.csv`
  - `student_IQdata.csv`
  - `student_completed_data.csv`
  - `student_data (2).csv`
  - `student_dataset_complete.csv`
  - `student_scores.csv`

Machine-Learning folder structure
- `Machine-Learning/Module 01/`
- `Machine-Learning/Module 02/`
- ...
- `Machine-Learning/Module 11/`  
Each module directory contains slides and materials for that module (lecture slides, PDFs, or other resources).

## Goals & scope

This repo is intended to:
- Provide lecture slides for an introductory Machine Learning course.
- Provide hands‑on Python notebooks that illustrate Python basics, data handling, visualization and early ML workflows.
- Give example datasets for practice and assignments.

Recommended audience: students and practitioners starting with Python for data analysis and introductory machine learning.

## Quick start

1. Clone the repository:
   ```bash
   git clone https://github.com/Marwanthe0/Artificial-Intelligence.git
   cd Artificial-Intelligence
   ```

2. Create a Python environment (recommended):
   - Using venv:
     ```bash
     python3 -m venv venv
     source venv/bin/activate   # macOS / Linux
     venv\Scripts\activate      # Windows (PowerShell)
     ```
   - Or using conda:
     ```bash
     conda create -n ai-course python=3.9
     conda activate ai-course
     ```

3. Install common dependencies (suggested):
   ```bash
   pip install --upgrade pip
   pip install jupyterlab notebook numpy pandas matplotlib seaborn scikit-learn
   ```
   - If you plan to run deep-learning examples, also install:
     ```bash
     pip install tensorflow keras
     ```
   - If a requirements file is added in the future, install with:
     ```bash
     pip install -r requirements.txt
     ```

4. Launch Jupyter and open notebooks:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
   - Open the notebooks under `Python-for-ML/` to run and interact.

## How to use the notebooks and datasets

- Open any notebook in `Python-for-ML/` with Jupyter. Notebooks are self‑contained; run cells sequentially.
- To load CSV data in a notebook:
  ```python
  import pandas as pd

  df = pd.read_csv("Python-for-ML/student_dataset_complete.csv")
  df.head()
  ```
- Check each notebook's first cells for explicit dependencies; some notebooks may use libraries such as `seaborn`, `matplotlib`, `scikit-learn` or `tensorflow`.

## Example workflows

- Basic data exploration (notebook pattern):
  1. Load dataset with pandas.
  2. Inspect (`.head()`, `.info()`, `.describe()`).
  3. Clean / transform data (missing values, encoding).
  4. Visualize features (matplotlib / seaborn).
  5. Build simple models using scikit-learn (train/test split, fit, evaluate).
  6. Record results and iterate.

- Typical commands used in notebooks:
  - Data split:
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
  - Train a model:
    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    ```

## Tips & best practices

- Use a dedicated virtual environment to avoid dependency conflicts.
- If a notebook runs slowly or uses large datasets, consider reading a subset of data or increasing machine resources.
- Save your executed notebook copies or export to HTML/PDF when sharing results.
- Use version control (commits, branches) when changing notebooks: store important checkpoints and share improved materials.

## Contributing

Contributions are welcome. Suggested ways to contribute:
- Report issues or suggestions via GitHub Issues.
- Submit improved notebooks or clearer slides as pull requests.
- Add a `requirements.txt` listing exact dependency versions.
- Add a `LICENSE` file if you'd like to specify reuse terms.

Guidelines:
- Keep notebooks deterministic where possible (set `random_state`).
- Add explanatory text and comments in notebooks so others can follow the steps.

## License

No license file was detected in the repository contents listed. If you want to allow reuse, consider adding an explicit license (for example, the MIT license). To add a license, create a `LICENSE` file at repository root and choose the license appropriate to your needs.

## Contact / attribution

Repository owner: Marwanthe0  
If you want this README adjusted (more details per-module, recorded dependency versions, examples extracted from particular notebooks, or a generated `requirements.txt`), add a short note in an issue or create a branch with the proposed changes.

---

Happy learning! Explore the slides in `Machine-Learning/` for theory and run the notebooks in `Python-for-ML/` for hands‑on practice.
