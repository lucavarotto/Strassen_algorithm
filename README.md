# Strassen's Algorithm for Matrices of Arbitrary Dimensions

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white)
![R](https://img.shields.io/badge/R-ggplot2-276DC3?logo=r&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-report-008080?logo=latex&logoColor=white)

In 1969 V. Strassen showed that two `[2 x 2]` matrices can be multiplied with
7 scalar products instead of 8. Applying the idea recursively to blocks brings
the cost of matrix multiplication down from `O(n^3)` to `O(n^log2(7))`, roughly
`O(n^2.807)`.

The catch is that splitting a matrix into four equally sized blocks requires an
even number of rows and columns at *every* level of the recursion. This
repository implements the original algorithm together with the **three
strategies proposed in the literature to handle matrices of any shape**, a
benchmark suite that compares them against NumPy's row-by-column product, the R
code that produces the plots, and the full report discussing the results.

**Academic context.** The project was written for the *Sistemi di Elaborazione 2*
(Computer Systems 2) exam of the BSc in *Statistics for Technology and Science*
at the University of Padua, April 2024. The report is in Italian, and so are the
comments in the source files.

---

## The four implementations

All of them live in [`strassen_algorithms.py`](strassen_algorithms.py). `A` is
`[a x b]`, `B` is `[b x c]`, and `Q` is the **crossover point**: the size below
which the recursion stops and the product is delegated to `numpy.matmul`,
because the extra additions and copies would no longer pay off.

| Function | Accepted shapes | Strategy for odd dimensions | Crossover criterion |
| --- | --- | --- | --- |
| `Strassen_squareM(A, B, Q=2)` | square, dimensions even at every split | none, raises `ValueError` | current dimension vs `Q` |
| `Strassen_rectM(A, B, Q)` | rectangular, dimensions even at every split | none, raises `ValueError` | `min(a, b, c) <= Q` |
| `Strassen_dynamic_padding(A, B, Q)` | any | at each recursive call, pads one row/column of zeros where needed | geometric mean of `a, b, c` `<= Q` |
| `Strassen_static_padding(A, B, Q)` | any | pads once, before the recursion, with enough zeros to keep every dimension even down to the crossover point | `min(a, b, c) <= Q`, inherited from `Strassen_rectM` |
| `Strassen_dynamic_peeling(A, B, Q)` | any | at each recursive call, peels off the trailing row/column and handles it separately through block matrix algebra | geometric mean of `a, b, c` `<= Q` |

**Dynamic padding** adds the bare minimum of zeros but pays the check at every
level. A row of zeros appended to `A` yields a row of zeros in `C`, which is
removed on the way back; a column added to `A` must be matched by a row added to
`B`, and their contribution to the product is null.

**Static padding** rounds each dimension up to the nearest multiple of `2^z`,
where `z` is the recursion depth needed to reach `Q`: a dimension survives `z`
halvings while staying even if and only if it is a multiple of `2^z`. All the
bookkeeping happens once, at the cost of possibly adding more zeros than
strictly necessary.

**Dynamic peeling** goes the other way and removes rows and columns instead of
adding them. With all three dimensions odd, the partition

```
A = | A11  a12 |        B = | B11  b12 |
    | a21  a22 |            | b21  b22 |
```

gives

```
C = | A11*B11 + a12*b21    A11*b12 + a12*b22 |
    | a21*B11 + a22*b21    a21*b12 + a22*b22 |
```

where only `A11*B11` is large enough to deserve a recursive call: every other
product involves a vector or a scalar and goes straight to `numpy.matmul`. The
function handles the eight parity combinations of `a`, `b` and `c` explicitly.

---

## Repository structure

```
strassen_algorithm/
├── strassen_algorithms.py        # the four implementations and their helpers
├── strassen_testing.py           # benchmark suite; writes the CSVs in dataset/
├── grafici.Rmd                   # R Markdown: reads dataset/, writes figures/
├── dataset/                      # benchmark results, CSV with a shared header
│   ├── q_test.txt                # effect of the crossover point Q
│   ├── minimum_q.txt             # same experiment with a very low Q (Q=8)
│   ├── sparse_vs_dense.txt       # sparse vs dense matrices
│   ├── lunghe_vs_larghe.txt      # strongly unbalanced shapes
│   ├── dimensioni.txt            # growing dimensions
│   ├── on_floating.txt           # floating point vs integer entries
│   └── error_floating.txt        # sample used for the numerical error analysis
├── figures/                      # plots produced by grafici.Rmd
├── Report/
│   ├── Varotto_Luca_report.pdf   # the report (Italian)
│   ├── Varotto_Luca_report.tex   # its LaTeX source
│   ├── tau-book.cls              # tau-book template, CC BY 4.0
│   ├── tau.sty                   #   "
│   └── tau-book-cls-README.md    # template documentation and attribution
├── .gitignore
└── README.md
```

Every CSV in `dataset/` shares the same header:

```
ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time
```

`ID` identifies the pair of matrices, `tipo` the experimental group, `Q` the
crossover point, `a`/`b`/`c` the three dimensions, and the last four columns the
CPU time in seconds of row-by-column product, dynamic padding, static padding
and dynamic peeling. `grafici.Rmd` reads these files directly.

---

## Requirements and setup

The algorithms need Python and NumPy only:

```bash
pip install numpy
```

Reproducing the plots additionally needs R with `ggplot2` and `rmarkdown`, and
rebuilding the report needs a LaTeX distribution. The code was developed with
Python 3.12.2 and NumPy 1.26.4.

---

## Usage

### Multiplying two matrices

```python
import numpy as np
import strassen_algorithms as sa

A = np.random.randint(-100, 100, size=(998, 825), dtype='l')
B = np.random.randint(-100, 100, size=(825, 659), dtype='l')

C = sa.Strassen_static_padding(A, B, 128)      # Q = 128

np.array_equal(C, np.matmul(A, B))             # True on integer matrices
```

The three variants accept any pair of conformable matrices, odd dimensions
included, and `Q = 128` is a reasonable default (see the benchmarks below).
Lowering `Q` deepens the recursion; raising it enlarges the matrices handed to
`numpy.matmul` at the leaves.

### Running the benchmarks

`strassen_testing.py` contains one function per experiment. Pick one in the
`__main__` block, leaving the others commented out, since each run overwrites
its own file and takes minutes; the `dataset/` directory must exist.

```python
if __name__ == "__main__":

    different_q(600, 1000, n_test=30, file_name="dataset/q_test.txt", qmin=4, qmax=10)
    #sparse_dense_matrix(600, 1000, 128, 40)
    #lunghe_vs_larghe(800, 1200, 64, 60)
    #increasing_dimension(200, 400, 128, 75)
    #floating_time(600, 1000, 128, 40)
    #floating_error(600, 1000, 128, 40, 8)
```

```bash
python strassen_testing.py
```

Every experiment fixes `random.seed(1234)`, so the shapes of the generated
matrices are reproducible. Timings use `time.process_time()`, which measures CPU
time and is therefore insensitive to other load on the machine.

### Rebuilding the plots

```bash
Rscript -e "rmarkdown::render('grafici.Rmd')"
```

---

## Results

The figures below are measured on an Intel Core i5-6300U at 2.40 GHz with 16 GB
of RAM, Python 3.12.2 and NumPy 1.26.4. Times are means over the sample, in
seconds, and the fastest algorithm of each group is in bold.

### Choosing the crossover point

Thirty pairs of matrices with sides between 600 and 1000, each multiplied with
every `Q` in `{16, 32, 64, 128, 256, 512}`. The best performance is reached
around `Q = 64` and `Q = 128`. Below that the recursion goes too deep and all
three variants lose badly to `numpy.matmul`; above it the matrices at the leaves
grow, and the three variants converge back to the row-by-column product.

`dataset/q_test.txt` was regenerated later on a newer machine, where the same
experiment places the optimum slightly higher, between `Q = 128` and `Q = 256`,
with the three variants about 2.3 times faster than `numpy.matmul` (0.20 s
against 0.47 s). The shape of the curve is unchanged.

With a deliberately low `Q = 8` (`dataset/minimum_q.txt`, 10 pairs) the gap is
dramatic:

| Algorithm | Mean time (s) |
| --- | --- |
| Row by column | **0.11** |
| Static padding | 1.53 |
| Dynamic peeling | 10.23 |
| Dynamic padding | 10.25 |

### Dense vs sparse matrices

Two samples of 40 pairs, sides between 600 and 1000, `Q = 128`. The sparse
matrices have 50% of their cells set to zero.

| Algorithm | Dense (s) | Sparse (s) |
| --- | --- | --- |
| Row by column | 0.703 | **0.063** |
| Dynamic padding | 0.354 | 0.198 |
| Static padding | **0.348** | 0.179 |
| Dynamic peeling | 0.350 | 0.192 |

On dense matrices the three variants are nearly twice as fast as the
row-by-column product. On sparse matrices the picture reverses: `numpy.matmul`
is about three times faster than any of them.

### Unbalanced shapes

Three samples of 60 pairs, `Q = 64`. A matrix is "long" when it has about ten
times more rows than columns, and "wide" in the opposite case.

| Algorithm | A wide (s) | A long, B wide (s) | B long (s) |
| --- | --- | --- | --- |
| Row by column | 0.165 | 0.083 | 0.095 |
| Dynamic padding | 0.095 | 0.146 | 0.109 |
| Static padding | 0.103 | **0.077** | 0.102 |
| Dynamic peeling | **0.073** | 0.152 | **0.076** |

Static padding is the most stable across the three groups, while dynamic padding
and dynamic peeling are roughly twice as slow in the "A long, B wide" case. The
crossover criteria explain much of this: static padding only looks at the
smallest dimension, so with a minimum between 80 and 120 and `Q = 64` it always
stops at the second recursive call, whereas the geometric mean used by the other
two allows a deeper recursion.

### Integer vs floating point entries

Forty pairs, sides between 600 and 1000, `Q = 128`.

| Algorithm | Integer (s) | Floating point (s) |
| --- | --- | --- |
| Row by column | 0.703 | **0.064** |
| Dynamic padding | 0.354 | 0.160 |
| Static padding | 0.348 | 0.117 |
| Dynamic peeling | 0.350 | 0.131 |

Floating point matrices are multiplied faster by every algorithm, but the
advantage over `numpy.matmul` disappears entirely.

### Numerical stability

Rounding the result matrices to 8 decimal places, the three variants share on
average only **98%** of their entries with the row-by-column product; at 10
decimal places the agreement collapses to **16%**. The instability is real, as
the extra additions and subtractions accumulate rounding error, but the
deviations are infinitesimal and may well be irrelevant in practice.

### Growing dimensions

Over 75 pairs of increasing size with `Q = 128`, the advantage of the three
variants grows with the geometric mean of the dimensions, and it is already
visible well below the `1000 x 1000` threshold the algorithm was long believed
to require.

### Summary

* `Q` between 64 and 128 is where the three variants perform best.
* On dense integer matrices they are about twice as fast as `numpy.matmul`, and
  the gap widens with size.
* None of them is advisable for sparse matrices or floating point entries.
* Dynamic padding and dynamic peeling degrade on strongly unbalanced shapes;
  static padding is the most consistent of the three overall.

---

## References

1. *Gaussian Elimination is not Optimal*, Volker Strassen, August 1969.
2. *Using Strassen's Algorithm to Accelerate the Solution of Linear Systems*,
   David H. Bailey, King Lee, Horst D. Simon, March 1990.
3. *Implementation of Strassen's Algorithm for Matrix Multiplication*,
   Steven Huss-Lederman, Elaine M. Jacobson, Jeremy R. Johnson, Anna Tsao,
   Thomas Turnbull, August 1996.
4. *On the arithmetic complexity of Strassen-like matrix multiplications*,
   Murat Cenk, M. Anwar Hasan, July 2016.

---

## Author

**Luca Varotto** — BSc in Statistics for Technology and Science, University of
Padua. Report written for the *Sistemi di Elaborazione 2* exam, 3 April 2024.

The report is typeset with the *tau-book* template by Guillermo Jimenez,
distributed under CC BY 4.0; see `Report/tau-book-cls-README.md`.
