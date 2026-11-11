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
Strassen_algorithm/
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
│   └── Varotto_Luca_report.pdf   # the report (Italian)
├── .gitignore
└── README.md
```

The LaTeX source of the report, the *tau-book* template files and the build
output live in the local working folder but stay out of the repository, see
`.gitignore`: of the report, only the compiled PDF is published here.

Every CSV in `dataset/` shares the same header:

```
ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time
```

`ID` identifies the pair of matrices, `tipo` the experimental group, `Q` the
crossover point, `a`/`b`/`c` the three dimensions, and the last four columns the
CPU time in seconds of row-by-column product, dynamic padding, static padding
and dynamic peeling. `grafici.Rmd` reads these files directly.

---

## Results

Measured on an Intel Core i5-6300U at 2.40 GHz with 16 GB of RAM, Python
3.12.2 and NumPy 1.26.4. Times are means over the sample, in seconds; the
fastest algorithm of each row is in bold, and the report discusses every
experiment in full.

| Sample | Row by column | Dynamic padding | Static padding | Dynamic peeling |
| --- | --- | --- | --- | --- |
| Dense integers, `Q=128`, n=40 | 0.703 | 0.354 | **0.348** | 0.350 |
| Sparse, `Q=128`, n=40 | **0.063** | 0.198 | 0.179 | 0.192 |
| Floating point, `Q=128`, n=40 | **0.064** | 0.160 | 0.117 | 0.131 |
| A long and B wide, `Q=64`, n=60 | 0.083 | 0.146 | **0.077** | 0.152 |
| Deliberately low `Q=8`, n=10 | **0.11** | 10.25 | 1.53 | 10.23 |

* `Q` between 64 and 128 is where the three variants perform best. Below that
  the recursion goes too deep and they lose badly to `numpy.matmul`; above it
  the matrices at the leaves grow and the advantage fades.
* On dense integer matrices they are about twice as fast as `numpy.matmul`, and
  the gap widens with the size of the operands, already well below the
  `1000 x 1000` threshold the algorithm was long believed to require.
* None of them is advisable for sparse matrices or floating point entries,
  where the row-by-column product wins by a factor of two or three.
* Dynamic padding and dynamic peeling degrade on strongly unbalanced shapes,
  because they stop the recursion on the geometric mean of the dimensions,
  while static padding looks at the smallest one and stays the most consistent.
* Rounding the results to 8 decimal places, the three variants share 98% of
  their entries with the row-by-column product; at 10 decimal places the
  agreement collapses to 16%. The instability is real but infinitesimal.

`dataset/q_test.txt` was regenerated later on a newer machine, where the
optimum sits between `Q = 128` and `Q = 256` and the variants are 2.3 times
faster than `numpy.matmul`. The shape of the curve is unchanged.

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
distributed under CC BY 4.0.
