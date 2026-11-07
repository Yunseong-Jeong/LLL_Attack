# LLL_Attack

Experimental lattice-reduction (LLL) attack on a hypersphere-ECC based face template protection scheme, the construction later published as **IronMask** (CVPR 2021).

> Research prototype written in January 2020 during the security analysis of the scheme. Not production code.

## Background

The scheme turns a (normalized) biometric feature vector `w ∈ R^n` into a reproducible key:

- **Gen**
  1. Sample a random orthogonal matrix `Q` (product of random Householder reflections).
  2. Sample a codeword `c`: a sparse unit vector with `k` non-zero entries of value `±1/√k`
     (here `n = 64`, `k = 4`, entries `±1/2`).
  3. Compute a rotation `R` that maps `Q·w` to `c`.
  4. Publish helper data `P = R·Q`; the key is `r = SHA512(c)`.
- **Rep**
  1. For a fresh sample `w'`, compute `P·w'`.
  2. Decode with the hypersphere ECC: keep the top-`k` coordinates by magnitude and round them to `±1/√k`.
  3. Output `r' = SHA512(c')`.

## Attack idea

Lattice reduction (LLL, δ = 0.75) is applied to a matrix derived from the public helper data, to test whether the reduced basis leaks the codeword structure. Concretely, for two enrollments `(w1, P1)` and `(w2, P2)`, the experiment checks whether

```
ECC( 4 · P_LLL · c1 )  ==  c2
```

i.e. whether the reduced matrix maps one enrollment's codeword to the other's, and compares the supports (non-zero indices) of the two codewords.

## Files

| File | Description |
|---|---|
| `1-1. Gen(dimension64).py` | Implements `Gen` / `Rep` (n = 64, k = 4). Samples a random `w`, saves `gen_w`, `gen_P`, `gen_r`. |
| `1-2. Proof.py` | Correctness check: verifies `SHA512(ECC(P·w)) == r`. |
| `1-3. LLL_Reduction.py` | Exact-arithmetic (Fraction) Gram–Schmidt and LLL reduction. Reduces `P.tensor` → `P_Reduction.tensor`. |
| `1-4. LLL_Attack.py` | Attack check using two enrollments (`gen_w1/P1`, `gen_w2/P2`) and the reduced matrix. |
| `*.tensor` | Saved PyTorch tensors used in the experiment. |

## Usage

```bash
pip install torch numpy

python "1-1. Gen(dimension64).py"   # run twice; rename outputs to gen_*1 / gen_*2
python "1-2. Proof.py"
python "1-3. LLL_Reduction.py"      # needs P.tensor
python "1-4. LLL_Attack.py"
```

## Notes

- `P.tensor` (the input to the LLL step) is `P2 · P1ᵀ`, the "transition" matrix between the two helper matrices (max error ≈ 1.3e-7). It is not produced by the scripts in this repository.
- `GenC` still contains branches for a 512-dimensional / 16-sparse codeword setting; the 64-dimensional setting (`GenC(3)`) is the one used.
- The LLL implementation uses `fractions.Fraction` and recomputes Gram–Schmidt after each update, so it is slow; it is meant for clarity, not performance.

## Result

- **LLL leaves the matrix unchanged** (`P_Reduction.tensor == P.tensor` exactly). `P2 · P1ᵀ` is orthogonal, and an orthonormal basis already satisfies the LLL conditions (all Gram–Schmidt coefficients μ are 0 and all norms are equal), so reduction has nothing to do.
- **The codeword is not recovered.** The decoded `ECC(4 · P_LLL · c1)` has support `{7, 35, 47, 53}`, while `c2` has support `{33, 34, 41, 51}`. Because `P1 · w1 = c1`, we get `P · c1 = P2 · w1`, so the check only tells you what `w1` decodes to under `P2`.
- In the saved data, `w1` and `w2` are independent random vectors (cosine ≈ 0.016), so they act as two different users.
- Conclusion: this LLL approach gives no advantage against orthogonal helper matrices.

## Reference

S. Kim, Y. Jeong, J. Kim, J. Kim, H. T. Lee, J. H. Seo,
"IronMask: Modular Architecture for Protecting Deep Face Template," *CVPR 2021*.
