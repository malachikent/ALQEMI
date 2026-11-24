"""Code generation.

Turns derived ALQEMI expressions into Python code for prototype programs: tensor
contractions written as ``np.einsum`` / ``opt_einsum.contract`` calls
(``alqemi.codegen.einsum.einsum_code``).
"""
