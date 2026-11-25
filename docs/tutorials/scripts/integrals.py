"""MO integral blocks in the layout ALQEMI's generated code reads (numpy only).

ALQEMI's generated functions take one object, ``moeris``, and read attributes named
after the orbital spaces of each tensor's axes, occupied axes first (OOVV order):

* ``f_OV``  Fock block  f^p_q -> f[p, q]
* ``g_OOVV`` antisymmetrized integral <pq||rs> -> g[p, q, r, s] (physicist's order)

Spin-orbital input (``SpinOrbitalIntegrals``) uses upper-case letters for the
spin-orbital spaces. UHF input (``UHFIntegrals``) uses upper case for alpha and lower
case for beta: ``g_OoVv[i, J, a, B]`` would be <iJ||aB>, and opposite-spin blocks
hold no exchange part.

Blocks are built on first use from AO integrals and MO coefficients, so the same
classes work with any SCF program (see scf.py).
"""
from __future__ import annotations

import numpy as np


def _mo(eri_ao, c1, c2, c3, c4):
    """Chemist's-notation MO integrals (pq|rs) from AO (mu nu|la si)."""
    return np.einsum("mnls,mp,nq,lr,st->pqrt", eri_ao, c1, c2, c3, c4, optimize=True)


class UHFIntegrals:
    """f and <pq||rs> blocks of a UHF (or RHF, with ca = cb) reference.

    ``one_body`` adds more one-electron operators by name, e.g. {"u": dipole_z_ao}
    gives blocks u_OV, u_ov, ... (spin-conserving, like f).
    """

    def __init__(self, h_ao, eri_ao, ca, cb, nocc_a, nocc_b, e_nuc=0.0, one_body=None):
        self.h_ao, self.eri_ao, self.e_nuc = h_ao, eri_ao, e_nuc
        self._one_body = {"h": h_ao, **(one_body or {})}
        self._c = {
            "O": ca[:, :nocc_a], "V": ca[:, nocc_a:],
            "o": cb[:, :nocc_b], "v": cb[:, nocc_b:],
        }
        da = ca[:, :nocc_a] @ ca[:, :nocc_a].T
        db = cb[:, :nocc_b] @ cb[:, :nocc_b].T
        j = np.einsum("mnls,ls->mn", eri_ao, da + db)
        self._fock_ao = {
            "alpha": h_ao + j - np.einsum("mlns,ls->mn", eri_ao, da),
            "beta": h_ao + j - np.einsum("mlns,ls->mn", eri_ao, db),
        }
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cache = self.__dict__.setdefault("_cache", {})
        if name not in cache:
            kind, _, block = name.partition("_")
            if kind == "f" and len(block) == 2:
                cache[name] = self._fock(block)
            elif kind in self._one_body and len(block) == 2:
                if block[0].isupper() != block[1].isupper():
                    raise AttributeError(f"{name} mixes spins")
                cache[name] = self._c[block[0]].T @ self._one_body[kind] @ self._c[block[1]]
            elif kind == "g" and len(block) == 4:
                cache[name] = self._antisymmetrized(block)
            else:
                raise AttributeError(name)
        return cache[name]

    def _fock(self, block):
        spin = "alpha" if block[0].isupper() else "beta"
        if block[0].isupper() != block[1].isupper():
            raise AttributeError(f"f_{block} mixes spins")
        return self._c[block[0]].T @ self._fock_ao[spin] @ self._c[block[1]]

    def _antisymmetrized(self, block):
        p, q, r, s = block
        c = self._c
        out = 0.0
        if p.isupper() == r.isupper() and q.isupper() == s.isupper():
            # <pq|rs> = (pr|qs)
            out = _mo(self.eri_ao, c[p], c[r], c[q], c[s]).transpose(0, 2, 1, 3)
        if p.isupper() == s.isupper() and q.isupper() == r.isupper():
            # - <pq|sr> = - (ps|qr)
            out = out - _mo(self.eri_ao, c[p], c[s], c[q], c[r]).transpose(0, 2, 3, 1)
        return np.asarray(out)


class SpinOrbitalIntegrals:
    """Spin-orbital f and <pq||rs> blocks (spaces O, V) from a UHF/RHF reference.

    Spin orbitals are ordered alpha occupied, beta occupied, then alpha virtual,
    beta virtual.
    """

    def __init__(self, h_ao, eri_ao, ca, cb, nocc_a, nocc_b, e_nuc=0.0, one_body=None):
        self._uhf = UHFIntegrals(h_ao, eri_ao, ca, cb, nocc_a, nocc_b, e_nuc, one_body)
        self.e_nuc = e_nuc
        self._sizes = {"O": (nocc_a, nocc_b), "V": (ca.shape[1] - nocc_a, cb.shape[1] - nocc_b)}
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cache = self.__dict__.setdefault("_cache", {})
        if name not in cache:
            kind, _, block = name.partition("_")
            if len(block) not in (2, 4) or not set(block) <= {"O", "V"}:
                raise AttributeError(name)
            cache[name] = self._assemble(kind, block)
        return cache[name]

    def _assemble(self, kind, block):
        shape = [sum(self._sizes[x]) for x in block]
        out = np.zeros(shape)
        for spins in np.ndindex(*(2,) * len(block)):
            letters = "".join(x if s == 0 else x.lower() for x, s in zip(block, spins))
            try:
                part = getattr(self._uhf, f"{kind}_{letters}")
            except AttributeError:
                if spins == (0,) * len(block):
                    raise  # the all-alpha block always exists: an unknown name
                continue  # a spin-forbidden block
            where = tuple(
                slice(0, self._sizes[x][0]) if s == 0 else slice(self._sizes[x][0], None)
                for x, s in zip(block, spins)
            )
            out[where] = part
        return out
