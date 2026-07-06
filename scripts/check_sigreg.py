"""Smoke-test the optional LeJEPA/SIGReg dependency before A1 training."""

from __future__ import annotations

import torch


def main() -> None:
    import lejepa

    x = torch.randn(512, 256, requires_grad=True)
    univariate_test = lejepa.univariate.EppsPulley(n_points=17)
    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=1024,
    )
    loss = sigreg(x)
    if not torch.isfinite(loss):
        raise RuntimeError(f"SIGReg loss is not finite: {loss}")
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("SIGReg gradients are missing or non-finite")
    print(f"SIGReg smoke test passed: loss={float(loss.detach()):.6f}", flush=True)


if __name__ == "__main__":
    main()
