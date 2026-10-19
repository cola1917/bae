from __future__ import annotations

from pathlib import Path
import sys

import pypose as pp
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bae.utils.pypose_ambient_grad import install_pypose_ambient_grad_monkeypatch  # noqa: E402
from bae.utils.parameter import trim_parameter_jacobian_values  # noqa: E402


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
]


def _fd_jacobian(func, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    y0 = func(x)
    jac = torch.zeros(y0.numel(), x.numel(), dtype=x.dtype, device=x.device)
    for i in range(x.numel()):
        xp = x.clone()
        xm = x.clone()
        xp.view(-1)[i] += eps
        xm.view(-1)[i] -= eps
        jac[:, i] = ((func(xp) - func(xm)) / (2.0 * eps)).reshape(-1)
    return jac.reshape(*y0.shape, *x.shape)


def test_pypose_se3_act_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)
    point = torch.randn(1, 3, device=device, dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Act(point))(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Act(point), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_mul_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    x = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)
    y = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda a: (pp.SE3(a) * pp.SE3(y)).tensor())(x)
    jac_fd = _fd_jacobian(lambda a: (pp.SE3(a) * pp.SE3(y)).tensor(), x)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_log_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Log().tensor())(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Log().tensor(), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_inv_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Inv().tensor())(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Inv().tensor(), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_exp_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    algebra = 0.2 * torch.randn(1, 6, device=device, dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.se3(x).Exp().tensor())(algebra)
    jac_fd = _fd_jacobian(lambda x: pp.se3(x).Exp().tensor(), algebra)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_pypose_se3_translation_monkeypatch_produces_ambient_gradient(batch_shape):
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = torch.tensor(
        [0.3, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float64
    ).expand(*batch_shape, 7).clone()

    def translation(x):
        return pp.SE3(x).translation()

    torch.testing.assert_close(translation(pose), pose[..., :3])
    jac = torch.func.jacrev(translation)(pose)
    jac_fd = _fd_jacobian(translation, pose)
    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_se3_relative_translation_localized_jacobian_matches_pose_perturbations(monkeypatch):
    monkeypatch.setenv("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algebra = torch.tensor(
        [[0.3, -0.2, 0.5, 0.2, 0.3, -0.1], [-0.4, 0.8, 0.2, -0.3, 0.1, 0.4]],
        device=device, dtype=torch.float64,
    )
    nodes = pp.se3(algebra).Exp().detach()

    def relative_translation(x):
        poses = pp.SE3(x)
        return (poses[0].Inv() @ poses[1]).translation()

    # BAE differentiates stored coordinates, then converts each parameter block
    # once to the six-dimensional left perturbation used by the optimizer.
    ambient = torch.func.jacrev(relative_translation)(nodes.tensor())
    local = trim_parameter_jacobian_values(nodes, ambient.transpose(0, 1)).transpose(0, 1)

    def perturbed_translation(delta):
        perturbed = pp.se3(delta).Exp() @ nodes
        return relative_translation(perturbed.tensor())

    local_fd = _fd_jacobian(perturbed_translation, torch.zeros_like(algebra))
    torch.testing.assert_close(local, local_fd, rtol=1e-7, atol=1e-7)
