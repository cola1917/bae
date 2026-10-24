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


@pytest.mark.parametrize("group", [pp.SO3, pp.SE3])
def test_ambient_log_identity_and_mixed_batch_gradients(group):
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quaternions = torch.tensor(
        [[0., 0., 0., 1.], [0., 0., 0., -1.],
         [0., 0., 0., 1.0000011496537984],
         [1e-8, -2e-8, 3e-8, 1.], [0.2, -0.3, 0.1, 0.9]],
        device=device, dtype=torch.float64,
    )
    values = quaternions
    if group is pp.SE3:
        translation = quaternions.new_tensor([0.3, -0.2, 0.5]).expand(5, 3)
        values = torch.cat((translation, quaternions), dim=-1)

    def log(x):
        return group(x).Log().tensor()

    # vmap/jacrev is the path used to differentiate individual sparse factors.
    jac = torch.func.vmap(torch.func.jacrev(log))(values)
    assert torch.isfinite(jac).all()
    for value, derivative in zip(values, jac):
        torch.testing.assert_close(derivative, _fd_jacobian(log, value), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ambient_so3_log_half_turn_is_finite(dtype):
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quaternion = torch.tensor([1., 0., 0., 0.], device=device, dtype=dtype, requires_grad=True)
    result = pp.SO3(quaternion).Log().tensor()
    torch.testing.assert_close(result, quaternion.new_tensor([torch.pi, 0., 0.]))
    # The principal Log is discontinuous here. Check numerical finiteness of
    # the selected branch, not agreement with a derivative across the cut.
    result.sum().backward()
    assert torch.isfinite(quaternion.grad).all()


def test_ambient_pgo_log_at_identity_has_correct_local_jacobian(monkeypatch):
    monkeypatch.setenv("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor(
        [[0., 0., 0., 0., 0., 0., 1.], [0.3, -0.2, 0.5, 0., 0., 0., 1.]],
        device=device, dtype=torch.float64,
    )
    nodes = pp.SE3(values)
    measurement = nodes[0].Inv() @ nodes[1]

    def residual(x):
        poses = pp.SE3(x)
        return (measurement.Inv() @ poses[0].Inv() @ poses[1]).Log().tensor()

    torch.testing.assert_close(residual(values), values.new_zeros(6))
    ambient = torch.func.jacrev(residual)(values)
    local = trim_parameter_jacobian_values(nodes, ambient.transpose(0, 1)).transpose(0, 1)

    def perturbed_residual(delta):
        return residual((pp.se3(delta).Exp() @ nodes).tensor())

    local_fd = _fd_jacobian(perturbed_residual, values.new_zeros(2, 6))
    torch.testing.assert_close(local, local_fd, rtol=1e-7, atol=1e-7)


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
