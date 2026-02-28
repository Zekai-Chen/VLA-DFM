import torch

from prismatic.discrete_flow.dfm_schedule import kappa, kappa_dot, time_grid


def test_kappa_endpoints_and_range():
    t = torch.tensor([0.0, 0.5, 1.0])
    for schedule in ("cosine", "sin", "linear", "poly2"):
        kt = kappa(t, schedule=schedule)
        assert torch.all(kt >= 0.0)
        assert torch.all(kt <= 1.0)
        assert torch.isclose(kt[0], torch.tensor(0.0))
        assert torch.isclose(kt[-1], torch.tensor(1.0))


def test_kappa_dot_nonnegative():
    t = torch.linspace(0.0, 1.0, steps=11)
    for schedule in ("cosine", "sin", "linear", "poly2"):
        kdot = kappa_dot(t, schedule=schedule)
        assert torch.all(kdot >= -1e-6)


def test_time_grid_bounds():
    t, dt = time_grid(num_steps=4, eps=1e-3)
    assert t.shape == (4,)
    assert dt.shape == (4,)
    assert torch.all(t >= 0.0)
    assert torch.all(t <= 1.0)
    assert torch.all(dt > 0.0)
    # Total span should be 1 - eps
    assert torch.isclose(dt.sum(), torch.tensor(1.0 - 1e-3), atol=1e-6)
