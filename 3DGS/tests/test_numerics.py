import numpy as np
import pytest
import torch

from lsot.gaussians import GMM, gaussian_pair_costs
from lsot.plans import average_lsot, minimum_lsot, lift_projection
from lsot.projections import sample_projection_bank
from lsot3dgs.geometry import GaussianSet,GaussianCost,bures_update
from lsot3dgs.plans import EdgePlan,SortedProjection,LiftedPlan,AveragePlan,plan_change
from lsot3dgs.solvers.lsot import LSOTSolver
from lsot3dgs.solvers.mw2 import MW2Solver,BudgetExceeded
from lsot3dgs.reduction import reduce_block

torch.set_num_threads(1)


def mixture(n,seed,device="cpu"):
    g = torch.Generator().manual_seed(seed)
    weights = torch.rand(n,generator=g,dtype=torch.float64)+.1
    means = torch.randn(n,3,generator=g,dtype=torch.float64)
    a = torch.randn(n,3,3,generator=g,dtype=torch.float64)
    cov = a@a.transpose(-1,-2)+.2*torch.eye(3,dtype=torch.float64)
    return GaussianSet(*(x.to(device) for x in (weights/weights.sum(),means,cov)))


@pytest.mark.parametrize("kind",["Mix","SMix","B","B1D"])
@pytest.mark.parametrize("mode",["min","avg"])
def test_streaming_matches_existing_lsot(kind,mode):
    x,y = mixture(9,1),mixture(4,2)
    bank = sample_projection_bank(3,7,kind=kind,seed=3)
    actual = LSOTSolver(x,f"{mode}-{kind}",{"projections":7,"projection_batch":3}, {}, bank).solve(y)
    xx,yy = GMM(x.weights,x.means,x.covariances),GMM(y.weights,y.means,y.covariances)
    reference = minimum_lsot(xx,yy,bank,kind) if mode=="min" else average_lsot(xx,yy,bank,kind)
    plan = reference.plan if mode=="min" else reference
    torch.testing.assert_close(actual.plan.dense(),plan.dense(),rtol=1e-10,atol=1e-12)
    assert actual.objective == pytest.approx(float(plan.cost(xx,yy)),rel=1e-10,abs=1e-12)
    if mode=="min":
        assert actual.diagnostics["projection_index"] == reference.projection_index


def test_tie_fibers_stream_product_lift():
    a = torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    b = torch.tensor([.4,.35,.25],dtype=torch.float64)
    x,y = torch.tensor([0.,0.,1.,1.]),torch.tensor([0.,0.,2.])
    plan = LiftedPlan(SortedProjection(x,a),SortedProjection(y,b))
    expected = lift_projection(x,y,a,b)
    torch.testing.assert_close(plan.dense(),expected.dense())
    assert max(len(p) for _,_,p in plan.edges(2)) <= 2
    assert plan.marginal_error(a,b) < 1e-12
    tied = LiftedPlan(SortedProjection(x*0,a),SortedProjection(y*0,b))
    torch.testing.assert_close(tied.dense(),a[:,None]*b[None,:])


def test_cost_certificate_and_ordering():
    x,y = mixture(12,4),mixture(5,5)
    exact = MW2Solver(x,{},{}).solve(y)
    minimum = LSOTSolver(x,"min-B1D",{"projections":12},{}).solve(y)
    average = LSOTSolver(x,"avg-B1D",{"projections":12},{}).solve(y)
    assert exact.objective <= minimum.objective+1e-10
    assert minimum.objective <= average.objective+1e-10
    i,j = torch.where(torch.ones(12,5,dtype=torch.bool))
    torch.testing.assert_close(GaussianCost(x,3).pairs(y,i,j),gaussian_pair_costs(x,y,i,j))
    assert exact.diagnostics["duality_gap_scaled"] < 1e-8
    with pytest.raises(BudgetExceeded):
        MW2Solver(x,{"max_dense_bytes":1},{}).solve(y)


def test_commuting_bures_has_known_solution():
    w = torch.tensor([.2,.3,.5],dtype=torch.float64)
    means = torch.tensor([[0.,1.,2.],[2.,3.,0.],[1.,0.,4.]],dtype=torch.float64)
    std = torch.tensor([[1.,2.,3.],[3.,1.,2.],[2.,4.,1.]],dtype=torch.float64)
    x = GaussianSet(w,means,torch.diag_embed(std.square()))
    y = GaussianSet(torch.ones(1,dtype=torch.float64),torch.zeros(1,3,dtype=torch.float64),
                    torch.eye(3,dtype=torch.float64)[None])
    p = EdgePlan(torch.arange(3),torch.zeros(3,dtype=torch.long),w,(3,1))
    result,info = bures_update(x,y,p,tolerance=1e-12)
    torch.testing.assert_close(result.means,(w[:,None]*means).sum(0)[None])
    torch.testing.assert_close(result.covariances,torch.diag_embed((w[:,None]*std).sum(0).square())[None])
    assert info["inner_converged"]


def test_noncommuting_bures_decreases_fixed_plan_cost():
    x,y = mixture(8,12),mixture(3,13)
    s = MW2Solver(x,{},{}); old = s.solve(y)
    result,info = bures_update(x,y,old.plan,steps=300,tolerance=1e-9)
    assert s.cost.plan_cost(result,old.plan) <= old.objective+1e-10
    assert info["inner_converged"]
    torch.testing.assert_close(result.weights,y.weights)


@pytest.mark.parametrize("method",["mw2","min-B1D","avg-SMix"])
def test_outer_reports_actual_selected_objective(method):
    x,y = mixture(10,16),mixture(3,17)
    history=[]
    settings={"outer_steps":3,"projections":5,"bures_steps":100,"bures_tolerance":1e-7}
    result,appearance,record = reduce_block(x,y,method,settings,{},history.append)
    solver = MW2Solver(x,settings,{}) if method=="mw2" else LSOTSolver(x,method,settings,{})
    assert solver.solve(result).objective == pytest.approx(record["objective"],rel=1e-10)
    assert record["objective"] == min(row["objective"] for row in history)
    assert appearance.marginal_error(x.weights,y.weights) < 1e-9
    assert record["status"] in ("converged","iteration_cap")


def test_average_change_is_an_upper_bound():
    x,y=mixture(6,21),mixture(3,22)
    a=LSOTSolver(x,"avg-B1D",{"projections":4},{}).solve(y).plan
    yy=GaussianSet(y.weights,y.means+torch.tensor([.1,.2,-.3]),y.covariances)
    b=LSOTSolver(x,"avg-B1D",{"projections":4},{}).solve(yy).plan
    assert plan_change(a,b)+1e-12 >= float((a.dense()-b.dense()).abs().sum())


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA device required")
@pytest.mark.parametrize("method",["mw2","min-Mix","min-SMix","min-B","min-B1D","avg-B1D"])
def test_cuda_cpu_parity(method):
    x,y=mixture(12,31),mixture(5,32)
    xx,yy=mixture(12,31,"cuda"),mixture(5,32,"cuda")
    def solve(a,b):
        options={"projections":8}
        solver=MW2Solver(a,options,{}) if method=="mw2" else LSOTSolver(a,method,options,{})
        return solver.solve(b)
    a,b=solve(x,y),solve(xx,yy)
    assert a.objective == pytest.approx(b.objective,rel=1e-8,abs=1e-10)
    torch.testing.assert_close(a.plan.dense(),b.plan.dense().cpu(),rtol=1e-7,atol=1e-9)
