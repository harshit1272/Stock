from app import (
    PipelineRequest,
    StockInput,
    phase1_market_regime,
    phase2_sieve,
    phase7_position_size,
    run_pipeline,
    demo_request,
)


def sample_stock(**kwargs):
    base = dict(
        symbol="ABC",
        sector="IT",
        revenue_growth=20,
        eps_growth=19,
        roe=18,
        debt_to_equity=0.2,
        pe=18,
        growth_rate=22,
        relative_strength_percentile=80,
        roic=20,
        wacc=11,
        gross_margin_5y=[40, 40, 41, 42, 42],
        cfo=120,
        net_income=100,
        sector_cagr_3y=16,
        close=1000,
        dma50=980,
        dma150=930,
        dma200=900,
        up_volume=2000,
        down_volume=1000,
        bb_width=0.1,
        sentiment=20,
        severity_rank=2,
        win_rate=0.57,
        win_loss_ratio=1.8,
        atr=25,
    )
    base.update(kwargs)
    return StockInput(**base)


def test_regime_expansion():
    regime = phase1_market_regime({"pmi": 55, "cpi": 5, "gdp_growth": 7, "yield_spread": 1.2, "vix": 18})
    assert regime.regime == "Expansion"
    assert regime.risk_off is False


def test_sieve_rejects_high_peg():
    stock = sample_stock(pe=35, growth_rate=20)
    passed, stages = phase2_sieve(stock)
    assert passed is False
    assert stages["stage2"] is False


def test_position_size_positive():
    stock = sample_stock()
    size = phase7_position_size(stock, capital=1_000_000)
    assert size > 0


def test_pipeline_emits_decision_for_good_stock():
    req = PipelineRequest(
        macro={"pmi": 53, "cpi": 5.2, "gdp_growth": 6.1, "yield_spread": 0.7, "vix": 17},
        nse_universe=[sample_stock()],
        capital=1_000_000,
    )
    decisions = run_pipeline(req)
    assert len(decisions) == 1
    assert decisions[0].symbol == "ABC"


def test_demo_request_generates_shortlist():
    decisions = run_pipeline(demo_request())
    assert len(decisions) >= 1
    assert decisions[0].conviction >= decisions[-1].conviction
