#!/usr/bin/env python3
"""
Scan de cointégration — toutes les combinaisons de paires crypto USDT.

Tests : ADF · Engle-Granger · Johansen · rolling cointegration
Sorties : heatmap matricielle N×N + graphiques détaillés top K paires
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ccxt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scipy import stats

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.robust.norms as sm_norms

from data_manager import DataManager


# ── Configuration ────────────────────────────────────────────────────────────
TIMEFRAME      = "1h"
DURATION       = "10an"
ROLLING_WINDOW = 50
OUTPUT_DIR     = "outputs"

TOP_N_SYMBOLS  = 50
TOP_K_RESULTS  = 10
EXCHANGE_ID    = "binance"
MAX_WORKERS    = 11

# Seuil de corrélation minimum pour ne pas tester des paires sans lien
CORR_MIN       = 0.5

# Solution 1 : winsorisation — seuil en σ pour les log-rendements
WINSOR_Z       = 3.0

# Solution 2 : instabilité du beta — poids max de la pénalité dans le score (0–1)
# CV(β) calculé sur N_SUBPERIODS sous-périodes ; pénalité progressive, jamais un filtre dur
BETA_INSTABILITY_WEIGHT = 0.3
N_SUBPERIODS   = 3




# Stablecoins et tokens fiat à exclure de l'univers de scan
_STABLECOINS: frozenset[str] = frozenset({
    # Stablecoins USD
    "USDT", "USDC", "BUSD", "TUSD", "USDP", "GUSD", "FRAX", "LUSD",
    "SUSD", "CUSD", "PUSD", "HUSD", "USTC", "UST", "FDUSD", "PYUSD",
    "USDD", "USDJ", "AUSD", "CEUR", "EURS", "EURT", "EURC", "XSGD",
    "DAI", "XDAI", "SDAI", "EDAI", "MXNT", "BIDR", "BVND", "IDRT",
    # Fiat tokenisés
    "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "ARS", "NGN", "UAH",
    # Wrapped tokens identiques à leur sous-jacent
    "WBTC", "WETH", "WBNB",
})

def get_top_usdt_symbols(exchange: ccxt.Exchange, top_n: int) -> list[str]:
    """
    Retourne les top_n paires spot /USDT actives triées par volume quote 24h,
    en excluant les stablecoins, les tokens fiat et les perpetuals.
    """
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    usdt_symbols = []
    for symbol, market in markets.items():
        base = market.get("base", "")

        if not (
            market.get("spot") is True           # spot uniquement
            and market.get("active") is True      # actif (strict)
            and symbol.endswith("/USDT")
            and ":" not in symbol                 # pas de perpetual/swap
            and base not in _STABLECOINS          # pas un stablecoin/fiat
            and base != ""
        ):
            continue

        usdt_symbols.append(symbol)

    # Tri par volume quote 24h (descending)
    def _quote_volume(sym: str) -> float:
        t = tickers.get(sym, {})
        return t.get("quoteVolume") or t.get("baseVolume") or 0.0

    usdt_symbols.sort(key=_quote_volume, reverse=True)
    filtered = usdt_symbols[:top_n]

    print(f"  → {len(usdt_symbols)} paires crypto actives trouvées, {len(filtered)} retenues")
    return filtered

# ── Tests statistiques ────────────────────────────────────────────────────────

def run_adf(series: pd.Series, name: str, fast: bool = False) -> dict:
    """
    Augmented Dickey-Fuller : H0 = racine unitaire (non-stationnaire).
    fast=True : lag fixé à 1, pas de recherche AIC → ~5x plus rapide.
    """
    kwargs = {"maxlag": 1, "autolag": None} if fast else {"autolag": "AIC"}
    result = adfuller(series.dropna(), **kwargs)
    return {
        "name": name,
        "stat": result[0],
        "pvalue": result[1],
        "used_lag": result[2],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def winsorize_series(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """
    Winsorise les log-rendements à ±z_thresh σ puis reconstruit le niveau.
    Réduit l'impact des flash-crashes et pics extrêmes sans supprimer d'observations.
    """
    log_prices = np.log(series)
    log_ret = log_prices.diff()          # premier élément = NaN
    mu    = log_ret.mean()               # skipna par défaut
    sigma = log_ret.std()
    log_ret_clipped = log_ret.clip(lower=mu - z_thresh * sigma,
                                   upper=mu + z_thresh * sigma)
    log_ret_clipped.iloc[0] = 0.0       # premier point : pas de changement
    log_level = log_prices.iloc[0] + log_ret_clipped.cumsum()
    return np.exp(log_level)


def compute_beta_stability(log_y: pd.Series, log_x: pd.Series,
                           n_periods: int = 3) -> float:
    """
    Découpe la série en n_periods sous-périodes et estime β sur chacune.
    Retourne le coefficient de variation CV = std(β) / |mean(β)|.
      CV ≈ 0  →  β stable dans le temps  (bonne paire)
      CV ≫ 1  →  β change de régime      (paire suspecte)
    Jamais utilisé comme filtre dur : sert de pénalité dans score_result().
    """
    n    = len(log_y)
    size = n // n_periods
    betas = []
    for i in range(n_periods):
        start = i * size
        end   = start + size if i < n_periods - 1 else n
        y_sub, x_sub = log_y.iloc[start:end], log_x.iloc[start:end]
        beta_sub, _, _ = compute_hedge_ratio(y_sub, x_sub)
        betas.append(beta_sub)
    betas    = np.array(betas)
    mean_abs = np.abs(np.mean(betas))
    return 0.0 if mean_abs < 1e-8 else float(np.std(betas) / mean_abs)


def compute_hedge_ratio(y: pd.Series, x: pd.Series):
    """OLS y ~ x  →  (beta, alpha, résidus)."""
    model = OLS(y, add_constant(x)).fit()
    alpha = model.params.iloc[0]
    beta  = model.params.iloc[1]
    return beta, alpha, y - (alpha + beta * x)


def run_engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Test de cointégration Engle-Granger."""
    stat, pvalue, crit = coint(y, x)
    return {
        "stat": stat,
        "pvalue": pvalue,
        "critical_values": crit,
        "is_cointegrated": pvalue < 0.05,
    }


def run_johansen(df: pd.DataFrame) -> dict:
    """Test de cointégration de Johansen (trace + valeur propre max)."""
    res = coint_johansen(df, det_order=0, k_ar_diff=1)

    trace_stat    = res.lr1          # shape (k,)
    trace_crit95  = res.cvt[:, 1]   # colonnes : 90 / 95 / 99 %
    max_stat      = res.lr2
    max_crit95    = res.cvm[:, 1]

    # Hypothèse H0 rejetée si stat > valeur critique 95 %
    r0_rejected = trace_stat[0] > trace_crit95[0]
    r1_rejected = trace_stat[1] > trace_crit95[1]

    n_coint = 0
    if r0_rejected:
        n_coint = 1
    if r1_rejected:
        n_coint = 2

    return {
        "trace_stat":   trace_stat,
        "trace_crit95": trace_crit95,
        "max_stat":     max_stat,
        "max_crit95":   max_crit95,
        "n_cointegrating_vectors": n_coint,
        "eigenvectors": res.evec,
    }


def run_rolling_cointegration(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """Métriques de cointégration en fenêtre roulante."""
    if window < 30:
        raise ValueError("ROLLING_WINDOW doit être >= 30")

    y, x = y.align(x, join="inner")
    y, x = y.dropna(), x.dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]

    if len(y) < window:
        raise ValueError(
            f"Pas assez d'observations pour fenêtre roulante: {len(y)} < {window}"
        )

    rows = []
    for end in range(window, len(y) + 1):
        y_w = y.iloc[end - window:end]
        x_w = x.iloc[end - window:end]

        beta_w, alpha_w, spread_w = compute_hedge_ratio(y_w, x_w)
        _, eg_p_w, _ = coint(y_w, x_w)
        adf_spread_p_w = adfuller(spread_w.dropna(), autolag="AIC")[1]

        spread_std_w = spread_w.std()
        if spread_std_w == 0 or np.isnan(spread_std_w):
            z_last_w = np.nan
        else:
            z_last_w = (spread_w.iloc[-1] - spread_w.mean()) / spread_std_w

        rows.append(
            {
                "date": y_w.index[-1],
                "eg_p": eg_p_w,
                "adf_spread_p": adf_spread_p_w,
                "beta": beta_w,
                "alpha": alpha_w,
                "z_last": z_last_w,
            }
        )

    return pd.DataFrame(rows).set_index("date")

# ── Graphiques ────────────────────────────────────────────────────────────────

def plot_results(btc, eth, spread, adf_btc, adf_eth, adf_spread,
                 eg, johansen, beta, alpha,
                 rolling_window, rolling_metrics,
                 output_png, symbol_1, symbol_2, show_plot=False):

    log_btc = np.log(btc)
    log_eth = np.log(eth)

    spread_mean = spread.mean()
    spread_std  = spread.std()
    zscore = (spread - spread.rolling(rolling_window).mean()) / spread.rolling(rolling_window).std()

    corr_window = min(rolling_window, max(10, len(btc) - 1))
    rolling_corr = btc.pct_change().rolling(corr_window).corr(eth.pct_change())

    # ── Mise en page ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 27))
    fig.suptitle(
        f"Analyse de Cointégration  {symbol_1} — {symbol_2}",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.35)

    BTC_COLOR = "#F7931A"
    ETH_COLOR = "#627EEA"

    # ── 1. Prix normalisés ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(btc.index, btc / btc.iloc[0], label=f"{symbol_1} (normalisé)",
             color=BTC_COLOR, linewidth=1.6)
    ax1.plot(eth.index, eth / eth.iloc[0], label=f"{symbol_2} (normalisé)",
             color=ETH_COLOR, linewidth=1.6)
    ax1.set_title("Prix normalisés (base = 1)", fontsize=12)
    ax1.set_ylabel("Prix normalisé")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. Spread avec bandes ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(spread.index, spread, color="#2ecc71", linewidth=1.2, label="Spread")
    ax2.axhline(spread_mean, color="black", linestyle="--", linewidth=1.0,
                label=f"Moyenne : {spread_mean:.4f}")
    ax2.axhline(spread_mean + 2 * spread_std, color="red",   linestyle=":", linewidth=1.0,
                label="±2σ")
    ax2.axhline(spread_mean - 2 * spread_std, color="red",   linestyle=":", linewidth=1.0)
    ax2.fill_between(spread.index,
                     spread_mean - spread_std, spread_mean + spread_std,
                     alpha=0.15, color="green", label="±1σ")
    ax2.set_title(
        f"Spread  log({symbol_1}) − {beta:.4f}×log({symbol_2}) − {alpha:.4f}"
        f"   |   ADF p = {adf_spread['pvalue']:.4f}"
        f"  {'(stationnaire ✅)' if adf_spread['is_stationary'] else '(non-stationnaire ❌)'}",
        fontsize=11,
    )
    ax2.set_ylabel("Spread (log)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. Z-score ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(zscore.index, zscore, color="#9b59b6", linewidth=1.2)
    ax3.axhline(0,  color="black", linewidth=0.9)
    ax3.axhline( 2, color="red",  linestyle="--", linewidth=1.0, label="+2σ")
    ax3.axhline(-2, color="blue", linestyle="--", linewidth=1.0, label="−2σ")
    ax3.fill_between(zscore.index, -1, 1, alpha=0.10, color="green", label="±1σ")
    ax3.set_title(f"Z-score du spread (fenêtre {rolling_window})", fontsize=12)
    ax3.set_ylabel("Z-score")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. Corrélation glissante ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(rolling_corr.index, rolling_corr, color="#e74c3c", linewidth=1.2)
    ax4.axhline(rolling_corr.mean(), color="black", linestyle="--", linewidth=1.0,
                label=f"Moyenne : {rolling_corr.mean():.3f}")
    ax4.set_title(f"Corrélation glissante des rendements  (fenêtre {corr_window})", fontsize=12)
    ax4.set_ylabel("Corrélation de Pearson")
    ax4.set_ylim(-1, 1)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── 5. Scatter + OLS ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(log_eth, log_btc, alpha=0.30, s=10, color="#3498db")
    x_line = np.linspace(log_eth.min(), log_eth.max(), 200)
    ax5.plot(x_line, alpha + beta * x_line, color="red", linewidth=2,
             label=f"OLS : y = {beta:.4f}x + {alpha:.4f}")
    ax5.set_xlabel(f"log({symbol_2})")
    ax5.set_ylabel(f"log({symbol_1})")
    ax5.set_title(f"Régression OLS  log({symbol_1}) ~ log({symbol_2})", fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── 6. QQ-plot du spread ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(spread.dropna(), dist="norm")
    ax6.scatter(osm, osr, alpha=0.35, s=10, color="#e67e22")
    x_qq = np.array([osm[0], osm[-1]])
    ax6.plot(x_qq, slope * x_qq + intercept, color="red", linewidth=2,
             label=f"R² = {r**2:.4f}")
    ax6.set_title("QQ-plot du spread vs loi normale", fontsize=12)
    ax6.set_xlabel("Quantiles théoriques")
    ax6.set_ylabel("Quantiles observés")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── 7. Cointégration roulante ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.plot(rolling_metrics.index, rolling_metrics["eg_p"],
             color="#1f77b4", linewidth=1.2, label="EG p-value (rolling)")
    ax7.plot(rolling_metrics.index, rolling_metrics["adf_spread_p"],
             color="#ff7f0e", linewidth=1.2, label="ADF spread p-value (rolling)")
    ax7.axhline(0.05, color="red", linestyle="--", linewidth=1.0, label="Seuil 5%")
    ax7.set_ylim(0, 1)
    ax7.set_title(f"Tests roulants de cointégration (fenêtre {rolling_window})", fontsize=12)
    ax7.set_ylabel("p-value")
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # ── Bloc résumé ───────────────────────────────────────────────────────────
    is_coint = eg["is_cointegrated"] or johansen["n_cointegrating_vectors"] > 0
    color_box = "#d4edda" if is_coint else "#f8d7da"
    conclusion = (
        f"Engle-Granger p = {eg['pvalue']:.4f}   |   "
        f"Johansen : {johansen['n_cointegrating_vectors']} vecteur(s)\n"
        f"{'COINTÉGRÉS ✅' if is_coint else 'NON COINTÉGRÉS ❌'}"
    )
    fig.text(
        0.5, 0.005, conclusion,
        ha="center", va="bottom", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color_box, alpha=0.8),
    )

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"Graphique sauvegardé : {output_png}")



def score_result(result: dict) -> float:
    """
    Score composite — plus bas = mieux.
      base    : 0.6 × EG_p + 0.4 × ADF_spread_p   (∈ [0, 1])
      pénalité: BETA_INSTABILITY_WEIGHT × min(CV_β / 2, 1)
                CV_β = 0 → +0   |   CV_β = 1 → +weight/2   |   CV_β ≥ 2 → +weight
    La pénalité dégrade le rang des paires instables sans les éliminer.
    """
    base    = 0.6 * result["eg"]["pvalue"] + 0.4 * result["adf_spread"]["pvalue"]
    beta_cv = result.get("beta_cv", 0.0)
    penalty = BETA_INSTABILITY_WEIGHT * min(beta_cv / 2.0, 1.0)
    return base + penalty


def is_significant(result: dict) -> bool:
    return result["eg"]["pvalue"] < 0.05 and result["adf_spread"]["pvalue"] < 0.05


def _sanitize_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _fast_scan_pair(
    symbol_1: str, s1: pd.Series,
    symbol_2: str, s2: pd.Series,
) -> dict:
    """
    Scan rapide d'une paire : corrélation → ADF (lag fixe) → Engle-Granger.
    Pas de rolling, pas de Johansen. Réservé au balayage initial.
    """
    try:
        s1, s2 = s1.align(s2, join="inner")
        s1, s2 = s1.dropna(), s2.dropna()
        common = s1.index.intersection(s2.index)
        s1, s2 = s1.loc[common], s2.loc[common]

        if len(s1) < ROLLING_WINDOW:
            raise ValueError(f"Observations insuffisantes : {len(s1)} < {ROLLING_WINDOW}")

        # ── Filtre corrélation (numpy pur, gratuit) ───────────────────────────
        corr = np.corrcoef(s1.pct_change().dropna(), s2.pct_change().dropna())[0, 1]
        if corr < CORR_MIN:
            raise ValueError(f"Corrélation trop faible ({corr:.2f} < {CORR_MIN})")

        # ── Solution 1 : winsorisation des log-rendements ─────────────────────
        s1 = winsorize_series(s1, WINSOR_Z)
        s2 = winsorize_series(s2, WINSOR_Z)

        log_s1 = np.log(s1)
        log_s2 = np.log(s2)

        # ADF lag fixe sur les niveaux (vérification I(1))
        adf_s1 = run_adf(log_s1, f"log({symbol_1})", fast=True)
        adf_s2 = run_adf(log_s2, f"log({symbol_2})", fast=True)

        beta, alpha, spread = compute_hedge_ratio(log_s1, log_s2)

        # ── Solution 2 : stabilité du beta (rupture structurelle légère) ─────
        beta_cv = compute_beta_stability(log_s1, log_s2, N_SUBPERIODS)

        # ADF lag fixe sur le spread
        adf_spread = run_adf(spread, "Spread", fast=True)

        # Engle-Granger
        eg = run_engle_granger(log_s1, log_s2)

        return {
            "symbol_1": symbol_1, "symbol_2": symbol_2,
            "n_obs": len(s1), "corr": corr,
            "s1": s1, "s2": s2, "spread": spread,
            "adf_s1": adf_s1, "adf_s2": adf_s2, "adf_spread": adf_spread,
            "eg": eg, "beta": beta, "alpha": alpha, "beta_cv": beta_cv,
            "error": None,
        }
    except Exception as exc:
        return {"symbol_1": symbol_1, "symbol_2": symbol_2, "error": str(exc)}


def _full_analysis_pair(r: dict) -> dict:
    """
    Complète un résultat de scan rapide avec rolling cointegration + Johansen.
    Appelé uniquement sur le top K.
    """
    log_s1 = np.log(r["s1"])
    log_s2 = np.log(r["s2"])

    log_df = pd.concat([log_s1, log_s2], axis=1)
    log_df.columns = [r["symbol_1"], r["symbol_2"]]

    johansen        = run_johansen(log_df)
    rolling_metrics = run_rolling_cointegration(log_s1, log_s2, ROLLING_WINDOW)
    rolling_latest  = rolling_metrics.iloc[-1]

    # Hedge ratio robuste (Huber M-estimator) pour les graphiques finaux
    beta_rlm, alpha_rlm, spread_rlm = (
        RLM(log_s1, add_constant(log_s2), M=sm_norms.HuberT())
        .fit()
        .params.pipe(lambda p: (p.iloc[1], p.iloc[0], log_s1 - (p.iloc[0] + p.iloc[1] * log_s2)))
    )
    r["beta"]   = beta_rlm
    r["alpha"]  = alpha_rlm
    r["spread"] = spread_rlm

    # ADF précis (autolag AIC) sur le spread robuste
    r["adf_s1"]     = run_adf(log_s1, f"log({r['symbol_1']})")
    r["adf_s2"]     = run_adf(log_s2, f"log({r['symbol_2']})")
    r["adf_spread"] = run_adf(spread_rlm, "Spread (résidus RLM)")

    r.update({
        "johansen":        johansen,
        "rolling_metrics": rolling_metrics,
        "rolling_latest":  rolling_latest,
    })
    return r


def plot_matrix_heatmap(symbols: list[str], results: list[dict], output_dir: Path):
    """
    Heatmap N×N symétrique des p-values Engle-Granger.
    Cellule (i, j) = min(EG_p(i→j), EG_p(j→i)) pour rendre la matrice symétrique.
    Diagonal = NaN.
    """
    n = len(symbols)
    sym_index = {s: i for i, s in enumerate(symbols)}

    eg_mat   = np.full((n, n), np.nan)
    adf_mat  = np.full((n, n), np.nan)
    score_mat = np.full((n, n), np.nan)

    for r in results:
        if r.get("error"):
            continue
        i = sym_index[r["symbol_1"]]
        j = sym_index[r["symbol_2"]]
        eg_p   = r["eg"]["pvalue"]
        adf_p  = r["adf_spread"]["pvalue"]
        sc     = r["score"]
        # Matrice symétrique : on prend la valeur existante ou la nouvelle, selon min
        for mat, val in [(eg_mat, eg_p), (adf_mat, adf_p), (score_mat, sc)]:
            if np.isnan(mat[i, j]):
                mat[i, j] = val
                mat[j, i] = val
            else:
                mat[i, j] = min(mat[i, j], val)
                mat[j, i] = min(mat[j, i], val)

    # Labels courts (BASE seulement)
    labels = [s.replace("/USDT", "") for s in symbols]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n // 2), max(10, n // 2)))
    fig.suptitle(
        f"Matrice de cointégration — Top {n} paires USDT\n"
        f"[{TIMEFRAME} / {DURATION} / fenêtre rolling {ROLLING_WINDOW}]",
        fontsize=13, fontweight="bold",
    )

    for ax, mat, title, cmap, vmax in [
        (axes[0], eg_mat,  "p-value Engle-Granger  (vert = cointégré)", "RdYlGn_r", 0.5),
        (axes[1], adf_mat, "p-value ADF spread       (vert = stationnaire)", "RdYlGn_r", 0.5),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=max(5, 9 - n // 10))
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=max(5, 9 - n // 10))
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Annotations uniquement si matrice raisonnable
        if n <= 30:
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    if np.isnan(val):
                        continue
                    txt = f"{val:.2f}" + ("*" if val < 0.05 else "")
                    norm_val = val / vmax
                    tc = "white" if norm_val > 0.55 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=max(4, 7 - n // 10), color=tc)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = output_dir / "matrix_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap matricielle sauvegardée : {out_path}")


def run_matrix_scan():
    """Teste toutes les combinaisons possibles parmi les top N symboles."""
    print(f"Récupération des top {TOP_N_SYMBOLS} paires USDT sur {EXCHANGE_ID}...")
    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_cls({"enableRateLimit": True})

    top_symbols = get_top_usdt_symbols(exchange, TOP_N_SYMBOLS)
    print(f"Symboles retenus : {len(top_symbols)}")

    # ── Phase 1 : téléchargement unique de tous les symboles ─────────────────
    # Un seul DataManager, une seule copie de chaque série en RAM.
    print(f"\nTéléchargement des données [{TIMEFRAME} / {DURATION}]...")
    dm = DataManager()
    close_data: dict[str, pd.Series] = {}
    failed_dl: list[str] = []

    for i, sym in enumerate(top_symbols, 1):
        print(f"  [{i:3d}/{len(top_symbols)}]  {sym}", end="  ")
        try:
            df = dm.fetch_symbol(sym, TIMEFRAME, DURATION, validate_symbol=False)
            close_data[sym] = df["close"]
            print(f"✓  {len(df)} bougies")
        except Exception as exc:
            print(f"✗  {exc}")
            failed_dl.append(sym)

    active_symbols = [s for s in top_symbols if s in close_data]
    if failed_dl:
        print(f"\n  {len(failed_dl)} symbole(s) ignoré(s) au téléchargement : {failed_dl}")

    # ── Phase 2 : scan rapide de toutes les paires (threads, mémoire partagée) ─
    pairs = list(itertools.combinations(active_symbols, 2))
    total = len(pairs)
    print(f"\nScan rapide : {total} combinaisons  ({len(active_symbols)} symboles)")
    print(f"  filtre corrélation ≥ {CORR_MIN}  |  ADF lag fixe  |  pas de rolling\n")

    scan_results = []

    def _worker(s1: str, s2: str) -> dict:
        return _fast_scan_pair(s1, close_data[s1], s2, close_data[s2])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, s1, s2): (s1, s2) for s1, s2 in pairs}
        for idx, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            scan_results.append(res)
            if idx % 50 == 0 or idx == total:
                print(f"  [{idx:4d}/{total}]  ...")

    ok_results  = [r for r in scan_results if not r.get("error")]
    err_results = [r for r in scan_results if r.get("error")]

    for r in ok_results:
        r["score"]       = score_result(r)
        r["significant"] = is_significant(r)

    ranked = sorted(ok_results, key=lambda r: (not r["significant"], r["score"]))
    top_k  = ranked[:TOP_K_RESULTS]

    # ── Résumé terminal ───────────────────────────────────────────────────────
    sig_count    = sum(1 for r in ok_results if r["significant"])
    filtered_out = sum(1 for r in err_results if "Corrélation" in r.get("error", ""))
    other_err    = len(err_results) - filtered_out
    print(f"\n{'=' * 72}")
    print(f"  {sig_count}/{len(ok_results)} paires cointégrées au seuil 5 %")
    print(f"  {filtered_out} paires éliminées par le filtre corrélation")
    print(f"  {other_err} erreur(s)")
    print(f"{'=' * 72}")
    print(f"\n  Top {TOP_K_RESULTS} paires (scan rapide) :\n")
    for i, r in enumerate(top_k, start=1):
        flag = "✅" if r["significant"] else "⚠️ "
        beta_cv  = r.get("beta_cv", 0.0)
        stab_flag = "⚠️ " if beta_cv > 1.0 else "  "
        print(
            f"  {i:>2}. {flag} {r['symbol_1']:12s} / {r['symbol_2']:12s} | "
            f"score={r['score']:.4f} | EG={r['eg']['pvalue']:.4f} | "
            f"ADF_sprd={r['adf_spread']['pvalue']:.4f} | corr={r['corr']:.2f} | "
            f"β_cv={beta_cv:.2f}{stab_flag}"
        )

    # ── Phase 3 : analyse complète du top K (rolling + Johansen) ─────────────
    print(f"\nAnalyse complète du top {TOP_K_RESULTS} (rolling + Johansen)...")
    for i, r in enumerate(top_k, 1):
        print(f"  [{i}/{TOP_K_RESULTS}]  {r['symbol_1']} / {r['symbol_2']}")
        _full_analysis_pair(r)

    # ── Sorties graphiques ────────────────────────────────────────────────────
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_matrix_heatmap(active_symbols, ok_results, output_dir)

    for i, r in enumerate(top_k, start=1):
        fname = (
            f"top_{i}_{_sanitize_symbol_for_filename(r['symbol_1'])}"
            f"__{_sanitize_symbol_for_filename(r['symbol_2'])}.png"
        )
        plot_results(
            r["s1"], r["s2"], r["spread"],
            r["adf_s1"], r["adf_s2"], r["adf_spread"],
            r["eg"], r["johansen"],
            r["beta"], r["alpha"],
            ROLLING_WINDOW, r["rolling_metrics"],
            output_png=str(output_dir / fname),
            symbol_1=r["symbol_1"],
            symbol_2=r["symbol_2"],
        )

    print(f"\nTerminé. Graphiques dans : {output_dir.resolve()}")
    if other_err:
        print(f"  {other_err} paire(s) en erreur :")
        for r in err_results:
            if "Corrélation" not in r.get("error", ""):
                print(f"    {r['symbol_1']} / {r['symbol_2']} — {r['error'][:60]}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_matrix_scan()
