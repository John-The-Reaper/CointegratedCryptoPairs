use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 0.0 || var_y <= 0.0 {
        return f64::NAN;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

#[pyfunction]
fn pct_change_corr(s1: PyReadonlyArray1<f64>, s2: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a = s1.as_slice()?;
    let b = s2.as_slice()?;

    if a.len() != b.len() {
        return Err(PyValueError::new_err("Series length mismatch"));
    }
    if a.len() < 2 {
        return Ok(f64::NAN);
    }

    let mut r1 = Vec::with_capacity(a.len() - 1);
    let mut r2 = Vec::with_capacity(b.len() - 1);

    for i in 1..a.len() {
        let p1 = a[i - 1];
        let p2 = b[i - 1];
        let c1 = a[i];
        let c2 = b[i];

        if p1 == 0.0 || p2 == 0.0 {
            continue;
        }

        let d1 = (c1 / p1) - 1.0;
        let d2 = (c2 / p2) - 1.0;
        if d1.is_finite() && d2.is_finite() {
            r1.push(d1);
            r2.push(d2);
        }
    }

    Ok(pearson_corr(&r1, &r2))
}

#[pyfunction]
fn winsorize_prices<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    z_thresh: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = prices.as_slice()?;
    if data.is_empty() {
        return Ok(PyArray1::from_vec(py, Vec::new()));
    }

    for &v in data {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err(
                "Prices must be finite and strictly positive",
            ));
        }
    }

    let n = data.len();
    let mut log_prices = Vec::with_capacity(n);
    for &v in data {
        log_prices.push(v.ln());
    }

    let mut log_ret = vec![0.0_f64; n];
    for i in 1..n {
        log_ret[i] = log_prices[i] - log_prices[i - 1];
    }

    let sample = &log_ret[1..];
    let m = sample.len() as f64;
    let mu = if m > 0.0 {
        sample.iter().sum::<f64>() / m
    } else {
        0.0
    };

    let sigma = if m > 0.0 {
        let var = sample.iter().map(|r| {
            let d = r - mu;
            d * d
        }).sum::<f64>() / m;
        var.sqrt()
    } else {
        0.0
    };

    let lower = mu - z_thresh * sigma;
    let upper = mu + z_thresh * sigma;
    for r in log_ret.iter_mut().skip(1) {
        if *r < lower {
            *r = lower;
        } else if *r > upper {
            *r = upper;
        }
    }

    let mut level = log_prices[0];
    let mut out = Vec::with_capacity(n);
    out.push(level.exp());
    for r in log_ret.iter().skip(1) {
        level += *r;
        out.push(level.exp());
    }

    Ok(PyArray1::from_vec(py, out))
}

#[pymodule]
fn cointegration_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pct_change_corr, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_prices, m)?)?;
    Ok(())
}
