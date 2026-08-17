--
-- TimescaleDB Schema — trading-core
--
-- 4 tablas:
--   bars          : hypertable, datos de entrenamiento (DRB + features)
--   studies       : metadatos de optimización bayesiana
--   failed_days   : días que fallaron en cada estudio
--   positioning   : funding rates de Binance Futures (separado de bars)
--

-- ============================================================
-- Tabla 1: bars (hypertable)
-- ============================================================

CREATE TABLE IF NOT EXISTS bars (
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,
    symbol VARCHAR(10) NOT NULL,

    -- OHLCV
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    n_ticks BIGINT,
    volume DOUBLE PRECISION,
    dollar_value DOUBLE PRECISION,

    -- Hyperparams aplicados (denormalizado para queries rápidas)
    exp_lambda DOUBLE PRECISION,
    init_exp_t INTEGER,
    study_source VARCHAR(50),
    completion_rate DOUBLE PRECISION,
    failed_day SMALLINT DEFAULT 0,
    sample_weight DOUBLE PRECISION DEFAULT 1.0,

    -- Features (15 columnas: 14 bar-level + 1 positioning)
    log_return DOUBLE PRECISION,
    frac_diff_return DOUBLE PRECISION,
    rolling_volatility DOUBLE PRECISION,
    bar_range DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    volume_z DOUBLE PRECISION,
    dollar_value_z DOUBLE PRECISION,
    n_ticks_z DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    price_to_vwap DOUBLE PRECISION,
    bar_duration_secs DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    bb_pct_b DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    funding_rate_mean DOUBLE PRECISION
);

-- Convertir en hypertable (chunk mensual)
SELECT create_hypertable('bars', 'open_time', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- Índices para queries frecuentes
CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars (symbol, open_time DESC);
CREATE INDEX IF NOT EXISTS idx_bars_study ON bars (study_source);

-- ============================================================
-- Tabla 2: studies (metadatos de optimización)
-- ============================================================

CREATE TABLE IF NOT EXISTS studies (
    symbol VARCHAR(10) NOT NULL,
    period VARCHAR(7) NOT NULL,           -- '2024-03'
    window_name VARCHAR(30) NOT NULL,     -- 'bayesian_2024_03_w2m'

    -- Best config
    exp_lambda DOUBLE PRECISION,
    init_exp_t INTEGER,
    composite_score DOUBLE PRECISION,
    quality_component DOUBLE PRECISION,
    coverage_component DOUBLE PRECISION,
    stability_component DOUBLE PRECISION,
    granularity_component DOUBLE PRECISION,

    -- Estadísticas del estudio
    total_trials INTEGER,
    completed_trials INTEGER,
    pruned_trials INTEGER,
    completion_rate DOUBLE PRECISION,
    avg_bars_per_day DOUBLE PRECISION,

    -- Search bounds usados
    t_min INTEGER,
    t_max INTEGER,
    lambda_min DOUBLE PRECISION,
    lambda_max DOUBLE PRECISION,
    sampler_type VARCHAR(20),
    device VARCHAR(10),

    -- YAML completo (search_bounds crudo como JSONB)
    search_bounds_json JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, period)
);

CREATE INDEX IF NOT EXISTS idx_studies_symbol ON studies (symbol, period);
CREATE INDEX IF NOT EXISTS idx_studies_score ON studies (composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_studies_window ON studies (window_name);

-- ============================================================
-- Tabla 3: failed_days (análisis de días fallidos)
-- ============================================================

CREATE TABLE IF NOT EXISTS failed_days (
    symbol VARCHAR(10) NOT NULL,
    period VARCHAR(7) NOT NULL,
    failed_date DATE NOT NULL,
    total_trials INTEGER,
    insufficient_bars INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    reasons TEXT[],

    PRIMARY KEY (symbol, period, failed_date),
    FOREIGN KEY (symbol, period) REFERENCES studies(symbol, period) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_failed_days_period ON failed_days (symbol, period);

-- ============================================================
-- Tabla 4: positioning (funding rates de Binance Futures)
-- ============================================================

CREATE TABLE IF NOT EXISTS positioning (
    symbol VARCHAR(10) NOT NULL,
    funding_time TIMESTAMPTZ NOT NULL,
    funding_rate DOUBLE PRECISION,
    mark_price DOUBLE PRECISION,

    PRIMARY KEY (symbol, funding_time)
);

CREATE INDEX IF NOT EXISTS idx_positioning_symbol ON positioning (symbol, funding_time DESC);

-- ============================================================
-- Vistas útiles
-- ============================================================

-- Vista: barras + positioning (JOIN pre-computado)
CREATE OR REPLACE VIEW bars_with_positioning AS
SELECT
    b.*,
    p.funding_rate,
    p.mark_price
FROM bars b
LEFT JOIN positioning p
    ON b.symbol = p.symbol
    AND p.funding_time <= b.close_time
    AND p.funding_time > b.close_time - INTERVAL '8 hours'
ORDER BY b.open_time;

-- Vista: estudios con resumen de calidad
CREATE OR REPLACE VIEW study_quality_summary AS
SELECT
    s.symbol,
    s.period,
    s.window_name,
    s.composite_score,
    s.completion_rate,
    COUNT(DISTINCT f.failed_date) AS n_failed_days,
    AVG(b.bar_duration_secs) AS avg_bar_duration,
    COUNT(*)::INT AS total_bars
FROM studies s
LEFT JOIN failed_days f ON s.symbol = f.symbol AND s.period = f.period
LEFT JOIN bars b ON b.study_source = s.window_name AND b.symbol = s.symbol
GROUP BY s.symbol, s.period, s.window_name, s.composite_score, s.completion_rate;
