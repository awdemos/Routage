use crate::domain::*;
use chrono::Utc;
use sqlx::{AnyPool, Row, migrate::MigrateDatabase};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbBackend {
    Sqlite,
    Postgres,
}

#[derive(Debug, Clone)]
pub struct Store {
    pool: AnyPool,
    backend: DbBackend,
}

impl Store {
    pub async fn new(db_url: &str) -> anyhow::Result<Self> {
        sqlx::any::install_default_drivers();
        let backend = if db_url.starts_with("sqlite:") {
            DbBackend::Sqlite
        } else if db_url.starts_with("postgres://") || db_url.starts_with("postgresql://") {
            DbBackend::Postgres
        } else {
            DbBackend::Sqlite
        };
        if backend == DbBackend::Sqlite {
            if let Ok(false) = sqlx::Sqlite::database_exists(db_url).await {
                sqlx::Sqlite::create_database(db_url).await?;
            }
        }
        let pool = AnyPool::connect(db_url).await?;
        Self::run_migrations(&pool).await?;
        Ok(Self { pool, backend })
    }

    async fn current_version(pool: &AnyPool) -> anyhow::Result<i32> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS _migrations (version INTEGER PRIMARY KEY)"
        )
        .execute(pool)
        .await?;
        let row = sqlx::query("SELECT COALESCE(MAX(version), 0) as v FROM _migrations")
            .fetch_one(pool)
            .await;
        match row {
            Ok(r) => Ok(r.try_get::<i32, _>("v")?),
            Err(_) => Ok(0),
        }
    }

    async fn bump_version(pool: &AnyPool, version: i32) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO _migrations (version) VALUES ($1) ON CONFLICT(version) DO NOTHING"
        )
        .bind(version)
        .execute(pool)
        .await?;
        Ok(())
    }

    async fn run_migrations(pool: &AnyPool) -> anyhow::Result<()> {
        let version = Self::current_version(pool).await.unwrap_or(0);

        if version < 1 {
            let stmts = [
                "CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL, agent_id TEXT NOT NULL DEFAULT 'default', prompt TEXT NOT NULL, features TEXT NOT NULL, candidate_models TEXT NOT NULL, selected_model TEXT NOT NULL, scores TEXT NOT NULL, created_at TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS inferences (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, model TEXT NOT NULL, provider_url TEXT, response_text TEXT NOT NULL, latency_ms INTEGER NOT NULL, tokens_used INTEGER, cost_usd REAL, error_kind TEXT, error_message TEXT, created_at TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, agent_id TEXT NOT NULL DEFAULT 'default', model TEXT NOT NULL, user_rating REAL, completion_success INTEGER, latency_ms INTEGER, tokens_used INTEGER, cost_usd REAL, metadata TEXT, created_at TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_decisions_tenant ON decisions(tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)",
            ];
            for stmt in &stmts {
                sqlx::query(stmt).execute(pool).await?;
            }
            Self::bump_version(pool, 1).await?;
        }

        if version < 2 {
            let stmts = [
                "CREATE TABLE IF NOT EXISTS model_failures (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, model_id TEXT NOT NULL, failure_kind TEXT NOT NULL, error_message TEXT NOT NULL, created_at TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS idx_failures_session ON model_failures(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_failures_model ON model_failures(model_id)",
                "CREATE TABLE IF NOT EXISTS model_health_snapshots (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, status TEXT NOT NULL, latency_ms INTEGER, error_message TEXT, created_at TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS idx_health_model ON model_health_snapshots(model_id)",
            ];
            for stmt in &stmts {
                sqlx::query(stmt).execute(pool).await?;
            }
            Self::bump_version(pool, 2).await?;
        }

        if version < 3 {
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS bandit_arms (tenant_id TEXT NOT NULL, agent_id TEXT NOT NULL, model_id TEXT NOT NULL, pulls INTEGER NOT NULL DEFAULT 0, total_reward REAL NOT NULL DEFAULT 0, sum_sq_reward REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (tenant_id, agent_id, model_id))"
            )
            .execute(pool)
            .await?;
            Self::bump_version(pool, 3).await?;
        }

        Ok(())
    }

    pub async fn log_decision(&self, log: &DecisionLog) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO decisions (id, session_id, tenant_id, agent_id, prompt, features, candidate_models, selected_model, scores, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            "#,
        )
        .bind(log.id.to_string())
        .bind(&log.session_id)
        .bind(&log.tenant_id)
        .bind(&log.agent_id)
        .bind(&log.prompt)
        .bind(serde_json::to_string(&log.features)?)
        .bind(serde_json::to_string(&log.candidate_models)?)
        .bind(&log.selected_model)
        .bind(serde_json::to_string(&log.scores)?)
        .bind(log.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn save_inference(&self, inf: &InferenceResult) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO inferences (id, session_id, model, provider_url, response_text, latency_ms, tokens_used, cost_usd, error_kind, error_message, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(&inf.session_id)
        .bind(&inf.model)
        .bind(&inf.provider_url)
        .bind(&inf.response_text)
        .bind(inf.latency_ms as i64)
        .bind(inf.tokens_used.map(|v| v as i64))
        .bind(inf.cost_usd)
        .bind(inf.error_kind.as_ref().map(|k| serde_json::to_string(k).unwrap_or_default()))
        .bind(&inf.error_message)
        .bind(inf.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn save_feedback(&self, fb: &FeedbackEvent) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO feedback (id, session_id, agent_id, model, user_rating, completion_success, latency_ms, tokens_used, cost_usd, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(&fb.session_id)
        .bind(fb.agent_id.as_deref().unwrap_or("default"))
        .bind(&fb.model)
        .bind(fb.user_rating)
        .bind(fb.completion_success.map(|b| if b { 1i64 } else { 0i64 }))
        .bind(fb.latency_ms.map(|v| v as i64))
        .bind(fb.tokens_used.map(|v| v as i64))
        .bind(fb.cost_usd)
        .bind(fb.metadata.as_ref().map(|m| serde_json::to_string(m).unwrap_or_default()))
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn save_failure(&self, failure: &ModelFailure) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO model_failures (id, session_id, model_id, failure_kind, error_message, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(failure.id.to_string())
        .bind(&failure.session_id)
        .bind(&failure.model_id)
        .bind(serde_json::to_string(&failure.failure_kind)?)
        .bind(&failure.error_message)
        .bind(failure.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn save_health_snapshot(
        &self,
        model_id: &str,
        status: HealthStatus,
        latency_ms: u64,
        error_message: Option<&str>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO model_health_snapshots (id, model_id, status, latency_ms, error_message, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(model_id)
        .bind(serde_json::to_string(&status)?)
        .bind(latency_ms as i64)
        .bind(error_message)
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetch average feedback per model for a tenant (simple offline signal).
    pub async fn model_performance(
        &self,
        tenant_id: &str,
    ) -> anyhow::Result<HashMap<String, f64>> {
        let rows = sqlx::query(
            r#"
            SELECT f.model, AVG(f.user_rating) as avg_rating
            FROM feedback f
            JOIN decisions d ON d.session_id = f.session_id
            WHERE d.tenant_id = $1 AND f.user_rating IS NOT NULL
            GROUP BY f.model
            "#,
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        let mut out = HashMap::new();
        for row in rows {
            let model: String = row.try_get("model")?;
            let avg: f64 = row.try_get("avg_rating")?;
            out.insert(model, avg);
        }
        Ok(out)
    }

    pub async fn get_decision_by_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<DecisionLog>> {
        let row = sqlx::query(
            r#"
            SELECT id, session_id, tenant_id, agent_id, prompt, features, candidate_models, selected_model, scores, created_at
            FROM decisions WHERE session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => {
                let scores_json: String = r.try_get("scores")?;
                let ranked_models: Vec<ModelScore> = serde_json::from_str(&scores_json)?;
                Ok(Some(DecisionLog {
                    id: Uuid::parse_str(&r.try_get::<String, _>("id")?).unwrap_or_else(|_| Uuid::new_v4()),
                    session_id: r.try_get("session_id")?,
                    tenant_id: r.try_get("tenant_id")?,
                    agent_id: r.try_get("agent_id")?,
                    prompt: r.try_get("prompt")?,
                    features: serde_json::from_str(&r.try_get::<String, _>("features")?).unwrap_or(serde_json::Value::Null),
                    candidate_models: serde_json::from_str(&r.try_get::<String, _>("candidate_models")?).unwrap_or(serde_json::Value::Null),
                    selected_model: r.try_get("selected_model")?,
                    scores: serde_json::to_value(&ranked_models)?,
                    created_at: r.try_get::<String, _>("created_at")?.parse().unwrap_or_else(|_| Utc::now()),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn get_performance(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let sql = match self.backend {
            DbBackend::Postgres => r#"
                SELECT
                    i.model,
                    COUNT(*) as total_requests,
                    AVG(i.latency_ms::FLOAT) as avg_latency_ms,
                    SUM(CASE WHEN i.error_kind IS NULL THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN i.error_kind IS NOT NULL THEN 1 ELSE 0 END) as failures,
                    AVG(f.user_rating::FLOAT) as avg_rating
                FROM inferences i
                LEFT JOIN feedback f ON f.session_id = i.session_id
                GROUP BY i.model
                ORDER BY total_requests DESC
            "#,
            DbBackend::Sqlite => r#"
                SELECT
                    i.model,
                    COUNT(*) as total_requests,
                    AVG(i.latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN i.error_kind IS NULL THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN i.error_kind IS NOT NULL THEN 1 ELSE 0 END) as failures,
                    AVG(f.user_rating) as avg_rating
                FROM inferences i
                LEFT JOIN feedback f ON f.session_id = i.session_id
                GROUP BY i.model
                ORDER BY total_requests DESC
            "#,
        };
        let rows = sqlx::query(sql)
            .fetch_all(&self.pool)
            .await?;

        let mut out = Vec::new();
        for row in rows {
            let total: i64 = row.try_get("total_requests")?;
            let successes: i64 = row.try_get("successes")?;
            let failures: i64 = row.try_get("failures")?;
            let success_rate = if total > 0 {
                (successes as f64 / total as f64) * 100.0
            } else {
                0.0
            };

            let avg_latency: Option<f64> = row.try_get("avg_latency_ms").ok();
            let avg_rating: Option<f64> = row.try_get("avg_rating").ok();
            out.push(serde_json::json!({
                "model": row.try_get::<String, _>("model")?,
                "total_requests": total,
                "avg_latency_ms": avg_latency.map(|v| (v * 100.0).round() / 100.0),
                "successes": successes,
                "failures": failures,
                "success_rate_percent": (success_rate * 100.0).round() / 100.0,
                "avg_rating": avg_rating.map(|v| (v * 1000.0).round() / 1000.0),
            }));
        }
        Ok(out)
    }

    pub async fn load_bandit_arms(
        &self,
    ) -> anyhow::Result<Vec<(String, String, String, ArmStats)>> {
        let rows = sqlx::query(
            r#"
            SELECT tenant_id, agent_id, model_id, pulls, total_reward, sum_sq_reward
            FROM bandit_arms
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::new();
        for row in rows {
            out.push((
                row.try_get("tenant_id")?,
                row.try_get("agent_id")?,
                row.try_get("model_id")?,
                ArmStats {
                    pulls: row.try_get::<i64, _>("pulls")? as u64,
                    total_reward: row.try_get::<f64, _>("total_reward")?,
                    sum_sq_reward: row.try_get::<f64, _>("sum_sq_reward")?,
                },
            ));
        }
        Ok(out)
    }

    pub async fn save_bandit_arm(
        &self,
        tenant_id: &str,
        agent_id: &str,
        model_id: &str,
        stats: &ArmStats,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO bandit_arms (tenant_id, agent_id, model_id, pulls, total_reward, sum_sq_reward, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT(tenant_id, agent_id, model_id) DO UPDATE SET
                pulls = excluded.pulls,
                total_reward = excluded.total_reward,
                sum_sq_reward = excluded.sum_sq_reward,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(tenant_id)
        .bind(agent_id)
        .bind(model_id)
        .bind(stats.pulls as i64)
        .bind(stats.total_reward)
        .bind(stats.sum_sq_reward)
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_explanation(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ExplainResponse>> {
        let decision_row = sqlx::query(
            r#"
            SELECT session_id, tenant_id, agent_id, prompt, features, candidate_models, selected_model, scores, created_at
            FROM decisions WHERE session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        let decision = match decision_row {
            Some(r) => r,
            None => return Ok(None),
        };

        let selected_model: String = decision.try_get("selected_model")?;
        let scores_json: String = decision.try_get("scores")?;
        let ranked_models: Vec<ModelScore> = serde_json::from_str(&scores_json)?;

        let feedback_rows = sqlx::query(
            r#"
            SELECT session_id, agent_id, model, user_rating, completion_success, latency_ms, tokens_used, cost_usd, metadata, created_at
            FROM feedback WHERE session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let feedback = feedback_rows.into_iter().next().map(|row| {
            let meta_str: Option<String> = row.try_get("metadata").ok();
            FeedbackEvent {
                session_id: row.try_get("session_id").unwrap_or_default(),
                agent_id: row.try_get("agent_id").ok(),
                model: row.try_get("model").unwrap_or_default(),
                user_rating: row.try_get("user_rating").ok(),
                completion_success: row.try_get::<i64, _>("completion_success").ok().map(|v| v == 1),
                latency_ms: row.try_get::<i64, _>("latency_ms").ok().map(|v| v as u64),
                tokens_used: row.try_get::<i64, _>("tokens_used").ok().map(|v| v as u64),
                cost_usd: row.try_get("cost_usd").ok(),
                metadata: meta_str.and_then(|s| serde_json::from_str(&s).ok()),
            }
        });

        let failure_rows = sqlx::query(
            r#"
            SELECT id, session_id, model_id, failure_kind, error_message, created_at
            FROM model_failures WHERE session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let mut failures = Vec::new();
        for row in failure_rows {
            let kind_str: String = row.try_get("failure_kind")?;
            let kind: FailureKind = serde_json::from_str(&kind_str).unwrap_or(FailureKind::Unknown);
            failures.push(ModelFailure {
                id: Uuid::parse_str(&row.try_get::<String, _>("id")?).unwrap_or_else(|_| Uuid::new_v4()),
                session_id: row.try_get("session_id")?,
                model_id: row.try_get("model_id")?,
                failure_kind: kind,
                error_message: row.try_get("error_message")?,
                created_at: row.try_get::<String, _>("created_at")?.parse().unwrap_or_else(|_| Utc::now()),
            });
        }

        let perf = self.model_performance("default").await.unwrap_or_default();
        let perf_summary = if perf.is_empty() {
            "No aggregated performance data available yet.".into()
        } else {
            perf.iter()
                .map(|(m, avg)| format!("{}: avg_rating={:.2}", m, avg))
                .collect::<Vec<_>>()
                .join("; ")
        };

        let explanation = format!(
            "Session {} routed to {} based on bandit scores with {} candidate(s).",
            session_id,
            selected_model,
            ranked_models.len()
        );

        Ok(Some(ExplainResponse {
            session_id: session_id.into(),
            selected_model,
            explanation,
            ranked_models,
            feedback,
            failures,
            performance_summary: perf_summary,
        }))
    }
}
