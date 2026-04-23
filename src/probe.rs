use crate::domain::{HealthStatus, ModelHealth, ModelSpec};
use crate::handlers::AppState;
use chrono::Utc;
use dashmap::DashMap;
use reqwest::StatusCode;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing;

pub type HealthMap = Arc<DashMap<String, ModelHealth>>;

pub fn build_health_map(models: &[(String, ModelSpec)]) -> HealthMap {
    let map = DashMap::new();
    for (id, spec) in models {
        map.insert(
            id.clone(),
            ModelHealth {
                model_id: id.clone(),
                provider_base_url: spec.provider_base_url.clone(),
                status: HealthStatus::Healthy,
                last_checked: Utc::now(),
                consecutive_failures: 0,
                last_error: None,
                probe_latency_ms: 0,
            },
        );
    }
    Arc::new(map)
}

async fn probe_once(url: &str, client: &reqwest::Client) -> Result<StatusCode, reqwest::Error> {
    // Try /health first (preferred for gateways like TensorZero).
    // For OpenAI-compatible endpoints at /openai/v1, check the root /health.
    let health_base = url.rsplit_once("/openai/v1").map(|(base, _)| base).unwrap_or(url);
    let health_url = format!("{}/health", health_base.trim_end_matches('/'));
    let health_res = client.get(&health_url).timeout(Duration::from_secs(10)).send().await;
    match health_res {
        Ok(r) if r.status().is_success() || r.status() == StatusCode::UNAUTHORIZED => return Ok(r.status()),
        _ => {}
    }

    // Fall back to /models for traditional OpenAI-compatible providers.
    let models_url = format!("{}/models", url.trim_end_matches('/'));
    let res = client.get(&models_url).timeout(Duration::from_secs(10)).send().await;
    match res {
        Ok(r) => Ok(r.status()),
        Err(e) => Err(e),
    }
}

fn classify_from_status(status: StatusCode) -> (HealthStatus, Option<String>) {
    if status.is_success() || status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        (HealthStatus::Healthy, None)
    } else if status == StatusCode::TOO_MANY_REQUESTS {
        (HealthStatus::Degraded, Some(format!("rate limited ({})" , status)))
    } else if status.is_server_error() {
        (HealthStatus::Offline, Some(format!("server error ({})" , status)))
    } else {
        (HealthStatus::Degraded, Some(format!("unexpected status ({})" , status)))
    }
}

pub async fn run_probes(
    state: AppState,
    interval: Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let mut tick = tokio::time::interval(interval);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                for (id, spec) in state.models.iter() {
                    let start = Instant::now();
                    let probe_res = probe_once(&spec.provider_base_url, &state.http).await;
                    let latency_ms = start.elapsed().as_millis() as u64;

                    let (mut status, error_msg) = match probe_res {
                        Ok(http_status) => classify_from_status(http_status),
                        Err(e) => (HealthStatus::Offline, Some(format!("probe failed: {}", e))),
                    };
                    let mut consecutive = 0u32;

                    state.health_map.entry(id.clone()).and_modify(|entry| {
                        entry.last_checked = Utc::now();
                        entry.probe_latency_ms = latency_ms;
                        if status == HealthStatus::Healthy {
                            entry.consecutive_failures = entry.consecutive_failures.saturating_sub(1);
                            if entry.consecutive_failures == 0 {
                                entry.status = HealthStatus::Healthy;
                            } else if entry.consecutive_failures <= 2 {
                                entry.status = HealthStatus::Degraded;
                            }
                        } else {
                            entry.consecutive_failures += 1;
                            if entry.consecutive_failures >= 3 {
                                entry.status = HealthStatus::Offline;
                            } else {
                                entry.status = HealthStatus::Degraded;
                            }
                        }
                        entry.last_error = error_msg.clone();
                        status = entry.status;
                        consecutive = entry.consecutive_failures;
                    });

                    let _ = state
                        .store
                        .save_health_snapshot(&id, status, latency_ms, error_msg.as_deref())
                        .await;

                    if status == HealthStatus::Offline {
                        tracing::warn!(
                            model = %id,
                            consecutive = consecutive,
                            "Model marked offline by health probe"
                        );
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("Health probe task shutting down");
                    break;
                }
            }
        }
    }
}
