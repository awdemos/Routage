mod auth;
mod bandit;
mod config;
mod domain;
mod handlers;
mod probe;
mod store;

use crate::auth::auth_and_trace_middleware;
use crate::bandit::BanditEngine;
use crate::config::Config;

use crate::handlers::{AppState, explain, feedback, health, infer, metrics_handler, openai_chat_completions, openai_models, performance, route};
use crate::probe::{build_health_map, run_probes};
use crate::store::Store;
use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "routage=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let recorder_handle = PrometheusBuilder::new().install_recorder().ok();

    let config = Config::from_env()?;
    let store = Store::new(&config.database_url).await?;

    let bandit = BanditEngine::new();
    match store.load_bandit_arms().await {
        Ok(arms) => {
            tracing::info!("Loaded {} bandit arms from database", arms.len());
            bandit.load(arms);
        }
        Err(e) => {
            tracing::warn!("Failed to load bandit arms: {}", e);
        }
    }

    let models = config.models.clone();
    let health_map = build_health_map(&models);

    let mut tenant_policies = HashMap::new();
    for tp in &config.tenant_policies {
        tenant_policies.insert(tp.tenant_id.clone(), tp.clone());
    }

    let mut policies = HashMap::new();
    for ap in &config.agent_policies {
        policies.insert(ap.agent_id.clone(), ap.clone());
    }

    let state = AppState {
        store,
        bandit,
        models: Arc::new(models),
        policies: Arc::new(policies),
        tenant_policies: Arc::new(tenant_policies),
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?,
        metrics_handle: recorder_handle,
        health_map: health_map.clone(),
        config: Arc::new(config.clone()),
    };

    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
    let (probe_shutdown_tx, probe_shutdown_rx) = tokio::sync::watch::channel(false);

    let probe_state = state.clone();
    let probe_interval = std::time::Duration::from_secs(config.probe_interval_secs);
    let probe_handle = tokio::spawn(async move {
        run_probes(probe_state, probe_interval, probe_shutdown_rx).await;
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/performance", get(performance))
        .route("/route", post(route))
        .route("/infer", post(infer))
        .route("/feedback", post(feedback))
        .route("/explain/:session_id", get(explain))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .layer(middleware::from_fn(auth_and_trace_middleware))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
        .with_state(state);

    let listener = TcpListener::bind(config.bind_addr).await?;
    tracing::info!("Routage listening on {}", config.bind_addr);

    let server = axum::serve(listener, app);

    let _shutdown_task = tokio::spawn(async move {
        let ctrl_c = async {
            let _ = signal::ctrl_c().await;
        };

        #[cfg(unix)]
        let terminate = async {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler");
            sigterm.recv().await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => tracing::info!("Received SIGINT, shutting down gracefully"),
            _ = terminate => tracing::info!("Received SIGTERM, shutting down gracefully"),
        }

        let _ = shutdown_tx.send(true);
        let _ = probe_shutdown_tx.send(true);
    });

    server.await?;

    tracing::info!("Server stopped, waiting for background tasks...");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), probe_handle).await;
    tracing::info!("Shutdown complete");

    Ok(())
}
