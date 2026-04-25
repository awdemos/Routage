use crate::bandit::BanditEngine;
use crate::config::Config;
use crate::domain::*;
use crate::probe::HealthMap;
use crate::store::Store;
use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Event, Sse}},
    Json,
};
use chrono::Utc;
use futures::StreamExt;
use metrics::{counter, histogram};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub bandit: BanditEngine,
    pub models: Arc<Vec<(String, ModelSpec)>>,
    pub policies: Arc<HashMap<String, AgentPolicy>>,
    pub tenant_policies: Arc<HashMap<String, TenantPolicy>>,
    pub http: reqwest::Client,
    pub metrics_handle: Option<metrics_exporter_prometheus::PrometheusHandle>,
    pub health_map: HealthMap,
    pub config: Arc<Config>,
}

#[derive(Debug, serde::Serialize)]
pub struct ApiError {
    error: String,
    failure_kind: Option<FailureKind>,
}

impl ApiError {
    pub fn new(error: impl Into<String>, failure_kind: Option<FailureKind>) -> Self {
        Self {
            error: error.into(),
            failure_kind,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.failure_kind {
            Some(FailureKind::Timeout) | Some(FailureKind::ConnectionError) => StatusCode::GATEWAY_TIMEOUT,
            Some(FailureKind::RateLimited) => StatusCode::TOO_MANY_REQUESTS,
            Some(FailureKind::AuthError) => StatusCode::UNAUTHORIZED,
            Some(FailureKind::ServerError) => StatusCode::BAD_GATEWAY,
            Some(FailureKind::BadResponseFormat) => StatusCode::BAD_GATEWAY,
            Some(FailureKind::InvalidInput) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self {
            error: e.to_string(),
            failure_kind: None,
        }
    }
}

fn validate_prompt(prompt: &str, max_len: usize) -> Result<(), ApiError> {
    if prompt.len() > max_len {
        return Err(ApiError {
            error: format!("Prompt exceeds maximum length of {} characters", max_len),
            failure_kind: Some(FailureKind::InvalidInput),
        });
    }
    if prompt.trim().is_empty() {
        return Err(ApiError {
            error: "Prompt cannot be empty".into(),
            failure_kind: Some(FailureKind::InvalidInput),
        });
    }
    Ok(())
}

fn validate_candidates(candidates: &[String], max: usize) -> Result<(), ApiError> {
    if candidates.len() > max {
        return Err(ApiError {
            error: format!("Too many candidate models (max {})", max),
            failure_kind: Some(FailureKind::InvalidInput),
        });
    }
    Ok(())
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let mut models_online = 0usize;
    let mut models_offline = 0usize;
    for entry in state.health_map.iter() {
        match entry.status {
            HealthStatus::Healthy | HealthStatus::Degraded => models_online += 1,
            HealthStatus::Offline => models_offline += 1,
        }
    }
    Json(serde_json::json!({
        "status": "ok",
        "models_online": models_online,
        "models_offline": models_offline,
    }))
}

pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    match state.metrics_handle {
        Some(ref h) => (StatusCode::OK, h.render()),
        None => (StatusCode::SERVICE_UNAVAILABLE, "Metrics unavailable".to_string()),
    }
}

pub(crate) fn resolve_policy(
    policies: &HashMap<String, AgentPolicy>,
    tenant_policies: &HashMap<String, TenantPolicy>,
    tenant_id: &str,
    agent_id: Option<&str>,
) -> AgentPolicy {
    if let Some(aid) = agent_id {
        if let Some(ap) = policies.get(aid) {
            return ap.clone();
        }
    }
    if let Some(tp) = tenant_policies.get(tenant_id) {
        return AgentPolicy {
            agent_id: agent_id.unwrap_or("default").into(),
            tenant_id: tenant_id.into(),
            role: AgentRole::Generic,
            allowed_models: tp.allowed_models.clone(),
            blocked_models: tp.blocked_models.clone(),
            latency_budget_ms: tp.latency_budget_ms,
            cost_budget_usd: tp.cost_budget_usd,
            exploration_rate: tp.exploration_rate,
            quality_weight: tp.quality_weight,
            latency_weight: tp.latency_weight,
            cost_weight: tp.cost_weight,
        };
    }
    AgentPolicy {
        agent_id: agent_id.unwrap_or("default").into(),
        tenant_id: tenant_id.into(),
        ..Default::default()
    }
}

pub async fn route(
    State(state): State<AppState>,
    Json(req): Json<RouteRequest>,
) -> Result<Json<RouteResponse>, ApiError> {
    let start = Instant::now();

    validate_prompt(&req.prompt, state.config.max_prompt_length)?;
    validate_candidates(&req.candidate_models, state.config.max_candidate_models)?;

    let policy = resolve_policy(
        &state.policies,
        &state.tenant_policies,
        &req.tenant_id,
        req.agent_id.as_deref(),
    );

    let decision = state.bandit.route(&req, &state.models, &policy, &state.health_map);

    let log = DecisionLog {
        id: Uuid::new_v4(),
        session_id: decision.session_id.clone(),
        tenant_id: req.tenant_id.clone(),
        agent_id: req.agent_id.clone().unwrap_or_else(|| "default".into()),
        prompt: req.prompt.clone(),
        features: serde_json::to_value(BanditEngine::extract_features(&req))?,
        candidate_models: serde_json::to_value(&req.candidate_models)?,
        selected_model: decision.selected_model.clone(),
        scores: serde_json::to_value(&decision.ranked_models)?,
        created_at: Utc::now(),
    };
    let _ = state.store.log_decision(&log).await;

    counter!("routage.route_requests_total", "tenant" => req.tenant_id.clone(), "agent" => policy.agent_id.clone()).increment(1);
    histogram!("routage.route_latency_ms").record(start.elapsed().as_millis() as f64);

    Ok(Json(decision))
}

#[derive(serde::Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<crate::domain::ChatMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(serde::Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: MessageContent,
}

#[derive(serde::Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(serde::Deserialize)]
struct Usage {
    total_tokens: u64,
}

// Streaming structs
#[derive(serde::Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(serde::Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
}

#[derive(serde::Deserialize)]
struct ChunkDelta {
    content: Option<String>,
}

fn classify_reqwest_error(e: &reqwest::Error) -> FailureKind {
    if e.is_timeout() {
        FailureKind::Timeout
    } else if e.is_connect() {
        FailureKind::ConnectionError
    } else if e.is_decode() {
        FailureKind::BadResponseFormat
    } else {
        FailureKind::Unknown
    }
}

fn classify_http_status(status: StatusCode) -> FailureKind {
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        FailureKind::AuthError
    } else if status == StatusCode::TOO_MANY_REQUESTS {
        FailureKind::RateLimited
    } else if status.is_server_error() {
        FailureKind::ServerError
    } else {
        FailureKind::Unknown
    }
}

fn is_retryable(kind: Option<FailureKind>) -> bool {
    matches!(
        kind,
        Some(FailureKind::Timeout)
            | Some(FailureKind::RateLimited)
            | Some(FailureKind::ServerError)
            | Some(FailureKind::ConnectionError)
    )
}

fn build_request_builder(
    state: &AppState,
    req: &InferenceRequest,
    spec: &ModelSpec,
    provider_url: &str,
    stream: bool,
) -> reqwest::RequestBuilder {
    let body = ChatCompletionRequest {
        model: spec.provider_model_id.clone(),
        messages: vec![crate::domain::ChatMessage {
            role: "user".into(),
            content: req.prompt.clone(),
        }],
        max_tokens: 1024,
        stream,
    };

    let mut request_builder = state
        .http
        .post(format!("{}/chat/completions", provider_url.trim_end_matches('/')))
        .json(&body)
        .timeout(std::time::Duration::from_secs(60));

    if let Some(key) = state.config.provider_keys.get(&spec.provider) {
        let has_auth = req.provider_headers.as_ref().map_or(false, |h| {
            h.iter().any(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        });
        if !has_auth {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", key));
        }
    }

    if let Some(headers) = &req.provider_headers {
        for (k, v) in headers {
            request_builder = request_builder.header(k, v);
        }
    }

    request_builder
}

async fn execute_inference_once(
    state: &AppState,
    req: &InferenceRequest,
    model_id: &str,
    spec: &ModelSpec,
    provider_url: &str,
) -> Result<InferenceResult, ApiError> {
    let start = Instant::now();
    let request_builder = build_request_builder(state, req, spec, provider_url, false);
    let response = request_builder.send().await;
    let latency_ms = start.elapsed().as_millis() as u64;

    let mut result = InferenceResult {
        session_id: req.session_id.clone(),
        model: model_id.into(),
        provider_url: Some(provider_url.into()),
        response_text: String::new(),
        latency_ms,
        tokens_used: None,
        cost_usd: None,
        error_kind: None,
        error_message: None,
        created_at: Utc::now(),
    };

    match response {
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                let kind = classify_http_status(status);
                let text = resp.text().await.unwrap_or_default();
                let msg = format!("Provider returned {}: {}", status, text);
                result.error_kind = Some(kind);
                result.error_message = Some(msg.clone());
                state.store.save_inference(&result).await?;
                state
                    .store
                    .save_failure(&ModelFailure {
                        id: Uuid::new_v4(),
                        session_id: req.session_id.clone(),
                        model_id: model_id.into(),
                        failure_kind: kind,
                        error_message: msg.clone(),
                        created_at: Utc::now(),
                    })
                    .await?;
                return Err(ApiError {
                    error: msg,
                    failure_kind: Some(kind),
                });
            }

            let data: ChatCompletionResponse = match resp.json().await {
                Ok(d) => d,
                Err(e) => {
                    let msg = format!("Failed to parse provider response: {}", e);
                    result.error_kind = Some(FailureKind::ServerError);
                    result.error_message = Some(msg.clone());
                    state.store.save_inference(&result).await?;
                    state
                        .store
                        .save_failure(&ModelFailure {
                            id: Uuid::new_v4(),
                            session_id: req.session_id.clone(),
                            model_id: model_id.into(),
                            failure_kind: FailureKind::ServerError,
                            error_message: msg.clone(),
                            created_at: Utc::now(),
                        })
                        .await?;
                    return Err(ApiError {
                        error: msg,
                        failure_kind: Some(FailureKind::ServerError),
                    });
                }
            };

            result.response_text = data
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default();
            result.tokens_used = data.usage.map(|u| u.total_tokens);
            if let Some(tokens) = result.tokens_used {
                result.cost_usd = Some((tokens as f32 / 1000.0) * spec.default_cost_per_1k_tokens_usd);
            }

            state.store.save_inference(&result).await?;
        }
        Err(e) => {
            let kind = classify_reqwest_error(&e);
            let msg = format!("Request to provider failed: {}", e);
            result.error_kind = Some(kind);
            result.error_message = Some(msg.clone());
            state.store.save_inference(&result).await?;
            state
                .store
                .save_failure(&ModelFailure {
                    id: Uuid::new_v4(),
                    session_id: req.session_id.clone(),
                    model_id: model_id.into(),
                    failure_kind: kind,
                    error_message: msg.clone(),
                    created_at: Utc::now(),
                })
                .await?;
            return Err(ApiError {
                error: msg,
                failure_kind: Some(kind),
            });
        }
    }

    Ok(result)
}

async fn execute_inference_with_retry(
    state: &AppState,
    req: &InferenceRequest,
    model_id: &str,
    spec: &ModelSpec,
    provider_url: &str,
) -> Result<InferenceResult, ApiError> {
    let mut last_error = None;
    for attempt in 0..3 {
        if attempt > 0 {
            let backoff = std::time::Duration::from_millis(100 * 2_u64.pow(attempt - 1));
            tracing::info!(
                model = %model_id,
                attempt = attempt + 1,
                "Retrying inference after {:?}",
                backoff
            );
            tokio::time::sleep(backoff).await;
        }
        match execute_inference_once(state, req, model_id, spec, provider_url).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if !is_retryable(e.failure_kind) {
                    return Err(e);
                }
                last_error = Some(e);
            }
        }
    }
    Err(last_error.unwrap())
}

async fn try_start_stream(
    state: &AppState,
    req: &InferenceRequest,
    model_id: &str,
    spec: &ModelSpec,
    provider_url: &str,
) -> Result<reqwest::Response, ApiError> {
    let request_builder = build_request_builder(state, req, spec, provider_url, true);
    let response = request_builder.send().await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                let kind = classify_http_status(status);
                let text = resp.text().await.unwrap_or_default();
                let msg = format!("Provider returned {}: {}", status, text);
                state
                    .store
                    .save_failure(&ModelFailure {
                        id: Uuid::new_v4(),
                        session_id: req.session_id.clone(),
                        model_id: model_id.into(),
                        failure_kind: kind,
                        error_message: msg.clone(),
                        created_at: Utc::now(),
                    })
                    .await?;
                return Err(ApiError {
                    error: msg,
                    failure_kind: Some(kind),
                });
            }
            Ok(resp)
        }
        Err(e) => {
            let kind = classify_reqwest_error(&e);
            let msg = format!("Request to provider failed: {}", e);
            state
                .store
                .save_failure(&ModelFailure {
                    id: Uuid::new_v4(),
                    session_id: req.session_id.clone(),
                    model_id: model_id.into(),
                    failure_kind: kind,
                    error_message: msg.clone(),
                    created_at: Utc::now(),
                })
                .await?;
            Err(ApiError {
                error: msg,
                failure_kind: Some(kind),
            })
        }
    }
}

async fn proxy_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<Result<Event, Infallible>>,
    model_id: String,
    spec: ModelSpec,
    session_id: String,
    store: Store,
) {
    let start = Instant::now();
    let mut byte_stream = response.bytes_stream();
    let mut full_text = String::new();
    let mut buffer = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        match chunk_result {
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer.drain(..=pos).collect::<String>();
                    let line = line.trim();
                    if let Some(data) = line.strip_prefix("data: ") {
                        let data = data.trim();
                        if data == "[DONE]" {
                            break;
                        }
                        if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) {
                            if let Some(content) = chunk.choices.get(0).and_then(|c| c.delta.content.as_ref()) {
                                full_text.push_str(content);
                                let event = Event::default().data(
                                    serde_json::json!({
                                        "token": content,
                                        "model": &model_id,
                                        "session_id": &session_id,
                                        "done": false,
                                    }).to_string()
                                );
                                if tx.send(Ok(event)).await.is_err() {
                                    tracing::info!("Client disconnected from SSE stream");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                let event = Event::default().data(
                    serde_json::json!({
                        "error": format!("Stream error: {}", e),
                        "done": true,
                    }).to_string()
                );
                let _ = tx.send(Ok(event)).await;
                break;
            }
        }
    }

    let latency_ms = start.elapsed().as_millis() as u64;
    let tokens_used = Some(full_text.len() as u64 / 4);
    let cost_usd = tokens_used.map(|t| (t as f32 / 1000.0) * spec.default_cost_per_1k_tokens_usd);

    let _ = tx.send(Ok(Event::default().data(
        serde_json::json!({
            "done": true,
            "model": &model_id,
            "session_id": &session_id,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
        }).to_string()
    ))).await;

    let result = InferenceResult {
        session_id,
        model: model_id,
        provider_url: Some(spec.provider_base_url),
        response_text: full_text,
        latency_ms,
        tokens_used,
        cost_usd,
        error_kind: None,
        error_message: None,
        created_at: Utc::now(),
    };
    let _ = store.save_inference(&result).await;
}

async fn build_fallback_chain(
    state: &AppState,
    req: &InferenceRequest,
) -> Vec<(String, ModelSpec, String)> {
    let mut chain = Vec::new();

    // Primary model
    if let Some((_, spec)) = state.models.iter().find(|(id, _)| id == &req.model) {
        let url = req.provider_url.clone().unwrap_or_else(|| spec.provider_base_url.clone());
        chain.push((req.model.clone(), spec.clone(), url));
    }

    // Fallback models from request
    for fb in &req.fallback_models {
        if fb == &req.model { continue; }
        if let Some((_, spec)) = state.models.iter().find(|(id, _)| id == fb) {
            chain.push((fb.clone(), spec.clone(), spec.provider_base_url.clone()));
        }
    }

    // Fallback models from decision log
    if chain.len() <= 1 {
        if let Ok(Some(decision)) = state.store.get_decision_by_session(&req.session_id).await {
            if let Ok(ranked) = serde_json::from_value::<Vec<ModelScore>>(decision.scores) {
                for score in ranked {
                    if score.model == req.model { continue; }
                    if chain.iter().any(|(id, _, _)| id == &score.model) { continue; }
                    if let Some((_, spec)) = state.models.iter().find(|(id, _)| id == &score.model) {
                        chain.push((score.model.clone(), spec.clone(), spec.provider_base_url.clone()));
                    }
                }
            }
        }
    }

    chain
}

async fn infer_stream(
    state: AppState,
    req: InferenceRequest,
) -> Result<Response, ApiError> {
    let chain = build_fallback_chain(&state, &req).await;

    for (model_id, spec, url) in chain {
        tracing::info!(model = %model_id, "Attempting streaming inference");
        match try_start_stream(&state, &req, &model_id, &spec, &url).await {
            Ok(response) => {
                let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(128);
                tokio::spawn(proxy_stream(
                    response,
                    tx,
                    model_id.clone(),
                    spec,
                    req.session_id.clone(),
                    state.store.clone(),
                ));
                counter!("routage.inference_stream_requests_total", "model" => model_id).increment(1);
                return Ok(Sse::new(ReceiverStream::new(rx)).into_response());
            }
            Err(e) => {
                tracing::warn!(model = %model_id, error = %e.error, "Streaming model failed, trying fallback");
                if !req.allow_fallback {
                    return Err(e);
                }
            }
        }
    }

    Err(ApiError {
        error: "All models failed to establish streaming connection".into(),
        failure_kind: Some(FailureKind::ServerError),
    })
}

pub async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> Result<Response, ApiError> {
    validate_prompt(&req.prompt, state.config.max_prompt_length)?;

    if req.stream {
        infer_stream(state, req).await
    } else {
        let start = Instant::now();

        let primary_spec = state
            .models
            .iter()
            .find(|(id, _)| id == &req.model)
            .map(|(_, s)| s.clone());

        let primary_spec = match primary_spec {
            Some(s) => s,
            None => {
                return Err(ApiError {
                    error: format!("Unknown model: {}", req.model),
                    failure_kind: Some(FailureKind::InvalidInput),
                });
            }
        };

        let primary_url = req
            .provider_url
            .clone()
            .unwrap_or_else(|| primary_spec.provider_base_url.clone());

        let primary_result = execute_inference_with_retry(
            &state,
            &req,
            &req.model,
            &primary_spec,
            &primary_url,
        )
        .await;

        if primary_result.is_ok() || !req.allow_fallback {
            let result = primary_result?;
            counter!("routage.inference_requests_total", "model" => req.model.clone()).increment(1);
            histogram!("routage.inference_latency_ms").record(start.elapsed().as_millis() as f64);
            return Ok(Json(result).into_response());
        }

        tracing::warn!(
            session = %req.session_id,
            model = %req.model,
            "Primary model failed, attempting fallback"
        );

        let chain = build_fallback_chain(&state, &req).await;
        for (fallback_model, spec, url) in chain.into_iter().skip(1) {
            match execute_inference_with_retry(&state, &req, &fallback_model, &spec, &url).await {
                Ok(mut result) => {
                    result.model = fallback_model.clone();
                    counter!("routage.inference_fallback_success_total", "primary" => req.model.clone(), "fallback" => fallback_model.clone()).increment(1);
                    histogram!("routage.inference_latency_ms").record(start.elapsed().as_millis() as f64);
                    return Ok(Json(result).into_response());
                }
                Err(e) => {
                    tracing::warn!(
                        fallback_model = %fallback_model,
                        error = %e.error,
                        "Fallback model also failed"
                    );
                }
            }
        }

        counter!("routage.inference_fallback_exhausted_total", "primary" => req.model.clone()).increment(1);
        Err(ApiError {
            error: format!("Model {} failed and no fallback succeeded", req.model),
            failure_kind: Some(FailureKind::ServerError),
        })
    }
}

pub async fn feedback(
    State(state): State<AppState>,
    Json(fb): Json<FeedbackEvent>,
) -> Result<StatusCode, ApiError> {
    let mut reward = 0.5_f64;
    if let Some(rating) = fb.user_rating {
        reward = rating as f64;
    }
    if let Some(true) = fb.completion_success {
        reward += 0.2;
    } else if let Some(false) = fb.completion_success {
        reward -= 0.2;
    }
    if let Some(latency) = fb.latency_ms {
        if latency < 500 {
            reward += 0.1;
        } else if latency > 3000 {
            reward -= 0.1;
        }
    }
    reward = reward.clamp(0.0, 1.0);

    let agent_id = fb.agent_id.as_deref().unwrap_or("default");
    let stats = state.bandit.update("default", agent_id, &fb.model, reward);
    state.store.save_feedback(&fb).await?;
    let _ = state.store.save_bandit_arm("default", agent_id, &fb.model, &stats).await;

    counter!("routage.feedback_events_total", "model" => fb.model.clone(), "agent" => agent_id.to_string()).increment(1);

    Ok(StatusCode::NO_CONTENT)
}

pub async fn performance(State(state): State<AppState>) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let data = state.store.get_performance().await?;
    Ok(Json(data))
}

pub async fn explain(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<ExplainResponse>, ApiError> {
    let exp = state
        .store
        .get_explanation(&session_id)
        .await?;

    match exp {
        Some(e) => Ok(Json(e)),
        None => Err(ApiError {
            error: format!("Session {} not found", session_id),
            failure_kind: None,
        }),
    }
}

// ------------------------------------------------------------------
// OpenAI-compatible endpoints
// ------------------------------------------------------------------

pub async fn openai_models(
    State(state): State<AppState>,
) -> Result<Json<crate::domain::OpenAIModelList>, ApiError> {
    let data = state
        .models
        .iter()
        .map(|(id, spec)| crate::domain::OpenAIModel {
            id: id.clone(),
            object: "model".into(),
            created: 0,
            owned_by: spec.provider.clone(),
        })
        .collect();

    Ok(Json(crate::domain::OpenAIModelList {
        object: "list".into(),
        data,
    }))
}

fn extract_prompt_from_messages(messages: &serde_json::Value) -> String {
    messages
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let role = m.get("role")?.as_str()?;
                    let content = m.get("content")?.as_str()?;
                    Some(format!("{}: {}", role, content))
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

async fn select_model_for_openai(
    state: &AppState,
    messages: &serde_json::Value,
    requested_model: Option<&str>,
) -> Result<(String, crate::domain::ModelSpec), ApiError> {
    // If client requested a known model, honour it.
    if let Some(model_id) = requested_model {
        if let Some((_, spec)) = state.models.iter().find(|(id, _)| id == model_id) {
            return Ok((model_id.to_string(), spec.clone()));
        }
    }

    // Otherwise let the bandit decide.
    let prompt = extract_prompt_from_messages(messages);
    let req = crate::domain::RouteRequest {
        tenant_id: "default".into(),
        agent_id: None,
        prompt,
        max_latency_ms: None,
        max_cost_usd: None,
        tool_names: vec![],
        candidate_models: vec![],
    };

    let policy = resolve_policy(&state.policies, &state.tenant_policies, "default", None);
    let decision = state.bandit.route(&req, &state.models, &policy, &state.health_map);

    let spec = state
        .models
        .iter()
        .find(|(id, _)| id == &decision.selected_model)
        .map(|(_, s)| s.clone())
        .ok_or_else(|| ApiError {
            error: format!("Bandit selected unknown model: {}", decision.selected_model),
            failure_kind: Some(FailureKind::ServerError),
        })?;

    Ok((decision.selected_model.clone(), spec))
}

pub async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(mut body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let messages = body.get("messages").cloned().unwrap_or_else(|| serde_json::json!([]));
    let requested_model = body.get("model").and_then(|m| m.as_str());
    let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    let (model_id, spec) = select_model_for_openai(&state, &messages, requested_model).await?;

    // Replace the model field with the TensorZero model reference.
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), serde_json::json!(spec.provider_model_id));
    }

    let url = format!(
        "{}/chat/completions",
        spec.provider_base_url.trim_end_matches('/')
    );

    let mut builder = state.http.post(&url).json(&body);
    if let Some(key) = state.config.provider_keys.get(&spec.provider) {
        builder = builder.header("Authorization", format!("Bearer {}", key));
    }

    let start = std::time::Instant::now();
    let response = builder.timeout(std::time::Duration::from_secs(60)).send().await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            let ct = resp.headers().get("content-type").cloned();

            // Log decision so the bandit has history.
            let log = crate::domain::DecisionLog {
                id: uuid::Uuid::new_v4(),
                session_id: uuid::Uuid::new_v4().to_string(),
                tenant_id: "default".into(),
                agent_id: "default".into(),
                prompt: extract_prompt_from_messages(&messages),
                features: serde_json::to_value(crate::bandit::BanditEngine::extract_features(&crate::domain::RouteRequest {
                    tenant_id: "default".into(),
                    agent_id: None,
                    prompt: extract_prompt_from_messages(&messages),
                    max_latency_ms: None,
                    max_cost_usd: None,
                    tool_names: vec![],
                    candidate_models: vec![],
                })).unwrap_or_default(),
                candidate_models: serde_json::json!(state.models.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>()),
                selected_model: model_id.clone(),
                scores: serde_json::Value::Null,
                created_at: chrono::Utc::now(),
            };
            let _ = state.store.log_decision(&log).await;

            if stream {
                let body = axum::body::Body::from_stream(resp.bytes_stream());
                let mut rb = Response::builder().status(status);
                if let Some(ct) = ct {
                    rb = rb.header("content-type", ct);
                }
                Ok(rb.body(body).unwrap())
            } else {
                let bytes = resp.bytes().await.map_err(|e| ApiError {
                    error: format!("Failed to read provider response: {}", e),
                    failure_kind: Some(FailureKind::BadResponseFormat),
                })?;

                let latency_ms = start.elapsed().as_millis() as u64;
                let result = crate::domain::InferenceResult {
                    session_id: log.session_id.clone(),
                    model: model_id,
                    provider_url: Some(spec.provider_base_url),
                    response_text: String::new(), // Not parsed for proxy path
                    latency_ms,
                    tokens_used: None,
                    cost_usd: None,
                    error_kind: if status.is_success() { None } else { Some(FailureKind::ServerError) },
                    error_message: if status.is_success() { None } else { Some(format!("Provider returned {}", status)) },
                    created_at: chrono::Utc::now(),
                };
                let _ = state.store.save_inference(&result).await;

                Ok(Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .body(Body::from(bytes))
                    .unwrap())
            }
        }
        Err(e) => {
            let kind = classify_reqwest_error(&e);
            Err(ApiError {
                error: format!("Provider request failed: {}", e),
                failure_kind: Some(kind),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandit::BanditEngine;
    use crate::config::Config;
    use crate::probe::build_health_map;
    use crate::store::Store;
    use axum::body::to_bytes;
    use metrics_exporter_prometheus::PrometheusBuilder;

    async fn test_state() -> AppState {
        let recorder_handle = PrometheusBuilder::new().install_recorder().ok();
        let db_path = format!("/tmp/routage_test_{}.db", Uuid::new_v4());
        let store = Store::new(&format!("sqlite:{}", db_path)).await.unwrap();
        let config = Config::from_env().unwrap();
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

        AppState {
            store,
            bandit: BanditEngine::new(),
            models: Arc::new(models),
            policies: Arc::new(policies),
            tenant_policies: Arc::new(tenant_policies),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
            metrics_handle: recorder_handle,
            health_map: health_map.clone(),
            config: Arc::new(config),
        }
    }

    #[tokio::test]
    async fn test_health() {
        let state = test_state().await;
        let response = health(State(state.clone())).await;
        let body = to_bytes(response.into_response().into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_route() {
        let state = test_state().await;
        let req = RouteRequest {
            tenant_id: "default".into(),
            agent_id: None,
            prompt: "Hello".into(),
            max_latency_ms: None,
            max_cost_usd: None,
            tool_names: vec![],
            candidate_models: vec!["kimi-k2.5".into()],
        };
        let result = route(State(state.clone()), Json(req)).await.unwrap();
        assert_eq!(result.0.selected_model, "kimi-k2.5");
        assert!(!result.0.ranked_models.is_empty());
    }

    #[tokio::test]
    async fn test_feedback() {
        let state = test_state().await;
        let fb = FeedbackEvent {
            session_id: "test-session".into(),
            agent_id: None,
            model: "kimi-k2.5".into(),
            user_rating: Some(0.9),
            completion_success: Some(true),
            latency_ms: Some(500),
            tokens_used: Some(100),
            cost_usd: Some(0.001),
            metadata: None,
        };
        let result = feedback(State(state.clone()), Json(fb)).await.unwrap();
        assert_eq!(result, StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_performance() {
        let state = test_state().await;
        let result = performance(State(state.clone())).await.unwrap();
        assert!(result.0.is_empty());
    }
}
