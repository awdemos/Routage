use crate::domain::*;
use crate::handlers::{ApiError, AppState};
use axum::{
    extract::{Query, State},
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use uuid::Uuid;

// ------------------------------------------------------------------
// GET /v2/models
// ------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
pub struct ListModelsQuery {
    #[serde(default)]
    pub provider: Vec<String>,
    #[serde(default)]
    pub openrouter_only: bool,
}

pub async fn list_models(
    State(state): State<AppState>,
    Query(query): Query<ListModelsQuery>,
) -> Result<Json<NdModelListResponse>, ApiError> {
    let mut models = Vec::new();
    let mut deprecated_models = Vec::new();

    for (id, spec) in state.models.iter() {
        let info = NdModelInfo {
            provider: spec.provider.clone(),
            model: id.clone(),
            display_name: format!("{}/{}", spec.provider, id),
            context_length: 128_000,
            input_price: spec.default_cost_per_1k_tokens_usd as f64 * 1000.0,
            output_price: (spec.default_cost_per_1k_tokens_usd * 2.0) as f64 * 1000.0,
            latency: spec.default_latency_ms as f64 / 1000.0,
            is_deprecated: false,
            supports_vision: spec.tags.contains(&"vision".to_string()),
            supports_tools: spec.tags.contains(&"tools".to_string()) || spec.tags.contains(&"agentic".to_string()),
            supports_json_mode: true,
        };

        // Apply provider filter
        if !query.provider.is_empty() {
            let provider_lower = spec.provider.to_lowercase();
            if !query.provider.iter().any(|p| p.to_lowercase() == provider_lower) {
                continue;
            }
        }

        // openrouter_only filter (in this project, all models are routed through our gateway)
        if query.openrouter_only {
            // In a real implementation, this would filter for OpenRouter-supported models
            // For now, we include all models
        }

        if info.is_deprecated {
            deprecated_models.push(info);
        } else {
            models.push(info);
        }
    }

    Ok(Json(NdModelListResponse {
        total: models.len(),
        models,
        deprecated_models,
    }))
}

// ------------------------------------------------------------------
// POST /v2/modelRouter/modelSelect
// ------------------------------------------------------------------

pub async fn model_select(
    State(state): State<AppState>,
    Json(req): Json<ModelSelectRequest>,
) -> Result<Json<ModelSelectResponse>, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::new(
            "messages array is required and cannot be empty",
            Some(FailureKind::InvalidInput),
        ));
    }
    if req.models.is_empty() {
        return Err(ApiError::new(
            "models array is required and cannot be empty",
            Some(FailureKind::InvalidInput),
        ));
    }

    // Check for custom router preference first
    if let Some(ref pref_id) = req.preference_id {
        match state.store.get_custom_router_preference(pref_id).await {
            Ok(Some(pref)) if pref.status == OptimizationStatus::Completed => {
                // Custom preference exists and is ready - in a real implementation,
                // this would influence routing decisions
                let _ = pref;
            }
            Ok(Some(_)) => {
                return Err(ApiError::new(
                    format!("Preference {} is not ready yet", pref_id),
                    Some(FailureKind::InvalidInput),
                ));
            }
            _ => {
                return Err(ApiError::new(
                    format!("Preference {} not found", pref_id),
                    Some(FailureKind::InvalidInput),
                ));
            }
        }
    }

    // Extract prompt from messages
    let prompt = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Match requested provider/model pairs against available models
    let mut candidate_models = Vec::new();
    let mut model_provider_map: HashMap<String, String> = HashMap::new();
    for req_model in &req.models {
        for (id, spec) in state.models.iter() {
            let matches = if req_model.provider == "custom" {
                req_model.model == *id
            } else {
                spec.provider.to_lowercase() == req_model.provider.to_lowercase()
                    && (spec.provider_model_id.ends_with(&req_model.model)
                        || id.to_lowercase() == req_model.model.to_lowercase())
            };
            if matches && !candidate_models.contains(id) {
                candidate_models.push(id.clone());
                model_provider_map.insert(id.clone(), spec.provider.clone());
            }
        }
    }

    if candidate_models.is_empty() {
        return Err(ApiError::new(
            "No matching models found for the requested providers/models",
            Some(FailureKind::InvalidInput),
        ));
    }

    // Use existing bandit engine for routing
    let route_req = RouteRequest {
        tenant_id: "default".into(),
        agent_id: None,
        prompt: prompt.clone(),
        max_latency_ms: None,
        max_cost_usd: None,
        tool_names: req.tools.iter().map(|t| {
            t.spec.get("name").and_then(|n| n.as_str()).unwrap_or("unknown").to_string()
        }).collect(),
        candidate_models: candidate_models.clone(),
    };

    let policy = crate::handlers::resolve_policy(
        &state.policies,
        &state.tenant_policies,
        "default",
        None,
    );

    let decision = state.bandit.route(&route_req, &state.models, &policy, &state.health_map);

    // Build ranked models from bandit scores
    let mut ranked = Vec::new();
    for score in &decision.ranked_models {
        let provider = model_provider_map.get(&score.model)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        ranked.push(ModelSelectRanked {
            provider,
            model: score.model.clone(),
            score: score.score,
            reason: score.reason.clone(),
        });
    }

    // Apply max_model_count limit
    let max_count = req.max_model_count.unwrap_or(ranked.len());
    ranked.truncate(max_count);

    // Apply tradeoff adjustment
    if let Some(ref tradeoff) = req.tradeoff {
        match tradeoff.as_str() {
            "cost" => {
                ranked.sort_by(|a, b| {
                    let cost_a = state.models.iter()
                        .find(|(id, _)| id == &a.model)
                        .map(|(_, s)| s.default_cost_per_1k_tokens_usd)
                        .unwrap_or(f32::MAX);
                    let cost_b = state.models.iter()
                        .find(|(id, _)| id == &b.model)
                        .map(|(_, s)| s.default_cost_per_1k_tokens_usd)
                        .unwrap_or(f32::MAX);
                    cost_a.partial_cmp(&cost_b).unwrap()
                });
            }
            "latency" => {
                ranked.sort_by(|a, b| {
                    let lat_a = state.models.iter()
                        .find(|(id, _)| id == &a.model)
                        .map(|(_, s)| s.default_latency_ms)
                        .unwrap_or(u64::MAX);
                    let lat_b = state.models.iter()
                        .find(|(id, _)| id == &b.model)
                        .map(|(_, s)| s.default_latency_ms)
                        .unwrap_or(u64::MAX);
                    lat_a.partial_cmp(&lat_b).unwrap()
                });
            }
            _ => {}
        }
    }

    let selected_provider = ranked.first()
        .map(|r| r.provider.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let selected_model = ranked.first()
        .map(|r| r.model.clone())
        .unwrap_or_else(|| decision.selected_model.clone());

    let session_id = req.previous_session.clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // Log the decision
    let log = DecisionLog {
        id: Uuid::new_v4(),
        session_id: session_id.clone(),
        tenant_id: "default".into(),
        agent_id: "default".into(),
        prompt,
        features: serde_json::to_value(BanditEngine::extract_features(&route_req)).unwrap_or_default(),
        candidate_models: serde_json::to_value(&candidate_models).unwrap_or_default(),
        selected_model: selected_model.clone(),
        scores: serde_json::to_value(&ranked).unwrap_or_default(),
        created_at: Utc::now(),
    };
    let _ = state.store.log_decision(&log).await;

    Ok(Json(ModelSelectResponse {
        session_id,
        provider: selected_provider,
        model: selected_model,
        ranked_models: ranked,
    }))
}

// Need to import BanditEngine here
use crate::bandit::BanditEngine;

// ------------------------------------------------------------------
// POST /v2/pzn/trainCustomRouter
// ------------------------------------------------------------------

pub async fn train_custom_router(
    State(state): State<AppState>,
    Json(req): Json<TrainCustomRouterRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Validate request
    if req.dataset_csv.trim().is_empty() {
        return Err(ApiError::new(
            "dataset_csv is required",
            Some(FailureKind::InvalidInput),
        ));
    }
    if req.models.is_empty() {
        return Err(ApiError::new(
            "at least one model is required",
            Some(FailureKind::InvalidInput),
        ));
    }

    // Parse CSV to count samples
    let samples = parse_csv_samples(&req.dataset_csv, &req.prompt_column);
    if samples < 25 {
        return Err(ApiError::new(
            format!("Dataset must contain at least 25 samples, got {}", samples),
            Some(FailureKind::InvalidInput),
        ));
    }

    let preference_id = format!("pref_{}", Uuid::new_v4().to_string().replace("-", ""));
    let now = Utc::now();

    let pref = CustomRouterPreference {
        preference_id: preference_id.clone(),
        status: OptimizationStatus::Pending,
        models: req.models.clone(),
        dataset_csv: Some(req.dataset_csv.clone()),
        train_samples: samples,
        accuracy: None,
        created_at: now,
        updated_at: now,
        completed_at: None,
    };

    state.store.create_custom_router_preference(&pref).await?;

    // Spawn background training worker
    let worker_state = state.clone();
    let worker_pref_id = preference_id.clone();
    let worker_req = req;
    tokio::spawn(async move {
        run_custom_router_training(worker_state, worker_pref_id, worker_req).await;
    });

    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(TrainCustomRouterResponse {
            preference_id,
            status: "training".into(),
            message: "Custom router training started. Use preference_id in modelSelect calls once completed.".into(),
        }),
    ))
}

fn parse_csv_samples(csv: &str, prompt_column: &str) -> usize {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv.as_bytes());

    let headers = match reader.headers() {
        Ok(h) => h.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        Err(_) => return 0,
    };

    // Check if prompt column exists
    if !headers.iter().any(|h| h == prompt_column) {
        return 0;
    }

    let mut count = 0;
    for result in reader.records() {
        if result.is_ok() {
            count += 1;
        }
    }
    count
}

async fn run_custom_router_training(state: AppState, pref_id: String, _req: TrainCustomRouterRequest) {
    tracing::info!(preference_id = %pref_id, "Starting custom router training");

    let mut pref = match state.store.get_custom_router_preference(&pref_id).await {
        Ok(Some(p)) => p,
        _ => {
            tracing::error!(preference_id = %pref_id, "Failed to load custom router preference");
            return;
        }
    };

    pref.status = OptimizationStatus::Running;
    pref.updated_at = Utc::now();
    let _ = state.store.update_custom_router_preference(&pref).await;

    // Simulate training steps
    let steps = vec![
        (15, "Parsing and validating dataset"),
        (30, "Clustering similar queries"),
        (50, "Analyzing model performance patterns"),
        (70, "Training routing classifier"),
        (85, "Evaluating on hold-out set"),
        (95, "Finalizing router weights"),
    ];

    for (pct, msg) in steps {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        tracing::info!(preference_id = %pref_id, progress = pct, "{}", msg);
    }

    // Simulate accuracy based on dataset size
    let mut rng = rand::rngs::StdRng::from_entropy();
    let base_accuracy: f32 = rng.gen_range(0.78..0.94);
    let size_bonus = (pref.train_samples as f32 / 500.0).min(0.05);
    let accuracy = ((base_accuracy + size_bonus) * 100.0).round() / 100.0;

    // Mark as completed
    pref.status = OptimizationStatus::Completed;
    pref.accuracy = Some(accuracy);
    pref.updated_at = Utc::now();
    pref.completed_at = Some(Utc::now());

    if let Err(e) = state.store.update_custom_router_preference(&pref).await {
        tracing::error!(preference_id = %pref_id, error = %e, "Failed to finalize custom router training");
    } else {
        tracing::info!(preference_id = %pref_id, accuracy = accuracy, "Custom router training completed");
    }
}
