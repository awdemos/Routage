use crate::domain::*;
use crate::handlers::{ApiError, AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

// ------------------------------------------------------------------
// Validation
// ------------------------------------------------------------------

fn validate_optimize_request(req: &PromptOptimizeRequest) -> Result<(), ApiError> {
    if req.system_prompt.trim().is_empty() {
        return Err(ApiError::new("system_prompt is required", Some(FailureKind::InvalidInput)));
    }
    if req.template.trim().is_empty() {
        return Err(ApiError::new("template is required", Some(FailureKind::InvalidInput)));
    }
    if req.fields.is_empty() {
        return Err(ApiError::new("fields are required", Some(FailureKind::InvalidInput)));
    }
    if req.target_models.is_empty() {
        return Err(ApiError::new("at least one target_model is required", Some(FailureKind::InvalidInput)));
    }

    // Check goldens vs train_goldens/test_goldens
    let has_legacy = !req.goldens.is_empty();
    let has_split = !req.train_goldens.is_empty() || !req.test_goldens.is_empty();

    if has_legacy && has_split {
        return Err(ApiError::new(
            "Cannot use both 'goldens' and 'train_goldens/test_goldens'. Use one or the other.",
            Some(FailureKind::InvalidInput),
        ));
    }

    let train_count = if has_legacy {
        req.goldens.len()
    } else {
        req.train_goldens.len()
    };

    let min_required = if req.prototype_mode { 3 } else { 25 };

    if train_count < min_required {
        return Err(ApiError::new(
            format!(
                "Insufficient training examples. Got {}, minimum required is {} (or enable prototype_mode for {})",
                train_count,
                if req.prototype_mode { 3 } else { 25 },
                3
            ),
            Some(FailureKind::InvalidInput),
        ));
    }

    if has_split && req.test_goldens.is_empty() {
        return Err(ApiError::new(
            "test_goldens is required when train_goldens is provided",
            Some(FailureKind::InvalidInput),
        ));
    }

    Ok(())
}

// ------------------------------------------------------------------
// POST /v2/prompt/optimize
// ------------------------------------------------------------------

pub async fn optimize(
    State(state): State<AppState>,
    Json(body): Json<PromptOptimizeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    validate_optimize_request(&body)?;

    let run_id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let run = OptimizationRun {
        id: run_id.clone(),
        status: OptimizationStatus::Pending,
        progress_percent: 0,
        message: Some("Optimization queued".into()),
        request_json: serde_json::to_string(&body)?,
        results_json: None,
        costs_json: None,
        created_at: now,
        updated_at: now,
        completed_at: None,
    };

    state.store.create_optimization_run(&run).await?;

    // Spawn background optimization worker
    let worker_state = state.clone();
    let worker_run_id = run_id.clone();
    let worker_req = body;
    tokio::spawn(async move {
        run_optimization(worker_state, worker_run_id, worker_req).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "optimization_run_id": run_id,
            "status": "pending",
            "message": "Optimization started. Poll /v2/prompt/optimizeStatus/{id} for progress."
        })),
    ))
}

// ------------------------------------------------------------------
// GET /v2/prompt/optimizeStatus/{id}
// ------------------------------------------------------------------

pub async fn optimize_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<PromptOptimizeStatusResponse>, ApiError> {
    let run = state
        .store
        .get_optimization_run(&id)
        .await?
        .ok_or_else(|| ApiError::new(format!("Optimization run {} not found", id), None))?;

    Ok(Json(PromptOptimizeStatusResponse {
        optimization_run_id: run.id,
        status: run.status,
        progress_percent: run.progress_percent,
        message: run.message,
        created_at: run.created_at,
        updated_at: run.updated_at,
        estimated_completion_at: if run.status == OptimizationStatus::Running {
            Some(run.created_at + chrono::Duration::minutes(20))
        } else {
            None
        },
    }))
}

// ------------------------------------------------------------------
// GET /v2/prompt/optimizeResults/{id}
// ------------------------------------------------------------------

pub async fn optimize_results(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<PromptOptimizeResultsResponse>, ApiError> {
    let run = state
        .store
        .get_optimization_run(&id)
        .await?
        .ok_or_else(|| ApiError::new(format!("Optimization run {} not found", id), None))?;

    if run.status != OptimizationStatus::Completed {
        return Err(ApiError::new(
            format!(
                "Optimization run {} is not completed yet (status: {:?})",
                id, run.status
            ),
            Some(FailureKind::InvalidInput),
        ));
    }

    let results: PromptOptimizeResultsResponse = match run.results_json {
        Some(json) => serde_json::from_str(&json).map_err(|e| ApiError::new(
            format!("Failed to parse results: {}", e),
            None,
        ))?,
        None => {
            return Err(ApiError::new("Results not available", None))
        }
    };

    Ok(Json(results))
}

// ------------------------------------------------------------------
// GET /v2/prompt/optimize/{id}/costs
// ------------------------------------------------------------------

pub async fn optimize_costs(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<OptimizationCostBreakdown>, ApiError> {
    let run = state
        .store
        .get_optimization_run(&id)
        .await?
        .ok_or_else(|| ApiError::new(format!("Optimization run {} not found", id), None))?;

    if run.status != OptimizationStatus::Completed {
        return Err(ApiError::new(
            format!(
                "Optimization run {} is not completed yet (status: {:?})",
                id, run.status
            ),
            Some(FailureKind::InvalidInput),
        ));
    }

    let costs: OptimizationCostBreakdown = match run.costs_json {
        Some(json) => serde_json::from_str(&json).map_err(|e| ApiError::new(
            format!("Failed to parse costs: {}", e),
            None,
        ))?,
        None => OptimizationCostBreakdown {
            optimization_run_id: id.clone(),
            total_tokens_used: 0,
            total_cost_usd: 0.0,
            model_costs: vec![],
        },
    };

    Ok(Json(costs))
}

// ------------------------------------------------------------------
// Background Optimization Worker
// ------------------------------------------------------------------

async fn run_optimization(state: AppState, run_id: String, req: PromptOptimizeRequest) {
    tracing::info!(run_id = %run_id, "Starting prompt optimization worker");

    // Mark as running
    let mut run = match state.store.get_optimization_run(&run_id).await {
        Ok(Some(r)) => r,
        _ => {
            tracing::error!(run_id = %run_id, "Failed to load optimization run");
            return;
        }
    };

    run.status = OptimizationStatus::Running;
    run.progress_percent = 5;
    run.message = Some("Initializing optimization pipeline".into());
    run.updated_at = Utc::now();
    let _ = state.store.update_optimization_run(&run).await;

    // Simulate optimization steps with progress updates
    let steps = vec![
        (10, "Analyzing origin model prompt performance"),
        (20, "Generating prompt variants for target models"),
        (35, "Running evaluation on training examples"),
        (50, "Scoring prompt variants"),
        (65, "Evaluating on test set"),
        (80, "Selecting best prompts per target model"),
        (90, "Compiling final results"),
        (95, "Finalizing"),
    ];

    for (pct, msg) in steps {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        run.progress_percent = pct;
        run.message = Some(msg.into());
        run.updated_at = Utc::now();
        if let Err(e) = state.store.update_optimization_run(&run).await {
            tracing::warn!(run_id = %run_id, error = %e, "Failed to update optimization progress");
        }
    }

    // Build simulated results
    let train_count = if !req.goldens.is_empty() {
        req.goldens.len()
    } else {
        req.train_goldens.len()
    };
    let test_count = if !req.goldens.is_empty() {
        0
    } else {
        req.test_goldens.len()
    };

    let mut optimized_prompts = Vec::new();
    let mut rng = rand::rngs::StdRng::from_entropy();

    for target in &req.target_models {
        // Simulate a tailored system prompt and template
        let adapted_system = adapt_prompt_for_model(&req.system_prompt, &target.model);
        let adapted_template = adapt_template_for_model(&req.template, &target.model);

        // Simulate score: base around 0.75-0.95 with some variance
        let base_score: f32 = rng.gen_range(0.75..0.95);
        let score = (base_score * 100.0).round() / 100.0;

        let improvement = if req.origin_model_evaluation_score.is_some() {
            let origin = req.origin_model_evaluation_score.unwrap();
            Some(((score - origin) / origin * 100.0 * 100.0).round() / 100.0)
        } else {
            None
        };

        optimized_prompts.push(OptimizedPrompt {
            provider: target.provider.clone(),
            model: target.model.clone(),
            system_prompt: adapted_system,
            template: adapted_template,
            score,
            improvement_percent: improvement,
        });
    }

    // Sort by score descending
    optimized_prompts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let results = PromptOptimizeResultsResponse {
        optimization_run_id: run_id.clone(),
        status: OptimizationStatus::Completed,
        origin_model: req.origin_model.clone(),
        origin_model_score: req.origin_model_evaluation_score,
        optimized_prompts,
        train_examples_used: train_count,
        test_examples_used: test_count,
        completed_at: Some(Utc::now()),
    };

    // Build simulated costs
    let mut model_costs = Vec::new();
    let mut total_tokens = 0u64;
    let mut total_cost = 0.0f32;

    // Cost for origin model evaluation (if not provided)
    if req.origin_model_evaluation_score.is_none() {
        if let Some(ref origin) = req.origin_model {
            let tokens = train_count as u64 * 150;
            let cost = tokens as f32 / 1000.0 * 0.003;
            model_costs.push(ModelCost {
                provider: origin.provider.clone(),
                model: origin.model.clone(),
                tokens_used: tokens,
                cost_usd: cost,
            });
            total_tokens += tokens;
            total_cost += cost;
        }
    }

    // Cost for each target model optimization
    for target in &req.target_models {
        let tokens = (train_count + test_count) as u64 * 200;
        let cost = tokens as f32 / 1000.0 * 0.005;
        model_costs.push(ModelCost {
            provider: target.provider.clone(),
            model: target.model.clone(),
            tokens_used: tokens,
            cost_usd: cost,
        });
        total_tokens += tokens;
        total_cost += cost;
    }

    let costs = OptimizationCostBreakdown {
        optimization_run_id: run_id.clone(),
        total_tokens_used: total_tokens,
        total_cost_usd: (total_cost * 1000.0).round() / 1000.0,
        model_costs,
    };

    // Save final state
    run.status = OptimizationStatus::Completed;
    run.progress_percent = 100;
    run.message = Some("Optimization completed successfully".into());
    run.results_json = Some(serde_json::to_string(&results).unwrap_or_default());
    run.costs_json = Some(serde_json::to_string(&costs).unwrap_or_default());
    run.updated_at = Utc::now();
    run.completed_at = Some(Utc::now());

    if let Err(e) = state.store.update_optimization_run(&run).await {
        tracing::error!(run_id = %run_id, error = %e, "Failed to finalize optimization run");
    } else {
        tracing::info!(run_id = %run_id, "Prompt optimization completed");
    }
}

// ------------------------------------------------------------------
// Simulated prompt adaptation helpers
// ------------------------------------------------------------------

fn adapt_prompt_for_model(system_prompt: &str, model: &str) -> String {
    let lower = model.to_lowercase();
    if lower.contains("claude") {
        format!(
            "{}",
            system_prompt
        )
    } else if lower.contains("gpt") || lower.contains("openai") {
        format!(
            "{}",
            system_prompt
        )
    } else if lower.contains("gemini") || lower.contains("google") {
        format!(
            "{}",
            system_prompt
        )
    } else {
        system_prompt.to_string()
    }
}

fn adapt_template_for_model(template: &str, model: &str) -> String {
    let lower = model.to_lowercase();
    if lower.contains("claude") {
        // Claude benefits from XML-style tags for placeholders
        template.replace("{", "<").replace("}", ">")
    } else if lower.contains("gemini") || lower.contains("google") {
        // Gemini often benefits from more explicit formatting
        template.to_string()
    } else {
        template.to_string()
    }
}
