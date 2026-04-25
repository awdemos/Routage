use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Incoming request to select a model.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteRequest {
    pub tenant_id: String,
    pub agent_id: Option<String>,
    pub prompt: String,
    pub max_latency_ms: Option<u64>,
    pub max_cost_usd: Option<f32>,
    #[serde(default)]
    pub tool_names: Vec<String>,
    #[serde(default)]
    pub candidate_models: Vec<String>,
}

/// Score + rationale for a single model.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelScore {
    pub model: String,
    pub score: f32,
    pub reason: String,
}

/// Outgoing routing decision.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteResponse {
    pub selected_model: String,
    pub ranked_models: Vec<ModelScore>,
    pub session_id: String,
    pub explanation: String,
}

/// Inference execution request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceRequest {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub model: String,
    pub prompt: String,
    pub provider_url: Option<String>,
    pub provider_headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub allow_fallback: bool,
    #[serde(default)]
    pub fallback_models: Vec<String>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceResult {
    pub session_id: String,
    pub model: String,
    pub provider_url: Option<String>,
    pub response_text: String,
    pub latency_ms: u64,
    pub tokens_used: Option<u64>,
    pub cost_usd: Option<f32>,
    pub error_kind: Option<FailureKind>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Feedback on a prior routing / inference decision.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeedbackEvent {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub model: String,
    pub user_rating: Option<f32>,         // 0.0 .. 1.0
    pub completion_success: Option<bool>,
    pub latency_ms: Option<u64>,
    pub tokens_used: Option<u64>,
    pub cost_usd: Option<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Specification for a model known to the router.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSpec {
    pub id: String,
    pub provider: String,
    pub provider_model_id: String,
    pub provider_base_url: String,
    pub default_quality: f32,
    pub default_latency_ms: u64,
    pub default_cost_per_1k_tokens_usd: f32,
    pub tags: Vec<String>,
}

/// Per-tenant constraints / weights.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TenantPolicy {
    pub tenant_id: String,
    pub allowed_models: Vec<String>,
    pub blocked_models: Vec<String>,
    pub latency_budget_ms: Option<u64>,
    pub cost_budget_usd: Option<f32>,
    pub exploration_rate: f32,
    pub quality_weight: f32,
    pub latency_weight: f32,
    pub cost_weight: f32,
}

impl Default for TenantPolicy {
    fn default() -> Self {
        Self {
            tenant_id: "default".into(),
            allowed_models: vec![],
            blocked_models: vec![],
            latency_budget_ms: None,
            cost_budget_usd: None,
            exploration_rate: 0.15,
            quality_weight: 0.5,
            latency_weight: 0.25,
            cost_weight: 0.25,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIModelList {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Orchestrator,
    Planner,
    Coder,
    Reviewer,
    Summarizer,
    Translator,
    Generic,
}

impl Default for AgentRole {
    fn default() -> Self {
        AgentRole::Generic
    }
}

/// Per-agent constraints / weights.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentPolicy {
    pub agent_id: String,
    pub tenant_id: String,
    pub role: AgentRole,
    pub allowed_models: Vec<String>,
    pub blocked_models: Vec<String>,
    pub latency_budget_ms: Option<u64>,
    pub cost_budget_usd: Option<f32>,
    pub exploration_rate: f32,
    pub quality_weight: f32,
    pub latency_weight: f32,
    pub cost_weight: f32,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self {
            agent_id: "default".into(),
            tenant_id: "default".into(),
            role: AgentRole::Generic,
            allowed_models: vec![],
            blocked_models: vec![],
            latency_budget_ms: None,
            cost_budget_usd: None,
            exploration_rate: 0.15,
            quality_weight: 0.5,
            latency_weight: 0.25,
            cost_weight: 0.25,
        }
    }
}

/// Compact feature vector used for bandit scoring.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PromptFeatures {
    pub prompt_length: usize,
    pub has_tools: bool,
    pub tool_count: usize,
    pub estimated_complexity: f32,
}

/// In-memory arm statistics for the contextual bandit.
#[derive(Debug, Clone, Default)]
pub struct ArmStats {
    pub pulls: u64,
    pub total_reward: f64,
    pub sum_sq_reward: f64,
}

/// A logged decision row for offline analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLog {
    pub id: Uuid,
    pub session_id: String,
    pub tenant_id: String,
    pub agent_id: String,
    pub prompt: String,
    pub features: serde_json::Value,
    pub candidate_models: serde_json::Value,
    pub selected_model: String,
    pub scores: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    Timeout,
    RateLimited,
    AuthError,
    ServerError,
    ContentFilter,
    BadResponseFormat,
    InvalidInput,
    CostExceeded,
    LatencyExceeded,
    ConnectionError,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Offline,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelHealth {
    pub model_id: String,
    pub provider_base_url: String,
    pub status: HealthStatus,
    pub last_checked: DateTime<Utc>,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
    pub probe_latency_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelFailure {
    pub id: Uuid,
    pub session_id: String,
    pub model_id: String,
    pub failure_kind: FailureKind,
    pub error_message: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExplainResponse {
    pub session_id: String,
    pub selected_model: String,
    pub explanation: String,
    pub ranked_models: Vec<ModelScore>,
    pub feedback: Option<FeedbackEvent>,
    pub failures: Vec<ModelFailure>,
    pub performance_summary: String,
}

// ------------------------------------------------------------------
// Prompt Optimization v2
// ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoldenRecord {
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestProvider {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TargetModel {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvaluationConfig {
    pub judge_model: String,
    pub evaluation_prompt: String,
    pub cutoff_score: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptOptimizeRequest {
    pub system_prompt: String,
    pub template: String,
    pub fields: Vec<String>,
    #[serde(default)]
    pub goldens: Vec<GoldenRecord>,
    #[serde(default)]
    pub train_goldens: Vec<GoldenRecord>,
    #[serde(default)]
    pub test_goldens: Vec<GoldenRecord>,
    pub origin_model: Option<RequestProvider>,
    pub target_models: Vec<TargetModel>,
    pub evaluation_metric: Option<String>,
    pub evaluation_config: Option<EvaluationConfig>,
    pub origin_model_evaluation_score: Option<f32>,
    #[serde(default)]
    pub prototype_mode: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptOptimizeStatusResponse {
    pub optimization_run_id: String,
    pub status: OptimizationStatus,
    pub progress_percent: u8,
    pub message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub estimated_completion_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptimizedPrompt {
    pub provider: String,
    pub model: String,
    pub system_prompt: String,
    pub template: String,
    pub score: f32,
    pub improvement_percent: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptOptimizeResultsResponse {
    pub optimization_run_id: String,
    pub status: OptimizationStatus,
    pub origin_model: Option<RequestProvider>,
    pub origin_model_score: Option<f32>,
    pub optimized_prompts: Vec<OptimizedPrompt>,
    pub train_examples_used: usize,
    pub test_examples_used: usize,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptimizationCostBreakdown {
    pub optimization_run_id: String,
    pub total_tokens_used: u64,
    pub total_cost_usd: f32,
    pub model_costs: Vec<ModelCost>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelCost {
    pub provider: String,
    pub model: String,
    pub tokens_used: u64,
    pub cost_usd: f32,
}

/// Internal representation of an optimization run stored in the DB.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptimizationRun {
    pub id: String,
    pub status: OptimizationStatus,
    pub progress_percent: u8,
    pub message: Option<String>,
    pub request_json: String,
    pub results_json: Option<String>,
    pub costs_json: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

// ------------------------------------------------------------------
// Model Listing v2
// ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NdModelInfo {
    pub provider: String,
    pub model: String,
    pub display_name: String,
    pub context_length: u64,
    pub input_price: f64,
    pub output_price: f64,
    pub latency: f64,
    pub is_deprecated: bool,
    pub supports_vision: bool,
    pub supports_tools: bool,
    pub supports_json_mode: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NdModelListResponse {
    pub models: Vec<NdModelInfo>,
    pub total: usize,
    pub deprecated_models: Vec<NdModelInfo>,
}

// ------------------------------------------------------------------
// Model Router / modelSelect v2
// ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectProvider {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectTool {
    #[serde(flatten)]
    pub spec: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectRequest {
    pub messages: Vec<ModelSelectMessage>,
    pub models: Vec<ModelSelectProvider>,
    #[serde(default)]
    pub tools: Vec<ModelSelectTool>,
    #[serde(default)]
    pub hash_content: bool,
    pub metric: Option<String>,
    pub max_model_count: Option<usize>,
    pub tradeoff: Option<String>,
    pub preference_id: Option<String>,
    pub previous_session: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectResponse {
    pub session_id: String,
    pub provider: String,
    pub model: String,
    pub ranked_models: Vec<ModelSelectRanked>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSelectRanked {
    pub provider: String,
    pub model: String,
    pub score: f32,
    pub reason: String,
}

// ------------------------------------------------------------------
// Custom Router Training v2
// ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CustomModelConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub is_custom: bool,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub context_length: Option<u64>,
    pub latency: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainCustomRouterRequest {
    pub dataset_csv: String,
    pub models: Vec<CustomModelConfig>,
    pub prompt_column: String,
    pub score_column_prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainCustomRouterResponse {
    pub preference_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CustomRouterPreference {
    pub preference_id: String,
    pub status: OptimizationStatus,
    pub models: Vec<CustomModelConfig>,
    pub dataset_csv: Option<String>,
    pub train_samples: usize,
    pub accuracy: Option<f32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}
