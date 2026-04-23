use crate::domain::{AgentPolicy, AgentRole, ModelSpec, TenantPolicy};
use std::collections::HashMap;
use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub bind_addr: SocketAddr,
    pub probe_interval_secs: u64,
    pub max_prompt_length: usize,
    pub max_candidate_models: usize,
    pub provider_keys: HashMap<String, String>,
    pub models: Vec<(String, ModelSpec)>,
    pub agent_policies: Vec<AgentPolicy>,
    pub tenant_policies: Vec<TenantPolicy>,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "sqlite:routage.db".into());
        let bind_addr = std::env::var("BIND_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:8080".into())
            .parse()?;
        let probe_interval_secs = std::env::var("PROBE_INTERVAL_SECS")
            .unwrap_or_else(|_| "30".into())
            .parse()?;
        let max_prompt_length = std::env::var("MAX_PROMPT_LENGTH")
            .unwrap_or_else(|_| "100000".into())
            .parse()?;
        let max_candidate_models = std::env::var("MAX_CANDIDATE_MODELS")
            .unwrap_or_else(|_| "50".into())
            .parse()?;

        let mut provider_keys = HashMap::new();
        for (key, val) in std::env::vars() {
            if key.ends_with("_API_KEY") {
                let provider = key.trim_end_matches("_API_KEY").to_lowercase();
                provider_keys.insert(provider, val);
            }
        }

        let models = Self::default_models();
        let agent_policies = Self::default_agent_policies();
        let tenant_policies = Self::default_tenant_policies();

        Ok(Config {
            database_url,
            bind_addr,
            probe_interval_secs,
            max_prompt_length,
            max_candidate_models,
            provider_keys,
            models,
            agent_policies,
            tenant_policies,
        })
    }

    fn default_models() -> Vec<(String, ModelSpec)> {
        let tensorzero_url = std::env::var("TENSORZERO_URL")
            .unwrap_or_else(|_| "http://localhost:3000/openai/v1".into());

        vec![
            (
                "kimi-k2.5".into(),
                ModelSpec {
                    id: "kimi-k2.5".into(),
                    provider: "tensorzero".into(),
                    provider_model_id: "tensorzero::model_name::kimi_k2_5".into(),
                    provider_base_url: tensorzero_url.clone(),
                    default_quality: 0.92,
                    default_latency_ms: 1200,
                    default_cost_per_1k_tokens_usd: 0.003,
                    tags: vec!["chat".into(), "long-context".into(), "agentic".into()],
                },
            ),
            (
                "gpt-4o-mini".into(),
                ModelSpec {
                    id: "gpt-4o-mini".into(),
                    provider: "tensorzero".into(),
                    provider_model_id: "tensorzero::model_name::gpt_4o_mini".into(),
                    provider_base_url: tensorzero_url.clone(),
                    default_quality: 0.85,
                    default_latency_ms: 800,
                    default_cost_per_1k_tokens_usd: 0.00015,
                    tags: vec!["chat".into(), "fast".into(), "cheap".into()],
                },
            ),
            (
                "zai-glm".into(),
                ModelSpec {
                    id: "zai-glm".into(),
                    provider: "tensorzero".into(),
                    provider_model_id: "tensorzero::model_name::zai_glm".into(),
                    provider_base_url: tensorzero_url,
                    default_quality: 0.88,
                    default_latency_ms: 1000,
                    default_cost_per_1k_tokens_usd: 0.002,
                    tags: vec!["chat".into(), "reasoning".into()],
                },
            ),
        ]
    }

    fn default_agent_policies() -> Vec<AgentPolicy> {
        vec![
            AgentPolicy {
                agent_id: "orchestrator".into(),
                tenant_id: "default".into(),
                role: AgentRole::Orchestrator,
                allowed_models: vec!["kimi-k2.5".into(), "gpt-4o-mini".into(), "zai-claude".into()],
                blocked_models: vec![],
                latency_budget_ms: Some(2000),
                cost_budget_usd: Some(0.01),
                exploration_rate: 0.10,
                quality_weight: 0.6,
                latency_weight: 0.2,
                cost_weight: 0.2,
            },
            AgentPolicy {
                agent_id: "coder".into(),
                tenant_id: "default".into(),
                role: AgentRole::Coder,
                allowed_models: vec!["kimi-k2.5".into(), "gpt-4o-mini".into(), "zai-claude".into()],
                blocked_models: vec![],
                latency_budget_ms: Some(5000),
                cost_budget_usd: Some(0.02),
                exploration_rate: 0.15,
                quality_weight: 0.7,
                latency_weight: 0.1,
                cost_weight: 0.2,
            },
            AgentPolicy {
                agent_id: "summarizer".into(),
                tenant_id: "default".into(),
                role: AgentRole::Summarizer,
                allowed_models: vec!["kimi-k2.5".into(), "gpt-4o-mini".into(), "zai-claude".into()],
                blocked_models: vec![],
                latency_budget_ms: Some(1500),
                cost_budget_usd: Some(0.005),
                exploration_rate: 0.20,
                quality_weight: 0.3,
                latency_weight: 0.5,
                cost_weight: 0.2,
            },
        ]
    }

    fn default_tenant_policies() -> Vec<TenantPolicy> {
        vec![TenantPolicy::default()]
    }
}
