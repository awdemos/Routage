use crate::domain::*;
use crate::probe::HealthMap;
use dashmap::DashMap;
use rand::{thread_rng, Rng};
use std::sync::Arc;

/// Shared in-memory bandit state keyed by "tenant_id:agent_id:model_id".
#[derive(Debug, Clone, Default)]
pub struct BanditEngine {
    arms: Arc<DashMap<String, ArmStats>>,
}

impl BanditEngine {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(tenant_id: &str, agent_id: &str, model_id: &str) -> String {
        format!("{}:{}:{}", tenant_id, agent_id, model_id)
    }

    /// Compute a prompt feature vector from the raw request.
    pub fn extract_features(req: &RouteRequest) -> PromptFeatures {
        let prompt_length = req.prompt.len();
        let has_tools = !req.tool_names.is_empty();
        let tool_count = req.tool_names.len();
        // Naive complexity heuristic: length + tool overhead
        let estimated_complexity = {
            let len_score = (prompt_length as f32 / 2000.0).min(1.0);
            let tool_score = (tool_count as f32 / 5.0).min(1.0);
            (len_score * 0.6 + tool_score * 0.4).min(1.0)
        };
        PromptFeatures {
            prompt_length,
            has_tools,
            tool_count,
            estimated_complexity,
        }
    }

    /// Score a single model given context, policy, and historical performance.
    pub fn score_model(
        &self,
        model_id: &str,
        model_spec: &ModelSpec,
        _features: &PromptFeatures,
        policy: &AgentPolicy,
        _session_id: &str,
    ) -> ModelScore {
        let key = Self::key(&policy.tenant_id, &policy.agent_id, model_id);
        let stats = self.arms.get(&key).map(|e| e.clone()).unwrap_or_default();

        // 1. Quality estimate (baseline + empirical average reward)
        let empirical_quality = if stats.pulls > 0 {
            (stats.total_reward / stats.pulls as f64) as f32
        } else {
            model_spec.default_quality
        };

        // 2. Latency penalty (inverse normalized)
        let latency_penalty = if model_spec.default_latency_ms > 0 {
            let budget = policy.latency_budget_ms.unwrap_or(5000) as f32;
            let ratio = model_spec.default_latency_ms as f32 / budget;
            ratio.min(1.0)
        } else {
            0.0
        };

        // 3. Cost penalty (inverse normalized)
        let cost_penalty = {
            let budget = policy.cost_budget_usd.unwrap_or(0.05).max(0.001);
            let ratio = model_spec.default_cost_per_1k_tokens_usd / budget;
            ratio.min(1.0)
        };

        // 4. Exploration bonus (UCB-like)
        let total_pulls: u64 = self
            .arms
            .iter()
            .filter(|e| e.key().starts_with(&format!("{}:{}", policy.tenant_id, policy.agent_id)))
            .map(|e| e.pulls)
            .sum();
        let exploration_bonus = if stats.pulls > 0 {
            let total = total_pulls.max(1) as f64;
            let pulls = stats.pulls as f64;
            ((2.0 * total.ln()) / pulls).sqrt() as f32
        } else {
            0.5 // encourage trying unseen arms
        };

        let score = policy.quality_weight * empirical_quality
            - policy.latency_weight * latency_penalty
            - policy.cost_weight * cost_penalty
            + policy.exploration_rate * exploration_bonus;

        let reason = format!(
            "quality={:.3} latency_pen={:.3} cost_pen={:.3} ucb={:.3} pulls={}",
            empirical_quality, latency_penalty, cost_penalty, exploration_bonus, stats.pulls
        );

        ModelScore {
            model: model_id.into(),
            score: score.max(0.0),
            reason,
        }
    }

    /// Main routing logic:
    /// 1. Hard-filter invalid / blocked / offline models.
    /// 2. Score survivors.
    /// 3. Epsilon-greedy exploration.
    /// 4. Return top model + full ranking + human-readable explanation.
    pub fn route(
        &self,
        req: &RouteRequest,
        models: &[(String, ModelSpec)],
        policy: &AgentPolicy,
        health_map: &HealthMap,
    ) -> RouteResponse {
        let features = Self::extract_features(req);
        let session_id = uuid::Uuid::new_v4().to_string();

        // 1. Hard filter: policy + health
        let mut candidates: Vec<(String, &ModelSpec)> = models
            .iter()
            .filter(|(id, _)| {
                if !policy.allowed_models.is_empty() && !policy.allowed_models.contains(id) {
                    return false;
                }
                if policy.blocked_models.contains(id) {
                    return false;
                }
                if !req.candidate_models.is_empty() && !req.candidate_models.contains(id) {
                    return false;
                }
                // Circuit breaker: exclude offline models
                if let Some(h) = health_map.get(id) {
                    if h.status == HealthStatus::Offline {
                        return false;
                    }
                }
                true
            })
            .map(|(id, spec)| (id.clone(), spec))
            .collect();

        let mut excluded_offline: Vec<String> = Vec::new();
        if candidates.is_empty() {
            // Fallback: if filtering emptied the pool because of offline models,
            // try to include any model except blocked ones.
            for (id, spec) in models.iter() {
                if policy.blocked_models.contains(id) {
                    continue;
                }
                if !req.candidate_models.is_empty() && !req.candidate_models.contains(id) {
                    continue;
                }
                if !policy.allowed_models.is_empty() && !policy.allowed_models.contains(id) {
                    continue;
                }
                candidates.push((id.clone(), spec));
                if let Some(h) = health_map.get(id) {
                    if h.status == HealthStatus::Offline {
                        excluded_offline.push(id.clone());
                    }
                }
            }
        }

        // 2. Score
        let mut scored: Vec<ModelScore> = candidates
            .iter()
            .map(|(id, spec)| self.score_model(id, spec, &features, policy, &session_id))
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // 3. Exploration
        let selected_model = if thread_rng().gen::<f32>() < policy.exploration_rate && !scored.is_empty() {
            let idx = thread_rng().gen_range(0..scored.len());
            scored[idx].model.clone()
        } else if let Some(top) = scored.first() {
            top.model.clone()
        } else {
            "unknown".into()
        };

        // 4. Build human-readable explanation
        let mut explanation_parts = vec![format!(
            "Selected {} for agent '{}' (role: {:?}).",
            selected_model,
            policy.agent_id,
            policy.role
        )];

        if let Some(top) = scored.first() {
            explanation_parts.push(format!(
                "Top score was {:.3} with rationale: {}.",
                top.score, top.reason
            ));
        }
        if !excluded_offline.is_empty() {
            explanation_parts.push(format!(
                "Excluded offline models: {}.",
                excluded_offline.join(", ")
            ));
        }
        if scored.len() > 1 {
            explanation_parts.push(format!(
                "Runner-up was {} with score {:.3}.",
                scored[1].model, scored[1].score
            ));
        }

        let explanation = explanation_parts.join(" ");

        RouteResponse {
            selected_model,
            ranked_models: scored,
            session_id,
            explanation,
        }
    }

    /// Load persisted arms into memory.
    pub fn load(&self, arms: Vec<(String, String, String, ArmStats)>) {
        for (tenant_id, agent_id, model_id, stats) in arms {
            let key = Self::key(&tenant_id, &agent_id, &model_id);
            self.arms.insert(key, stats);
        }
    }

    /// Apply online update from feedback and return the updated stats.
    pub fn update(&self, tenant_id: &str, agent_id: &str, model_id: &str, reward: f64) -> ArmStats {
        let key = Self::key(tenant_id, agent_id, model_id);
        let mut stats = self.arms.entry(key).or_insert(ArmStats {
            pulls: 0,
            total_reward: 0.0,
            sum_sq_reward: 0.0,
        });
        stats.pulls += 1;
        stats.total_reward += reward;
        stats.sum_sq_reward += reward * reward;
        stats.clone()
    }
}
