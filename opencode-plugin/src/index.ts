import { Plugin, tool } from "@opencode-ai/plugin";

const ROUTER_BASE = process.env.MODEL_ROUTER_URL || "http://localhost:8080";

export const ModelRouterPlugin: Plugin = async () => {
  return {
    name: "model-router",
    version: "0.1.0",

    tool: {
      route_model: tool({
        description:
          "Route an inference request to the best-suited model via the model-router service.",
        args: {
          prompt: tool.schema.string(),
          tenant_id: tool.schema.string().optional(),
          candidate_models: tool.schema.array(tool.schema.string()).optional(),
          max_latency_ms: tool.schema.number().optional(),
          max_cost_usd: tool.schema.number().optional(),
        },
        execute: async (args) => {
          const body = {
            tenant_id: args.tenant_id ?? "default",
            prompt: args.prompt,
            candidate_models: args.candidate_models ?? [],
            max_latency_ms: args.max_latency_ms,
            max_cost_usd: args.max_cost_usd,
            tool_names: [],
          };

          const res = await fetch(`${ROUTER_BASE}/route`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          });

          if (!res.ok) {
            throw new Error(`Router error: ${res.status} ${await res.text()}`);
          }

          return res.json();
        },
      }),

      submit_feedback: tool({
        description:
          "Submit outcome feedback for a prior routing decision so the router can learn.",
        args: {
          session_id: tool.schema.string(),
          model: tool.schema.string(),
          user_rating: tool.schema.number().optional(),
          completion_success: tool.schema.boolean().optional(),
          latency_ms: tool.schema.number().optional(),
        },
        execute: async (args) => {
          const res = await fetch(`${ROUTER_BASE}/feedback`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(args),
          });

          if (!res.ok) {
            throw new Error(`Feedback error: ${res.status} ${await res.text()}`);
          }

          return { ok: true };
        },
      }),
    },

    // Optional: intercept chat messages and rewrite model params before execution.
    hooks: {
      beforeChat: async (ctx) => {
        const prompt =
          typeof ctx.message.content === "string"
            ? ctx.message.content
            : JSON.stringify(ctx.message.content);

        const routeRes = await fetch(`${ROUTER_BASE}/route`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            tenant_id: ctx.tenantId ?? "default",
            prompt,
            tool_names: ctx.tools?.map((t: any) => t.name) ?? [],
            candidate_models: [],
          }),
        });

        if (routeRes.ok) {
          const decision = await routeRes.json();
          // Attach routing metadata so downstream tools / logs know which model was chosen.
          ctx.metadata = {
            ...ctx.metadata,
            routed_model: decision.selected_model,
            router_session_id: decision.session_id,
            router_ranking: decision.ranked_models,
          };
        }

        return ctx;
      },
    },
  };
};

export default ModelRouterPlugin;
