# routage

Async Rust service that routes LLM inference requests to the best-suited model, learns from feedback via a contextual multi-armed bandit, and integrates with OpenCode via an in-process plugin.

## Stack

- **Rust**: `axum` + `tokio` + `tower-http`
- **Persistence**: `sqlx` + SQLite (migratable to Postgres)
- **Learning**: in-memory contextual multi-armed bandit (UCB + epsilon-greedy)
- **Observability**: `tracing`, Prometheus metrics
- **Plugin**: TypeScript OpenCode plugin with tool hooks + `beforeChat`

## Quick start

### Local (Rust required)

```bash
cd routage
cargo run
```

Service binds to `0.0.0.0:8080`.

### Docker + TensorZero

Routage ships with a Docker Compose setup that includes the [TensorZero](https://www.tensorzero.com/) gateway for model inference.

```bash
# 1. Copy the example environment file and add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Build and start everything
docker compose up --build

# 3. In another terminal, test the router
curl -s http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{
    "tenant_id": "default",
    "prompt": "Explain quantum computing in one sentence.",
    "candidate_models": ["gpt-4o", "gpt-4o-mini"]
  }' | jq .
```

Services:
- **TensorZero Gateway** → `http://localhost:3000`
- **Routage** → `http://localhost:8080`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/metrics` | Prometheus metrics |
| GET | `/performance` | Per-model aggregated stats |
| POST | `/route` | Rank models and select the best one |
| POST | `/infer` | Run inference through TensorZero |
| POST | `/feedback` | Send outcome feedback to update the bandit |
| GET | `/explain/:session_id` | Explain a past routing decision |

## Example

```bash
curl -s http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{
    "tenant_id": "default",
    "prompt": "Explain quantum computing in one sentence.",
    "candidate_models": ["gpt-4o", "gpt-4o-mini"]
  }' | jq .
```

## Learning roadmap

1. Static heuristic router ✓
2. Request logging + feedback storage ✓
3. OpenCode plugin ✓
4. Online bandit updates ✓
5. TensorZero + Docker integration ✓
6. Offline policy evaluation (next)
7. Provider failover + caching (next)

## OpenCode plugin

See `opencode-plugin/`. Build with `npm run build` (requires `@opencode-ai/plugin` stub or real package).

```ts
import ModelRouterPlugin from "@model-router/opencode-plugin";
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:routage.db` | SQLite database path |
| `RUST_LOG` | `routage=debug` | Tracing filter |
| `TENSORZERO_URL` | `http://localhost:3000/openai/v1` | TensorZero gateway endpoint |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for TensorZero |
