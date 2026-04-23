use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use uuid::Uuid;

const SKIP_AUTH_PATHS: &[&str] = &["/health", "/metrics"];

pub async fn auth_and_trace_middleware(
    req: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let path = req.uri().path().to_string();

    // Inject request_id into tracing span
    let span = tracing::info_span!("request", %request_id, %path);
    let _enter = span.enter();

    // Auth check
    let api_key = std::env::var("ROUTAGE_API_KEY").ok();
    let needs_auth = api_key.is_some() && !SKIP_AUTH_PATHS.contains(&path.as_str());

    if needs_auth {
        let auth_header = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        let expected = format!("Bearer {}", api_key.unwrap());
        if auth_header != Some(&expected) {
            tracing::warn!("Unauthorized request to {}", path);
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({
                    "error": "Unauthorized",
                    "failure_kind": "auth_error"
                })),
            )
                .into_response();
        }
    }

    let mut response = next.run(req).await;

    // Attach request-id to response headers
    response
        .headers_mut()
        .insert("x-request-id", request_id.parse().unwrap());

    response
}
