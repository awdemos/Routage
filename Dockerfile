# syntax=docker/dockerfile:1
FROM rust:1.86-slim-bookworm AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Cache dependencies: copy manifests and build a dummy project
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release

# Now copy real source and rebuild (only our code recompiles)
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    touch src/main.rs && cargo build --release

# Copy the final binary out of the cache mount
RUN --mount=type=cache,target=/app/target \
    cp /app/target/release/routage /app/routage

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/routage /usr/local/bin/routage

ENV DATABASE_URL=sqlite:/app/data/routage.db
ENV RUST_LOG=routage=info
ENV BIND_ADDR=0.0.0.0:8080

EXPOSE 8080

VOLUME ["/app/data"]

ENTRYPOINT ["/usr/local/bin/routage"]
