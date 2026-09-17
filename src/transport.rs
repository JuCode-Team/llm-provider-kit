//! Blocking HTTP transport for the wire protocols: one request at a time,
//! retries for transient failures, and stream decoding into [`WireEvent`]s.
//!
//! The client owns what every dialect shares — timeouts, auth headers, the
//! prompt-cache session id, Codex turn-state stickiness — and dispatches to the
//! parser of the protocol the endpoint speaks. No async runtime: ureq plus
//! threads, matching the rest of the crate.

use crate::{anthropic, chat, responses, Protocol, Usage, WireEvent};
use serde_json::Value;
use std::{
    env,
    sync::{Arc, OnceLock},
    thread,
    time::Duration,
};

const RETRY_BACKOFF_BASE_MS: u64 = 250;
const RETRY_BACKOFF_MAX_MS: u64 = 4_000;
/// Prompt-cache stickiness header the Codex backend returns.
const X_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
/// The Codex backend version-gates model availability on this value.
const CODEX_CLIENT_VERSION: &str = "0.153.0";
const CODEX_BETA_RESPONSES: &str = "responses=experimental";
/// Azure API revision used when `AZURE_OPENAI_API_VERSION` is unset.
const AZURE_DEFAULT_API_VERSION: &str = "v1";

/// A transport-level event: connection lifecycle plus the decoded wire events.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// The request was accepted (after any retries).
    Connected,
    /// The request is being re-sent; `attempt` is the next attempt number.
    Retrying { attempt: usize },
    /// A decoded protocol event.
    Wire(WireEvent),
}

/// A failed request: the HTTP status when the server answered, otherwise a
/// transport failure (timeout, DNS, dropped connection).
#[derive(Debug, Clone)]
pub struct RequestError {
    pub status: Option<u16>,
    pub message: String,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

impl RequestError {
    /// Retry on transport failures, 429 rate limits, and 5xx responses. Other
    /// 4xx responses are client errors and are never retried.
    pub fn is_retryable(&self) -> bool {
        match self.status {
            Some(code) => code == 429 || code >= 500,
            None => true,
        }
    }
}

/// Everything the client needs that is not per-request.
pub struct ClientConfig<'a> {
    pub api_key: &'a str,
    /// Session id sent as `session-id`/`thread-id` and the Responses
    /// `prompt_cache_key`; providers key their cache on it.
    pub prompt_cache_key: &'a str,
    /// How this client identifies itself where a provider expects a client
    /// name (the Codex `originator` header).
    pub client_name: &'a str,
    pub connect_timeout: Duration,
    /// Retries after the first attempt (transport errors, 429, 5xx only).
    pub retry_attempts: usize,
    /// Print prompt-cache diagnostics to stderr.
    pub cache_debug: bool,
}

/// Cloning shares the prompt-cache turn state, so a derived client (a subagent,
/// say) keeps the session's stickiness while choosing its own timeouts.
#[derive(Clone)]
pub struct Client {
    api_key: String,
    prompt_cache_key: String,
    client_name: String,
    connect_timeout: Duration,
    retry_attempts: usize,
    cache_debug: bool,
    turn_state: Arc<OnceLock<String>>,
}

impl Client {
    pub fn new(config: ClientConfig<'_>) -> Self {
        Self {
            api_key: config.api_key.to_string(),
            prompt_cache_key: config.prompt_cache_key.to_string(),
            client_name: config.client_name.to_string(),
            connect_timeout: config.connect_timeout,
            retry_attempts: config.retry_attempts,
            cache_debug: config.cache_debug,
            turn_state: Arc::new(OnceLock::new()),
        }
    }

    /// One attempt, with the client's read timeout. Retry policy belongs to
    /// the caller so a one-shot probe can fail fast.
    pub fn send(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
    ) -> Result<ureq::Response, RequestError> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(self.connect_timeout)
            .timeout_read(read_timeout)
            .build();
        let mut request = agent
            .post(url)
            .set("Accept", "text/event-stream")
            .set("Content-Type", "application/json")
            .set("session-id", &self.prompt_cache_key)
            .set("thread-id", &self.prompt_cache_key)
            .set("x-client-request-id", &self.prompt_cache_key);
        // Azure authenticates with an `api-key` header, the official Anthropic
        // API with x-api-key; everything else, the Codex backend included,
        // takes the Bearer scheme.
        if protocol == Protocol::AzureOpenAiResponses {
            request = request.set("api-key", &self.api_key);
        } else if anthropic::is_official_url(url) {
            request = request.set("x-api-key", &self.api_key);
        } else {
            request = request.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        if protocol == Protocol::AnthropicMessages {
            request = request.set("anthropic-version", anthropic::ANTHROPIC_VERSION);
        }
        if protocol == Protocol::OpenAiCodexResponses {
            request = request
                .set("OpenAI-Beta", CODEX_BETA_RESPONSES)
                .set("originator", &self.client_name)
                .set("version", CODEX_CLIENT_VERSION);
            if let Some(account_id) = responses::codex_account_id(&self.api_key) {
                request = request.set("chatgpt-account-id", &account_id);
            }
        }
        if let Some(turn_state) = self.turn_state.get() {
            request = request.set(X_CODEX_TURN_STATE_HEADER, turn_state);
        }
        if self.cache_debug {
            eprintln!(
                "[llm-cache] send protocol={} turn_state={}",
                protocol.as_str(),
                self.turn_state.get().is_some()
            );
        }
        let response = request.send_json(body.clone()).map_err(map_ureq_error)?;
        let saw_turn_state = capture_turn_state(&response, &self.turn_state);
        if self.cache_debug {
            eprintln!(
                "[llm-cache] response turn_state_header={saw_turn_state} turn_state_stored={}",
                self.turn_state.get().is_some()
            );
        }
        Ok(response)
    }

    /// `send` with the configured retry policy.
    pub fn send_with_retry(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
        emit: &mut impl FnMut(StreamEvent) -> Result<(), String>,
    ) -> Result<ureq::Response, String> {
        let max_attempts = self.max_attempts();
        for attempt in 1..=max_attempts {
            match self.send(protocol, url, body, read_timeout) {
                Ok(response) => return Ok(response),
                Err(error) if attempt < max_attempts && error.is_retryable() => {
                    emit(StreamEvent::Retrying {
                        attempt: attempt + 1,
                    })?;
                    thread::sleep(retry_backoff(attempt));
                }
                Err(error) => return Err(error.message),
            }
        }
        unreachable!("retry loop always returns a response or error")
    }

    /// Sends one streaming request and returns its completed output items.
    /// Text deltas stream live; items and usage are buffered until the stream
    /// completes, so a retry cannot duplicate session state.
    pub fn stream(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
        emit: &mut impl FnMut(StreamEvent) -> Result<(), String>,
    ) -> Result<Vec<Value>, String> {
        let max_attempts = self.max_attempts();
        for attempt in 1..=max_attempts {
            // Single send per attempt: this loop owns all retries, so send and
            // stream failures cannot multiply into nested retry rounds.
            let response = match self.send(protocol, url, body, read_timeout) {
                Ok(response) => response,
                Err(error) if attempt < max_attempts && error.is_retryable() => {
                    emit(StreamEvent::Retrying {
                        attempt: attempt + 1,
                    })?;
                    thread::sleep(retry_backoff(attempt));
                    continue;
                }
                Err(error) => return Err(error.message),
            };
            emit(StreamEvent::Connected)?;
            let content_type = response
                .header("content-type")
                .unwrap_or_default()
                .to_string();
            if !content_type.contains("text/event-stream")
                && !content_type.contains("application/json")
            {
                let body = response.into_string().map_err(|error| error.to_string())?;
                return Err(non_json_response_error(protocol, url, &content_type, &body));
            }
            if content_type.contains("application/json") {
                let body = response.into_string().map_err(|error| error.to_string())?;
                let value =
                    serde_json::from_str::<Value>(&body).map_err(|error| error.to_string())?;
                let (output_items, usage) = json_items(protocol, &value);
                for item in &output_items {
                    let text = item_text(item);
                    if !text.is_empty() {
                        emit(StreamEvent::Wire(WireEvent::Delta(text)))?;
                    }
                    emit(StreamEvent::Wire(WireEvent::ResponseItem(item.clone())))?;
                }
                if let Some(usage) = usage {
                    emit(StreamEvent::Wire(WireEvent::Usage(usage)))?;
                }
                return Ok(output_items);
            }
            let mut buffered: Vec<WireEvent> = Vec::new();
            let read = read_sse(protocol, response.into_reader(), |event| match event {
                event @ (WireEvent::Delta(_) | WireEvent::ReasoningDelta(_)) => {
                    emit(StreamEvent::Wire(event))
                }
                other => {
                    buffered.push(other);
                    Ok(())
                }
            });
            match read {
                Ok(output_items) => {
                    for event in buffered {
                        emit(StreamEvent::Wire(event))?;
                    }
                    return Ok(output_items);
                }
                Err(error) if attempt < max_attempts && is_retryable_stream_error(&error) => {
                    emit(StreamEvent::Retrying {
                        attempt: attempt + 1,
                    })?;
                    thread::sleep(retry_backoff(attempt));
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("streaming retry loop always returns")
    }

    /// Reads a response body as text: streamed deltas are forwarded to `emit`
    /// as they arrive, non-streaming JSON is flattened to its message text.
    pub fn read_text(
        &self,
        response: ureq::Response,
        protocol: Protocol,
        emit: &mut impl FnMut(WireEvent) -> Result<(), String>,
    ) -> Result<String, String> {
        let content_type = response
            .header("content-type")
            .unwrap_or_default()
            .to_string();
        let mut text = String::new();
        let mut accumulate = |event: WireEvent| {
            if let WireEvent::Delta(delta) = event {
                emit(WireEvent::Delta(delta.clone()))?;
                text.push_str(&delta);
            }
            Ok(())
        };
        if content_type.contains("application/json") {
            let body = response.into_string().map_err(|error| error.to_string())?;
            let value = serde_json::from_str::<Value>(&body).map_err(|error| error.to_string())?;
            for item in json_items(protocol, &value).0 {
                let delta = item_text(&item);
                if !delta.is_empty() {
                    emit(WireEvent::Delta(delta.clone()))?;
                    text.push_str(&delta);
                }
            }
            return Ok(text);
        }
        read_sse(protocol, response.into_reader(), &mut accumulate)?;
        Ok(text)
    }

    /// Attempts including the first one.
    fn max_attempts(&self) -> usize {
        self.retry_attempts.saturating_add(1).max(1)
    }
}

/// Endpoint for a protocol on `base_url`. The Responses dialects share one
/// event stream but differ in path and query.
pub fn endpoint(protocol: Protocol, base_url: &str) -> String {
    match protocol {
        Protocol::OpenAiCodexResponses => responses::codex_responses_url(base_url),
        Protocol::AzureOpenAiResponses => {
            responses::azure_responses_url(base_url, &azure_api_version())
        }
        Protocol::OpenAiResponses => responses::responses_url(base_url),
        Protocol::AnthropicMessages => anthropic::messages_url(base_url),
        Protocol::OpenAiChatCompletions => chat::completions_url(base_url),
    }
}

/// Azure API revision, overridable the same way the Azure SDKs do it.
pub fn azure_api_version() -> String {
    env::var("AZURE_OPENAI_API_VERSION")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| AZURE_DEFAULT_API_VERSION.to_string())
}

/// Dispatches to the protocol's SSE parser.
fn read_sse(
    protocol: Protocol,
    reader: impl std::io::Read,
    emit: impl FnMut(WireEvent) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    match protocol {
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => responses::read_sse_stream(reader, emit),
        Protocol::AnthropicMessages => anthropic::read_sse_stream(reader, emit),
        Protocol::OpenAiChatCompletions => chat::read_sse_stream(reader, emit),
    }
}

/// Output items and usage from a non-streaming response body.
fn json_items(protocol: Protocol, value: &Value) -> (Vec<Value>, Option<Usage>) {
    match protocol {
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => (
            value
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            responses::extract_usage(value),
        ),
        Protocol::AnthropicMessages => (
            anthropic::message_items(value),
            anthropic::message_usage(value),
        ),
        Protocol::OpenAiChatCompletions => chat::completion_to_items(value),
    }
}

/// Text of an output item's parts, regardless of part type.
fn item_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>()
}

/// Error for a body that is neither SSE nor JSON — usually a wrong base URL.
fn non_json_response_error(
    protocol: Protocol,
    url: &str,
    content_type: &str,
    body: &str,
) -> String {
    let (label, hint) = match protocol {
        Protocol::AnthropicMessages => ("Anthropic API", ""),
        Protocol::OpenAiChatCompletions => (
            "Chat Completions API",
            " Check base_url; OpenAI-compatible endpoints usually end with /v1.",
        ),
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => (
            "OpenAI API",
            " Check base_url; OpenAI-compatible endpoints usually end with /v1.",
        ),
    };
    format!(
        "{label} returned non-JSON response from {url} (content-type: {content_type}).{hint} Body starts: {}",
        truncate_error_body(body)
    )
}

fn capture_turn_state(response: &ureq::Response, turn_state: &OnceLock<String>) -> bool {
    if let Some(value) = response
        .header(X_CODEX_TURN_STATE_HEADER)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let _ = turn_state.set(value.to_string());
        true
    } else {
        false
    }
}

fn retry_backoff(attempt: usize) -> Duration {
    let multiplier = 1u64 << attempt.saturating_sub(1).min(4);
    Duration::from_millis((RETRY_BACKOFF_BASE_MS * multiplier).min(RETRY_BACKOFF_MAX_MS))
}

fn truncate_error_body(body: &str) -> String {
    let mut chars = body.chars();
    let snippet = chars.by_ref().take(180).collect::<String>();
    if chars.next().is_some() {
        format!("{snippet}...")
    } else {
        snippet
    }
}

/// ureq's error split into status and transport failures.
fn map_ureq_error(error: ureq::Error) -> RequestError {
    match error {
        ureq::Error::Status(code, response) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<failed to read error body>".to_string());
            RequestError {
                status: Some(code),
                message: format!("LLM API returned HTTP {code}: {body}"),
            }
        }
        error => RequestError {
            status: None,
            message: error.to_string(),
        },
    }
}

/// True for transport-level failures that occur while reading the streamed body
/// (e.g. a dropped/garbled chunked connection). Re-sending the request is safe
/// and usually succeeds; data errors (bad JSON, `response.failed`) won't match.
fn is_stream_decode_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "decoding chunk",
        "while decoding",
        "timed out",
        "timeout",
        "connection reset",
        "connection closed",
        "connection aborted",
        "peer closed connection",
        "broken pipe",
        "tls close_notify",
        "stream closed before response.completed",
        "stream closed before message_stop",
        "stream closed before finish_reason",
        "unexpected end of file",
        "unexpected eof",
        "unexpected-eof",
        "eof while",
        "io error",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn is_retryable_stream_error(message: &str) -> bool {
    is_stream_decode_error(message)
        || is_retryable_response_failed(message)
        || is_retryable_anthropic_error(message)
}

/// Anthropic in-stream `error` events for transient conditions are safe to
/// re-send; other error types (invalid_request, authentication, ...) are not.
fn is_retryable_anthropic_error(message: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(message) else {
        return false;
    };
    if value.get("type").and_then(Value::as_str) != Some("error") {
        return false;
    }
    let kind = value
        .get("error")
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    matches!(kind, "overloaded_error" | "api_error")
}

fn is_retryable_response_failed(message: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(message) else {
        return false;
    };
    if value.get("type").and_then(Value::as_str) != Some("response.failed") {
        return false;
    }
    let code = value
        .get("response")
        .and_then(|response| response.get("error"))
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    matches!(
        code,
        "server_error" | "rate_limit_exceeded" | "internal_error"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    fn test_client() -> Client {
        Client::new(ClientConfig {
            api_key: "test-key",
            prompt_cache_key: "cache-key",
            client_name: "test-client",
            connect_timeout: Duration::from_secs(2),
            retry_attempts: 1,
            cache_debug: false,
        })
    }

    /// Serves `responses` in order, one per accepted connection.
    fn serve(responses: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            for response in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                // Drain the request line + headers before answering.
                let mut buffer = [0_u8; 2048];
                let _ = stream.read(&mut buffer);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    fn sse_response(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn stream_decodes_a_responses_sse_body() {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        );
        let base = serve(vec![sse_response(body)]);
        let client = test_client();
        let mut deltas = Vec::new();
        let mut connected = 0;
        let items = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    match event {
                        StreamEvent::Connected => connected += 1,
                        StreamEvent::Wire(WireEvent::Delta(delta)) => deltas.push(delta),
                        _ => {}
                    }
                    Ok(())
                },
            )
            .expect("stream succeeds");

        assert_eq!(connected, 1);
        assert_eq!(deltas, vec!["hi".to_string()]);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn stream_retries_a_server_error_once() {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let base = serve(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom"
                .to_string(),
            sse_response(body),
        ]);
        let client = test_client();
        let mut retries = 0;
        let mut deltas = Vec::new();
        let items = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    match event {
                        StreamEvent::Retrying { attempt } => {
                            assert_eq!(attempt, 2);
                            retries += 1;
                        }
                        StreamEvent::Wire(WireEvent::Delta(delta)) => deltas.push(delta),
                        _ => {}
                    }
                    Ok(())
                },
            )
            .expect("stream recovers");

        assert_eq!(retries, 1);
        assert_eq!(deltas, vec!["ok".to_string()]);
        assert!(items.is_empty());
    }

    #[test]
    fn client_errors_are_not_retried() {
        let base = serve(vec![
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\nContent-Length: 8\r\nConnection: close\r\n\r\nno token"
                .to_string(),
        ]);
        let client = test_client();
        let mut retries = 0;
        let error = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    if matches!(event, StreamEvent::Retrying { .. }) {
                        retries += 1;
                    }
                    Ok(())
                },
            )
            .expect_err("401 fails");

        assert_eq!(retries, 0);
        assert!(error.contains("401"), "{error}");
    }

    #[test]
    fn read_text_flattens_a_json_body() {
        let body = r#"{"output":[{"type":"message","content":[{"type":"output_text","text":"summary"}]}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let base = serve(vec![response]);
        let client = test_client();
        let sent = client
            .send(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
            )
            .expect("send succeeds");
        let text = client
            .read_text(sent, Protocol::OpenAiResponses, &mut |_| Ok(()))
            .expect("text reads");

        assert_eq!(text, "summary");
    }

    #[test]
    fn retry_classification_matches_transport_and_server_failures() {
        let error = |status| RequestError {
            status,
            message: String::new(),
        };
        assert!(error(Some(500)).is_retryable());
        assert!(error(Some(429)).is_retryable());
        assert!(!error(Some(401)).is_retryable());
        assert!(error(None).is_retryable());
        assert!(is_retryable_stream_error(
            "stream closed before response.completed"
        ));
        assert!(!is_retryable_stream_error(
            "{\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"invalid_prompt\"}}}"
        ));
        assert!(is_retryable_stream_error(
            "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}"
        ));
    }

    #[test]
    fn retry_backoff_increases_and_caps() {
        assert_eq!(retry_backoff(1), Duration::from_millis(250));
        assert_eq!(retry_backoff(2), Duration::from_millis(500));
        assert_eq!(retry_backoff(3), Duration::from_millis(1000));
        assert_eq!(retry_backoff(5), Duration::from_millis(4000));
        assert_eq!(retry_backoff(99), Duration::from_millis(4000));
    }

    #[test]
    fn stream_decode_errors_are_retryable_but_data_errors_are_not() {
        assert!(is_stream_decode_error("Error while decoding chunks"));
        assert!(is_stream_decode_error("connection reset by peer"));
        assert!(is_stream_decode_error("the operation timed out"));
        assert!(is_stream_decode_error(
            "peer closed connection without sending TLS close_notify"
        ));
        assert!(is_stream_decode_error(
            "tls connection init failed: unexpected end of file"
        ));
        assert!(is_stream_decode_error(
            "stream closed before response.completed"
        ));
        assert!(!is_stream_decode_error(
            "{\"type\":\"response.failed\",\"response\":{}}"
        ));
        assert!(!is_stream_decode_error("expected value at line 1 column 1"));
        assert!(!is_retryable_stream_error(
            r#"{"type":"response.failed","response":{"error":{"code":"invalid_request_error","message":"bad request"}}}"#
        ));
    }

    #[test]
    fn anthropic_transient_stream_errors_are_retryable() {
        assert!(is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
        ));
        assert!(is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"api_error","message":"Internal server error"}}"#
        ));
        assert!(!is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}"#
        ));
    }

    /// The parsers' terminal error messages must stay in sync with the retry
    /// classification: truncated streams are transport failures (safe to
    /// re-send), while in-stream data errors are not.
    #[test]
    fn vendor_stream_errors_classify_as_the_parsers_report_them() {
        let error = responses::read_sse_stream(
            "data: {\"type\":\"response.created\"}\n\n".as_bytes(),
            |_| Ok(()),
        )
        .expect_err("stream without response.completed should fail");
        assert!(error.contains("stream closed before response.completed"));
        assert!(is_retryable_stream_error(&error));

        let sse = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        );
        let error = anthropic::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("truncated stream should fail");
        assert!(error.contains("stream closed before message_stop"));
        assert!(is_retryable_stream_error(&error));

        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";
        let error =
            chat::read_sse_stream(sse.as_bytes(), |_| Ok(())).expect_err("truncated chat stream");
        assert!(error.contains("stream closed before finish_reason"));
        assert!(is_retryable_stream_error(&error));

        let sse =
            "data: {\"type\":\"error\",\"code\":\"invalid_api_key\",\"message\":\"bad key\"}\n\n";
        let error = responses::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("in-stream error event should fail");
        assert!(error.contains("invalid_api_key"));
        assert!(!is_retryable_stream_error(&error));

        // A truncated tool call is a data error: retrying would replay the
        // whole (expensive) response for the same likely outcome.
        let sse = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"write\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a.txt\\\",\\\"content\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let error = anthropic::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("truncated tool_use should fail");
        assert!(!is_retryable_stream_error(&error));
    }

    #[test]
    fn codex_turn_state_is_captured_once() {
        let turn_state = OnceLock::new();
        let response: ureq::Response = "HTTP/1.1 200 OK\r\n\
             x-codex-turn-state: sticky-1\r\n\
             \r\n"
            .parse()
            .unwrap();
        capture_turn_state(&response, &turn_state);
        assert_eq!(turn_state.get().map(String::as_str), Some("sticky-1"));

        let response: ureq::Response = "HTTP/1.1 200 OK\r\n\
             x-codex-turn-state: sticky-2\r\n\
             \r\n"
            .parse()
            .unwrap();
        capture_turn_state(&response, &turn_state);
        assert_eq!(turn_state.get().map(String::as_str), Some("sticky-1"));
    }

    #[test]
    fn endpoint_places_each_dialect_at_its_own_path() {
        assert_eq!(
            endpoint(Protocol::OpenAiResponses, "https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            endpoint(
                Protocol::OpenAiCodexResponses,
                "https://chatgpt.com/backend-api"
            ),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            endpoint(Protocol::AnthropicMessages, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert!(endpoint(
            Protocol::AzureOpenAiResponses,
            "https://res.openai.azure.com/openai/v1"
        )
        .ends_with("/responses?api-version=v1"));
    }
}
