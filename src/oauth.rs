//! OAuth primitives shared by the declared login flows: PKCE, loopback
//! callbacks, URL encoding, browser launch, and JSON request plumbing.
//!
//! Nothing here knows about a specific provider — the flows in [`crate::auth`]
//! drive these pieces from the catalog's compiled auth rules.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::Write, process::Command};

/// Seconds since the unix epoch; 0 when the clock is before it.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `bytes` of OS randomness, base64url (no padding) encoded.
pub fn random_token(bytes: usize) -> Result<String, String> {
    let mut data = vec![0_u8; bytes];
    getrandom::getrandom(&mut data).map_err(|error| error.to_string())?;
    Ok(URL_SAFE_NO_PAD.encode(data))
}

/// RFC 7636 S256 challenge for a verifier.
pub fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// Opens `url` in the platform's default browser. Errors when no handler
/// exists; callers surface the URL for manual opening.
pub fn open_browser(url: &str) -> Result<(), String> {
    let status = if cfg!(windows) {
        Command::new("rundll32")
            .arg("url.dll,FileProtocolHandler")
            .arg(url)
            .status()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(url).status()
    } else {
        Command::new("xdg-open").arg(url).status()
    }
    .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("failed to open browser: {status}"))
    }
}

/// Percent-encodes everything outside the unreserved set.
pub fn url_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                vec![byte as char]
            }
            _ => format!("%{byte:02X}").chars().collect(),
        })
        .collect()
}

/// Percent-decodes, treating `+` as a space (form-encoded query semantics).
pub fn url_decode(value: &str) -> String {
    let mut output = Vec::new();
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(hex) = u8::from_str_radix(&value[index + 1..index + 3], 16) {
                output.push(hex);
                index += 3;
                continue;
            }
        }
        output.push(if bytes[index] == b'+' {
            b' '
        } else {
            bytes[index]
        });
        index += 1;
    }
    String::from_utf8_lossy(&output).to_string()
}

/// Query parameters of a loopback callback's HTTP request line.
pub fn parse_callback_query(request_line: &str) -> Result<HashMap<String, String>, String> {
    let path = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "invalid OAuth callback request".to_string())?;
    let query = path
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or_default();
    Ok(query
        .split('&')
        .filter_map(|part| {
            let (key, value) = part.split_once('=')?;
            Some((url_decode(key), url_decode(value)))
        })
        .collect())
}

/// Answers the browser that hit the loopback callback with `body`, so the tab
/// shows a result instead of hanging. The wording belongs to the caller.
pub fn write_callback_response(stream: &mut impl Write, body: &str) -> Result<(), String> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .map_err(|error| error.to_string())
}

/// Reads a JSON body, turning HTTP failures into `"HTTP {code}: {body}"`.
/// Callers that need to name the service prefix the message.
pub fn json_response(response: Result<ureq::Response, ureq::Error>) -> Result<Value, String> {
    match response {
        Ok(response) => response
            .into_json::<Value>()
            .map_err(|error| error.to_string()),
        Err(ureq::Error::Status(code, response)) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<failed to read error body>".to_string());
            Err(format!("HTTP {code}: {body}"))
        }
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_the_rfc_7636_vector() {
        // Appendix B of RFC 7636.
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn url_codec_round_trips_reserved_and_space_characters() {
        for value in ["a b", "a+b", "x/y?z=1&w=2", "ünicode", "~ok-._"] {
            assert_eq!(url_decode(&url_encode(value)), value, "{value}");
        }
        assert_eq!(url_encode("a b"), "a%20b");
        assert_eq!(url_decode("a+b"), "a b");
    }

    #[test]
    fn callback_query_parses_the_request_line() {
        let params =
            parse_callback_query("GET /auth/callback?code=abc&state=xyz HTTP/1.1").unwrap();
        assert_eq!(params.get("code").map(String::as_str), Some("abc"));
        assert_eq!(params.get("state").map(String::as_str), Some("xyz"));
        assert!(parse_callback_query("garbage").is_err());
    }

    #[test]
    fn random_tokens_are_unpadded_url_safe_and_distinct() {
        let first = random_token(16).unwrap();
        let second = random_token(16).unwrap();
        assert_ne!(first, second);
        assert!(!first.contains('=') && !first.contains('+') && !first.contains('/'));
    }
}
