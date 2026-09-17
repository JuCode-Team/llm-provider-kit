//! Unverified JWT payload decoding, shared by the credential projection in
//! [`crate::auth`] and the provider dialects that read claims off a token.

use serde_json::Value;

/// Decodes a JWT payload without verifying the signature; None when the token
/// is not a well-formed three-part JWT.
pub(crate) fn decode_jwt_payload(token: &str) -> Option<Value> {
    use base64::Engine;
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_payload_and_rejects_non_jwts() {
        // {"sub":"abc"}
        let token = "header.eyJzdWIiOiJhYmMifQ.signature";
        assert_eq!(
            decode_jwt_payload(token).and_then(|v| v.get("sub").cloned()),
            Some(Value::String("abc".to_string()))
        );
        assert!(decode_jwt_payload("not-a-jwt").is_none());
        assert!(decode_jwt_payload("header.!!!.signature").is_none());
    }
}
