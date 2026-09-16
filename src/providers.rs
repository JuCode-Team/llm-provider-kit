//! Built-in provider templates: id, default base URL, wire protocol, and the
//! default model table. Callers map these into their own config and users can
//! override base_url/models.
//!
//! Protocol choices are deliberate: official OpenAI speaks the Responses API;
//! DeepSeek exposes an Anthropic-compatible endpoint; Ollama and OpenRouter are
//! driven over Chat Completions, the protocol their OpenAI-compatible endpoints
//! support best. A caller fronting its own gateway adds a [`ProviderTemplate`]
//! with the same body and its own base URL.

use crate::Protocol;

/// A built-in provider template. Add an entry to [`templates`] to ship a new
/// provider.
pub struct ProviderTemplate {
    pub id: &'static str,
    pub base_url: &'static str,
    pub protocol: Protocol,
    pub models: &'static [ModelTemplate],
}

/// Default model entry offered for a provider. Costs are intentionally not
/// part of the template: they change too often to hard-code.
pub struct ModelTemplate {
    pub name: &'static str,
    pub context_window: u64,
    pub max_output_tokens: u64,
    pub reasoning_efforts: &'static [&'static str],
}

const GPT_EFFORTS: &[&str] = &["none", "low", "medium", "high", "xhigh"];
const CODEX_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh"];
/// Chat-protocol models without a reasoning-effort knob.
const NO_EFFORTS: &[&str] = &["none"];

/// The gpt-5 family served by official OpenAI and Responses-speaking gateways.
const GPT_MODELS: &[ModelTemplate] = &[
    ModelTemplate {
        name: "gpt-5.5",
        context_window: 272_000,
        max_output_tokens: 128_000,
        reasoning_efforts: GPT_EFFORTS,
    },
    ModelTemplate {
        name: "gpt-5.4",
        context_window: 272_000,
        max_output_tokens: 128_000,
        reasoning_efforts: GPT_EFFORTS,
    },
    ModelTemplate {
        name: "gpt-5.4-mini",
        context_window: 400_000,
        max_output_tokens: 128_000,
        reasoning_efforts: GPT_EFFORTS,
    },
    ModelTemplate {
        name: "gpt-5.3-codex",
        context_window: 400_000,
        max_output_tokens: 128_000,
        reasoning_efforts: CODEX_EFFORTS,
    },
    ModelTemplate {
        name: "gpt-5.2",
        context_window: 400_000,
        max_output_tokens: 128_000,
        reasoning_efforts: GPT_EFFORTS,
    },
];

const DEEPSEEK_MODELS: &[ModelTemplate] = &[
    ModelTemplate {
        name: "deepseek-v4-pro",
        context_window: 1_000_000,
        max_output_tokens: 384_000,
        reasoning_efforts: &["high", "max"],
    },
    ModelTemplate {
        name: "deepseek-v4-flash",
        context_window: 1_000_000,
        max_output_tokens: 384_000,
        reasoning_efforts: &["high", "max"],
    },
];

/// Common local coding models; users typically override with whatever
/// `ollama list` shows.
const OLLAMA_MODELS: &[ModelTemplate] = &[
    ModelTemplate {
        name: "qwen3-coder:30b",
        context_window: 128_000,
        max_output_tokens: 32_000,
        reasoning_efforts: NO_EFFORTS,
    },
    ModelTemplate {
        name: "gpt-oss:20b",
        context_window: 128_000,
        max_output_tokens: 32_000,
        reasoning_efforts: NO_EFFORTS,
    },
    ModelTemplate {
        name: "llama3.3:70b",
        context_window: 128_000,
        max_output_tokens: 32_000,
        reasoning_efforts: NO_EFFORTS,
    },
];

const OPENROUTER_MODELS: &[ModelTemplate] = &[
    ModelTemplate {
        name: "openai/gpt-5.5",
        context_window: 272_000,
        max_output_tokens: 128_000,
        reasoning_efforts: GPT_EFFORTS,
    },
    ModelTemplate {
        name: "anthropic/claude-sonnet-4.5",
        context_window: 200_000,
        max_output_tokens: 64_000,
        reasoning_efforts: NO_EFFORTS,
    },
    ModelTemplate {
        name: "qwen/qwen3-coder",
        context_window: 262_000,
        max_output_tokens: 32_000,
        reasoning_efforts: NO_EFFORTS,
    },
];

/// All built-in providers, in the order UIs should list them.
pub fn templates() -> &'static [ProviderTemplate] {
    &[
        // Official OpenAI speaks the Responses API; a gateway fronting the same
        // body differs only in base URL and adds its own template.
        ProviderTemplate {
            id: "openai",
            base_url: "https://api.openai.com/v1",
            protocol: Protocol::OpenAiResponses,
            models: GPT_MODELS,
        },
        // DeepSeek exposes an Anthropic-compatible endpoint; route via Messages.
        ProviderTemplate {
            id: "deepseek",
            base_url: "https://api.deepseek.com/anthropic",
            protocol: Protocol::AnthropicMessages,
            models: DEEPSEEK_MODELS,
        },
        // Local Ollama serves OpenAI-compatible Chat Completions under /v1.
        ProviderTemplate {
            id: "ollama",
            base_url: "http://127.0.0.1:11434/v1",
            protocol: Protocol::OpenAiChatCompletions,
            models: OLLAMA_MODELS,
        },
        // OpenRouter multiplexes many vendors behind Chat Completions.
        ProviderTemplate {
            id: "openrouter",
            base_url: "https://openrouter.ai/api/v1",
            protocol: Protocol::OpenAiChatCompletions,
            models: OPENROUTER_MODELS,
        },
    ]
}

/// Looks up a built-in provider template by id.
pub fn template(id: &str) -> Option<&'static ProviderTemplate> {
    templates().iter().find(|template| template.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn template_ids_and_base_urls_are_unique() {
        let templates = templates();
        let ids: HashSet<_> = templates.iter().map(|t| t.id).collect();
        assert_eq!(ids.len(), templates.len(), "duplicate provider id");
        let urls: HashSet<_> = templates.iter().map(|t| t.base_url).collect();
        assert_eq!(urls.len(), templates.len(), "duplicate base_url");
    }

    #[test]
    fn expected_providers_ship_with_expected_protocols() {
        let expect = [
            ("openai", Protocol::OpenAiResponses),
            ("deepseek", Protocol::AnthropicMessages),
            ("ollama", Protocol::OpenAiChatCompletions),
            ("openrouter", Protocol::OpenAiChatCompletions),
        ];
        assert_eq!(templates().len(), expect.len());
        for (id, protocol) in expect {
            let found = template(id).unwrap_or_else(|| panic!("missing provider {id}"));
            assert_eq!(found.protocol, protocol, "{id}");
        }
    }

    #[test]
    fn every_template_has_models_with_unique_names_and_efforts() {
        for provider in templates() {
            assert!(!provider.models.is_empty(), "{} has no models", provider.id);
            let names: HashSet<_> = provider.models.iter().map(|m| m.name).collect();
            assert_eq!(
                names.len(),
                provider.models.len(),
                "duplicate model in {}",
                provider.id
            );
            for model in provider.models {
                assert!(model.context_window > 0, "{}", model.name);
                assert!(model.max_output_tokens > 0, "{}", model.name);
                assert!(!model.reasoning_efforts.is_empty(), "{}", model.name);
            }
        }
    }

    #[test]
    fn chat_defaults_point_at_ollama_and_openrouter_endpoints() {
        assert_eq!(
            template("ollama").unwrap().base_url,
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            template("openrouter").unwrap().base_url,
            "https://openrouter.ai/api/v1"
        );
    }
}
