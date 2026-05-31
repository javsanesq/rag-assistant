# Security Policy

## Supported Scope

This is a portfolio-ready, single-tenant RAG assistant. It includes demo-safe hardening for URL ingestion, upload validation, secret handling, Docker deployment, and structured error reporting. It does not claim enterprise authentication, tenant isolation, compliance controls, or production secret rotation.

## Secrets

Do not commit real secrets. Use `.env.example` as the documented template and keep real values in an untracked `.env`, deployment environment variables, or a secret manager.

The repository ignores `.env`, `.env.local`, runtime databases, uploaded documents, generated outputs, and local cache directories.

## URL Ingestion Safety

URL ingestion blocks private/local targets by default, applies timeout and response-size limits, validates content type, and limits sitemap/list batch size. Review these settings before deploying the service in a network that can reach internal systems.

## Reporting

If you find a vulnerability, open a private report through GitHub Security Advisories when available, or contact the maintainer directly. Please include reproduction steps, expected impact, and affected configuration.
