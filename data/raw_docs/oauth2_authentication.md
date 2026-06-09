# OAuth2 Authentication and Authorization Policies

Our enterprise microservices architecture implements OAuth2 protocols to secure API endpoints. This document outlines token validation, signature verification, and claims-based scope verification.

## Token Validation Process

Every incoming request to a protected endpoint must include a Bearer token in the `Authorization` header:

```http
Authorization: Bearer <JWT_access_token>
```

The authentication filter intercepts the request and performs the following checks:

1. **Format Validation**: Ensures the token is a well-formed JSON Web Token (JWT) consisting of three Base64URL-encoded parts: Header, Payload, and Signature.
2. **Signature Verification**: Verifies the signature using the public keys retrieved from the Identity Provider's (IdP) JSON Web Key Set (JWKS) endpoint. The gateway caches these keys for 24 hours to reduce latency.
3. **Expiration check**: Rejects any token if the current system time is past the value of the `exp` (expiration) claim. We allow a clock skew window of 60 seconds.
4. **Audience & Issuer verification**: The `aud` (audience) claim must match the service ID, and the `iss` (issuer) claim must match the configured identity provider URL.

## Scope Verification

Authorization is checked using the `scope` claim in the JWT payload. The API Gateway maps required scopes to specific URI routes.

Example configuration in `authz.yml`:

```yaml
authorization:
  policies:
    - path: /api/v1/billing/*
      methods: [POST, PUT, DELETE]
      required_scopes: ["billing:write", "admin"]
    - path: /api/v1/billing/*
      methods: [GET]
      required_scopes: ["billing:read", "billing:write", "admin"]
    - path: /api/v1/users/*
      methods: [GET]
      required_scopes: ["profile:read", "admin"]
```

If a client attempts to access an endpoint without the required scope, the microservice returns an HTTP Status Code `403 Forbidden` response.

## Token Lifetime and Refresh Token Flow

Access tokens are short-lived, with a default expiration time of 15 minutes. To obtain a new access token without re-prompting the user, clients must perform the Refresh Token flow:

1. Send a `POST` request to the token endpoint `/oauth2/token` with:
   - `grant_type`: `refresh_token`
   - `refresh_token`: The long-lived refresh token (valid for 30 days)
   - `client_id` & `client_secret` (if confidential client)
2. The authorization server invalidates the old access token and issues a new access token (15 mins) and optionally a rotated refresh token.