# API Gateway Configuration Guide: Rate Limiting Middleware

Rate limiting is crucial for preventing denial-of-service (DoS) attacks, brute-force attempts, and API abuse. In our enterprise API Gateway, rate limiting is configured via the `limiter` middleware.

## Global Configuration

The `limiter` middleware can be applied globally to all routes or specific route groups. Under the hood, it utilizes a Redis token bucket algorithm to store client request states.

Here is an example configuration block inside `gateway.config.yml`:

```yaml
gateway:
  middlewares:
    limiter:
      enabled: true
      driver: redis
      redis_url: redis://localhost:6379/0
      rules:
        - key: client_ip
          rate: 100/m  # 100 requests per minute
          burst: 20
        - key: api_key
          rate: 1000/h # 1000 requests per hour
          burst: 100
```

## Configuration Parameters

1. **driver**: The backend store for tracking request counts. We currently support `redis` (recommended for production) and `in-memory` (for development testing only).
2. **redis_url**: The connection string to the Redis database when the `redis` driver is selected.
3. **rules**: An ordered list of evaluation rules applied to incoming requests.
   - **key**: The criteria used to identify the client. Valid options are `client_ip` (rate limits based on client IP address), `api_key` (rate limits based on the `X-API-Key` header), or `user_id` (rate limits based on the JWT subject claim).
   - **rate**: The maximum request threshold. Formatted as `count/interval` (e.g., `100/m` for 100 requests per minute, `10/s` for 10 requests per second, `5000/d` for 5000 requests per day).
   - **burst**: The capacity of the token bucket. Allows temporary burst traffic above the steady-state rate.

## Header injection

When the rate limit is exceeded, the gateway responds with HTTP Status Code `429 Too Many Requests`. Additionally, it injects the following rate limiting headers into the HTTP response:

- `X-RateLimit-Limit`: The maximum number of requests allowed within the current period.
- `X-RateLimit-Remaining`: The number of requests remaining in the current window.
- `X-RateLimit-Reset`: The Unix timestamp indicating when the current window resets.

## Custom Error Responses

You can customize the error message returned to clients when they hit the rate limit. Modify the `error_response` block:

```yaml
limiter:
  error_response:
    code: 429
    message: "Rate limit exceeded. Please wait before retrying."
    headers:
      Retry-After: 60
```