# Service Mesh Routing and Traffic Management

Our platform uses Istio Service Mesh to orchestrate microservice communication, route traffic dynamically, and build resilient services. This guide covers VirtualServices, DestinationRules, and Circuit Breaker configuration.

## VirtualServices

A `VirtualService` defines a set of traffic routing rules to apply when a host is addressed. It enables traffic splitting, header-based routing, and URI path rewrites.

Example VirtualService configuration for a payment service:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: payment-service-routes
spec:
  hosts:
    - payment-service
  http:
    - match:
        - headers:
            version:
              exact: v2
      route:
        - destination:
            host: payment-service
            subset: v2
    - route:
        - destination:
            host: payment-service
            subset: v1
          weight: 90
        - destination:
            host: payment-service
            subset: v2
          weight: 10
```

In the configuration above:
- Requests containing the header `version: v2` are routed exclusively to subset `v2`.
- Regular traffic is split: 90% goes to `v1` and 10% goes to `v2` (Canary deployment).

## DestinationRules

A `DestinationRule` defines policies that apply to traffic intended for a service after routing has occurred. This is where you configure load balancing models, TLS settings, and circuit breakers.

Example DestinationRule with a Circuit Breaker:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: payment-service-policies
spec:
  host: payment-service
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 10
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

## Circuit Breaker Parameters

The `outlierDetection` block controls circuit breaking behavior:

- **consecutive5xxErrors**: The number of consecutive 5xx errors from a pod before it is ejected from the load-balancing pool. Set to `3` in the example.
- **interval**: The time interval for tracking error counts. Set to `10s`.
- **baseEjectionTime**: The duration that the faulty host is ejected from the pool. Set to `30s`. During this time, no traffic will be routed to it.
- **maxEjectionPercent**: The maximum percentage of pods for a service that can be ejected at the same time. This prevents the entire service pool from going down. Set to `50%`.