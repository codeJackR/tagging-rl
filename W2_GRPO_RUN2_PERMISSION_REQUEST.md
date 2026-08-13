# Run 2 confirmation-source permission request

**Purpose:** obtain auditable written authorization before accessing a
merchant's Shopify `products.json` endpoint for the locked Run 2 confirmation
study.

This is an operational template, not legal advice. Do not mark a source
approved from a reply that is ambiguous about the domain, automated access, or
intended uses.

## Message template

Subject: Permission request for low-rate product-metadata research access

Hello,

I am conducting a non-commercial research project that evaluates how accurately
language models assign controlled fashion attributes to product listings. I am
requesting written permission to access the public Shopify product endpoint for
`<DOMAIN>` at:

`https://<DOMAIN>/products.json?limit=250&page=<PAGE>`

If you approve, the collector will:

- make at most one request per second across all participating stores;
- identify itself with the repository's descriptive research user agent;
- stop the complete collection immediately on HTTP 403 or 429;
- collect listing metadata such as title, description, vendor, product type,
  tags, and image URLs;
- retain that metadata for research, human labeling, and model evaluation;
- never use the data to train the SFT or GRPO models in this study;
- exclude the store if permission is declined, revoked, or remains unclear.

Please confirm whether you authorize all four of the following for `<DOMAIN>`:

1. automated access to the endpoint above under the stated rate limit;
2. retention of the collected listing metadata for this research;
3. human annotation of that metadata into controlled fashion attributes;
4. use of the metadata and annotations to evaluate language models.

Separately, please state whether the raw collected listing metadata may be
published with the research artifacts. “No” is acceptable: the dataset can be
kept private while only hashes, protocol details, and aggregate results are
published.

If you approve, please include your role or authority to grant access and any
expiration date, additional restrictions, or revocation process.

Thank you.

## Accept/reject rule

A response can be recorded as `approved` only when it is written, names the
exact domain, comes from a person with a stated merchant role, and explicitly
covers all four requested scopes. A partial yes, unclear authority, unanswered
scope, or verbal-only response remains `unresolved`. A refusal or contradictory
terms remains `prohibited`.

Permission to publish raw metadata is recorded independently as `allowed` or
`not_allowed`; it is not required for private evaluation. The eventual storage
and Git policy must obey that choice.

## Evidence handling

For every response:

1. export the complete message thread, including headers and timestamps, to a
   stable file in a private evidence store;
2. do not commit personal email addresses or message contents to the public
   repository;
3. compute SHA-256 over the exact retained evidence file;
4. assign a stable private reference such as
   `run2-confirmation-permission-<domain>-v1`;
5. record the grantor's role, exact domain, grant UTC time, scopes, publication
   choice, restrictions, expiration, and evidence hash in a new terms audit;
6. have a second person check the transcription before changing the decision to
   `approved`;
7. rerun `python -m training.run2_confirmation_source_gate` and require at least
   eight approved domains before any product endpoint request.

## Approved-entry shape

```json
{
  "decision": "approved",
  "endpoint_domain": "store.example",
  "permission_basis": {
    "evidence_type": "written_merchant_authorization",
    "evidence_reference": "run2-confirmation-permission-store-example-v1",
    "evidence_sha256": "<64 lowercase hexadecimal characters>",
    "authorized_domain": "store.example",
    "granted_at_utc": "<ISO-8601 UTC timestamp>",
    "grantor_role": "<merchant role>",
    "scopes": [
      "automated_products_json_access",
      "research_retention",
      "human_labeling",
      "model_evaluation"
    ],
    "raw_metadata_publication": "allowed or not_allowed"
  }
}
```

The public terms URL, point-in-time review UTC timestamp, and concise evidence
summary remain required alongside this object.

## Current stopping condition

The 2026-08-13 terms audit has 20 candidates: 16 prohibited, 4 unresolved, and
0 approved. No request should be sent to a product endpoint until at least eight
new written approvals pass the executable gate. Merchant outreach itself is an
external action and requires explicit user authorization.
