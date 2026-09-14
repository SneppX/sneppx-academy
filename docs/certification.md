# SneppX Academy Certification

Certification is a paid add-on demonstrating completion of the SneppX
Academy curriculum and practical competence with the SneppX ecosystem.

## How it works

1. **Complete the modules**: finish modules 01-04 (free content) and pass
   each module quiz with ≥75%.
2. **Submit your work**: the certification form requires a link to your
   GitHub repository containing completed exercises for each module.
3. **Review**: a SneppX team member reviews your exercises against the
   rubric (correctness, code quality, documentation).
4. **Receive your certificate**: on approval you receive a signed
   certificate (PDF, Ed25519-signed via `sneppx-shield`) and a public
   badge on your GitHub profile.

## Modules covered

| Module | Topic | Status |
|--------|-------|--------|
| 01 | Tensor engine | Published |
| 02 | Autograd & nn | Outlined |
| 03 | Distributed training | Draft |
| 04 | AI security layers | Draft |

## Pricing

Certification pricing will be announced when modules 01-04 are published.
Early supporters (sponsor tier ≥ License) receive a discounted rate.

## Verification

Every certificate is signed with Ed25519 and can be verified with
`sneppx-shield verify`:

```bash
sneppx-shield verify certificate.pdf.sig --public-key <signer-key-hex>
```

This guarantees the certificate was issued by SneppX and has not been
tampered with.