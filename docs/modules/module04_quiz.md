# Module 04 Quiz

1. What is an SBOM?
   a) Software Bill of Materials
   b) Security Binary Output Module
   c) Standard Backend Optimization Method

   **Answer: a**

2. Which algorithm does `sneppx-shield` use to sign model artifacts?
   a) SHA-1
   b) Ed25519
   c) CRC32

   **Answer: b**

3. What is a detached signature?
   a) A signature stored separately from the signed artifact
   b) A signature embedded in the model weights
   c) A signed certificate authority

   **Answer: a**

4. Why bind the signature to a `message_sha256` digest?
   a) So large models can be verified anywhere without copying the file
   b) To speed up training
   c) To encrypt the model

   **Answer: a**

5. Which framework governs AI compliance in the EU?
   a) EU AI Act
   b) GDPR only
   c) RFC 8032

   **Answer: a**

6. In `sneppx-shield audit --ci`, what does an annotation like `::error` do?
   a) Writes an audit to a database
   b) Emits a GitHub Actions annotation and exit code
   c) Deletes the model

   **Answer: b**

7. Why should you run `sneppx-shield keygen` with `--out` inside a restricted directory?
   a) So keys are stored in version control
   b) So the secret seed stays protected with file permissions
   c) To make keys reproducible

   **Answer: b**

8. What does `sneppx-shield sbom` produce?
   a) A list of files with size and sha256 for an artifact tree
   b) A compiler artifact
   c) A test log

   **Answer: a**

## Answer key
| Q | Answer |
|---|--------|
| 1 | a |
| 2 | b |
| 3 | a |
| 4 | a |
| 5 | a |
| 6 | b |
| 7 | b |
| 8 | a |