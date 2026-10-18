# Fictional chatbot audit samples

These six small shop-policy cases are fictional. All shop rules, answers, and document IDs were invented for this project; they do not describe an actual company or verified source. Each case supplies relevant policy as a separate reference document. The gift-card case intentionally supplies two contradictory documents.

`cases.jsonl` contains a question, candidate answer, references, explicit requirements, and a manually assigned `expected_verdict` (`pass`, `fail`, or `review`). Examples cover a grounded delivery answer, invented return rules, missed reply instructions, incomplete cancellation information, a correct admission of unknown shipping coverage, and conflicting gift-card policies.

`demo_judgments.json` contains **illustrative, manually authored judgments, not actual model outputs**. Scores range from 0 to 4, and cited quotes are exact substrings of supplied references. Judgments omit verdicts: the evaluator derives `fail` if any score is at most 1; otherwise `review` if human review is requested or any score is 2; otherwise `pass`.

A correct admission that a policy does not contain an answer can pass. A faithful answer identifying contradictory policies can require review because the underlying business rule remains unresolved. The source-conflict findings explain that review need without alleging a false answer claim.

Expected labels and demo judgments were authored together for a reproducible demonstration. They are **not independent formal benchmark validation**, measured model accuracy, or evidence of judge reliability. Use independently reviewed cases and actual model runs before making performance claims.
