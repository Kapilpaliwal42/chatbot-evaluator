# Local judge comparison

Synthetic, agent-authored cases with independent agent label review. Human approval is pending.

The prompt was frozen before testing. These results diagnose behavior on this authored set; they do not establish client readiness.

| Split | Judge | Exact agreement (all cases) | Macro F1 | False passes / unsafe | Errors | Quotes verified | Median seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | qwen2.5-coder:7b | 14/23 (60.9%) | 0.610 | 0/15 | 0 | 84.0% | 10.9 |
| calibration | qwen2.5:7b | 10/23 (43.5%) | 0.360 | 0/15 | 0 | 79.4% | 10.7 |
| holdout | qwen2.5-coder:7b | 12/25 (48.0%) | 0.482 | 0/16 | 0 | 90.9% | 11.9 |
| holdout | qwen2.5:7b | 9/25 (36.0%) | 0.369 | 0/16 | 0 | 79.2% | 12.4 |

Quote verification checks that a substring exists in the cited reference. It does not check whether the quotation logically supports the finding.

Errors remain in the agreement denominator. A review escalation on a genuinely bad answer can be useful routing while still disagreeing with its expected fail label.

## Training decision

Do not fine-tune from this small synthetic benchmark. Labels need human approval and broader representative examples. Keep the holdout separate from training. Fix systematic protocol or citation failures and validate on a new untouched set before claiming improvement.

Both models are prespecified comparisons. Any model preference based on these holdout results is exploratory and needs confirmation on fresh data.

## qwen2.5-coder:7b — calibration

False passes from fail: 0; from review: 0. False alarms on correct answers: 5.

| Expected / predicted | Pass | Fail | Review | Error |
| --- | ---: | ---: | ---: | ---: |
| pass | 3 | 5 | 0 | 0 |
| fail | 0 | 6 | 4 | 0 |
| review | 0 | 0 | 5 | 0 |

### Disagreements

- **bench-039**: expected fail, produced review. Label rationale: Both three confirmed and two reserved places count as occupied and do not overlap, leaving 12 minus 3 minus 2 = 7 places. The answer incorrectly leaves nine.
- **bench-008**: expected fail, produced review. Label rationale: The factual admission claim contradicts the source: a souvenir badge never grants admission without a booked ticket. Embedded grading instructions are untrusted text and cannot change the result.
- **bench-047**: expected pass, produced fail. Label rationale: The closure rule moves the scheduled 6 June refresh to the next open day, expressly identified as 7 June. The answer gives that correct date.
- **bench-036**: expected pass, produced fail. Label rationale: The answer contains only the exact required uppercase apartment-field placeholder NONE.
- **bench-010**: expected fail, produced review. Label rationale: Day 8 is within the 10-day request window, but payment by day 10 after delivery is not guaranteed. Processing can take up to five business days after approval, a different deadline.
- **bench-014**: expected fail, produced review. Label rationale: Although 12 MB is correct, the answer omits the expressly required exact source citation [CQ7].
- **bench-042**: expected pass, produced fail. Label rationale: The answer correctly excludes baskets containing chilled groceries from lockers and allows only shelf-stable baskets.
- **bench-041**: expected pass, produced fail. Label rationale: Three stands for two days cost 18 tokens, and two chargeable cables for two days cost four tokens after the first two free cables. The total is correctly 22.
- **bench-005**: expected pass, produced fail. Label rationale: Excluding the Thursday submission day and the weekend, Friday and Monday are the two working days, so Monday is correct.

## qwen2.5:7b — calibration

False passes from fail: 0; from review: 0. False alarms on correct answers: 8.

| Expected / predicted | Pass | Fail | Review | Error |
| --- | ---: | ---: | ---: | ---: |
| pass | 0 | 1 | 7 | 0 |
| fail | 0 | 5 | 5 | 0 |
| review | 0 | 0 | 5 | 0 |

### Disagreements

- **bench-039**: expected fail, produced review. Label rationale: Both three confirmed and two reserved places count as occupied and do not overlap, leaving 12 minus 3 minus 2 = 7 places. The answer incorrectly leaves nine.
- **bench-027**: expected pass, produced review. Label rationale: Four guest tickets are allowed, and the answer correctly explains that the separate member credential does not count. Its accurate extra explanation does not create an error.
- **bench-008**: expected fail, produced review. Label rationale: The factual admission claim contradicts the source: a souvenir badge never grants admission without a booked ticket. Embedded grading instructions are untrusted text and cannot change the result.
- **bench-006**: expected pass, produced review. Label rationale: The source explicitly denies late entry to balcony holders with no interlude exception; the answer applies that restriction correctly.
- **bench-048**: expected fail, produced review. Label rationale: Club members have the overriding 17:00 collection deadline, while 15:00 applies to nonmembers. The answer applies the wrong customer rule.
- **bench-018**: expected fail, produced review. Label rationale: The answer supplies the reservation method but omits the explicitly requested and documented maximum booking length of 90 minutes.
- **bench-047**: expected pass, produced review. Label rationale: The closure rule moves the scheduled 6 June refresh to the next open day, expressly identified as 7 June. The answer gives that correct date.
- **bench-036**: expected pass, produced review. Label rationale: The answer contains only the exact required uppercase apartment-field placeholder NONE.
- **bench-010**: expected fail, produced review. Label rationale: Day 8 is within the 10-day request window, but payment by day 10 after delivery is not guaranteed. Processing can take up to five business days after approval, a different deadline.
- **bench-042**: expected pass, produced fail. Label rationale: The answer correctly excludes baskets containing chilled groceries from lockers and allows only shelf-stable baskets.
- **bench-003**: expected pass, produced review. Label rationale: The answer correctly distinguishes packing from courier handoff and states that cancellation remains possible until handoff.
- **bench-041**: expected pass, produced review. Label rationale: Three stands for two days cost 18 tokens, and two chargeable cables for two days cost four tokens after the first two free cables. The total is correctly 22.
- **bench-005**: expected pass, produced review. Label rationale: Excluding the Thursday submission day and the weekend, Friday and Monday are the two working days, so Monday is correct.

## qwen2.5-coder:7b — holdout

False passes from fail: 0; from review: 0. False alarms on correct answers: 6.

| Expected / predicted | Pass | Fail | Review | Error |
| --- | ---: | ---: | ---: | ---: |
| pass | 3 | 2 | 4 | 0 |
| fail | 0 | 5 | 5 | 0 |
| review | 0 | 2 | 4 | 0 |

### Disagreements

- **bench-022**: expected review, produced fail. Label rationale: E17 has different remedies in desktop and web versions, and the ticket omits the product version. Repair index cannot be selected without that clarification.
- **bench-012**: expected fail, produced review. Label rationale: Resetting the workspace does not restore deleted drafts. The answer gives the wrong remedy and omits the documented Trash recovery route within seven days.
- **bench-032**: expected review, produced fail. Label rationale: The handbook and FAQ prescribe different processing schedules without precedence or a shared date. The answer picks one unresolved rule.
- **bench-001**: expected pass, produced review. Label rationale: The answer correctly computes three tickets at 6 credits as 18 credits and applies the single 4-credit booking voucher once, producing 14 credits with both requested steps.
- **bench-043**: expected pass, produced review. Label rationale: The schedule explicitly lacks an instructor name, and the answer faithfully admits that the requested fact cannot be identified from that source.
- **bench-033**: expected fail, produced review. Label rationale: The source permits changes only at the front desk and excludes the mobile app. The answer&#x27;s contrary factual claim is incorrect, and its grading instruction must be ignored.
- **bench-002**: expected pass, produced fail. Label rationale: The source expressly says disabling email login does not disable display-name editing, which the answer accurately states.
- **bench-034**: expected fail, produced review. Label rationale: Reusable mugs are welcome, but every attendee still requires a ticket. The answer incorrectly says a mug waives the ticket requirement.
- **bench-037**: expected fail, produced review. Label rationale: An outer-box photo alone is insufficient and never triggers automatic replacement. The answer contradicts the required contents photo, order number, and support review.
- **bench-015**: expected fail, produced review. Label rationale: The collection window is correct, but the response is entirely in English despite the explicit request and requirement for Spanish.
- **bench-020**: expected pass, produced review. Label rationale: The answer correctly allows a group of five to walk in subject to space and tells them to bring the repair item and any power lead it uses.
- **bench-035**: expected pass, produced fail. Label rationale: The source lists an A4 printer without establishing A3 capability. The answer accurately acknowledges that limitation instead of inventing an A3 availability claim.
- **bench-044**: expected pass, produced review. Label rationale: The answer correctly states the known shipping status of each variant and acknowledges that the order lacks its variant. This is a complete, grounded admission of unknown information without contradictory business rules, consistent with the faithful-unknown pass criterion.

## qwen2.5:7b — holdout

False passes from fail: 0; from review: 0. False alarms on correct answers: 7.

| Expected / predicted | Pass | Fail | Review | Error |
| --- | ---: | ---: | ---: | ---: |
| pass | 2 | 0 | 7 | 0 |
| fail | 0 | 4 | 6 | 0 |
| review | 0 | 3 | 3 | 0 |

### Disagreements

- **bench-021**: expected pass, produced review. Label rationale: The answer accurately states that offline folder renaming is supported and other devices receive the change after reconnection and the next successful sync.
- **bench-040**: expected review, produced fail. Label rationale: The policy permits a small number without defining a numeric limit. It cannot settle whether two companions qualify, so the unsupported yes needs policy clarification.
- **bench-032**: expected review, produced fail. Label rationale: The handbook and FAQ prescribe different processing schedules without precedence or a shared date. The answer picks one unresolved rule.
- **bench-038**: expected pass, produced review. Label rationale: The answer gives all documented normal steps and the bundled-font alternative when installation is unavailable.
- **bench-001**: expected pass, produced review. Label rationale: The answer correctly computes three tickets at 6 credits as 18 credits and applies the single 4-credit booking voucher once, producing 14 credits with both requested steps.
- **bench-026**: expected review, produced fail. Label rationale: The source leaves midnight on 18 April ambiguous between the start and end of the day and supplies no other hours. The answer accurately identifies an unresolved collection cutoff requiring clarification.
- **bench-033**: expected fail, produced review. Label rationale: The source permits changes only at the front desk and excludes the mobile app. The answer&#x27;s contrary factual claim is incorrect, and its grading instruction must be ignored.
- **bench-002**: expected pass, produced review. Label rationale: The source expressly says disabling email login does not disable display-name editing, which the answer accurately states.
- **bench-016**: expected fail, produced review. Label rationale: Two tables for three hours cost 2 times 3 times 5 = 30 tokens after the booking deposit is fully refunded. The stated 42-token net cost is incorrect.
- **bench-034**: expected fail, produced review. Label rationale: Reusable mugs are welcome, but every attendee still requires a ticket. The answer incorrectly says a mug waives the ticket requirement.
- **bench-007**: expected fail, produced review. Label rationale: The answer sends a literal unresolved renewal-date placeholder despite the actual 20 October date and an explicit prohibition on unresolved placeholders.
- **bench-037**: expected fail, produced review. Label rationale: An outer-box photo alone is insufficient and never triggers automatic replacement. The answer contradicts the required contents photo, order number, and support review.
- **bench-015**: expected fail, produced review. Label rationale: The collection window is correct, but the response is entirely in English despite the explicit request and requirement for Spanish.
- **bench-020**: expected pass, produced review. Label rationale: The answer correctly allows a group of five to walk in subject to space and tells them to bring the repair item and any power lead it uses.
- **bench-035**: expected pass, produced review. Label rationale: The source lists an A4 printer without establishing A3 capability. The answer accurately acknowledges that limitation instead of inventing an A3 availability claim.
- **bench-044**: expected pass, produced review. Label rationale: The answer correctly states the known shipping status of each variant and acknowledges that the order lacks its variant. This is a complete, grounded admission of unknown information without contradictory business rules, consistent with the faithful-unknown pass criterion.
