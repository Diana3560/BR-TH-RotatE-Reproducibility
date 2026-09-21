# Sanitized extraction examples

The examples below are **paraphrased structural examples**, not verbatim quotations from the regulation. They document how source semantics are converted into entities and relations while keeping the public package free of original labels/text. All pseudonymous triples listed below occur in the released `R14_train.tsv` file and can be directly verified using the reported row numbers.

| Example | Sanitized source meaning | Structured result | file location |
|---|---|---|---|
| 1 | A particular train-model variant is applicable to a maintenance component. | `(TM_001, APPLIES_TO, C_001)` | R14_train.tsv, row 15362 |
| 2 | A component is associated with a defined failure/defect mode. | `(C_031, HAS_FAILURE, F_001)` | R14_train.tsv, row 138 |
| 3 | A failure mode is constrained by a technical limit for a plate-thickness interval. | `(F_001, LIMITED_BY, S_003)` | R14_train.tsv, row 4228 |
| 4 | A technical standard applies when its corresponding thickness condition is satisfied. | `(S_003, APPLIES_UNDER, CR_003)` | R14_train.tsv, row 12840 |
| 5 | A failure/defect mode is handled through a designated repair plan. | `(F_001, FIXED_BY, PP_005)` | R14_train.tsv, row 4227 |
| 6 | A repair/inspection plan contains a specific process step. | `(PP_003, HAS_STEP, P_004)` | R14_train.tsv, row 8406 |
| 7 | After one process step, execution proceeds to the next defined step. | `(P_004, NEXT_STEP, P_005)` | R14_train.tsv, row 10551 |
| 8 | A process step requires a specified tool/resource. | `(P_014, REQUIRES, T_003)` | R14_train.tsv, row 10560 |
| 9 | A technical standard is associated with a specified disposition requirement. | `(S_003, HAS_DISPOSITION, DR_015)` | R14_train.tsv, row 12841 |

These examples are intended to make the extraction logic reviewable without releasing the private ID-to-original-name mapping.
