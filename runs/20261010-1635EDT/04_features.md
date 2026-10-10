# 04 — feature selection (exp-006-feature-selection)

Model **random_forest**, params re-tuned once on this experiment's training rows and then frozen: `{'clf__max_depth': None, 'clf__min_samples_leaf': 3, 'clf__n_estimators': 400}`.

Rule, declared before testing: keep a feature iff score(with) − score(without) ≥ **0.003**, on 5×3 repeated stratified CV with both arms on identical folds. SE is the standard error of the 15 paired fold differences; repeated-CV folds overlap, so it understates the true uncertainty.

- Baseline (12 features): **0.8360** (`20261010T203602-d38cdc`)
- Selected (6 features): **0.8401** (`20261010T203640-986301`)
- Tests run: 36 of a 50 cap — every planned test ran; 46s

| # | test | value (with − without) | paired SE | verdict | cv after |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | add `AgeBin` | -0.0029 | 0.0028 | drop | 0.8330 |
| 2 | add `IsChild` | -0.0006 | 0.0026 | drop | 0.8354 |
| 3 | add `WomanOrChild` | -0.0006 | 0.0030 | drop | 0.8354 |
| 4 | add `PclassCat` | -0.0047 | 0.0027 | drop | 0.8313 |
| 5 | add `SexPclass` | -0.0041 | 0.0028 | drop | 0.8319 |
| 6 | add `FamilyBucket` | -0.0035 | 0.0028 | drop | 0.8325 |
| 7 | add `FareLog` | -0.0012 | 0.0029 | drop | 0.8348 |
| 8 | add `FareBin` | -0.0047 | 0.0031 | drop | 0.8313 |
| 9 | add `FarePerPerson` | -0.0012 | 0.0039 | drop | 0.8348 |
| 10 | add `FareZero` | -0.0012 | 0.0030 | drop | 0.8348 |
| 11 | add `TitleFine` | +0.0012 | 0.0030 | drop | 0.8372 |
| 12 | add `TicketPrefix` | -0.0064 | 0.0042 | drop | 0.8296 |
| 13 | add `TicketNumeric` | -0.0017 | 0.0030 | drop | 0.8342 |
| 14 | add `TicketLength` | -0.0047 | 0.0037 | drop | 0.8313 |
| 15 | add `HasCabin` | -0.0006 | 0.0035 | drop | 0.8354 |
| 16 | add `CabinCount` | -0.0023 | 0.0031 | drop | 0.8336 |
| 17 | add `CabinSide` | -0.0064 | 0.0026 | drop | 0.8295 |
| 18 | add `DeckGroup` | -0.0029 | 0.0028 | drop | 0.8330 |
| 19 | add `NameLength` | -0.0000 | 0.0041 | drop | 0.8360 |
| 20 | add `NameHasParens` | -0.0006 | 0.0022 | drop | 0.8354 |
| 21 | add `IsMother` | -0.0018 | 0.0020 | drop | 0.8342 |
| 22 | add `AgeClass` | +0.0006 | 0.0046 | drop | 0.8366 |
| 23 | add `HasSibSp` | +0.0000 | 0.0024 | drop | 0.8360 |
| 24 | add `HasParch` | -0.0018 | 0.0021 | drop | 0.8342 |
| 25 | remove `Sex` | +0.0053 | 0.0048 | keep | 0.8307 |
| 26 | remove `Pclass` | +0.0094 | 0.0049 | keep | 0.8266 |
| 27 | remove `Age` | +0.0024 | 0.0028 | drop | 0.8336 |
| 28 | remove `Fare` | +0.0058 | 0.0034 | keep | 0.8278 |
| 29 | remove `SibSp` | +0.0006 | 0.0032 | drop | 0.8330 |
| 30 | remove `Parch` | -0.0012 | 0.0021 | drop | 0.8342 |
| 31 | remove `Embarked` | +0.0082 | 0.0047 | keep | 0.8260 |
| 32 | remove `FamilySize` | +0.0070 | 0.0025 | keep | 0.8272 |
| 33 | remove `IsAlone` | +0.0012 | 0.0019 | drop | 0.8330 |
| 34 | remove `Title` | +0.0152 | 0.0052 | keep | 0.8178 |
| 35 | remove `Deck` | -0.0047 | 0.0035 | drop | 0.8378 |
| 36 | remove `AgeMissing` | -0.0023 | 0.0022 | drop | 0.8401 |

**Added:** none  
**Removed:** Age, SibSp, Parch, IsAlone, Deck, AgeMissing  
**Final set:** Sex, Pclass, Fare, Embarked, FamilySize, Title

## Candidate rationale

- `AgeBin` — Life-stage buckets; survival is non-linear in age (children first), and a bucket with an explicit Unknown level does not depend on median imputation.
- `IsChild` — Age < 16. 'Women and children first' was the loading rule; a hard threshold says that directly.
- `WomanOrChild` — Female, or a boy (Master / under 16): the evacuation rule itself as one binary, rather than something the trees must reassemble.
- `PclassCat` — Pclass one-hot instead of ordinal. Tests whether 1->2->3 being evenly spaced is a fair assumption.
- `SexPclass` — Sex x class interaction. Third-class women survived at roughly half the rate of first-class women; the effect of sex depends on class.
- `FamilyBucket` — Alone / small (2-4) / large (5+). Family-size survival rises then collapses; three buckets capture the shape with few parameters.
- `FareLog` — log1p(Fare). Fare is heavily right-skewed; the log spreads the dense low end where most of the class-3 variation lives.
- `FareBin` — Fare bands in round pounds. A coarse wealth signal less sensitive to the handful of extreme fares.
- `FarePerPerson` — Fare / FamilySize. Fares were paid per ticket, often for a whole family, so the raw fare overstates individual wealth for large groups.
- `FareZero` — Fare == 0. Free passage (crew-adjacent staff, guarantee group) is a distinct population, not just a cheap ticket.
- `TitleFine` — Finer titles: Mlle/Ms->Miss, Mme->Mrs, and Clergy, Officer, Noble and Professional split out of 'Rare', which mixes 0% and 100% survival groups.
- `TicketPrefix` — Alphabetic ticket prefix (PC, CA, SOTON...). Encodes the issuing office / route, which tracks class and embarkation point.
- `TicketNumeric` — Ticket is purely numeric. A coarse version of the prefix with no rare levels.
- `TicketLength` — Digits in the ticket number. Ticket series were issued in ranges by office, so length is a proxy for where and how the ticket was sold.
- `HasCabin` — Cabin recorded at all. Cabin records survived mostly for first class and for survivors' accounts; presence is the signal, the value is sparse.
- `CabinCount` — Number of cabins on one booking. Several cabins means a wealthy family travelling together.
- `CabinSide` — Odd cabin numbers were starboard, even port. The two sides were loaded under different officers with different rules for men.
- `DeckGroup` — Decks pooled ABC / DE / FG / Unknown. Individual deck letters are too sparse to estimate; pooled decks share distance to the boat deck.
- `NameLength` — Characters in Name. Long names are formal ones: married women listed under the husband's name plus their own, and titled passengers.
- `NameHasParens` — Name contains a parenthesised part — almost always a married woman's own name, i.e. a wife travelling with her husband.
- `IsMother` — Female, not Miss, travelling with children (Parch > 0). Mothers were boarded with their children.
- `AgeClass` — Age x Pclass. Older passengers in third class fared worst; a product lets a single split separate young first-class from old third-class.
- `HasSibSp` — Any sibling or spouse aboard. Someone to look for, or to help.
- `HasParch` — Any parent or child aboard; distinct from siblings/spouse.
